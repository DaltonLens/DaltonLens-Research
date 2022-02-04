#!/usr/bin/env python3

from charts.common.dataset import LabeledImage
import charts.pytorch.segmentation_transforms as segmentation_transforms

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms

import timm

import cv2

import numpy as np

from pathlib import Path
import sys

from charts.common.utils import swap_rb

from icecream import ic

from typing import List

class ImagePreprocessor:   
    def __init__(self, device: torch.device, target_size: int = None):
        Both = segmentation_transforms.ApplyOnBoth
        self.device = device
        transform_list = [
            Both(transforms.ToTensor()),
            # Both(transforms.Lambda(lambda x: x.to(device))),
            Both(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        ]
        if target_size is not None:
            transform_list.append(segmentation_transforms.RandomCropWithRegressionLabels(target_size))
        
        self.transform = segmentation_transforms.Compose(transform_list)

    def denormalize_and_clip_as_tensor (self, im: Tensor) -> Tensor:
        return torch.clip(im * 0.5 + 0.5, 0.0, 1.0)

    def denormalize_and_clip_as_numpy (self, im: Tensor) -> np.ndarray:
        return self.denormalize_and_clip_as_tensor(im).permute(1, 2, 0).detach().cpu().numpy()

class ColorRegressionImageDataset(Dataset):
    def __init__(self, img_dir, preprocessor: ImagePreprocessor, max_length=sys.maxsize):
        self.img_dir = img_dir
        self.transform = preprocessor.transform
        self.max_length = max_length

        json_files = sorted(img_dir.glob("img-?????-???.json"))
        self.labeled_images = [LabeledImage(f) for f in json_files]

    def __len__(self):
        return min(self.max_length, len(self.labeled_images))

    def __getitem__(self, idx):
        labeled_img = self.labeled_images[idx]
        labeled_img.ensure_images_loaded()
        labeled_img.compute_labels_as_rgb()
        image = labeled_img.rendered_image
        labels_image = labeled_img.labels_as_rgb
        labeled_img.release_images()
        if self.transform:
            image, labels_image = self.transform(image, labels_image)
        assert (image is not None)
        return image, labels_image, repr(self.labeled_images[idx])

    def __repr__(self):
        return f"{len(self)} images, first is {self.labeled_images[0]}, last is {self.labeled_images[-1]}"

class ResnetBlock (nn.Module):
    def __init__(self, in_features, out_features, first_stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=first_stride, padding=3//2, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=3//2, bias=False),
            nn.BatchNorm2d(out_features),
        )

        self.downsample_x = None
        if first_stride != 1 or out_features != in_features:
            self.downsample_x = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=first_stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        x_adjusted = x if self.downsample_x is None else self.downsample_x(x)
        y = self.block(x)
        return x_adjusted + y

class UnetBlock(nn.Module):
    def __init__(self, left_features, down_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(left_features + down_features, down_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_features),
            nn.ReLU(),

            nn.Conv2d(down_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

    def forward(self, left, down):
        down_upsampled = F.interpolate(down, scale_factor=2.0)
        x = torch.cat((down_upsampled, left), dim=1)
        return self.block (x)

class UnetDecoder(nn.Module):
    def __init__(self, residual_mode):
        super().__init__()
        self.residual_mode = residual_mode
        self.enc0 = ResnetBlock(3, 64)

        self.dec4     = UnetBlock(512,512, out_features=512)
        self.dec3     = UnetBlock(256,512, out_features=256)
        self.dec2     = UnetBlock(128,256, out_features=128)
        self.dec1     = UnetBlock(64,128,  out_features=64)
        self.dec1_act = UnetBlock(64,64,   out_features=64)
        self.dec0     = UnetBlock(64,64,   out_features=64)

        self.bottom = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),            
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        n_classes = 3 # RGB
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1x1 conv
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0),
        )

    def set_extra_state(self, state):
        self.residual_mode = state['residual_mode']

    def get_extra_state(self):
        return {'residual_mode': self.residual_mode}

    def forward(self, img, encoded_layers: List[torch.Tensor]):
        act1 = encoded_layers[0]
        layer1 = encoded_layers[1]
        layer2 = encoded_layers[2]
        layer3 = encoded_layers[3]
        layer4 = encoded_layers[4]

        x = self.bottom(layer4)
        x = self.dec4(layer4, x)
        x = self.dec3(layer3, x)
        x = self.dec2(layer2, x)
        x = self.dec1(layer1, x)
        x = self.dec1_act(act1, x)
        x = self.dec0(self.enc0(img), x)
        
        if self.residual_mode:
            return self.head(x) + img
        else:
            return self.head(x)
class RegressionNet_Unet1(nn.Module):
    def __init__(self, residual_mode = False):
        super().__init__()
        self.encoder = timm.create_model('resnet18', features_only=True, pretrained=True, scriptable=True)
        self.decoder = UnetDecoder(residual_mode)
        # {'module': 'act1', 'num_chs': 64, 'reduction': 2},
        # {'module': 'layer1', 'num_chs': 64, 'reduction': 4},
        # {'module': 'layer2', 'num_chs': 128, 'reduction': 8},
        # {'module': 'layer3', 'num_chs': 256, 'reduction': 16},
        # {'module': 'layer4', 'num_chs': 512, 'reduction': 32}]

        # act1 torch.Size([8, 64, 48, 64]),
        # layer1 torch.Size([8, 64, 24, 32]),
        # layer2 torch.Size([8, 128, 12, 16]),
        # X torch.Size([8, 256, 6, 8]),
        # X torch.Size([8, 512, 3, 4])

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, img):
        encoded_layers = self.encoder(img)
        return self.decoder(img, encoded_layers)

def compute_average_loss (dataset_loader: DataLoader, net: nn.Module, criterion: nn.Module, device: torch.device):
    with torch.no_grad():
        running_loss = 0.0
        for data in dataset_loader:
            inputs, labels = [x.to(device) for x in data[0:2]]
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataset_loader)

def compute_accuracy (dataset_loader: DataLoader, net: nn.Module, criterion: nn.Module, device: torch.device):
    with torch.no_grad():
        num_good = 0
        num_pixels = 0
        for data in dataset_loader:
            inputs, labels = [x.to(device) for x in data[0:2]]
            outputs = net(inputs)
            diff = torch.abs(outputs-labels)
            max_diff = torch.max(diff, dim=1)[0]
            num_good += torch.count_nonzero(max_diff < ((20/255.0)*2.0)).item()
            num_pixels += max_diff.numel()
    return num_good / num_pixels

class Processor:
    def __init__(self, torchscript_model_pt: Path):
        self.device = torch.device("cpu")
        self.preprocessor = ImagePreprocessor(self.device)
        # self.net = torch.load (network_model_pt, map_location=self.device)
        self.net = torch.jit.load(torchscript_model_pt, map_location=self.device)
        self.net.eval()
        # In case only a checkpoint was saved, and not the full model
        # self.net = RegressionNet_Unet1(residual_mode=True)
        # checkpoint = torch.load (network_model_pt, map_location=self.device)
        # self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.net.eval()
        # torch.save(self.net, network_model_pt.with_suffix('.model.pt'))

    def process_image(self, image_rgb: np.ndarray):
        input: Tensor = self.preprocessor.transform (image_rgb, image_rgb)[0]
        if (image_rgb.shape[0] % 32 != 0) or (image_rgb.shape[1] % 32 != 0):
            input = transforms.CenterCrop(128)(input)
        input.unsqueeze_ (0) # add the batch dim
        output = self.net (input)
        output_im = self.preprocessor.denormalize_and_clip_as_numpy (output[0])
        output_im = (output_im * 255).astype(np.uint8)

        input_cropped = self.preprocessor.denormalize_and_clip_as_numpy (input[0])
        input_cropped = (input_cropped * 255).astype(np.uint8)
        
        cv2.namedWindow("original - filtered", cv2.WINDOW_NORMAL)
        cv2.imshow ('original - filtered', np.hstack([input_cropped, output_im]))

        return output_im

if __name__ == "__main__":
    processor = Processor(Path(__file__).parent / "regression_unetres_v1_scripted.pt")
    im_rgb = swap_rb(cv2.imread(sys.argv[1], cv2.IMREAD_COLOR))
    output_image = processor.process_image (im_rgb)
    cv2.imwrite("output.png", swap_rb(output_image))
    cv2.waitKey(0)

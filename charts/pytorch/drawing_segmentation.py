#!/usr/bin/env python3

from charts.common.dataset import Lab_from_sRGB, LabeledImage

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms

import cv2

import numpy as np
import math

from pathlib import Path
import sys

from charts.common.utils import swap_rb
from charts.pytorch.similar_colors import ImagePreprocessor, ColorRegressionImageDataset

nbins_L = 8
nbins_ab = 16
class Preprocessor:
    def __init__(self, device: torch.device):
        self.device = device
        self.transform_srgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(self.device)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform_lab = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(self.device)),
            # Normalize as if it was a uniform distribution
            transforms.Normalize((50.0, 0., 0.), (100.0/math.sqrt(12), 255/math.sqrt(12), 255/math.sqrt(12))),
        ])

        # No need for normalization, it expects the class index.
        self.target_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(self.device)),
        ])
    
    def denormalize_and_clip_as_tensor (self, im: Tensor) -> Tensor:
        return torch.clip(im * 0.5 + 0.5, 0.0, 1.0)

    def denormalize_and_clip_as_numpy (self, im: Tensor) -> np.ndarray:
        return self.denormalize_and_clip_as_tensor(im).permute(1, 2, 0).detach().cpu().numpy()

    def quantize_bgr2lab(self, im_bgr: np.ndarray) -> np.ndarray:
        # L [0, 100] a and b [-127,127]
        im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)
        im_lab[0] = np.around(im_lab[0] / (100.0/nbins_L)) # -> 0 and 1    
        im_lab[1:3] = (im_lab[1:3] + 127.0) / (254.0/nbins_ab) # -> 0 and 1
        return np.around(im_lab)

class DrawingSegmentationDataset(Dataset):
    def __init__(self, img_dir, preprocessor: Preprocessor, max_length=sys.maxsize):
        self.img_dir = img_dir
        self.preprocessor = preprocessor
        self.max_length = max_length

        json_files = sorted(img_dir.glob("img-?????-???.json"))
        self.labeled_images = [LabeledImage(f) for f in json_files]

    def __len__(self):
        return min(self.max_length, len(self.labeled_images))

    def __getitem__(self, idx):
        labeled_img = self.labeled_images[idx]
        labeled_img.ensure_images_loaded()
        labeled_img.compute_quantized_Lab() # we need that

        image_srgb_uint8 = labeled_img.rendered_image
        
        # FIXME: do it on the GPU!
        image_lab = Lab_from_sRGB(image_srgb_uint8).astype(np.float32)

        drawing_segmentation = labeled_img.labels_image
        drawing_segmentation = np.where(drawing_segmentation > 0, 1, 0)
        drawing_segmentation = torch.as_tensor(drawing_segmentation, dtype=torch.long)

        L_target = labeled_img.labels_quantized_Lab[...,0]
        L_target = torch.as_tensor(L_target, dtype=torch.long)

        a_target = labeled_img.labels_quantized_Lab[...,0]
        a_target = torch.as_tensor(a_target, dtype=torch.long)

        b_target = labeled_img.labels_quantized_Lab[...,0]
        b_target = torch.as_tensor(b_target, dtype=torch.long)

        labeled_img.release_images()
        
        image_srgb_uint8 = self.preprocessor.transform_srgb(image_srgb_uint8)
        image_lab = self.preprocessor.transform_lab(image_lab)

        drawing_segmentation = self.preprocessor.target_transform(drawing_segmentation)
        L_target = self.preprocessor.target_transform(L_target)
        a_target = self.preprocessor.target_transform(a_target)
        b_target = self.preprocessor.target_transform(b_target)

        assert (image_srgb_uint8 is not None)
        return (image_srgb_uint8, image_lab), (drawing_segmentation, L_target, a_target, b_target), repr(self.labeled_images[idx])

    def __repr__(self):
        return f"{len(self)} images, first is {self.labeled_images[0]}, last is {self.labeled_images[-1]}"

class DrawSegmentation_Unet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.downThenUp = nn.Sequential(
            nn.Conv2d(6, 16, 5, padding='same'),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 5, padding='same'),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, 5, padding='same'),
            nn.ReLU(inplace=True),
        )

        # self.final_conv = nn.Conv2d(16 + 3, 32, 5, padding='same')
        self.drawing_conv = nn.Sequential (
            # +3 because of the concatenation.
            nn.Conv2d(16 + 6, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 2, 5, padding='same'),
            nn.ReLU(),
        )

        self.L_conv = nn.Sequential (
            # +3 because of the concatenation.
            nn.Conv2d(16 + 6, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 64, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, nbins_L, 5, padding='same'),
            nn.ReLU(),
        )

        self.a_conv = nn.Sequential (
            # +3 because of the concatenation.
            nn.Conv2d(16 + 6, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 64, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, nbins_ab, 5, padding='same'),
            nn.ReLU(),
        )

        self.b_conv = nn.Sequential (
            # +3 because of the concatenation.
            nn.Conv2d(16 + 6, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 128, 5, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 64, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, nbins_ab, 5, padding='same'),
            nn.ReLU(),
        )

    def combine(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        diffX_div2 = torch.div(diffX, 2, rounding_mode='floor')
        diffY_div2 = torch.div(diffY, 2, rounding_mode='floor')


        x1 = F.pad(x1, [diffX_div2, diffX - diffX_div2,
                        diffY_div2, diffY - diffY_div2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x_rgb, x_lab):
        # x = self.pool(self.conv1(x))
        # x = self.up(x)

        x = torch.concat((x_rgb, x_lab), dim=1)

        x_features = self.downThenUp(x)
        x = self.combine(x_features, x)
        drawing = self.drawing_conv(x)
        L = self.L_conv(x)
        a = self.a_conv(x)
        b = self.b_conv(x)        
        return (drawing, L, a, b)

def compute_average_loss (dataset_loader: DataLoader, net: nn.Module, criterion: nn.Module):
    with torch.no_grad():
        running_loss = 0.0
        for data in dataset_loader:
            inputs, targets, _ = data
            outputs = net(*inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataset_loader)

class DrawingSegmentor:
    def __init__(self, network_model_pt: Path):
        self.device = torch.device("cpu")
        self.preprocessor = ImagePreprocessor(self.device)
        self.net = torch.load (network_model_pt, map_location=self.device)

    def process_image(self, image_rgb: np.ndarray):
        input: Tensor = self.preprocessor.transform (image_rgb)
        input.unsqueeze_ (0) # add the batch dim
        output = self.net (input)
        output_im = self.preprocessor.denormalize_and_clip_as_numpy (output[0])
        output_im = (output_im * 255).astype(np.uint8)
        return output_im

if __name__ == "__main__":
    processor = DrawingSegmentor(Path(__file__).parent / "drawing_segmentation_v1.pt")
    im_rgb = swap_rb(cv2.imread(sys.argv[1], cv2.IMREAD_COLOR))
    output_image = processor.process_image (im_rgb)
    cv2.imwrite("output.png", swap_rb(output_image))

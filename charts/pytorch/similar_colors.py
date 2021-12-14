from charts.common.dataset import LabeledImage

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms

import numpy as np

import sys

class ImagePreprocessor:   
    def __init__(self, device: torch.device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(self.device)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def denormalize_and_clip_as_tensor (self, im: Tensor) -> Tensor:
        return torch.clip(im * 0.5 + 0.5, 0.0, 1.0)

    def denormalize_and_clip_as_numpy (self, im: Tensor) -> np.ndarray:
        return self.denormalize_and_clip_as_tensor(im).permute(1, 2, 0).detach().cpu().numpy()

class ColorRegressionImageDataset(Dataset):
    def __init__(self, img_dir, preprocessor: ImagePreprocessor, max_length=sys.maxsize):
        self.img_dir = img_dir
        self.transform = preprocessor.transform
        self.target_transform = preprocessor.transform
        self.max_length = max_length

        json_files = sorted(img_dir.glob("img-?????-???.json"))
        self.labeled_images = list(map(LabeledImage, json_files))

    def __len__(self):
        return min(self.max_length, len(self.labeled_images))

    def __getitem__(self, idx):
        labeled_img = self.labeled_images[idx]
        labeled_img.ensure_images_loaded()
        image = labeled_img.rendered_image
        labels_image = labeled_img.labels_as_rgb
        labeled_img.release_images()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels_image = self.target_transform(labels_image)
        assert (image is not None)
        return image, labels_image, repr(self.labeled_images[idx])

    def __repr__(self):
        return f"{len(self)} images, first is {self.labeled_images[0]}, last is {self.labeled_images[-1]}"

class RegressionNet_Unet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.downThenUp = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding='same'),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 5, padding='same'),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, 5, padding='same'),
            nn.ReLU(inplace=True),
        )

        # self.final_conv = nn.Conv2d(16 + 3, 32, 5, padding='same')
        self.final_conv = nn.Sequential (
            # +3 do to the concatenation.
            nn.Conv2d(16 + 3, 32, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, padding='same')
        )

        # # self.conv1.weight.data[...] = 1.0 / (5*5)
        # nn.init.constant_(self.conv1.bias, 0.0)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)
        # # nn.init.constant_(self.up.weight, 0.001)
        # nn.init.constant_(self.up.bias, 0.0)

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

    def forward(self, x):
        # x = self.pool(self.conv1(x))
        # x = self.up(x)
        x_features = self.downThenUp(x)
        x = self.combine(x_features, x)
        x = self.final_conv(x)
        return x

def compute_average_loss (dataset_loader: DataLoader, net: nn.Module, criterion: nn.Module):
    with torch.no_grad():
        running_loss = 0.0
        for data in dataset_loader:
            inputs, labels, _ = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataset_loader)

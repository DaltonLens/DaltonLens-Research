#!/usr/bin/env python3

from ..common.dataset import LabeledImage
from . import segmentation_transforms

from zv.log import zvlog

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

from ..common.utils import swap_rb

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
            transform_list += [
                segmentation_transforms.RandomCropWithRegressionLabels(target_size),
                segmentation_transforms.RandomHorizontalFlip(0.5),
            ]
        
        self.transform = segmentation_transforms.Compose(transform_list)

    def denormalize_and_clip_as_tensor (self, im: Tensor) -> Tensor:
        return torch.clip(im * 0.5 + 0.5, 0.0, 1.0)

    def denormalize_and_clip_as_numpy (self, im: Tensor) -> np.ndarray:
        return self.denormalize_and_clip_as_tensor(im).permute(1, 2, 0).detach().cpu().numpy()

class ColorRegressionImageDataset(Dataset):
    def __init__(self, img_dir, preprocessor: ImagePreprocessor, max_length=sys.maxsize):
        self.img_dir = img_dir
        self.preprocessor = preprocessor
        self.transform = preprocessor.transform
        self.max_length = max_length

        json_files = sorted(img_dir.glob("img-?????*.json"))
        if not json_files:
            raise Exception("No files in dataset {img_dir}")
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
            # zvlog.image ("original", image)
            image, labels_image = self.transform(image, labels_image)
            # zvlog.image ("augmented", self.preprocessor.denormalize_and_clip_as_numpy(image))
        assert (image is not None)
        return image, labels_image, repr(self.labeled_images[idx])

    def __repr__(self):
        return f"{len(self)} images, first is {self.labeled_images[0]}, last is {self.labeled_images[-1]}"

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
        if (image_rgb.shape[0] % 64 != 0) or (image_rgb.shape[1] % 64 != 0):
            new_size_y, new_size_x = 64 * (image_rgb.shape[0] // 64), 64 * (image_rgb.shape[1] // 64)
            ic(new_size_x, new_size_y)
            input = transforms.CenterCrop(min(new_size_x, new_size_y))(input)
        input.unsqueeze_ (0) # add the batch dim
        output = self.net (input)
        output_im = self.preprocessor.denormalize_and_clip_as_numpy (output[0])
        output_im = (output_im * 255).astype(np.uint8)

        input_cropped = self.preprocessor.denormalize_and_clip_as_numpy (input[0])
        input_cropped = (input_cropped * 255).astype(np.uint8)
        
        zvlog.image ("original", input_cropped)
        zvlog.image ("filtered", output_im)
        return output_im, input_cropped

if __name__ == "__main__":
    processor = Processor(Path(__file__).parent / "regression_unetres_v1_scripted.pt")
    im_rgb = swap_rb(cv2.imread(sys.argv[1], cv2.IMREAD_COLOR))
    output_image = processor.process_image (im_rgb)
    cv2.imwrite("output.png", swap_rb(output_image))
    cv2.waitKey(0)

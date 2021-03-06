#!/usr/bin/env python3

from collections import namedtuple

from dlcharts.pytorch.models_regression import RegressionModelOutput

from ..common.dataset import LabeledImage
from . import segmentation_transforms
from .segmentation_transforms import ApplyOnAll, ApplyOnFloatOnly

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

from ..common.utils import swap_rb

from icecream import ic

from dataclasses import dataclass
from pathlib import Path
import sys
import typing
from typing import List, NamedTuple, Dict, Tuple

class ImagePreprocessor:   
    class ToTensor:
        def __init__(self):
            self.to_tensor = transforms.ToTensor()

        def __call__(self, images: List):
            assert len(images) == 3
            return [self.to_tensor(images[0]), self.to_tensor(images[1]), torch.from_numpy(images[2])]

    def __init__(self, device: torch.device, do_augmentations: bool, cropping_border: int = 32):

        self.device = device
        transform_list = [
            self.ToTensor(),
            # Both(transforms.Lambda(lambda x: x.to(device))),
            ApplyOnFloatOnly(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        ]
        if do_augmentations:
            transform_list += [
                segmentation_transforms.RandomCropWithRegressionLabels(cropping_border),
                segmentation_transforms.RandomHorizontalFlip(0.5),
                segmentation_transforms.RandomVerticalFlip(0.5),
                # segmentation_transforms.RandomRotate90(0.2),
            ]
        
        self.transform = segmentation_transforms.Compose(transform_list)

    def denormalize_and_clip_as_tensor (self, im: Tensor) -> Tensor:
        return torch.clip(im * 0.5 + 0.5, 0.0, 1.0)

    def denormalize_and_clip_as_numpy (self, im: Tensor) -> np.ndarray:
        return np.ascontiguousarray(self.denormalize_and_clip_as_tensor(im).permute(1, 2, 0).detach().cpu().numpy())



class ColorRegressionImageDataset(Dataset):
    class Sample(NamedTuple):
        image: torch.FloatTensor
        labels_rgb: torch.FloatTensor
        labels_mask: torch.ByteTensor
        random_fg_label: int
        source: str

        def to(self, device: torch.device):
            return ColorRegressionImageDataset.Sample(
                image=self.image.to(device), 
                labels_rgb=self.labels_rgb.to(device),
                labels_mask=self.labels_mask.to(device),
                random_fg_label=self.random_fg_label,
                source=self.source)

    # Make it compatible with ClusteredDataset
    cluster_index: Tuple[int,int]
    debug: bool = False

    def __init__(self, img_dir, preprocessor: ImagePreprocessor, max_length=sys.maxsize):
        self.img_dir = img_dir
        self.preprocessor = preprocessor
        self.transform = preprocessor.transform
        self.max_length = max_length

        json_files = sorted(img_dir.glob("*.json"))
        if not json_files:
            raise Exception("No files in dataset {img_dir}")
        self.labeled_images = [LabeledImage(f) for f in json_files]
        self.labeled_images[0].ensure_images_loaded()
        self.rows = self.labeled_images[0].labels_image.shape[0]
        self.cols = self.labeled_images[0].labels_image.shape[1]
        self.cluster_index = (self.cols, self.rows)
        self.labeled_images[0].release_images()

    def __len__(self):
        return min(self.max_length, len(self.labeled_images))

    def __getitem__(self, idx) -> Sample:
        labeled_img = self.labeled_images[idx]
        labeled_img.ensure_images_loaded()
        labeled_img.compute_labels_as_rgb()
        image = labeled_img.rendered_image
        labels_image = labeled_img.labels_as_rgb
        labels_mask = labeled_img.labels_image
        # This guy could be used to compute a deviation loss on one label.
        # Pick semi-randomly to avoid issues where e.g. the fist fg label
        # is always the grid.
        fg_label_idx = 1 + (idx % (len(labeled_img.json['labels']) - 1))
        assert fg_label_idx != 0
        random_fg_label = labeled_img.json['labels'][fg_label_idx]
        labeled_img.release_images()
        if self.transform:
            if self.debug:
                zvlog.image ("original", image)
                zvlog.image ("original-labels-rgb", labels_image)
                zvlog.image ("original-labels-mask", labels_mask)
            image, labels_image, labels_mask = self.transform([image, labels_image, labels_mask])
            if self.debug:
                zvlog.image ("augmented", self.preprocessor.denormalize_and_clip_as_numpy(image))
                zvlog.image ("augmented-labels", self.preprocessor.denormalize_and_clip_as_numpy(labels_image))
                zvlog.image ("augmented-labels-mask", labels_mask.detach().numpy())
        assert (image is not None)
        return self.Sample(image=image,
                           labels_rgb=labels_image,
                           labels_mask=labels_mask,
                           random_fg_label=random_fg_label,
                           source=repr(self.labeled_images[idx]))

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

def pad_width (size: int, multiple: int):
    return 0 if size % multiple == 0 else multiple - (size%multiple)

# pad image to a multiple of 64
def pad_image(im, multiple=64):
    # H,W,C
    rows = im.shape[0]
    cols = im.shape[1]
    rows_to_pad = pad_width(rows, multiple)
    cols_to_pad = pad_width(cols, multiple)
    if rows_to_pad == 0 and cols_to_pad == 0:
        return im
    return np.pad (im, ((0, rows_to_pad), (0, cols_to_pad), (0, 0)), mode='reflect')

class Processor:
    def __init__(self, model_or_torchscript):
        self.device = torch.device("cpu")
        self.preprocessor = ImagePreprocessor(self.device, do_augmentations=False)
        # self.net = torch.load (network_model_pt, map_location=self.device)
        if isinstance(model_or_torchscript, torch.nn.Module):
            self.net = model_or_torchscript.to (self.device)
        else:
            self.net = torch.jit.load(model_or_torchscript, map_location=self.device)
        self.net.eval()
        # In case only a checkpoint was saved, and not the full model
        # self.net = RegressionNet_Unet1(residual_mode=True)
        # checkpoint = torch.load (network_model_pt, map_location=self.device)
        # self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.net.eval()
        # torch.save(self.net, network_model_pt.with_suffix('.model.pt'))

    def process_image(self, image_rgb_raw: np.ndarray):
        raw_cols, raw_rows = image_rgb_raw.shape[:2]
        too_big = (raw_cols > 1280) or (raw_rows > 1280)
        if too_big:
            if raw_rows > raw_cols:
                target_size = (int(1280*raw_cols/raw_rows), 1280)
            else:
                target_size = (1280, int(1280*raw_rows/raw_cols))
            image_rgb = cv2.resize (image_rgb_raw, target_size)
        else:
            image_rgb = image_rgb_raw
        image_rgb = pad_image (image_rgb)
        
        # Faking the GT data.
        fake_target_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
        fake_fg_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        input: Tensor = list(self.preprocessor.transform ([image_rgb, fake_target_rgb, fake_fg_mask]))[0]
        input.unsqueeze_ (0) # add the batch dim
        output = self.net (input)
        if isinstance(output, tuple): # RegressionModelOutput downcasted to tuple by torch.jit
            output_im = self.preprocessor.denormalize_and_clip_as_numpy (output.rgb[0])
        else:
            output_im = self.preprocessor.denormalize_and_clip_as_numpy (output[0])
        output_im = (output_im * 255).astype(np.uint8)

        input_cropped = self.preprocessor.denormalize_and_clip_as_numpy (input[0])
        input_cropped = (input_cropped * 255).astype(np.uint8)
        output_im = output_im[:image_rgb_raw.shape[0], :image_rgb_raw.shape[1]]
        return output_im, input_cropped

if __name__ == "__main__":
    processor = Processor(Path(__file__).parent / "regression_unetres_v1_scripted.pt")
    im_rgb = swap_rb(cv2.imread(sys.argv[1], cv2.IMREAD_COLOR))
    output_image = processor.process_image (im_rgb)
    cv2.imwrite("output.png", swap_rb(output_image))
    cv2.waitKey(0)

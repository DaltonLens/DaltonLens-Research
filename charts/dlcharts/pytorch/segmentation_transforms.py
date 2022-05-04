# Imported and tweaked from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py

import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import List

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size()[-2:])
    if min_size < size:
        ow, oh = img.size()[-2:]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class RandomCropWithRegressionLabels:
    def __init__(self, cropping_border, divider=64):
        self.cropping_border = cropping_border
        self.divider = divider # force the final size to be a multiple of divider

    def __call__(self, images):
        # images = [pad_if_smaller(image, self.size) for image in images]
        images = list(images)
        hw_size = images[0].size()[-2:] # height, width
        hw_size = (hw_size[0] - self.cropping_border, hw_size[1] - self.cropping_border)
        hw_size = (hw_size[0]//self.divider * self.divider, hw_size[1]//self.divider * self.divider)
        crop_params = T.RandomCrop.get_params(images[0], hw_size)
        outputs = [F.crop(image, *crop_params) for image in images]
        return outputs

class ApplyOnAll:
    def __init__(self, image_transform):
        self.image_transform = image_transform
    
    def __call__(self, images: List[torch.Tensor]):
        return map(self.image_transform, images)

class ApplyOnFloatOnly:
    def __init__(self, image_transform):
        self.image_transform = lambda x: image_transform(x) if x.is_floating_point() else x
    
    def __call__(self, images: List[torch.Tensor]):
        return map(self.image_transform, images)

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images: List[torch.Tensor]):
        for t in self.transforms:
            images = t(images)
        return images


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomRotate90:
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob

    def __call__(self, images: List[torch.Tensor]):
        if random.random() >= self.rotate_prob:
            return images
        return [image.transpose(-2,-1) for image in images]

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images: List):
        if random.random() >= self.flip_prob:
            return images
        return [F.hflip(image) for image in images]
    
class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images: List):
        if random.random() >= self.flip_prob:
            return images
        return [F.vflip(image) for image in images]

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

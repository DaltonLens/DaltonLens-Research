#!/usr/bin/env python3

from dlcharts.common.dataset import LabeledImage
from dlcharts.common.utils import swap_rb
from dlcharts.pytorch import segmentation_transforms
from dlcharts.pytorch import color_regression as cr

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

from icecream import ic

if __name__ == "__main__":
    checkpoint_file = sys.argv[1]
    net = cr.RegressionNet_Unet1(residual_mode=True)
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    scripted_module = torch.jit.script(net)
    torch.jit.save(scripted_module, sys.argv[2])

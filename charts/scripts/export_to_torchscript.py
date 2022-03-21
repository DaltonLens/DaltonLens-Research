#!/usr/bin/env python3

import dlcharts
from dlcharts.common.dataset import LabeledImage
from dlcharts.common.utils import swap_rb
from dlcharts.pytorch import segmentation_transforms
from dlcharts.pytorch import color_regression as cr

import torch
from torch import Tensor
import torch.onnx

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

def save_torchscript(net, output_file):
    scripted_module = torch.jit.script(net)
    torch.jit.save(scripted_module, output_file)
    return scripted_module

def save_onnx(net, output_file):
    sample_input = torch.zeros(1,3,256,256)
    torch.onnx.export(net,                   # model being run
                  sample_input,                        # model input (or a tuple for multiple inputs)
                  output_file,               # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2 : 'width', 3 : 'height' },    # variable length axes
                                'output' : {0 : 'batch_size', 2 : 'width', 3 : 'height'}})

if __name__ == "__main__":
    checkpoint_file = sys.argv[1]
    net = dlcharts.pytorch.models.RegressionNet_UResNet18(residual_mode=True)
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    scripted_module = save_torchscript(net, sys.argv[2])
    # save_onnx(scripted_module, "test.onnx")
    # Do not use the scripted module, it fails to convert because of some dict.
    save_onnx(net, "test.onnx")

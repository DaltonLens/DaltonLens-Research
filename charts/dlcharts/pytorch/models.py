from .layers import ResnetBlock, UnetBlock

import torch
import torch.nn.functional as F
from torch import nn

from enum import Enum

import timm

from icecream import ic

from typing import List, Dict

class UnetDecoder(nn.Module):
    def __init__(self, residual_mode, self_attention, icnr_shuffle):
        super().__init__()
        self.residual_mode = residual_mode
        self.self_attention = self_attention
        self.icnr_shuffle = icnr_shuffle
        self.enc0 = ResnetBlock(3, 64)

        self.dec4     = UnetBlock(512,512, out_features=512, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec3     = UnetBlock(256,512, out_features=256, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec2     = UnetBlock(128,256, out_features=128, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec1     = UnetBlock(64,128,  out_features=64,  self_attention=False, icnr_shuffle=icnr_shuffle)
        self.dec1_act = UnetBlock(64,64,   out_features=64,  self_attention=False, icnr_shuffle=icnr_shuffle)
        self.dec0     = UnetBlock(64,64,   out_features=64,  self_attention=False, icnr_shuffle=icnr_shuffle)

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
        # Backward compat, we used to save it as a dictionary.
        # But this is a problem for ONNX tracing, it needs tensors.
        if isinstance(state, dict):
            self.residual_mode = state['residual_mode']            
        else:
            values = state.item()
            self.residual_mode = values[0]
            self.self_attention = values[1]
            self.icnr_shuffle = values[2]

    def get_extra_state(self):
        return torch.tensor([self.residual_mode, self.self_attention, self.icnr_shuffle])

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

class RegressionNet_UResNet18(nn.Module):
    def __init__(self, residual_mode=False, self_attention=False, icnr_shuffle=False):
        super().__init__()
        self.encoder = timm.create_model('resnet18', features_only=True, pretrained=True, scriptable=True, exportable=True)
        self.decoder = UnetDecoder(residual_mode, self_attention, icnr_shuffle)
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

def create_regression_model(name):
    model = None
    if name == 'uresnet18-no-residual':
        model = RegressionNet_UResNet18(residual_mode=False)
    elif name == 'uresnet18':
        model = RegressionNet_UResNet18(residual_mode=True)
    elif name == 'uresnet18-shuffle':
        model = RegressionNet_UResNet18(residual_mode=True, icnr_shuffle=True)
    elif name == 'uresnet18-sa':
        model = RegressionNet_UResNet18(residual_mode=True, self_attention=True)
    elif name == 'uresnet18-sa-shuffle':
        model = RegressionNet_UResNet18(residual_mode=True, self_attention=True, icnr_shuffle=True)
    assert model
    return model

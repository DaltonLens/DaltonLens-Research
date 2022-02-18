import torch
import torch.nn.functional as F
from torch import nn

import timm

from icecream import ic

from typing import List, Dict

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

class RegressionNet_UResNet18(nn.Module):
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

def create_regression_model(name):
    if name == 'uresnet18-v1':
        model = RegressionNet_UResNet18()
        return model

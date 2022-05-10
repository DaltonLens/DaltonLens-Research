from .fastai_layers import PixelShuffle_ICNR, SelfAttention

import torch
import torch.nn.functional as F
from torch import nn

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
    def __init__(self, left_features, down_features, out_features, self_attention=False, icnr_shuffle=False):
        super().__init__()

        self.shuf_upsample = None
        if icnr_shuffle:
            self.shuf_upsample = PixelShuffle_ICNR(down_features, down_features, blur=False, act_cls=nn.ReLU, norm_type=None)

        self.block = nn.Sequential(
            nn.Conv2d(left_features + down_features, down_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(down_features),
            nn.ReLU(),

            nn.Conv2d(down_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),

            SelfAttention(out_features) if self_attention else nn.Identity(),
        )

    def forward(self, left, down):
        if self.shuf_upsample is not None:
            down_upsampled = self.shuf_upsample (down)
        else:
            down_upsampled = F.interpolate(down, scale_factor=2.0)
        x = torch.cat((down_upsampled, left), dim=1)
        return self.block (x)

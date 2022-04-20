from json import encoder
from .layers import ResnetBlock, UnetBlock
from .utils import get_model_complexity_info

import torch
import torch.nn.functional as F
from torch import nn

from enum import Enum
import re

import timm

from icecream import ic

from typing import List, Dict

from dataclasses import dataclass

from logging import info

# For the layers output by timm with features_only=True
encoder_config = dict(
    mobilenetv2_100 = dict(channels=[16, 16, 24, 32, 96, 320]),
    resnet18        = dict(channels=[64, 64, 64, 128, 256, 512])
)

decoder_config = dict(
    # Same channels as resnet 18
    invresnet18    = dict(channels=[64, 64,  64, 128, 256, 512]),

    # Same channels as mobilenet
    invmobilenetv2 = dict(channels=[16, 16,  24,  32,  96, 320]),

    # Mix between resnet and mobilenet
    medium = dict(channels=[32, 64,  64,  128,  256, 320]),

    # Larger than resnet 18
    large       = dict(channels=[64, 64, 128, 256, 512, 1024]),
)

class UnetDecoder(nn.Module):
    def __init__(self,
                 self_attention,
                 icnr_shuffle,
                 encoder_config: dict,
                 decoder_config: dict):
        super().__init__()
        self.self_attention = self_attention
        self.icnr_shuffle = icnr_shuffle
        
        # ResNet: [64, 64, 64, 128, 256, 512]
        # Assuming a /2 in size at every layer.
        nfenc0, nfenc1_act, nfenc1, nfenc2, nfenc3, nfenc4 = encoder_config['channels']
        nfdec0, nfdec1_act, nfdec1, nfdec2, nfdec3, nfdec4 = decoder_config['channels']

        self.enc0 = ResnetBlock(3, nfenc0)

        self.dec4     = UnetBlock(nfenc4, nfdec4, out_features=nfdec4, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec3     = UnetBlock(nfenc3, nfdec4, out_features=nfdec3, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec2     = UnetBlock(nfenc2, nfdec3, out_features=nfdec2, self_attention=self.self_attention, icnr_shuffle=icnr_shuffle)
        self.dec1     = UnetBlock(nfenc1, nfdec2, out_features=nfdec1,  self_attention=False, icnr_shuffle=icnr_shuffle)
        self.dec1_act = UnetBlock(nfenc1_act, nfdec1, out_features=nfdec1_act,  self_attention=False, icnr_shuffle=icnr_shuffle)
        self.dec0     = UnetBlock(nfenc0, nfdec1_act, out_features=nfdec0,  self_attention=False, icnr_shuffle=icnr_shuffle)

        self.bottom = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(nfenc4, nfdec4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nfdec4),
            nn.ReLU(),

            nn.Conv2d(nfdec4, nfdec4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nfdec4),
            nn.ReLU(),
        )

        n_classes = 4 # RGB and mask
        self.head = nn.Sequential(
            nn.Conv2d(nfdec0, nfdec0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nfdec0),
            nn.ReLU(),

            nn.Conv2d(nfdec0, nfdec0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nfdec0),
            nn.ReLU(),

            # 1x1 conv
            nn.Conv2d(nfdec0, n_classes, kernel_size=1, padding=0),
        )

    def set_extra_state(self, state):
        # Backward compat, we used to save it as a dictionary.
        # But this is a problem for ONNX tracing, it needs tensors.
        self.self_attention = state[1].item()
        self.icnr_shuffle = state[2].item()

    def get_extra_state(self):
        return torch.tensor([self.self_attention, self.icnr_shuffle])

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
        
        # B,4,H,W
        # First 3 channels are RGB for the fg pixels
        # Last channel is the mask
        outputs = self.head(x)
        return outputs

class GatedRegressionUnet(nn.Module):
    def __init__(self, encoder_model='resnet18', decoder_model='invresnet18', pretrained_encoder=True, self_attention=False, icnr_shuffle=False):
        super().__init__()
        # self.encoder = timm.create_model('resnet18', features_only=True, pretrained=True, scriptable=True, exportable=True)
        self.pretrained_encoder = pretrained_encoder
        self.encoder = timm.create_model(encoder_model, features_only=True, pretrained=pretrained_encoder, scriptable=True, exportable=True)
        # get_model_complexity_info (self.encoder)        
        self.decoder = UnetDecoder(self_attention, icnr_shuffle, encoder_config[encoder_model], decoder_config[decoder_model])
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
        if not self.pretrained_encoder:
            info ("The encoder is not pretrained, not freezing anything")
            return
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        if not self.pretrained_encoder:
            return
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, img):
        encoded_layers = self.encoder(img)
        return self.decoder(img, encoded_layers)

def decode_gated_regression(outputs: torch.Tensor, input_rgb: torch.Tensor):
    """
    outputs: B,4,H,W output of the model
    input_rgb: B,3,H,W raw input image
    
    Returns the final RGB image by applying the mask on the outputs.
    """
    single_item = (outputs.dim() == 3)
    if single_item:
        outputs = outputs.unsqueeze(0)
        input_rgb = input_rgb.unsqueeze(0)

    # The mask is the last channel.
    fg_mask = outputs[:,-1,:,:] > 0.0
    fg_mask_rgb = fg_mask.unsqueeze(1).expand(-1,3,-1,-1)
    outputs_rgb = outputs[:,0:3,:,:]
    final_rgb = torch.where(fg_mask_rgb, outputs_rgb, input_rgb)

    if single_item:
        final_rgb = final_rgb.squeeze(0)
        fg_mask = fg_mask.squeeze(0)
    return (final_rgb, fg_mask)


def create_gated_regression_model(name):
    model = None

    # Examples:
    # unet-rn18-rn18-sa
    # unet-mobv2-medium-shuffle

    opts = name.split('-')
    arch = opts[0]
    assert arch == 'unet'

    enc = opts[1]
    dec = opts[2]

    args = {}

    # Focus on the remaining options
    opts = opts[3:]
    args['self_attention'] = 'sa' in opts
    args['icnr_shuffle'] = 'shuffle' in opts
    args['pretrained_encoder'] = not 'nopretrain' in opts

    encs = dict(mobv2='mobilenetv2_100', rn18='resnet18')
    assert enc in encs
    args['encoder_model'] = encs[enc]

    decs = dict(mobv2='invmobilenetv2', rn18='invresnet18', large='large', medium='medium', small='small')
    assert dec in decs
    args['decoder_model'] = decs[dec]

    print ("Creating a RegressionNet_UResNet18, options = ", args)

    model = GatedRegressionUnet(**args)
    return model

"""
These modules were basically imported from fast.ai, with only minor changes.
It's useful to quickly prototype new modules.
"""

import torch
import torch.nn.functional as F
from torch import nn

from enum import Enum

from typing import List, Dict

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: nn.init.normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = nn.init.kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)

def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')

def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs):
    "InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('InstanceNorm', nf, ndim, zero=norm_type==NormType.InstanceZero, affine=affine, **kwargs)

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = nn.utils.weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = nn.utils.spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

# Adapted from fast.ai https://github.com/fastai/fastai1/blob/a8327427ad5137c4899a1b4f74745193c9ea5be3/fastai/layers.py#L81
class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
        # kaiming_uniform_ is used by default for conv2d.

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)
        # return nn.utils.spectral_norm (nn.Conv2d(n_in, n_out, kernel_size=1, bias=False))

    def forward(self, x):
        # Notation from the paper.
        size = x.size() # x [B,C,H,W]
        x = x.view(*size[:2],-1) # x [B,C,N]
        # f [B,C//8,N] g [B,C//8,N] h [B,C,N]
        f,g,h = self.query(x),self.key(x),self.value(x)
        # beta [B, N, N]
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        # o [B, C, N] (comes from B,C,N x B,N,N)
        o = self.gamma * torch.bmm(h, beta) + x
        # o [B, C, H, W]
        return o.view(*size).contiguous()

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

# From fast.ai
class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=nn.ReLU):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
                  nn.PixelShuffle(scale)]
        # if norm_type == NormType.Weight:
        #     layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
        #     layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3])**0.5)[:,None,None,None])
        # else:
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))

        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

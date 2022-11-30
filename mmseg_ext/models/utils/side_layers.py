#!/usr/bin/env python3

import torch.nn as nn

from mmcv.cnn import ConvModule
from mmseg.ops import resize


class SideConv(nn.Module):
    """'Improved' Basic Side Convolution

    instead of deconv, we use
    upsample -> 3x3conv

    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_cfg=None,
        norm_cfg=None,
        bias=False,
        act_cfg=dict(type="ReLU"),
        interpolation="bilinear",
        align_corners=False,
    ):
        super().__init__()
        self.conv_reduce = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,  # NOTE: worse results
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.post_resize = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=None,  # no bn
                bias=bias,
                act_cfg=None,  # no activation -> edge logit
            ),
        )
        self._interp = interpolation
        self._align_corners = align_corners

    def forward(self, x, size):
        x = resize(  # (B, out_channels, H, W)
            self.conv_reduce(x),
            size=size,
            mode=self._interp,
            align_corners=self._align_corners,
        )
        x = self.post_resize(x)
        return x

#!/usr/bin/env python3

"""Shape Stream (GSCNN)

Differences:
- Added activations between blocks
- Separate edge detection and features used in later layers
"""

import torch.nn as nn

from mmcv.cnn import ConvModule, build_activation_layer
from mmseg.ops import resize
from mmseg.models.backbones.resnet import BasicBlock

from .gated_conv import GatedSpatialConv2d


class ResNetGSCNNShapeStream(nn.Module):
    """GSCNN ShapeStream for ResNet Backbone

    Original implementation uses WiderResNet with features:
    (64, 256, 512, 4096)

    For ResNet, we use (stem, res1, res2, res4) which is:
    (64, 256, 512, 2048)

    Returns:
        binary feat, edge pred.
    """

    def __init__(
        self,
        in_channels,  # (64, 256, 512, 1024, 2048)
        num_classes,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        interpolation="bilinear",
        align_corners=False,
    ):
        super().__init__()

        assert isinstance(in_channels, (tuple, list))
        assert len(in_channels) >= 4

        self._interp = interpolation
        self._align_corners = align_corners

        self.dsn1 = ConvModule(
            in_channels=in_channels[0],  # 64
            out_channels=64,  # no feature reduction
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,  # bias=True in original
            act_cfg=act_cfg,
        )
        self.dsn2 = ConvModule(
            in_channels=in_channels[1],  # 256
            out_channels=1,  # gating feature
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,  # bias=True in original
            act_cfg=act_cfg,
        )
        self.dsn3 = ConvModule(
            in_channels=in_channels[2],  # 512
            out_channels=1,  # gating feature
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,  # bias=True in original
            act_cfg=act_cfg,
        )
        self.dsn4 = ConvModule(
            in_channels=in_channels[4],  # 2048
            out_channels=1,  # gating feature
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,  # bias=True in original
            act_cfg=act_cfg,
        )
        self.res1 = BasicBlock(
            inplanes=in_channels[0],  # 64
            planes=in_channels[0],  # 64
            stride=1,
            downsample=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.d1 = ConvModule(
            in_channels=in_channels[0],
            out_channels=32,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.res2 = BasicBlock(
            inplanes=32,
            planes=32,
            stride=1,
            downsample=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.d2 = ConvModule(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.res3 = BasicBlock(
            inplanes=16,
            planes=16,
            stride=1,
            downsample=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.d3 = ConvModule(
            in_channels=16,
            out_channels=8,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.gate1 = GatedSpatialConv2d(
            in_channels=32,
            out_channels=32,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.gate2 = GatedSpatialConv2d(
            in_channels=16,
            out_channels=16,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.gate3 = GatedSpatialConv2d(
            in_channels=8,
            out_channels=8,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.fuse = ConvModule(
            in_channels=8,
            out_channels=num_classes,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,  # norm_cfg,
            bias=True,  # originally false
            act_cfg=None,
        )

        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        bs, c, h, w = x[-1].shape
        resize_to = (h, w)  # TODO: might be too large

        # side outputs
        side1 = resize(
            self.dsn1(x[0]),
            size=resize_to,
            mode=self._interp,
            align_corners=self._align_corners,
        )
        side2 = resize(
            self.dsn2(x[1]),
            size=resize_to,
            mode=self._interp,
            align_corners=self._align_corners,
        )
        side3 = resize(
            self.dsn3(x[2]),
            size=resize_to,
            mode=self._interp,
            align_corners=self._align_corners,
        )
        side4 = resize(
            self.dsn4(x[4]),
            size=resize_to,
            mode=self._interp,
            align_corners=self._align_corners,
        )

        cs = self.d1(
            resize(
                self.res1(side1),
                size=resize_to,
                mode=self._interp,
                align_corners=self._align_corners,
            )
        )
        cs = self.gate1(cs, side2)
        cs = self.act(cs)
        cs = self.d2(
            resize(
                self.res2(cs),
                size=resize_to,
                mode=self._interp,
                align_corners=self._align_corners,
            )
        )
        cs = self.gate2(cs, side3)
        cs = self.act(cs)
        cs = self.d3(
            resize(
                self.res3(cs),
                size=resize_to,
                mode=self._interp,
                align_corners=self._align_corners,
            )
        )
        cs = self.gate3(cs, side4)
        cs = self.act(cs)
        edge_out = self.fuse(cs)

        return edge_out

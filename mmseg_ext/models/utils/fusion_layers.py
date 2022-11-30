#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


class GroupedConvFuse(nn.Module):
    """Basic multi-layer side fusion used in CaseNet

    https://github.com/Lavender105/DFF/blob/master/exps/models/casenet.py

    Changes:
    - bias=False: no bias in the last layer
    - flexible: number of sides could channge (CASENet, DDS, etc...)
    """

    def __init__(
        self,
        num_classes,
        num_sides,
        conv_cfg=None,
        bias=True,
    ):
        super().__init__()

        self.num_sides = num_sides

        # fuse (grouped convolution)
        self.fuse = ConvModule(
            in_channels=num_classes * num_sides,
            out_channels=num_classes,
            kernel_size=1,
            groups=num_classes,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            bias=bias,  # originally True
            act_cfg=None,
        )

    def forward(self, sides):
        assert isinstance(sides, list)
        assert len(sides) == self.num_sides, f"number of sides: {len(sides)}"

        side5 = sides.pop(-1)

        slice5 = side5[:, 0:1, :, :]
        fuse = torch.cat((slice5, *sides), 1)
        for i in range(side5.size(1) - 1):
            slice5 = side5[:, i + 1 : i + 2, :, :]
            fuse = torch.cat((fuse, slice5, *sides), 1)

        fuse = self.fuse(fuse)

        return fuse

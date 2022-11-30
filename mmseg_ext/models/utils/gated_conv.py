#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


class GatedSpatialConv2d(ConvModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        **kwargs,
    ):
        """Gated Spatial Conv2D

        Args:
            in_channels
            out_channels
            kernel_size
            stride
            padding
            dilation
            groups
            bias

        Original implementation:
        - BN -> conv2d(bias=True) -> ReLU -> conv2d(bias=True) -> BN -> sigmoid

        This implementation:
        - conv2d(bias=False) -> BN -> ReLU -> conv2d(bias=True) -> BN -> sigmoid

        This is because BN and activations are already applied to side outputs.
        """

        assert norm_cfg is not None
        assert act_cfg is not None

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs,
        )

        self._gate_conv = nn.Sequential(
            ConvModule(
                in_channels=in_channels + 1,
                out_channels=in_channels + 1,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            ConvModule(
                in_channels=in_channels + 1,
                out_channels=1,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        gating_features: torch.Tensor,
        activate: bool = False,  # default: don't activate after conv
        norm: bool = True,
    ) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                alphas = self._gate_conv(
                    torch.cat(
                        [x, gating_features],
                        dim=1,
                    )
                )
                # residual connection
                x = x * (alphas + 1)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)

        return x

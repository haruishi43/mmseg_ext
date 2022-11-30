#!/usr/bin/env python3

"""Reference model for GSCNN and Multi-Label GSCNN

Difference:
- Different feature fusion
- Removed sigmoid activations in the model because loss uses logits
- Added final convolution for edge prediction
- Different canny edge (image gradient)
- Shape stream has activations between gated-conv
- Depthwise Separable Convs at the end

Multi-Label GSCNN
- edge attention loss produces `nan` whle training possibily due to learning rate
    and network architecture.
"""

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.ops import resize
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPModule

from mmseg_ext.models.builder import SEG_HEADS
from .multitask_decode_head import BaseMultiTaskDecodeHead
from ..utils import ResNetGSCNNShapeStream, UnNormalizedCanny


@SEG_HEADS.register_module()
class GSCNNHead(BaseMultiTaskDecodeHead):
    """GSCNN head with minor edits for stable training"""

    def __init__(
        self,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        binary_edge=True,
        cv2_canny=False,
        loss_edge=dict(type="BinaryEdgeLoss", loss_weight=1.0 * 20),
        **kwargs,
    ):
        super().__init__(
            input_transform="multiple_select",
            binary_edge=binary_edge,
            loss_edge=loss_edge,
            **kwargs,
        )

        if binary_edge:
            num_classes = 1
        else:
            num_classes = self.num_classes

        self.shape_stream = ResNetGSCNNShapeStream(
            in_channels=self.in_channels,
            num_classes=num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )
        self.cw = ConvModule(
            in_channels=num_classes + 1,  # logit + canny
            out_channels=1,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.aspp_edge_conv = ConvModule(
            in_channels=1,
            out_channels=self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels[4],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        self.aspp_modules = DepthwiseSeparableASPPModule(
            in_channels=self.in_channels[4],
            channels=self.channels,
            dilations=dilations,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.c1_bottleneck = ConvModule(
            in_channels=c1_in_channels,
            out_channels=c1_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.bottleneck = ConvModule(
            (len(dilations) + 2) * self.channels,  # (img_pool, edge, *dilations)
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.channels + c1_channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            DepthwiseSeparableConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        # cv2, fast, non-differentiable
        self.canny = UnNormalizedCanny()

    def forward(self, inputs):
        """Forward function"""

        # [stem, layer1, layer2, layer3, layer4, input_image]
        # h: 1/2, 1/4, 1/8, 1/8, 1/8, 1
        x = [i for i in inputs]

        assert isinstance(x, list)
        assert len(x) == 6

        bs, c, h, w = x[-1].shape
        resize_to = (h, w)

        # Shape Stream
        edge_out = self.shape_stream(x)

        # Canny Edge Fusion
        canny = self.canny(x[-1])
        edge_feat = resize(
            torch.sigmoid(edge_out),
            size=resize_to,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        edge_feat = torch.cat((edge_feat, canny), dim=1)
        edge_feat = self.cw(edge_feat)
        edge_feat = resize(
            edge_feat,
            size=x[4].size()[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        # ASPP
        aspp_outs = [
            resize(
                self.image_pool(x[4]),
                size=x[4].size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        ]
        aspp_outs.append(self.aspp_edge_conv(edge_feat))
        aspp_outs.extend(self.aspp_modules(x[4]))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        bot_out = self.bottleneck(aspp_outs)

        # Bottlenecks
        c1_output = self.c1_bottleneck(x[1])
        bot_out = resize(
            input=bot_out,
            size=c1_output.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        bot_out = torch.cat([bot_out, c1_output], dim=1)
        bot_out = self.sep_bottleneck(bot_out)

        seg_out = self.cls_seg(bot_out)
        return seg_out, edge_out

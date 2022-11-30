#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize

from mmseg_ext.models.builder import build_seg_loss, build_edge_loss, build_joint_loss
from mmseg_ext.models.losses import accuracy, edge_accuracy


class BaseMultiTaskDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseMultiTaskDecodeHead.

    The key differences are:
    - `loss_decode` is now `loss_seg`
    - added `loss_edge`, and `loss_joint`
    """

    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes,
        binary_edge=False,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        in_index=-1,
        input_transform=None,
        loss_seg=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_edge=None,
        loss_joint=None,
        ignore_index=255,
        sampler=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01, override=dict(name="conv_seg")),
        no_conv_seg=False,  # HACK: some models use their own convolution at the end
    ):
        super(BaseMultiTaskDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        assert num_classes > 1, f"ERR: number of classes > 1: {num_classes}"
        self.num_classes = num_classes
        self.binary_edge = binary_edge
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        # setting up loss_seg
        if isinstance(loss_seg, dict):
            if "num_classes" in loss_seg.keys():
                loss_seg.update(dict(num_classes=self.num_classes))
            self.loss_seg = build_seg_loss(loss_seg)
        elif isinstance(loss_seg, (list, tuple)):
            self.loss_seg = nn.ModuleList()
            for loss in loss_seg:
                if "num_classes" in loss.keys():
                    loss.update(dict(num_classes=self.num_classes))
                self.loss_seg.append(build_seg_loss(loss))
        else:
            raise TypeError(
                f"loss_seg must be a dict or sequence of dict,\
                but got {type(loss_seg)}"
            )

        # setting up loss_edge
        if loss_edge is None:
            self.loss_edge = None
        else:
            if isinstance(loss_edge, dict):
                if "num_classes" in loss_edge.keys():
                    loss_edge.update(dict(num_classes=self.num_classes))
                self.loss_edge = build_edge_loss(loss_edge)
            elif isinstance(loss_edge, (list, tuple)):
                self.loss_edge = nn.ModuleList()
                for loss in loss_edge:
                    if "num_classes" in loss.keys():
                        loss.update(dict(num_classes=self.num_classes))
                    self.loss_edge.append(build_edge_loss(loss))
            else:
                raise TypeError(
                    f"loss_edge must be a dict or sequence of dict,\
                    but got {type(loss_joint)}"
                )

        # setting up loss_joint
        if loss_joint is None:
            self.loss_joint = None
        else:
            if isinstance(loss_joint, dict):
                if "num_classes" in loss_joint.keys():
                    loss_joint.update(dict(num_classes=self.num_classes))
                self.loss_joint = build_joint_loss(loss_joint)
            elif isinstance(loss_joint, (list, tuple)):
                self.loss_joint = nn.ModuleList()
                for loss in loss_joint:
                    if "num_classes" in loss.keys():
                        loss.update(dict(num_classes=self.num_classes))
                    self.loss_joint.append(build_joint_loss(loss))
            else:
                raise TypeError(
                    f"loss_joint must be a dict or sequence of dict,\
                    but got {type(loss_joint)}"
                )

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        if no_conv_seg:
            pass
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
            if dropout_ratio > 0:
                self.dropout = nn.Dropout2d(dropout_ratio)
            else:
                self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = (
            f"input_transform={self.input_transform}, "
            f"ignore_index={self.ignore_index}, "
            f"align_corners={self.align_corners}"
        )
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, **kwargs):
        """Placeholder of forward function."""
        pass

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        gt_semantic_edge,
        train_cfg,
    ):
        # NOTE: segmentation results are logits, but edges may not
        seg_logits, edge_logits = self(inputs)

        # convert multilabel to binary edge
        if self.binary_edge and gt_semantic_edge.shape[1] != 1:
            # convert to binary
            gt_semantic_edge = (
                (torch.sum(gt_semantic_edge, axis=1) > 0).unsqueeze(1).float()
            )

        # Assume that there are always seg_losses
        losses = self.seg_losses(
            seg_logit=seg_logits,
            seg_label=gt_semantic_seg,
        )
        if self.loss_edge is not None:
            losses.update(
                self.edge_losses(
                    edge_logit=edge_logits,
                    edge_label=gt_semantic_edge,
                )
            )
        if self.loss_joint is not None:
            losses.update(
                self.joint_losses(
                    seg_logit=seg_logits,
                    edge_logit=edge_logits,
                    seg_label=gt_semantic_seg,
                    edge_label=gt_semantic_edge,
                )
            )

        return losses

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward_test(self, inputs, img_metas, test_cfg, return_edge=False, **kwargs):
        seg, edge = self(inputs)

        if return_edge:
            return dict(seg=seg, edge=edge)
        else:
            return dict(seg=seg)

    @force_fp32(apply_to=("seg_logit"))
    def seg_losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_seg, nn.ModuleList):
            losses_seg = [self.loss_seg]
        else:
            losses_seg = self.loss_seg
        for loss_seg in losses_seg:
            if loss_seg.loss_name not in loss:
                loss[loss_seg.loss_name] = loss_seg(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )
            else:
                loss[loss_seg.loss_name] += loss_seg(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )

        loss["acc_seg"] = accuracy(seg_logit, seg_label)
        return loss

    @force_fp32(apply_to=("edge_logit"))
    def edge_losses(self, edge_logit, edge_label):
        """Compute edge loss."""
        loss = dict()

        edge_logit = resize(
            input=edge_logit,
            size=edge_label.shape[2:],  # (b, cls, h, w)
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if not isinstance(self.loss_edge, nn.ModuleList):
            losses_edge = [self.loss_edge]
        else:
            losses_edge = self.loss_edge
        for loss_edge in losses_edge:
            if loss_edge.loss_name not in loss:
                loss[loss_edge.loss_name] = loss_edge(
                    edge_logit,
                    edge_label,
                    ignore_index=self.ignore_index,
                )
            else:
                loss[loss_edge.loss_name] += loss_edge(
                    edge_logit,
                    edge_label,
                    ignore_index=self.ignore_index,
                )

        for metric_name, v in edge_accuracy(edge_logit, edge_label).items():
            loss["edge" + "_" + metric_name] = v

        return loss

    @force_fp32(apply_to=("seg_logit", "edge_logit"))
    def joint_losses(self, seg_logit, edge_logit, seg_label, edge_label):
        """Compute joint loss.

        Currently only supports binary edges
        """
        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        edge_logit = resize(
            input=edge_logit,
            size=edge_label.shape[2:],  # (b, 19, h, w)
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_joint, nn.ModuleList):
            losses_joint = [self.loss_joint]
        else:
            losses_joint = self.loss_joint
        for loss_joint in losses_joint:
            if loss_joint.loss_name not in loss:
                loss[loss_joint.loss_name] = loss_joint(
                    cls_score=seg_logit,
                    label=seg_label,
                    edge=edge_logit,
                    edge_label=edge_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )
            else:
                loss[loss_joint.loss_name] += loss_joint(
                    cls_score=seg_logit,
                    label=seg_label,
                    edge=edge_logit,
                    edge_label=edge_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )

        return loss

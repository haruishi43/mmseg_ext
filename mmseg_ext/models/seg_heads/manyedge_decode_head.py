#!/usr/bin/env python3

import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize

from mmseg_ext.models.builder import build_seg_loss, build_edge_loss, build_joint_loss
from mmseg_ext.models.losses import accuracy, edge_accuracy


class BaseManyEdgeDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseManyEdgeDecodeHead.

    Useful when there are multiple types and multiple edge logits that needs to be
    supervised.
    """

    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes,
        edge_key,
        log_edge_keys=[],
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        in_index=-1,
        binary_edge=False,
        input_transform=None,
        binary_edge_keys=[],
        multilabel_edge_keys=[],
        loss_joint_edge_key=None,
        loss_seg=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_binary_edge=None,
        loss_multilabel_edge=None,
        loss_joint=None,
        ignore_index=255,
        sampler=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01, override=dict(name="conv_seg")),
        no_conv_seg=False,  # HACK: some models use their own convolution at the end
    ):
        super().__init__(init_cfg)

        assert isinstance(edge_key, str)
        self.edge_key = edge_key
        if log_edge_keys is None:
            log_edge_keys = []
        elif isinstance(log_edge_keys, str):
            log_edge_keys = [log_edge_keys]
        assert isinstance(log_edge_keys, (tuple, list))
        self.log_edge_keys = log_edge_keys

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        assert num_classes > 1, f"ERR: number of classes > 1: {num_classes}"
        self.num_classes = num_classes
        # binary classication has 2 classes (class + background), but edge is just a single class
        self.num_edge_classes = 1 if binary_edge else num_classes
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

        # setting up loss_multilabel_edge
        if loss_multilabel_edge is None:
            if len(multilabel_edge_keys) > 0:
                warnings.warn(
                    "multilabel edge loss is None, but there seems to be some keys, "
                    f"{multilabel_edge_keys},"
                    "removing keys..."
                )
                multilabel_edge_keys = []
            self.loss_multilabel_edge = None
        else:
            if isinstance(loss_multilabel_edge, dict):
                if "num_classes" in loss_multilabel_edge.keys():
                    loss_multilabel_edge.update(dict(num_classes=self.num_edge_classes))
                self.loss_multilabel_edge = build_edge_loss(loss_multilabel_edge)
            elif isinstance(loss_multilabel_edge, (list, tuple)):
                self.loss_multilabel_edge = nn.ModuleList()
                for loss in loss_multilabel_edge:
                    if "num_classes" in loss.keys():
                        loss.update(dict(num_classes=self.num_edge_classes))
                    self.loss_multilabel_edge.append(build_edge_loss(loss))
            else:
                raise TypeError(
                    f"loss_multilabel_edge must be a dict or sequence of dict,\
                    but got {type(loss_multilabel_edge)}"
                )

        # setting up loss_binary_edge
        if loss_binary_edge is None:
            assert (
                self.loss_multilabel_edge is not None
            ), "needs either binary or multilabel loss"
            if len(binary_edge_keys) > 0:
                warnings.warn(
                    "binary edge loss is None, but there seems to be some keys, "
                    f"{binary_edge_keys},"
                    "removing keys..."
                )
                binary_edge_keys = []
            self.loss_binary_edge = None
        else:
            if isinstance(loss_binary_edge, dict):
                self.loss_binary_edge = build_edge_loss(loss_binary_edge)
            elif isinstance(loss_binary_edge, (list, tuple)):
                self.loss_binary_edge = nn.ModuleList()
                for loss in loss_binary_edge:
                    self.loss_binary_edge.append(build_edge_loss(loss))
            else:
                raise TypeError(
                    f"loss_binary must be a dict or sequence of dict,\
                    but got {type(loss_binary_edge)}"
                )
        assert (
            len(multilabel_edge_keys) + len(binary_edge_keys) > 0
        ), f"there seems to be no edges to supervise: {multilabel_edge_keys}, {binary_edge_keys}"
        self.binary_edge_keys = binary_edge_keys
        self.multilabel_edge_keys = multilabel_edge_keys

        # setting up loss_joint
        if loss_joint is None:
            if loss_joint_edge_key is not None:
                warnings.warn(
                    f"loss_joint is None, but was given and edge key {loss_joint_edge_key}"
                )
            self.loss_joint = None
        else:
            assert loss_joint_edge_key is not None, "loss_joint_edge_key is None"
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
        self.loss_joint_edge_key = loss_joint_edge_key

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

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

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
        seg_logits, edge_logits = self(inputs)

        # Assume that there are always seg_losses
        losses = self.seg_losses(
            seg_logit=seg_logits,
            seg_label=gt_semantic_seg,
        )
        if self.loss_binary_edge is not None:
            losses.update(
                self.binary_edge_losses(
                    edge_logit=edge_logits,
                    edge_label=gt_semantic_edge,
                )
            )
        if self.loss_multilabel_edge is not None:
            losses.update(
                self.multilabel_edge_losses(
                    edge_logit=edge_logits,
                    edge_label=gt_semantic_edge,
                )
            )
        if self.loss_joint is not None:
            edge_logit = edge_logits.get(self.loss_joint_edge_key, None)
            assert edge_logit is not None
            losses.update(
                self.joint_losses(
                    seg_logit=seg_logits,
                    edge_logit=edge_logit,
                    seg_label=gt_semantic_seg,
                    edge_label=gt_semantic_edge,
                )
            )

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, return_edge=False, **kwargs):
        seg, edge = self(inputs)
        if return_edge:
            out_edge = edge.get(self.edge_key, None)
            assert out_edge is not None, f"could not find {self.edge_key}"
            return dict(seg=seg, edge=out_edge)
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
    def binary_edge_losses(self, edge_logit, edge_label):
        """Compute binary edge loss."""

        loss = dict()

        if edge_logit is None:
            return loss
        if isinstance(edge_logit, torch.Tensor):
            # if a tensor is passed
            edge_logit = dict(bin_edge=edge_logit)
        assert isinstance(edge_logit, dict)

        # convert multilabel to binary edge, if needed
        if edge_label.shape[1] != 1:
            # convert to binary
            edge_label = (torch.sum(edge_label, axis=1) > 0).unsqueeze(1).float()

        for k, logit in edge_logit.items():
            if k in self.binary_edge_keys:
                logit = resize(
                    input=logit,
                    size=edge_label.shape[2:],  # (b, cls, h, w)
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                assert (
                    edge_label.shape == logit.shape
                ), f"label, pred: {edge_label.shape}, {logit.shape}"

                if not isinstance(self.loss_binary_edge, nn.ModuleList):
                    losses_edge = [self.loss_binary_edge]
                else:
                    losses_edge = self.loss_binary_edge

                for loss_edge in losses_edge:
                    if loss_edge.loss_name not in loss:
                        loss[loss_edge.loss_name] = loss_edge(
                            logit,
                            edge_label,
                            ignore_index=self.ignore_index,
                        )
                    else:
                        loss[loss_edge.loss_name] += loss_edge(
                            logit,
                            edge_label,
                            ignore_index=self.ignore_index,
                        )

                if k in self.log_edge_keys:
                    for name, v in edge_accuracy(logit, edge_label).items():
                        loss[k + "_" + name] = v

        return loss

    @force_fp32(apply_to=("edge_logit"))
    def multilabel_edge_losses(self, edge_logit, edge_label):
        """Compute multilabel edge loss."""
        loss = dict()

        if edge_logit is None:
            return loss
        if isinstance(edge_logit, torch.Tensor):
            # if a tensor is passed
            edge_logit = dict(ml_edge=edge_logit)
        assert isinstance(edge_logit, dict)

        for k, logit in edge_logit.items():
            if k in self.multilabel_edge_keys:
                logit = resize(
                    input=logit,
                    size=edge_label.shape[2:],  # (b, cls, h, w)
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                assert (
                    edge_label.shape == logit.shape
                ), f"label, pred: {edge_label.shape}, {logit.shape}"

                if not isinstance(self.loss_multilabel_edge, nn.ModuleList):
                    losses_edge = [self.loss_multilabel_edge]
                else:
                    losses_edge = self.loss_multilabel_edge

                for loss_edge in losses_edge:
                    if loss_edge.loss_name not in loss:
                        loss[loss_edge.loss_name] = loss_edge(
                            logit,
                            edge_label,
                            ignore_index=self.ignore_index,
                        )
                    else:
                        loss[loss_edge.loss_name] += loss_edge(
                            logit,
                            edge_label,
                            ignore_index=self.ignore_index,
                        )

                if k in self.log_edge_keys:
                    for name, v in edge_accuracy(logit, edge_label).items():
                        loss[k + "_" + name] = v

        return loss

    @force_fp32(apply_to=("seg_logit", "edge_logit"))
    def joint_losses(self, seg_logit, edge_logit, seg_label, edge_label):
        """Compute joint loss.

        Currently supports only semantic edge.
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

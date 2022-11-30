#!/usr/bin/env python3

"""Old losses that is converted for mmsegmentation

FIXME: update the losses so that we can reuse them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SEG_LOSSES, JOINT_LOSSES


def calculate_weights(target, num_classes=19, norm=False, upper_bound=1.0):
    """
    Calculate weights of classes based on the training crop
    """

    bins = torch.histc(
        target,
        bins=num_classes,
        min=0.0,
        max=num_classes,
    )
    hist_norm = bins.float() / bins.sum()
    if norm:
        hist = ((bins != 0).float() * upper_bound * (1 / hist_norm)) + 1.0
    else:
        hist = ((bins != 0).float() * upper_bound * (1.0 - hist_norm)) + 1.0
    return hist


@SEG_LOSSES.register_module()
class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(
        self,
        num_classes,
        norm=False,
        upper_bound=1.0,
        loss_weight=1.0,
        loss_name="loss_imgwt_ce",
    ):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = num_classes
        self.norm = norm
        self.upper_bound = upper_bound
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        cls_score,  # logits
        label,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        # NOTE: calculate with batch
        weight = calculate_weights(
            label,
            num_classes=self.num_classes,
            norm=self.norm,
            upper_bound=self.upper_bound,
        )

        # I think this works since cls_score and labels are batches
        return self.loss_weight * F.cross_entropy(
            cls_score, label, weight=weight, reduction="mean", ignore_index=ignore_index
        )

    @property
    def loss_name(self):
        return self._loss_name


@JOINT_LOSSES.register_module()
class EdgeAttentionLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        threshold=0.8,
        norm=False,
        upper_bound=1.0,
        loss_weight=1.0,
        loss_name="loss_edge_att",
    ):
        super().__init__()
        self.num_classes = num_classes
        assert 0 < threshold < 1
        self.threshold = threshold
        self.norm = norm
        self.upper_bound = upper_bound
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        cls_score,  # logits
        label,
        edge,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        filler = torch.ones_like(label) * 255
        weight = calculate_weights(
            label,
            num_classes=self.num_classes,
            norm=self.norm,
            upper_bound=self.upper_bound,
        )

        # turn edge to probability
        edge = torch.sigmoid(edge)

        assert len(edge.shape) == 4, f"edge shape should be 4dim, {edge.shape}"

        return self.loss_weight * F.cross_entropy(
            cls_score,
            torch.where(edge.max(1)[0] > self.threshold, label, filler),  # binary edge
            weight=weight,
            reduction="mean",
            ignore_index=ignore_index,
        )

    @property
    def loss_name(self):
        return self._loss_name

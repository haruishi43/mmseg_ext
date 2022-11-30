#!/usr/bin/env python3

from .accuracy import Accuracy, accuracy, edge_accuracy
from .cross_entropy_loss import (
    CrossEntropyLoss,
    binary_cross_entropy,
    cross_entropy,
    mask_cross_entropy,
)
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .old_losses import (
    EdgeAttentionLoss,
    ImageBasedCrossEntropyLoss2d,
)
from .dual_task_loss import DualTaskLoss

from .binary_edge_loss import BinaryEdgeLoss
from .multilabel_edge_loss import MultiLabelEdgeLoss

__all__ = [
    "Accuracy",
    "accuracy",
    "edge_accuracy",
    "CrossEntropyLoss",
    "cross_entropy",
    "binary_cross_entropy",
    "mask_cross_entropy",
    "reduce_loss",
    "weight_reduce_loss",
    "weighted_loss",
    "EdgeAttentionLoss",
    "ImageBasedCrossEntropyLoss2d",
    "DualTaskLoss",
    "BinaryEdgeLoss",
    "MultiLabelEdgeLoss",
]

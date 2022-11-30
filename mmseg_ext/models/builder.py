#!/usr/bin/env python3

import warnings

from mmcv.utils import Registry
from mmseg.models.builder import (
    MODELS,
    BACKBONES,
    NECKS,
    HEADS as SEG_HEADS,
    LOSSES as SEG_LOSSES,
    SEGMENTORS,
    build_segmentor,
)

EDGE_MODELS = Registry("edge_models", parent=MODELS)
EDGE_HEADS = EDGE_MODELS
EDGE_LOSSES = EDGE_MODELS
DETECTORS = EDGE_MODELS

JOINT_LOSSES = MODELS

__all__ = [
    "BACKBONES",
    "NECKS",
    "SEG_HEADS",
    "EDGE_HEADS",
    "SEG_LOSSES",
    "EDGE_LOSSES",
    "JOINT_LOSSES",
    "SEGMENTORS",
    "build_backbone",
    "build_neck",
    "build_seg_head",
    "build_edge_head",
    "build_seg_loss",
    "build_edge_loss",
    "build_joint_loss",
    "build_segmentor",
]


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_seg_head(cfg):
    """Build seg head."""
    return SEG_HEADS.build(cfg)


def build_edge_head(cfg):
    """Build edge head."""
    return EDGE_HEADS.build(cfg)


def build_seg_loss(cfg):
    """Build segmentation loss."""
    return SEG_LOSSES.build(cfg)


def build_edge_loss(cfg):
    """Build edge loss."""
    return EDGE_LOSSES.build(cfg)


def build_joint_loss(cfg):
    """Build joint task loss"""
    return JOINT_LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            "train_cfg and test_cfg is deprecated, " "please specify them in model",
            UserWarning,
        )
    assert (
        cfg.get("train_cfg") is None or train_cfg is None
    ), "train_cfg specified in both outer field and model field "
    assert (
        cfg.get("test_cfg") is None or test_cfg is None
    ), "test_cfg specified in both outer field and model field "
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )

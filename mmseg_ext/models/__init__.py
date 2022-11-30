#!/usr/bin/env python3

from .builder import (
    BACKBONES,
    NECKS,
    SEG_HEADS,
    EDGE_HEADS,
    SEG_LOSSES,
    EDGE_LOSSES,
    JOINT_LOSSES,
    SEGMENTORS,
    build_backbone,
    build_seg_head,
    build_edge_head,
    build_neck,
    build_seg_loss,
    build_edge_loss,
    build_joint_loss,
    build_segmentor,
    build_detector,
)

# update registries
from .backbones import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .seg_heads import *  # noqa: F401,F403
from .edge_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .detectors import *  # noqa: F401,F403

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
    "build_detector",
]

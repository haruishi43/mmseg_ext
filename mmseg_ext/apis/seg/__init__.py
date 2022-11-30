#!/usr/bin/env python3

from .train import train_seg_det
from .test import single_gpu_seg_test, multi_gpu_seg_test

__all__ = [
    "train_seg_det",
    "single_gpu_seg_test",
    "multi_gpu_seg_test",
]

#!/usr/bin/env python3

from .train import train_det
from .test import single_gpu_edge_test, multi_gpu_edge_test

__all__ = [
    "train_det",
    "single_gpu_edge_test",
    "multi_gpu_edge_test",
]

#!/usr/bin/env python3

import copy

from mmcv.utils import Registry, build_from_cfg
from mmseg.datasets.builder import (
    DATASETS as SEG_DATASETS,
    PIPELINES as MMSEG_PIPELINES,
    build_dataloader,
    build_dataset as build_seg_dataset,
    _concat_dataset,
)

EDGE_DATASETS = Registry("dataset")
PIPELINES = MMSEG_PIPELINES


def build_edge_dataset(cfg, default_args=None):
    """Build datasets."""
    from mmseg.datasets.dataset_wrappers import (
        ConcatDataset,
        MultiImageMixDataset,
        RepeatDataset,
    )

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_edge_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_edge_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "MultiImageMixDataset":
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg["dataset"] = build_edge_dataset(cp_cfg["dataset"])
        cp_cfg.pop("type")
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get("img_dir"), (list, tuple)) or isinstance(
        cfg.get("split", None), (list, tuple)
    ):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, EDGE_DATASETS, default_args)

    return dataset


__all__ = [
    "SEG_DATASETS",
    "EDGE_DATASETS",
    "PIPELINES",
    "build_dataloader",
    "build_seg_dataset",
    "build_edge_dataset",
]

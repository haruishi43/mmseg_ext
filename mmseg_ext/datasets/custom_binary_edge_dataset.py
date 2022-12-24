#!/usr/bin/env python3

import os.path as osp

import mmcv
from mmcv.utils import print_log
from pyEdgeEval.edge_tools.transforms import Mask2Edge

from mmseg_ext.utils import get_root_logger
from .builder import EDGE_DATASETS
from .base_edge_dataset import BaseBinaryEdgeDataset
from .pipelines import Compose, BinaryEdgeFormatBundle, LoadAnnotations, LoadEdges


@EDGE_DATASETS.register_module()
class CustomBinaryLabelEdgeDataset(BaseBinaryEdgeDataset):
    def __init__(
        self,
        edge_dir=None,
        edge_map_suffix="_edge.png",
        gt_loader_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert edge_dir is not None, f"ERR: edge_dir is not valid: {edge_dir}"
        self.edge_dir = edge_dir
        self.edge_map_suffix = edge_map_suffix

        if gt_loader_cfg is None:
            gt_loader_cfg = dict(format=True, binary=True)
        else:
            gt_loader_cfg = dict(
                format=True,
                binary=True,
                **gt_loader_cfg,
            )
        self.gt_loader = Compose([LoadEdges(**gt_loader_cfg)])

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.edge_dir is None or osp.isabs(self.edge_dir)):
                self.edge_dir = osp.join(self.data_root, self.edge_dir)

        # load annotations
        assert self.edge_dir is not None
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            img_suffix=self.img_suffix,
            edge_map_suffix=self.edge_map_suffix,
            split=self.split,
        )

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        edge_map_suffix,
        split,
    ):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    edge_map = img_name + edge_map_suffix
                    img_info["ann"] = dict(edge_map=edge_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                edge_map = img.replace(img_suffix, edge_map_suffix)
                img_info["ann"] = dict(edge_map=edge_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["edge_prefix"] = self.edge_dir


@EDGE_DATASETS.register_module()
class OTFCustomBinaryLabelEdgeDataset(BaseBinaryEdgeDataset):

    IDS = None
    mask2edge = None

    def __init__(
        self,
        ann_dir=None,
        seg_map_suffix=".png",
        gt_loader_cfg=None,
        ignore_indices=[],
        labelIds=None,
        label2trainId=None,
        radius=2,
        selected_label=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix

        # initialize mask2edge
        labelIds = labelIds if labelIds else self.IDS
        self.mask2edge = Mask2Edge(
            labelIds=labelIds,
            ignore_indices=ignore_indices,
            label2trainId=label2trainId,
            radius=radius,
            use_cv2=True,
            quality=0,
        )

        assert selected_label is not None
        self.gt_loader = Compose(
            [
                LoadAnnotations()
                if gt_loader_cfg is None
                else LoadAnnotations(**gt_loader_cfg),
                BinaryEdgeFormatBundle(selected_label=selected_label),
            ]
        )

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        assert self.ann_dir is not None
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            split=self.split,
        )

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        seg_map_suffix,
        split,
    ):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    seg_map = img_name + seg_map_suffix
                    img_info["ann"] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["seg_prefix"] = self.ann_dir
        results["inst_sensitive"] = False
        results["mask2edge"] = self.mask2edge

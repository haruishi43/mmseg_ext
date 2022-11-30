#!/usr/bin/env python3

import os.path as osp

import mmcv
from mmcv.utils import print_log
from pyEdgeEval.edge_tools.transforms import Mask2Edge, InstanceMask2Edge

from mmseg_ext.utils import get_root_logger
from .pipelines import Compose, EdgeFormatBundle, LoadAnnotations, LoadEdges
from .base_edge_dataset import BaseMultiLabelEdgeDataset
from .builder import EDGE_DATASETS


@EDGE_DATASETS.register_module()
class CustomMultiLabelEdgeDataset(BaseMultiLabelEdgeDataset):
    """Loads Preprocessed Edges (useful for validation/evaluation)"""

    def __init__(
        self,
        edge_dir=None,
        edge_map_suffix="_edge.png",
        inst_sensitive=True,
        gt_loader_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert edge_dir is not None, f"ERR: edge_dir is not valid: {edge_dir}"
        self.edge_dir = edge_dir
        self.edge_map_suffix = edge_map_suffix

        # FIXME: not being used
        self.inst_sensitive = inst_sensitive

        if gt_loader_cfg is None:
            gt_loader_cfg = dict(format=True)
        else:
            gt_loader_cfg = dict(
                format=True,
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
                    ann = dict(
                        edge_map=edge_map,
                    )
                    img_info["ann"] = ann
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                edge_map = img.replace(img_suffix, edge_map_suffix)
                ann = dict(
                    edge_map=edge_map,
                )
                img_info["ann"] = ann
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["edge_prefix"] = self.edge_dir
        # need for converting labelids to trainids
        results["num_classes"] = len(self.CLASSES)


@EDGE_DATASETS.register_module()
class OTFCustomMultiLabelEdgeDataset(BaseMultiLabelEdgeDataset):
    """On-the-fly multilabe edge generation

    - creates semantic edges from semantic segmentation
    - supports instance masks
    """

    IDS = None
    inst_sensitive = None
    mask2edge = None

    def __init__(
        self,
        ann_dir=None,
        inst_dir=None,
        seg_map_suffix=".png",
        inst_map_suffix="_inst.png",
        inst_sensitive=True,
        gt_loader_cfg=None,
        class_names=None,
        labelIds=None,
        inst_labelIds=None,
        ignore_indices=[],
        label2trainId=None,
        radius=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert ann_dir is not None, f"ERR: ann_dir is not valid: {ann_dir}"
        if inst_sensitive:
            assert inst_dir is not None, f"ERR: inst_dir is not valid: {inst_dir}"

        if class_names:
            if self.CLASSES is not None:
                print_log(
                    "Resetting cls.CLASSES, which might be unintentional",
                    logger=get_root_logger(),
                )
            self.CLASSES = class_names
            self.IDS = list(range(self.CLASSES))

        self.ann_dir = ann_dir
        self.inst_dir = inst_dir
        self.seg_map_suffix = seg_map_suffix
        self.inst_map_suffix = inst_map_suffix
        self.inst_sensitive = inst_sensitive

        # initialize mask2edge
        labelIds = labelIds if labelIds else self.IDS
        if self.inst_sensitive:
            assert (
                inst_labelIds
            ), "ERR: `inst_labelIds` is needed for instance sensitive OTF"
            self.mask2edge = InstanceMask2Edge(
                labelIds=labelIds,
                inst_labelIds=inst_labelIds,
                ignore_indices=ignore_indices,
                label2trainId=label2trainId,
                radius=radius,
                use_cv2=True,
                quality=0,
            )
        else:
            self.mask2edge = Mask2Edge(
                labelIds=labelIds,
                ignore_indices=ignore_indices,
                label2trainId=label2trainId,
                radius=radius,
                use_cv2=True,
                quality=0,
            )

        # FIXME: formatting is complecated for OT
        self.gt_loader = Compose(
            [
                LoadAnnotations()
                if gt_loader_cfg is None
                else LoadAnnotations(**gt_loader_cfg),
                EdgeFormatBundle(),
            ]
        )

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.inst_dir is None or osp.isabs(self.inst_dir)):
                self.inst_dir = osp.join(self.data_root, self.inst_dir)

        # load annotations
        assert self.ann_dir is not None
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            inst_map_suffix=self.inst_map_suffix,
            inst_sensitive=self.inst_sensitive,
            split=self.split,
        )

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        seg_map_suffix,
        inst_map_suffix,
        inst_sensitive,
        split,
    ):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    seg_map = img_name + seg_map_suffix
                    if inst_sensitive:
                        # need instance map for instance aware segmentation (otf)
                        inst_map = img_name + inst_map_suffix
                        ann = dict(
                            seg_map=seg_map,
                            inst_map=inst_map,
                        )
                    else:
                        ann = dict(
                            seg_map=seg_map,
                        )
                    img_info["ann"] = ann
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                seg_map = img.replace(img_suffix, seg_map_suffix)
                if inst_sensitive:
                    inst_map = img.replace(img_suffix, inst_map_suffix)
                    ann = dict(
                        seg_map=seg_map,
                        inst_map=inst_map,
                    )
                else:
                    ann = dict(
                        seg_map=seg_map,
                    )
                img_info["ann"] = ann
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["seg_prefix"] = self.ann_dir
        results["inst_prefix"] = self.inst_dir
        results["inst_sensitive"] = self.inst_sensitive
        results["mask2edge"] = self.mask2edge

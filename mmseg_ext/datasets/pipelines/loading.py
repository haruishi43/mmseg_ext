#!/usr/bin/env python3

"""Loading data (images and annotations)."""

import os.path as osp

import numpy as np
from PIL import Image

import mmcv
from mmseg.datasets.pipelines.loading import LoadAnnotations as MMSEG_LoadAnnotations

from ..builder import PIPELINES


@PIPELINES.register_module(force=True)
class LoadAnnotations(MMSEG_LoadAnnotations):
    """LoadAnnotations (extention of mmseg)

    - loads instance segmentation for instance-sensitive edges
    """

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = (
            mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
            .squeeze()
            .astype(np.uint8)
        )
        # modify if custom classes (NOTE: currently not being used)
        if results.get("label_map", None) is not None:
            for old_id, new_id in results["label_map"].items():
                # Add deep copy to solve bug of repeatedly replacing
                # gt_semantic_seg
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")

        # load instance map
        inst_sensitive = results.get("inst_sensitive", False)
        if inst_sensitive:
            # load instance segmentation image
            inst_prefix = results.get("inst_prefix", None)
            if inst_prefix is not None:
                filename = osp.join(inst_prefix, results["ann_info"]["inst_map"])
            else:
                filename = results["ann_info"]["inst_map"]
            img_bytes = self.file_client.get(filename)
            gt_inst_seg = (
                mmcv.imfrombytes(
                    img_bytes, flag="unchanged", backend=self.imdecode_backend
                )
                .squeeze()
                .astype(np.int32)  # NOTE: needs to be int32
            )
            results["gt_inst_seg"] = gt_inst_seg
            results["seg_fields"].append("gt_inst_seg")

        return results


@PIPELINES.register_module()
class LoadEdges(MMSEG_LoadAnnotations):
    def __init__(
        self,
        format=False,
        binary=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.format = format
        self.binary = binary

    def _load_binary(self, filename):
        edge = Image.open(filename).convert("L")
        gt_semantic_edge = np.array(edge).astype(np.uint8)
        # FIXME: might need to format
        return gt_semantic_edge

    def _load_multilabel(self, filename, num_classes):
        # NOTE: only supports RGB inputs (need to change to binary or tif for more classes)
        # NOTE: for tif, we need to implement 32bit transforms
        img_bytes = self.file_client.get(filename)
        gt_semantic_edge = (
            mmcv.imfrombytes(
                img_bytes,
                flag="color",
                channel_order="rgb",
                backend="pillow",
            )
            .squeeze()
            .astype(np.uint8)
        )
        if self.format:
            assert (
                num_classes is not None
            ), "num_classes should be set for multilabel edges"
            gt_semantic_edge = np.unpackbits(
                gt_semantic_edge,
                axis=2,
            )[:, :, -1 : -(num_classes + 1) : -1]
            # outputs (c, h, w)
            gt_semantic_edge = np.ascontiguousarray(gt_semantic_edge.transpose(2, 0, 1))
        return gt_semantic_edge

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("edge_prefix", None) is not None:
            filename = osp.join(results["edge_prefix"], results["ann_info"]["edge_map"])
        else:
            filename = results["ann_info"]["edge_map"]

        if self.binary:
            gt_semantic_edge = self._load_binary(filename)
        else:
            num_classes = results.get("num_classes", None)
            gt_semantic_edge = self._load_multilabel(filename, num_classes)

        results["gt_semantic_edge"] = gt_semantic_edge
        results["seg_fields"].append("gt_semantic_edge")

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label},"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

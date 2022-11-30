#!/usr/bin/env python3

import os.path as osp
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset
from prettytable import PrettyTable

import mmcv
from mmcv.utils import print_log

from mmseg_ext.datasets.pipelines import Compose
from mmseg_ext.core.metrics import (
    eval_metrics as edge_metrics,
    pre_calc,
    pre_eval_to_metrics,
)

__all__ = ["BaseEdgeDataset", "BaseBinaryEdgeDataset", "BaseMultiLabelEdgeDataset"]


class BaseEdgeDataset(Dataset, metaclass=ABCMeta):

    CLASSES = None
    PALETTE = None
    img_infos = None
    gt_loader = None

    def __init__(
        self,
        pipeline,
        img_dir,
        img_suffix=".png",
        split=None,
        data_root=None,
        test_mode=False,
    ):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix

        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]["ann"]

    @abstractmethod
    def load_annotations(self):
        """Placeholder for loading annotations"""
        pass

    @abstractmethod
    def pre_pipeline(self, results):
        """Prepare results dict for edge pipeline."""
        pass

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)

        # needed for formatting
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir

        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    @abstractmethod
    def evaluate(self):
        """Placeholder for evaluation function"""
        pass


class BaseBinaryEdgeDataset(BaseEdgeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

    def get_gt_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)

        # needed for formatting
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir

        self.pre_pipeline(results)
        self.gt_loader(results)
        return results["gt_semantic_edge"]

    def get_gt_edge_maps(self):
        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            results["seg_fields"] = []
            results["img_prefix"] = self.img_dir
            self.pre_pipeline(results)
            self.gt_loader(results)
            yield results["gt_semantic_edge"]

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            edge_map = self.get_gt_by_idx(index)
            pre_eval_results.append(
                pre_calc(
                    pred,
                    edge_map,
                    num_classes=1,  # single class
                    thresh=0.7,
                )
            )

        return pre_eval_results

    def evaluate(
        self,
        results,
        metric="Fscore",
        gt_edge_maps=None,
        logger=None,
        **kwargs,
    ):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict segmentation map for computing evaluation
                metric.
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["Fscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))

        eval_results = {}
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_edge_maps is None:
                gt_edge_maps = self.get_gt_edge_maps()
            ret_metrics = edge_metrics(
                results,
                gt_edge_maps,
                num_classes=1,
                metrics=metric,
                thresh=0.7,
            )
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        return eval_results


class BaseMultiLabelEdgeDataset(BaseEdgeDataset):
    """BaseDataset for multi-label edge detection"""

    def __init__(
        self,
        pipeline,
        img_dir,
        img_suffix=".png",
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        palette=None,
    ):
        super().__init__(
            pipeline=pipeline,
            img_dir=img_dir,
            img_suffix=img_suffix,
            split=split,
            data_root=data_root,
            test_mode=test_mode,
        )
        # FIXME: might not need
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        assert self.CLASSES is not None, "`cls.CLASSES` should be specified"

        if self.PALETTE is None:
            self.PALETTE = self.get_palette_for_classes(self.CLASSES, palette)

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

    def get_palette_for_classes(self, class_names, palette=None):

        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def get_gt_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)

        # needed for formatting
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir

        self.pre_pipeline(results)
        self.gt_loader(results)
        return results["gt_semantic_edge"]

    def get_gt_edge_maps(self):
        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            results["seg_fields"] = []
            results["img_prefix"] = self.img_dir
            self.pre_pipeline(results)
            self.gt_loader(results)
            yield results["gt_semantic_edge"]

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            edge_map = self.get_gt_by_idx(index)
            pre_eval_results.append(
                pre_calc(
                    pred,
                    edge_map,
                    len(self.CLASSES),
                    thresh=0.7,
                )
            )

        return pre_eval_results

    def evaluate(
        self,
        results,
        metric="Fscore",
        gt_edge_maps=None,
        logger=None,
        **kwargs,
    ):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict segmentation map for computing evaluation
                metric.
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["Fscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))

        eval_results = {}
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_edge_maps is None:
                gt_edge_maps = self.get_gt_edge_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = edge_metrics(
                results,
                gt_edge_maps,
                num_classes=num_classes,
                metrics=metric,
                thresh=0.7,
            )
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": self.CLASSES})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {
                    key + "." + str(name): value[idx] / 100.0
                    for idx, name in enumerate(self.CLASSES)
                }
            )

        return eval_results

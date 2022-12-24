#!/usr/bin/env python3

import os.path as osp
import warnings
from copy import deepcopy

import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.runner import auto_fp16
from mmseg.models.segmentors import BaseSegmentor
from mmseg.core import add_prefix
from mmseg.ops import resize

from .. import builder
from ..seg_heads.multitask_decode_head import BaseMultiTaskDecodeHead
from mmseg_ext.visualize import beautify_edge


class BaseJointSegDet(BaseSegmentor):
    """Base class for Joint Modeling.

    Differrences:
    - support edge outputs
    - pass input image to decode head
    - logits passed from heads are dict
    """

    CLASSES = None
    PALETTE = None

    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        pass_input_image=False,
    ):
        super(BaseJointSegDet, self).__init__(init_cfg)
        if pretrained is not None:
            assert (
                backbone.get("pretrained") is None
            ), "both backbone and segmentor set pretrained weight"
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # pass input image to decode head
        self.pass_input_image = pass_input_image

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_seg_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_seg_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_seg_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)  # returns tuple
        if self.with_neck:
            x = self.neck(x)

        if self.pass_input_image:
            # HACK: append input
            x = (*x, img)

        return x

    def encode_decode(self, img, img_metas, return_edge=False, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.extract_feat(img)
        logits = self._decode_head_forward_test(
            x,
            img_metas,
            return_edge=return_edge,
            **kwargs,
        )
        seg = resize(
            input=logits["seg"],
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if return_edge:
            # need to check
            edge = logits["edge"]
            edge = resize(
                input=edge,
                size=img.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            return (seg, edge)
        else:
            return seg

    def _decode_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_seg,
        gt_semantic_edge,
    ):
        """Run forward function and calculate loss for decode head in
        training.

        - support ``gt_semantic_edge``
        """
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            inputs=x,
            img_metas=img_metas,
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_edge=gt_semantic_edge,
            train_cfg=self.train_cfg,
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas, return_edge=False, **kwargs):
        """Run forward function and calculate loss for decode head in
        inference."""
        logits = self.decode_head.forward_test(
            inputs=x,
            img_metas=img_metas,
            test_cfg=self.test_cfg,
            return_edge=return_edge,
            **kwargs,
        )
        return logits  # dict(seg, edge, etc...)

    def _auxiliary_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_seg,
        gt_semantic_edge=None,
    ):
        """Run forward function and calculate loss for auxiliary head in
        training.

        NOTE: needed to make sure that mmseg's aux heads are also supported
        """
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                # need to check if aux_head supports ``gt_semantic_edge``

                if isinstance(aux_head, BaseMultiTaskDecodeHead):
                    loss_aux = aux_head.forward_train(
                        inputs=x,
                        img_metas=img_metas,
                        gt_semantic_seg=gt_semantic_seg,
                        gt_semantic_edge=gt_semantic_edge,
                        train_cfg=self.train_cfg,
                    )
                    losses.update(add_prefix(loss_aux, f"aux_{idx}"))
                else:
                    loss_aux = aux_head.forward_train(
                        inputs=x,
                        img_metas=img_metas,
                        gt_semantic_seg=gt_semantic_seg,
                        train_cfg=self.train_cfg,
                    )
                    losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            if isinstance(self.auxiliary_head, BaseMultiTaskDecodeHead):
                loss_aux = self.auxiliary_head.forward_train(
                    inputs=x,
                    img_metas=img_metas,
                    gt_semantic_seg=gt_semantic_seg,
                    gt_semantic_edge=gt_semantic_edge,
                    train_cfg=self.train_cfg,
                )
                losses.update(add_prefix(loss_aux, "aux"))
            else:
                loss_aux = self.auxiliary_head.forward_train(
                    inputs=x,
                    img_metas=img_metas,
                    gt_semantic_seg=gt_semantic_seg,
                    train_cfg=self.train_cfg,
                )
                losses.update(add_prefix(loss_aux, "aux"))

        return losses

    def forward_dummy(self, img, return_edge=False, **kwargs):
        """Dummy forward function."""
        return self.encode_decode(img, None, return_edge=return_edge, **kwargs)

    def forward_train(
        self, img, img_metas, gt_semantic_seg, gt_semantic_edge, **kwargs
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg_ext/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg, gt_semantic_edge
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(
        self,
        img,
        img_meta,
        rescale,
        return_edge=False,
        **kwargs,
    ):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        seg_preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        if return_edge:
            edge_preds = img.new_zeros((batch_size, num_classes, h_img, w_img))

        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                logits = self.encode_decode(
                    crop_img, img_meta, return_edge=return_edge, **kwargs
                )

                if isinstance(logits, tuple):
                    seg_logit = logits[0]
                    edge = logits[1]
                    seg_preds += F.pad(
                        seg_logit,
                        (
                            int(x1),
                            int(seg_preds.shape[3] - x2),
                            int(y1),
                            int(seg_preds.shape[2] - y2),
                        ),
                    )
                    edge_preds += F.pad(
                        edge,
                        (
                            int(x1),
                            int(edge_preds.shape[3] - x2),
                            int(y1),
                            int(edge_preds.shape[2] - y2),
                        ),
                    )
                else:
                    seg_preds += F.pad(
                        logits,
                        (
                            int(x1),
                            int(seg_preds.shape[3] - x2),
                            int(y1),
                            int(seg_preds.shape[2] - y2),
                        ),
                    )

                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        seg_preds = seg_preds / count_mat
        if return_edge:
            edge_preds = edge_preds / count_mat

        if rescale:
            seg_preds = resize(
                seg_preds,
                size=img_meta[0]["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )
            if return_edge:
                edge_preds = resize(
                    edge_preds,
                    size=img_meta[0]["ori_shape"][:2],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                )

        if return_edge:
            return (seg_preds, edge_preds)
        else:
            return seg_preds

    def whole_inference(
        self,
        img,
        img_meta,
        rescale,
        return_edge=False,
        **kwargs,
    ):
        """Inference with full image."""

        logits = self.encode_decode(img, img_meta, return_edge=return_edge, **kwargs)

        if isinstance(logits, tuple):
            seg_logit = logits[0]
            edge = logits[1]

            if rescale:
                size = img_meta[0]["ori_shape"][:2]
                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                )
                edge = resize(
                    edge,
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                )

            return (seg_logit, edge)

        else:
            if rescale:
                size = img_meta[0]["ori_shape"][:2]
                logits = resize(
                    logits,
                    size=size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                )
            return logits

    def inference(self, img, img_meta, rescale, return_edge=False, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg_ext/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)

        if return_edge:
            if self.test_cfg.mode == "slide":
                seg_logit, edge = self.slide_inference(
                    img,
                    img_meta,
                    rescale,
                    return_edge=True,
                    **kwargs,
                )
            else:
                seg_logit, edge = self.whole_inference(
                    img,
                    img_meta,
                    rescale,
                    return_edge=True,
                    **kwargs,
                )

            seg_out = F.softmax(seg_logit, dim=1)

            flip = img_meta[0]["flip"]
            if flip:
                flip_direction = img_meta[0]["flip_direction"]
                assert flip_direction in ["horizontal", "vertical"]
                if flip_direction == "horizontal":
                    seg_out = seg_out.flip(dims=(3,))
                    edge = edge.flip(dims=(3,))
                elif flip_direction == "vertical":
                    seg_out = seg_out.flip(dims=(2,))
                    edge = edge.flip(dims=(2,))
            return (seg_out, edge)
        else:
            if self.test_cfg.mode == "slide":
                seg_logit = self.slide_inference(
                    img,
                    img_meta,
                    rescale,
                    return_edge=False,
                    **kwargs,
                )
            else:
                seg_logit = self.whole_inference(
                    img,
                    img_meta,
                    rescale,
                    return_edge=False,
                    **kwargs,
                )

            output = F.softmax(seg_logit, dim=1)

            flip = img_meta[0]["flip"]
            if flip:
                flip_direction = img_meta[0]["flip_direction"]
                assert flip_direction in ["horizontal", "vertical"]
                if flip_direction == "horizontal":
                    output = output.flip(dims=(3,))
                elif flip_direction == "vertical":
                    output = output.flip(dims=(2,))
            return output

    def simple_test(self, img, img_meta, rescale=True, return_edge=False, **kwargs):
        """Simple test with single image."""
        if return_edge:
            seg_logit, edge = self.inference(
                img,
                img_meta,
                rescale,
                return_edge=True,
                **kwargs,
            )
            seg_pred = seg_logit.argmax(dim=1)
            edge_pred = edge.sigmoid_()

            # NOTE: converts to numpy
            seg_pred = seg_pred.cpu().numpy()
            edge_pred = edge_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            edge_pred = list(edge_pred)
            return seg_pred, edge_pred
        else:
            seg_logit = self.inference(
                img,
                img_meta,
                rescale,
                return_edge=False,
                **kwargs,
            )
            seg_pred = seg_logit.argmax(dim=1)

            # NOTE: converts to numpy
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True, return_edge=False, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        if return_edge:
            # to save memory, we get augmented seg logit inplace
            seg_logit, edge = self.inference(
                imgs[0],
                img_metas[0],
                rescale,
                return_edge=True,
                **kwargs,
            )
            for i in range(1, len(imgs)):
                cur_seg_logit, cur_edge = self.inference(
                    imgs[i],
                    img_metas[i],
                    rescale,
                    return_edge=True,
                    **kwargs,
                )
                seg_logit += cur_seg_logit
                edge += cur_edge
            seg_logit /= len(imgs)
            edge /= len(imgs)
            seg_pred = seg_logit.argmax(dim=1)
            edge_pred = edge.sigmoid_()
            seg_pred = seg_pred.cpu().numpy()
            edge_pred = edge_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            edge_pred = list(edge_pred)
            return seg_pred, edge_pred
        else:
            # to save memory, we get augmented seg logit inplace
            seg_logit = self.inference(
                imgs[0],
                img_metas[0],
                rescale,
                return_edge=False,
                **kwargs,
            )
            for i in range(1, len(imgs)):
                cur_seg_logit = self.inference(
                    imgs[i],
                    img_metas[i],
                    rescale,
                    return_edge=False,
                    **kwargs,
                )
                seg_logit += cur_seg_logit
            seg_logit /= len(imgs)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred

    def forward_test(self, imgs, img_metas, return_edge=False, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) != "
                f"num of image meta ({len(img_metas)})"
            )
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_["ori_shape"] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_["img_shape"] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_["pad_shape"] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(
                imgs[0], img_metas[0], return_edge=return_edge, **kwargs
            )
        else:
            return self.aug_test(imgs, img_metas, return_edge=return_edge, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, return_edge=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        ``return_edge=True`` will return edges if the model outputs edges

        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, return_edge=return_edge, **kwargs)

    def show_result(self, img, result, **kwargs):
        """Compatible function with mmseg"""
        seg = result[0]
        return self.show_seg_result(
            img=img,
            seg=seg,
            **kwargs,
        )

    def show_seg_result(
        self,
        img,
        seg,
        palette=None,
        win_name="",
        show=False,
        wait_time=0,
        out_file=None,
        opacity=0.5,
    ):
        img = mmcv.imread(img)
        img = img.copy()
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn(
                "show==False and out_file is not specified, only "
                "result image will be returned"
            )
            return img[..., ::-1]  # convert back to RGB

    def show_edge_result(
        self,
        edges,  # not batched (but has multiple classes)
        out_dir,
        out_prefix,
        beautify=False,
        palette=None,
        beautify_threshold=0.5,
    ):
        is_binary = edges.shape[0] == 1

        if beautify:

            if is_binary:
                raise ValueError("beautify cannot be applied in binary mode")

            out_file = osp.join(
                out_dir,
                out_prefix + ".png",
            )

            mmcv.mkdir_or_exist(osp.dirname(out_file))

            if palette is None:
                if self.PALETTE is None:
                    # Get random state before set seed,
                    # and restore random state later.
                    # It will prevent loss of randomness, as the palette
                    # may be different in each iteration if not specified.
                    # See: https://github.com/open-mmlab/mmdetection/issues/5844
                    state = np.random.get_state()
                    np.random.seed(42)
                    # random palette
                    palette = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
                    np.random.set_state(state)
                else:
                    palette = self.PALETTE
            palette = np.array(palette)
            assert palette.shape[0] == len(self.CLASSES)
            assert palette.shape[1] == 3
            assert len(palette.shape) == 2
            assert 0 < beautify_threshold < 1

            out = beautify_edge(edges, palette, beautify_threshold)
            out = Image.fromarray(out)
            out.save(out_file)
        else:
            if is_binary:
                # we can try to save in a directory that is easy to see
                # but classes can have spaces, which makes it complicated for
                # pyEdgeEval
                out_file = osp.join(
                    out_dir,
                    out_prefix + ".png",
                )

                mmcv.mkdir_or_exist(osp.dirname(out_file))

                edge = deepcopy(edges[0])

                # assuming edge is 0 ~ 1
                edge = (edge * 255).astype(np.uint8)

                out = Image.fromarray(edge)
                out.save(out_file)
            else:
                num_classes, h, w = edges.shape
                assert num_classes == len(self.CLASSES)
                # maybe self.num_classes

                for c in range(num_classes):

                    # we can try to save in a directory that is easy to see
                    # but classes can have spaces, which makes it complicated for
                    # pyEdgeEval
                    # out_file = osp.join(
                    #     out_dir,
                    #     f"class_{c + 1}",  # NOTE: start from 1
                    #     out_prefix + ".png",
                    # )
                    # NOTE: requires zfill filepath
                    out_file = osp.join(
                        out_dir,
                        f"class_{str(c + 1).zfill(3)}",
                        out_prefix + ".png",
                    )

                    mmcv.mkdir_or_exist(osp.dirname(out_file))

                    edge = deepcopy(edges[c])

                    # assuming edge is 0 ~ 1
                    edge = (edge * 255).astype(np.uint8)

                    out = Image.fromarray(edge)
                    out.save(out_file)

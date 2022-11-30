#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize

from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module()
class BasicDetector(BaseDetector):
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
        super(BasicDetector, self).__init__(init_cfg)
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

        self.pass_input_image = pass_input_image

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_edge_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_edge_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_edge_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)  # returns tuple
        if self.with_neck:
            x = self.neck(x)

        if self.pass_input_image:
            # HACK: append input
            x = (*x, img)

        return x

    def encode_decode(self, img, img_metas, **kwargs):
        x = self.extract_feat(img)
        logits = self._decode_head_forward_test(
            x,
            img_metas,
            **kwargs,
        )

        # TODO: check if resize is necessary
        edge = resize(
            input=logits,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return edge

    def _decode_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_edge,
    ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            inputs=x,
            img_metas=img_metas,
            gt_semantic_edge=gt_semantic_edge,
            train_cfg=self.train_cfg,
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas, **kwargs):
        """Run forward function and calculate loss for decode head in
        inference."""
        logits = self.decode_head.forward_test(
            inputs=x,
            img_metas=img_metas,
            test_cfg=self.test_cfg,
            **kwargs,
        )
        return logits

    def _auxiliary_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_edge,
    ):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    inputs=x,
                    img_metas=img_metas,
                    gt_semantic_edge=gt_semantic_edge,
                    train_cfg=self.train_cfg,
                )
                losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                inputs=x,
                img_metas=img_metas,
                gt_semantic_edge=gt_semantic_edge,
                train_cfg=self.train_cfg,
            )
            losses.update(add_prefix(loss_aux, "aux"))

        return losses

    def forward_dummy(self, img, **kwargs):
        """Dummy forward function."""
        logits = self.encode_decode(img, None, **kwargs)
        return logits

    def forward_train(self, img, img_metas, gt_semantic_edge, **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_edge)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x,
                img_metas,
                gt_semantic_edge=gt_semantic_edge,
            )
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(
        self,
        img,
        img_meta,
        rescale,
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

        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))

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
                logits = self.encode_decode(crop_img, img_meta, **kwargs)

                preds += F.pad(
                    logits,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat

        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )

        return preds

    def whole_inference(
        self,
        img,
        img_meta,
        rescale,
        **kwargs,
    ):
        """Inference with full image."""

        logits = self.encode_decode(img, img_meta, **kwargs)

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

    def inference(self, img, img_meta, rescale, **kwargs):
        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)

        if self.test_cfg.mode == "slide":
            edge = self.slide_inference(
                img,
                img_meta,
                rescale,
                **kwargs,
            )
        else:
            edge = self.whole_inference(
                img,
                img_meta,
                rescale,
                **kwargs,
            )

        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                edge = edge.flip(dims=(3,))
            elif flip_direction == "vertical":
                edge = edge.flip(dims=(2,))
        return edge

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        edge = self.inference(
            img,
            img_meta,
            rescale,
            **kwargs,
        )
        edge_pred = edge.sigmoid_()

        # NOTE: converts to numpy
        edge_pred = edge_pred.cpu().numpy()
        # unravel batch dim
        edge_pred = list(edge_pred)
        return edge_pred

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        edge = self.inference(
            imgs[0],
            img_metas[0],
            rescale,
            **kwargs,
        )
        for i in range(1, len(imgs)):
            cur_edge = self.inference(
                imgs[i],
                img_metas[i],
                rescale,
                **kwargs,
            )
            edge += cur_edge
        edge /= len(imgs)
        edge_pred = edge.sigmoid_()
        edge_pred = edge_pred.cpu().numpy()
        # unravel batch dim
        edge_pred = list(edge_pred)
        return edge_pred

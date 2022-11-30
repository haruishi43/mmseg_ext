#!/usr/bin/env python3

import torch.nn.functional as F
from mmseg.ops import resize

from ..builder import SEGMENTORS
from .base import BaseJointSegDet


@SEGMENTORS.register_module()
class BinaryJointEncoderDecoder(BaseJointSegDet):
    """JointEncoderDecoder for binary datasets

    - Segmentation maps have two classes (background, target)
    - Edges only have one class (edge or not; binary)
    """

    num_classes = 2
    num_edge_classes = 1

    def __init__(
        self,
        **kwargs,
    ):
        super(BinaryJointEncoderDecoder, self).__init__(**kwargs)
        assert (
            self.num_classes == 2
        ), f"for binary segmentation, num_classes should be 2, but got {self.num_classes}"

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        gt_semantic_edge,
        **kwargs,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `potato/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if len(gt_semantic_edge.shape) == 3:
            # might not have channel dim
            gt_semantic_edge = gt_semantic_edge.unsqueeze(1)

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
            num_edge_classes = self.num_edge_classes
            edge_preds = img.new_zeros((batch_size, num_edge_classes, h_img, w_img))

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

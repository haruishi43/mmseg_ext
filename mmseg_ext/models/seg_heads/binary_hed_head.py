#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmseg.ops import resize

from mmseg_ext.models.builder import SEG_HEADS
from mmseg_ext.models.losses import accuracy, edge_accuracy
from .multitask_decode_head import BaseMultiTaskDecodeHead


@SEG_HEADS.register_module()
class BinaryHEDHead(BaseMultiTaskDecodeHead):
    def __init__(
        self,
        merging="attention",
        deep_supervision=True,
        **kwargs,
    ):
        super().__init__(
            input_transform="multiple_select",
            binary_edge=True,
            no_conv_seg=True,
            init_cfg=dict(type="Normal", std=0.01),
            **kwargs,
        )

        self.seg_branch = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=i,
                    out_channels=self.num_classes,  # 2
                    kernel_size=1,
                )
                for i in self.in_channels
            ]
        )

        self.edge_branch = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=i,
                    out_channels=1,  # 1
                    kernel_size=1,
                )
                for i in self.in_channels
            ]
        )

        self.deep_supervision = deep_supervision
        self.merging = merging
        if merging == "attention":
            self.seg_queries = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=i,
                        out_channels=self.num_classes,  # 2
                        kernel_size=1,
                    )
                    for i in self.in_channels
                ]
            )
            self.edge_queries = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=i,
                        out_channels=1,  # 1
                        kernel_size=1,
                    )
                    for i in self.in_channels
                ]
            )
        elif merging == "learned":
            self.merge_seg = nn.Conv2d(
                in_channels=self.num_classes * len(self.in_channels),
                out_channels=self.num_classes,  # 2
                kernel_size=1,
            )
            self.merge_edge = nn.Conv2d(
                in_channels=self.num_classes * len(self.in_channels),
                out_channels=1,  # 1
                kernel_size=1,
            )
        else:
            # no merging
            pass

    def forward(self, inputs):
        inputs = list(inputs)
        img = inputs.pop(-1)  # pop img
        b, _, h, w = img.shape

        # seg and edge branch
        seg_preds = []
        edge_preds = []
        for x, sconv, econv in zip(inputs, self.seg_branch, self.edge_branch):
            seg_preds.append(sconv(x))
            edge_preds.append(econv(x))

        if self.merging == "attention":
            seg_queries = torch.cat(
                [
                    resize(
                        input=q(feat),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for q, feat in zip(self.seg_queries, inputs)
                ],
                dim=1,
            )
            edge_queries = torch.cat(
                [
                    resize(
                        input=q(feat),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for q, feat in zip(self.edge_queries, inputs)
                ],
                dim=1,
            )
            segs = torch.cat(
                [
                    resize(
                        input=p,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for p in seg_preds
                ],
                dim=1,
            )
            edges = torch.cat(
                [
                    resize(
                        input=p,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for p in edge_preds
                ],
                dim=1,
            )
            seg_queries = seg_queries.reshape(b, -1, 2, h, w)
            attn = torch.softmax(seg_queries, dim=1)
            segs = segs.reshape(b, -1, 2, h, w)
            combined_seg = torch.sum(attn * segs, dim=1)

            edge_queries = edge_queries.reshape(b, -1, 1, h, w)
            attn = torch.softmax(edge_queries, dim=1)
            edges = edges.reshape(b, -1, 1, h, w)
            combined_edge = torch.sum(attn * edges, dim=1)

        elif self.merging == "learned":
            segs = torch.cat(
                [
                    resize(
                        input=p,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for p in seg_preds
                ],
                dim=1,
            )
            edges = torch.cat(
                [
                    resize(
                        input=p,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for p in edge_preds
                ],
                dim=1,
            )
            combined_seg = self.merge_seg(segs)
            combined_edge = self.merge_edge(edges)
        else:
            combined_seg = seg_preds[-1]
            combined_edge = edge_preds[-1]

        if self.deep_supervision:
            seg_preds.append(combined_seg)
            edge_preds.append(combined_edge)
            return seg_preds, edge_preds
        else:
            return combined_seg, combined_edge

    @force_fp32(apply_to=("seg_logit",))
    def multi_seg_losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()

        for s in seg_logit:
            s = resize(
                input=s,
                size=seg_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            if self.sampler is not None:
                seg_weight = self.sampler.sample(s, seg_label)
            else:
                seg_weight = None
            l = seg_label.squeeze(1)

            if not isinstance(self.loss_seg, nn.ModuleList):
                losses_seg = [self.loss_seg]
            else:
                losses_seg = self.loss_seg
            for loss_seg in losses_seg:
                if loss_seg.loss_name not in loss:
                    loss[loss_seg.loss_name] = loss_seg(
                        s,
                        l,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                    )
                else:
                    loss[loss_seg.loss_name] += loss_seg(
                        s,
                        l,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                    )

        loss["acc_seg"] = accuracy(seg_logit[-1], seg_label.squeeze(1))
        return loss

    @force_fp32(apply_to=("edge_logit",))
    def multi_edge_losses(self, edge_logit, edge_label):
        """Compute edge loss."""
        loss = dict()

        for e in edge_logit:
            e = resize(
                input=e,
                size=edge_label.shape[2:],  # (b, cls, h, w)
                mode="bilinear",
                align_corners=self.align_corners,
            )

            if not isinstance(self.loss_edge, nn.ModuleList):
                losses_edge = [self.loss_edge]
            else:
                losses_edge = self.loss_edge
            for loss_edge in losses_edge:
                if loss_edge.loss_name not in loss:
                    loss[loss_edge.loss_name] = loss_edge(
                        e,
                        edge_label,
                        ignore_index=self.ignore_index,
                    )
                else:
                    loss[loss_edge.loss_name] += loss_edge(
                        e,
                        edge_label,
                        ignore_index=self.ignore_index,
                    )

        for metric_name, v in edge_accuracy(edge_logit[-1], edge_label).items():
            loss["edge" + "_" + metric_name] = v

        return loss

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        gt_semantic_edge,
        train_cfg,
    ):
        # convert multilabel to binary edge
        if self.binary_edge and gt_semantic_edge.shape[1] != 1:
            # convert to binary
            gt_semantic_edge = (
                (torch.sum(gt_semantic_edge, axis=1) > 0).unsqueeze(1).float()
            )

        seg_logits, edge_logits = self(inputs)

        if self.deep_supervision:
            losses = self.multi_seg_losses(
                seg_logit=seg_logits,
                seg_label=gt_semantic_seg,
            )
            losses.update(
                self.multi_edge_losses(
                    edge_logit=edge_logits,
                    edge_label=gt_semantic_edge,
                )
            )
        else:
            # Assume that there are always seg_losses
            losses = self.seg_losses(
                seg_logit=seg_logits,
                seg_label=gt_semantic_seg,
            )
            losses.update(
                self.edge_losses(
                    edge_logit=edge_logits,
                    edge_label=gt_semantic_edge,
                )
            )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, return_edge=False, **kwargs):
        seg, edge = self(inputs)
        if self.deep_supervision:
            seg = seg[-1]
            edge = edge[-1]

        if return_edge:
            return dict(seg=seg, edge=edge)
        else:
            return dict(seg=seg)

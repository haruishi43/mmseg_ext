#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..builder import SEG_LOSSES
from mmseg_ext.ops.image_processing import compute_grad_mag


def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input


def _sample_gumbel(shape, eps=1e-10, device=torch.device("cpu")):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).to(device)
    return -torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """

    device = labels.device

    y = torch.eye(num_classes).to(device)
    return y[labels].permute(0, 3, 1, 2)


@SEG_LOSSES.register_module()
class DualTaskLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        loss_weight=1.0,
        loss_name="loss_dual_task",
    ):
        super(DualTaskLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        ignore_pixel=255,
        **kwargs,
    ):
        """
        Args:
            input_logits: NxCxHxW
            gt_semantic_masks: NxCxHxW

        Returns:
            final loss
        """
        N, C, H, W = cls_score.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        device = cls_score.device
        ignore_mask = (label == ignore_pixel).detach()
        cls_score = torch.where(
            ignore_mask.view(N, 1, H, W).expand(N, self.num_classes, H, W),
            torch.zeros(N, C, H, W).to(device),
            cls_score,
        )
        gt_semantic_masks = label.detach()
        gt_semantic_masks = torch.where(
            ignore_mask, torch.zeros(N, H, W).long().to(device), gt_semantic_masks
        )
        gt_semantic_masks = _one_hot_embedding(
            gt_semantic_masks, self.num_classes
        ).detach()

        g = _gumbel_softmax_sample(cls_score.view(N, C, -1), tau=0.5)
        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g)  # FIXME: what is this for?

        g_hat = compute_grad_mag(gt_semantic_masks)

        # general pixel-wise loss
        g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction="none")

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (
            torch.sum(p_plus_g_mask) + eps
        )

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (
            torch.sum(p_plus_g_hat_mask) + eps
        )

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return self.loss_weight * total_loss

    @property
    def loss_name(self):
        return self._loss_name

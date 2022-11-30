#!/usr/bin/env python3

"""Canny Edge"""

from copy import deepcopy

import cv2
import numpy as np
import torch


class UnNormalize(object):
    def __init__(
        self,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        inplace=True,
    ):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
                Supports batched tensor.
        Returns:
            Tensor: Normalized image.
        """

        if tensor.ndim == 4:
            tensor = torch.permute(tensor, (1, 0, 2, 3)).contiguous()
        assert tensor.shape[0] == len(self.mean) == len(self.std)

        if self.inplace:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)

            if tensor.ndim == 4:
                tensor = torch.permute(tensor, (1, 0, 2, 3)).contiguous()

            return tensor
        else:
            dtype = tensor.dtype
            device = tensor.device
            mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
            std = torch.as_tensor(self.std, dtype=dtype, device=device)

            out = torch.zeros_like(tensor)
            for i, (t, m, s) in enumerate(zip(tensor, mean, std)):
                out[i] = t * s + m

            if out.ndim == 4:
                out = torch.permute(out, (1, 0, 2, 3)).contiguous()

            return out


def canny_cv2(
    img: np.ndarray,
    thresh_low: int = 10,
    thresh_high: int = 100,
):
    """Canny Edge

    Args:
        img (np.ndarray): (batch, c, h, w), np.uint8
    """

    if img.ndim == 3:
        # add batch direction
        img = img[None, ...]
    assert img.ndim == 4

    bs = img.size(0)
    bs, h, w, c = img.shape
    out = np.zeros((bs, 1, h, w))
    for i in range(bs):
        # should the input be RGB or Gray?
        _img = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)

        # values are 0 ~ 255
        out[i] = cv2.Canny(_img, thresh_low, thresh_high)

    # NOTE: normalize output
    out = out / 255

    if out.ndim == 4 and out.shape[0] == 1:
        out = out.squeeze(0)

    return out


class UnNormalizedCanny(object):
    def __init__(
        self,
        low_threshold=10,
        high_threshold=100,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        inplace=True,
    ):
        self.thresh_low = low_threshold
        self.thresh_high = high_threshold
        self.unnorm = UnNormalize(
            mean=mean,
            std=std,
            inplace=inplace,
        )

    def __call__(self, img):
        non_batch = False
        if img.ndim == 3:
            # add batch direction
            non_batch = True
            img = img.unsqueeze(0)
        assert img.ndim == 4
        bs, c, h, w = img.shape
        device = img.device

        # NOTE: need to normalize since the threashold values are larger than 1
        im_ten = self.unnorm(deepcopy(img.detach().cpu()))
        # NOTE: cllip before casting
        im_arr = im_ten.numpy().transpose((0, 2, 3, 1)).clip(0, 255).astype(np.uint8)
        canny = np.zeros((bs, 1, h, w))
        for i in range(bs):
            # should the input be RGB or Gray?
            _img = cv2.cvtColor(im_arr[i], cv2.COLOR_RGB2GRAY)
            # values are 0 ~ 255
            canny[i] = cv2.Canny(_img, self.thresh_low, self.thresh_high)
        # NOTE: normalize output
        canny = canny / 255.0
        canny = torch.from_numpy(canny).to(device).float()

        if non_batch:
            canny = canny.squeeze(0)

        return canny

#!/usr/bin/env python3

import os.path as osp
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import mmcv
from mmcv.runner import BaseModule, auto_fp16


def apply_mask(image, mask, color):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] + color[c], image[:, :, c])
    return image


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for Joint Modeling."""

    CLASSES = None
    PALETTE = None

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self, "auxiliary_head") and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, "decode_head") and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas, **kwargs):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
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
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"])
        )

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        # NOTE: not used: cfg.workflow = [('train', 1)]
        # could be used when 'val' is added to the workflow
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"])
        )

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (
                f"rank {dist.get_rank()}"
                + f" len(log_vars): {len(log_vars)}"
                + " keys: "
                + ",".join(log_vars.keys())
                + "\n"
            )
            assert log_var_length == len(log_vars) * dist.get_world_size(), (
                "loss log variables are different across GPUs!\n" + message
            )

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(
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

            n, h, w = edges.shape
            out = np.zeros((h, w, 3))
            edges = np.where(edges >= beautify_threshold, 1, 0).astype(bool)
            edge_sum = np.zeros((h, w))

            for i in range(n):
                color = palette[i]
                edge = edges[i, :, :]
                edge_sum = edge_sum + edge
                masked_out = apply_mask(out, edge, color)

            edge_sum = np.array([edge_sum, edge_sum, edge_sum])
            edge_sum = np.transpose(edge_sum, (1, 2, 0))
            idx = edge_sum > 0
            masked_out[idx] = masked_out[idx] / edge_sum[idx]
            masked_out[~idx] = 255

            out = masked_out.astype(np.uint8)

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

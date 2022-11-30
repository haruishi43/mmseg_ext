#!/usr/bin/env python3
# Modified mmseg (Apache License)

import random
import warnings

import numpy as np
import torch
import torch.distributed as dist

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner, get_dist_info

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook
from mmseg.utils import find_latest_checkpoint

from mmseg_ext.datasets import build_dataloader, build_seg_dataset
from mmseg_ext.core import EvalSegHook, DistEvalSegHook
from mmseg_ext.utils import get_root_logger


def init_random_seed(seed=None, device="cuda"):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_seg_det(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    validate_edge=False,  # add edge validation
    timestamp=None,
    meta=None,
):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version(
                "1.4.4"
            ), "Please use MMCV >= 1.4.4 for CPU training!"
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get("runner") is None:
        cfg.runner = {"type": "IterBasedRunner", "max_iters": cfg.total_iters}
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_seg_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"

        if cfg.model.type == "JointEncoderDecoder":
            # joint model
            eval_hook = DistEvalSegHook if distributed else EvalSegHook
        else:
            # only seg
            eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

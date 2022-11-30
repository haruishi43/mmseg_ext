#!/usr/bin/env python3

import os.path as osp
import tempfile

import numpy as np
import torch

import mmcv
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir=tmpdir
        ).name
    np.save(temp_file_name, array)
    return temp_file_name


def inference(
    model,
    data_loader,
    out_dir=None,
    save=False,
    vis_edge=False,
    vis_dir=None,
    rescale=True,
):
    if save:
        mmcv.mkdir_or_exist(out_dir)

    if vis_edge:
        assert save, "need to have `save=True` in order to visualize"
        mmcv.mkdir_or_exist(vis_dir)

    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            edges = model(
                return_loss=False,
                rescale=rescale,
                **data,
            )
            assert isinstance(
                edges, list
            ), f"edges should be a list but got {type(edges)}"

        img_metas = data["img_metas"][0].data[0]

        assert len(edges) == len(img_metas)

        if save:
            for edge, img_meta in zip(edges, img_metas):  # NOTE: edges
                model.module.show_result(
                    edges=edge,
                    out_dir=out_dir,
                    out_prefix=osp.splitext(img_meta["ori_filename"])[0],
                    beautify=False,
                    palette=dataset.PALETTE,
                )
                if vis_edge:
                    # save visualized edge if needed
                    model.module.show_result(
                        edges=edge,
                        out_dir=vis_dir,
                        out_prefix=osp.splitext(img_meta["ori_filename"])[0],
                        beautify=True,
                        palette=dataset.PALETTE,
                        beautify_threshold=0.7,  # TODO: which value should we use?
                    )

        batch_size = len(edges)
        for _ in range(batch_size):
            prog_bar.update()


def single_gpu_edge_test(
    model,
    data_loader,
    out_dir=None,
    pre_eval=False,
    save=False,
    vis_edge=False,
    vis_dir=None,
):
    if save:
        mmcv.mkdir_or_exist(out_dir)

    if vis_edge:
        assert save, "need to have `save=True` in order to visualize"
        mmcv.mkdir_or_exist(vis_dir)

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            edges = model(
                return_loss=False,
                **data,
            )
            assert isinstance(
                edges, list
            ), f"edges should be a list but got {type(edges)}"

        img_metas = data["img_metas"][0].data[0]

        assert len(edges) == len(img_metas)

        if save:
            for edge, img_meta in zip(edges, img_metas):  # NOTE: edges
                model.module.show_result(
                    edges=edge,
                    out_dir=out_dir,
                    out_prefix=osp.splitext(img_meta["ori_filename"])[0],
                    beautify=False,
                    palette=dataset.PALETTE,
                )
                if vis_edge:
                    # save visualized edge if needed
                    model.module.show_result(
                        edges=edge,
                        out_dir=vis_dir,
                        out_prefix=osp.splitext(img_meta["ori_filename"])[0],
                        beautify=True,
                        palette=dataset.PALETTE,
                        beautify_threshold=0.7,  # TODO: which value should we use?
                    )

        if pre_eval:
            # technically not edge but result
            edges = dataset.pre_eval(edges, batch_indices)

        results.extend(edges)

        batch_size = len(edges)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_edge_test(
    model,
    data_loader,
    pre_eval=False,
    tmpdir=None,
    gpu_collect=False,
):
    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if pre_eval:
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

#!/usr/bin/env python3

from collections import OrderedDict

import torch


def pre_calc(
    result,
    gt_edge_map,
    num_classes,
    thresh=0.7,
):
    tp = torch.zeros((num_classes,), dtype=torch.float64)
    preds = torch.zeros((num_classes,), dtype=torch.float64)
    targets = torch.zeros((num_classes,), dtype=torch.float64)
    acc = torch.zeros((num_classes,), dtype=torch.float64)

    pred = torch.from_numpy(result)
    label = torch.from_numpy(gt_edge_map)

    # FIXME: very ugly
    if num_classes == 1:
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if label.ndim == 2:
            label = label.unsqueeze(0)

    assert pred.shape == label.shape
    assert pred.ndim == 3

    # print('pred', pred.shape)
    # print('lab', label.shape)

    # threshold
    t_pred = pred > thresh
    t_label = label > thresh

    for i in range(num_classes):
        _pred = t_pred[i]
        _label = t_label[i]

        tp[i] = _pred[_pred == _label].sum()
        preds[i] = _pred.sum()
        targets[i] = _label.sum()
        acc[i] = _pred.eq(_label).sum() / _label.numel()
        # print(_label.numel(), acc[i])

    return (tp, preds, targets, acc)


def pre_eval_to_metrics(
    pre_eval_results,
    metrics=["Fscore"],
):
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4
    all_tp = sum(pre_eval_results[0])
    all_preds = sum(pre_eval_results[1])
    all_targets = sum(pre_eval_results[2])
    all_acc = sum(pre_eval_results[3]) / len(pre_eval_results[3])

    num_classes = len(all_acc)

    # organize metrics that we should return
    g_acc = all_acc.sum() / num_classes
    ret_metrics = OrderedDict({"aAcc": g_acc})
    for metric in metrics:
        if metric == "Fscore":
            # ret_metrics["Acc"] = all_acc

            prec = all_tp / all_preds
            rec = all_tp / all_targets
            f1 = 2 * (prec * rec) / (prec + rec)

            ret_metrics["Precision"] = prec
            ret_metrics["Recall"] = rec
            ret_metrics["Fscore"] = f1

    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}

    return ret_metrics


def eval_metrics(
    results,
    gt_edge_maps,
    num_classes,
    metrics=["Fscore"],
    thresh=0.7,
):
    all_tp = torch.zeros((num_classes,), dtype=torch.float64)
    all_preds = torch.zeros((num_classes,), dtype=torch.float64)
    all_targets = torch.zeros((num_classes,), dtype=torch.float64)
    all_acc = torch.zeros((num_classes,), dtype=torch.float64)

    for result, gt_edge_map in zip(results, gt_edge_maps):
        # convert to numpy
        pred = torch.from_numpy(result)
        label = torch.from_numpy(gt_edge_map)

        # FIXME: very ugly
        if num_classes == 1:
            if pred.ndim == 2:
                pred = pred.unsqueeze(0)
            if label.ndim == 2:
                label = label.unsqueeze(0)

        assert pred.shape == label.shape
        assert pred.ndim == 3

        # threshold
        t_pred = pred > thresh
        t_label = label > thresh

        for i in range(num_classes):
            _pred = t_pred[i]
            _label = t_label[i]

            tp = _pred[_pred == _label].sum()
            num_preds = _pred.sum()
            num_labels = _label.sum()
            acc = _pred.eq(_label).sum() / _label.numel()

            all_tp[i] += tp
            all_preds[i] += num_preds
            all_targets[i] += num_labels
            all_acc[i] += acc

    # organize metrics that we should return
    g_acc = all_acc.sum() / len(results) / num_classes
    ret_metrics = OrderedDict({"aAcc": g_acc})
    for metric in metrics:
        if metric == "Fscore":
            # ret_metrics["Acc"] = all_acc

            prec = all_tp / all_preds
            rec = all_tp / all_targets
            f1 = 2 * (prec * rec) / (prec + rec)

            ret_metrics["Precision"] = prec
            ret_metrics["Recall"] = rec
            ret_metrics["Fscore"] = f1

    return ret_metrics

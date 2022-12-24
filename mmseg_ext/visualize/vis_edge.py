#!/usr/bin/env python3

import numpy as np


def beautify_edge(
    edges,
    palette,
    beautify_threshold=0.5,
):
    assert 0 < beautify_threshold < 1

    n, h, w = edges.shape
    out = np.zeros((h, w, 3))
    edges = np.where(edges >= beautify_threshold, 1, 0).astype(bool)
    edge_sum = np.zeros((h, w))

    for i in range(n):
        color = palette[i]
        edge = edges[i, :, :]
        edge_sum = edge_sum + edge
        for c in range(3):
            out[:, :, c] = np.where(edge == 1, out[:, :, c] + color[c], out[:, :, c])

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    out[idx] = out[idx] / edge_sum[idx]
    out[~idx] = 255

    edges = out.astype(np.uint8)
    assert edges.ndim == 3, f"ERR: should not be {edges.ndim}dim output"
    return edges

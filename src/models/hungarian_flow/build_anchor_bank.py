#!/usr/bin/env python
"""Build a high-resolution, Replogle-compatible fixed-capacity control bank."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def resolve_anchor_count(
    num_cells,
    capacity=128,
    anchors_per_cell_line=0,
    target_cells_per_anchor=256,
    minimum=16,
):
    """Resolve an offline active count while keeping model capacity fixed."""
    if int(anchors_per_cell_line) > 0:
        count = int(anchors_per_cell_line)
    else:
        count = int(round(float(num_cells) / float(target_cells_per_anchor)))
        count = max(int(minimum), count)
    return min(int(capacity), int(num_cells), count)


def batch_center(values, batches):
    """Remove batch means only in the clustering representation."""
    centered = values.astype(np.float32, copy=True)
    global_mean = centered.mean(axis=0, keepdims=True)
    for batch in np.unique(batches):
        mask = batches == batch
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
        centered[mask] += global_mean
    return centered


def representative_indices(distances, pool_size):
    """Cover cluster cores and edges deterministically by distance quantiles."""
    order = np.argsort(np.asarray(distances), kind="stable")
    if len(order) >= int(pool_size):
        positions = np.linspace(0, len(order) - 1, int(pool_size)).round().astype(int)
        return order[positions]
    repeats = int(np.ceil(float(pool_size) / float(max(len(order), 1))))
    return np.tile(order, repeats)[: int(pool_size)]


def merge_small_clusters(latent, labels, centers, minimum_size):
    """Merge unsupported outlier clusters into their nearest supported center."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    counts = np.bincount(labels, minlength=len(centers))
    supported = np.flatnonzero(counts >= int(minimum_size))
    if len(supported) == 0:
        supported = np.asarray([int(np.argmax(counts))])
    unsupported = np.setdiff1d(np.arange(len(centers)), supported)
    for anchor in unsupported:
        indices = np.flatnonzero(labels == anchor)
        if len(indices) == 0:
            continue
        distances = (
            latent[indices, None, :] - centers[supported][None, :, :]
        )
        distances = np.square(distances).sum(axis=-1)
        labels[indices] = supported[np.argmin(distances, axis=1)]

    active = np.unique(labels)
    remap = {int(old): new for new, old in enumerate(active.tolist())}
    repaired = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    repaired_centers = np.stack(
        [latent[repaired == index].mean(axis=0) for index in range(len(active))]
    )
    return repaired, repaired_centers



def build_cell_line_context(
    raw_values,
    batches,
    num_anchors,
    pca_dim,
    pool_size,
    seed,
    minimum_cluster_size=32,
):
    discovery = batch_center(raw_values, batches)
    components = min(int(pca_dim), discovery.shape[0] - 1, discovery.shape[1])
    latent = PCA(
        n_components=max(2, components),
        svd_solver="randomized",
        random_state=int(seed),
    ).fit_transform(discovery)
    clustering = MiniBatchKMeans(
        n_clusters=int(num_anchors),
        batch_size=min(2048, max(256, len(latent))),
        n_init=10,
        random_state=int(seed),
        reassignment_ratio=0.0,
    )
    labels = clustering.fit_predict(latent)
    labels, repaired_centers = merge_small_clusters(
        latent, labels, clustering.cluster_centers_, minimum_cluster_size
    )
    active_anchors = int(len(repaired_centers))

    means, stds, weights, pools, counts = [], [], [], [], []
    compactness = []
    for anchor in range(active_anchors):
        indices = np.flatnonzero(labels == anchor)
        values = raw_values[indices]
        local_distance = np.linalg.norm(
            latent[indices] - repaired_centers[anchor], axis=1
        )
        selected = representative_indices(local_distance, pool_size)
        means.append(values.mean(axis=0))
        stds.append(values.std(axis=0))
        weights.append(float(len(indices)) / float(len(raw_values)))
        pools.append(values[selected])
        counts.append(len(indices))
        compactness.append(float(np.mean(local_distance ** 2)))

    sampled = np.arange(len(latent))
    if len(sampled) > 10000:
        rng = np.random.default_rng(int(seed))
        sampled = rng.choice(sampled, size=10000, replace=False)
    silhouette = float(silhouette_score(latent[sampled], labels[sampled]))
    return {
        "states": [f"fine_{index:03d}" for index in range(active_anchors)],
        "prototype_means": np.stack(means).astype(np.float32),
        "prototype_stds": np.stack(stds).astype(np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "weighted_mean": np.average(
            np.stack(means), axis=0, weights=np.asarray(weights)
        ).astype(np.float32),
        "cluster_cell_tokens": np.stack(pools).astype(np.float32),
        "cluster_cell_counts": np.asarray(counts, dtype=np.int64),
        "diagnostics": {
            "num_cells": int(len(raw_values)),
            "num_requested_anchors": int(num_anchors),
            "num_active_anchors": active_anchors,
            "mean_cells_per_anchor": float(np.mean(counts)),
            "min_cells_per_anchor": int(np.min(counts)),
            "max_cells_per_anchor": int(np.max(counts)),
            "latent_silhouette": silhouette,
            "mean_within_anchor_squared_distance": float(np.mean(compactness)),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", default="X_hvg")
    parser.add_argument("--perturbation-key", default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--cell-line-key", default="cell_line")
    parser.add_argument("--batch-key", default="gem_group")
    parser.add_argument("--anchor-capacity", type=int, choices=(64, 128), default=128)
    parser.add_argument("--anchors-per-cell-line", type=int, default=0)
    parser.add_argument("--target-cells-per-anchor", type=int, default=256)
    parser.add_argument("--minimum-anchors", type=int, default=16)
    parser.add_argument("--minimum-cluster-size", type=int, default=32)
    parser.add_argument("--cells-per-anchor-pool", type=int, default=128)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    adata = ad.read_h5ad(args.h5ad, backed="r")
    contexts = {}
    try:
        obs = adata.obs
        control_mask = obs[args.perturbation_key].astype(str).eq(args.control_label)
        for offset, cell_line in enumerate(
            sorted(obs.loc[control_mask, args.cell_line_key].astype(str).unique())
        ):
            mask = control_mask & obs[args.cell_line_key].astype(str).eq(cell_line)
            positions = np.flatnonzero(mask.to_numpy())
            raw = np.asarray(adata.obsm[args.feature_key][positions], dtype=np.float32)
            batches = obs.iloc[positions][args.batch_key].astype(str).to_numpy()
            count = resolve_anchor_count(
                len(positions),
                capacity=args.anchor_capacity,
                anchors_per_cell_line=args.anchors_per_cell_line,
                target_cells_per_anchor=args.target_cells_per_anchor,
                minimum=args.minimum_anchors,
            )
            context = build_cell_line_context(
                raw,
                batches,
                num_anchors=count,
                pca_dim=args.pca_dim,
                pool_size=args.cells_per_anchor_pool,
                seed=args.seed + offset,
                minimum_cluster_size=args.minimum_cluster_size,
            )
            contexts[cell_line] = context
            print(cell_line, context["diagnostics"])
    finally:
        adata.file.close()

    bank = {
        "schema": "hungarian_flow_fixed_anchor_bank_v1",
        "context_by_cell_line": contexts,
        "metadata": {
            "source_h5ad": str(args.h5ad.resolve()),
            "feature_key": args.feature_key,
            "control_label": args.control_label,
            "anchor_capacity": args.anchor_capacity,
            "anchors_per_cell_line": args.anchors_per_cell_line,
            "target_cells_per_anchor": args.target_cells_per_anchor,
            "minimum_anchors": args.minimum_anchors,
            "minimum_cluster_size": args.minimum_cluster_size,
            "cells_per_anchor_pool": args.cells_per_anchor_pool,
            "pca_dim": args.pca_dim,
            "seed": args.seed,
            "clustering_space": "batch-centered PCA",
            "prototype_space": "original X_hvg",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(bank, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

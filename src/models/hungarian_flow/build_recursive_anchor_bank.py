#!/usr/bin/env python
"""Build a non-uniform recursive control-state Child bank."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from .build_anchor_bank import batch_center
from .context_artifact import RECURSIVE_CONTEXT_SCHEMA, sha256_file


@dataclass(frozen=True)
class RecursivePartition:
    labels: np.ndarray
    active_children: int
    within_child_sse: np.ndarray
    split_gains: np.ndarray
    fallback_splits: int
    stopped_reason: str


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    result = (
        np.square(values).sum(axis=1, keepdims=True)
        + np.square(centers).sum(axis=1)[None, :]
        - 2.0 * values @ centers.T
    )
    return np.maximum(result, 0.0)


def _repair_minimum_binary_support(
    labels: np.ndarray,
    distances: np.ndarray,
    minimum: int,
) -> np.ndarray:
    """Repair a two-way assignment with the smallest incremental SSE moves."""

    labels = np.asarray(labels, dtype=np.int64).copy()
    counts = np.bincount(labels, minlength=2)
    for small in (0, 1):
        need = int(minimum) - int(counts[small])
        if need <= 0:
            continue
        large = 1 - small
        candidates = np.flatnonzero(labels == large)
        if len(candidates) - need < int(minimum):
            raise ValueError("binary split cannot satisfy minimum leaf support")
        incremental = distances[candidates, small] - distances[candidates, large]
        moved = candidates[np.argsort(incremental, kind="stable")[:need]]
        labels[moved] = small
        counts[small] += need
        counts[large] -= need
    if (np.bincount(labels, minlength=2) < int(minimum)).any():
        raise RuntimeError("minimum-support repair failed")
    return labels


def _fallback_binary_labels(
    values: np.ndarray,
    minimum: int,
) -> np.ndarray:
    """Deterministically bisect even a degenerate/duplicate-valued leaf."""

    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    variance = np.square(centered).sum(axis=0)
    axis = int(np.argmax(variance))
    order = np.argsort(values[:, axis], kind="stable")
    cut = int(np.clip(len(values) // 2, int(minimum), len(values) - int(minimum)))
    labels = np.ones(len(values), dtype=np.int64)
    labels[order[:cut]] = 0
    return labels


def _constrained_binary_split(
    values: np.ndarray,
    *,
    minimum: int,
    seed: int,
    n_init: int,
    refinement_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run two-means followed by a local minimum-support constrained Lloyd step."""

    if len(values) < 2 * int(minimum):
        raise ValueError("leaf is too small for a legal binary split")
    model = KMeans(
        n_clusters=2,
        n_init=int(n_init),
        max_iter=100,
        algorithm="lloyd",
        random_state=int(seed),
    ).fit(values)
    labels = np.asarray(model.labels_, dtype=np.int64)
    fallback = np.unique(labels).size != 2
    if fallback:
        labels = _fallback_binary_labels(values, minimum)
    centers = np.stack(
        [values[labels == side].mean(axis=0) for side in range(2)]
    ).astype(np.float32)
    for _ in range(int(refinement_iterations)):
        distances = _squared_distances(values, centers)
        updated = distances.argmin(axis=1).astype(np.int64)
        if np.unique(updated).size != 2:
            updated = _fallback_binary_labels(values, minimum)
            fallback = True
        updated = _repair_minimum_binary_support(updated, distances, minimum)
        new_centers = np.stack(
            [values[updated == side].mean(axis=0) for side in range(2)]
        ).astype(np.float32)
        if np.array_equal(updated, labels):
            labels, centers = updated, new_centers
            break
        labels, centers = updated, new_centers
    return labels, centers, fallback


def _leaf_sse(values: np.ndarray, indices: np.ndarray) -> float:
    local = values[indices].astype(np.float64)
    center = local.mean(axis=0, dtype=np.float64)
    return float(np.square(local - center).sum())


def recursive_sse_partition(
    latent: np.ndarray,
    *,
    max_children: int = 128,
    min_leaf_size: int = 20,
    seed: int = 42,
    n_init: int = 5,
) -> RecursivePartition:
    """Recursively split the current highest-SSE/large legal leaf.

    The priority is ``SSE * (1 + log1p(size/min_leaf_size))``.  If one chosen
    leaf is degenerate, a stable median split keeps the recursion moving.  The
    search stops only at ``max_children`` or when no leaf has at least
    ``2*min_leaf_size`` members.
    """

    values = np.asarray(latent, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("latent must have non-empty shape [N,D]")
    if not np.isfinite(values).all():
        raise ValueError("latent contains non-finite values")
    max_children = int(max_children)
    min_leaf_size = int(min_leaf_size)
    if max_children < 1 or min_leaf_size < 2:
        raise ValueError("max_children >= 1 and min_leaf_size >= 2 are required")
    if len(values) < min_leaf_size:
        raise ValueError("not enough cells for one legal leaf")

    next_id = 1
    leaves = {
        0: {
            "indices": np.arange(len(values), dtype=np.int64),
            "sse": _leaf_sse(values, np.arange(len(values), dtype=np.int64)),
        }
    }
    gains = []
    fallback_splits = 0
    split_number = 0
    while len(leaves) < max_children:
        legal = []
        for leaf_id, leaf in leaves.items():
            size = len(leaf["indices"])
            if size < 2 * min_leaf_size:
                continue
            priority = float(leaf["sse"]) * (
                1.0 + math.log1p(float(size) / float(min_leaf_size))
            )
            legal.append((priority, float(leaf["sse"]), size, -leaf_id, leaf_id))
        if not legal:
            stopped_reason = "no_legal_leaf"
            break
        leaf_id = max(legal)[-1]
        parent = leaves.pop(leaf_id)
        indices = parent["indices"]
        local_labels, _, fallback = _constrained_binary_split(
            values[indices],
            minimum=min_leaf_size,
            seed=int(seed) + 1009 * split_number + 17 * int(leaf_id),
            n_init=n_init,
        )
        fallback_splits += int(fallback)
        children = []
        child_sse = 0.0
        for side in range(2):
            child_indices = indices[local_labels == side]
            sse = _leaf_sse(values, child_indices)
            child_sse += sse
            children.append((child_indices, sse))
        gains.append(float(parent["sse"]) - child_sse)
        for child_indices, sse in children:
            leaves[next_id] = {"indices": child_indices, "sse": sse}
            next_id += 1
        split_number += 1
    else:
        stopped_reason = "max_children"

    ordered = sorted(leaves.items())
    labels = np.full(len(values), -1, dtype=np.int64)
    sse = []
    for child, (_, leaf) in enumerate(ordered):
        labels[leaf["indices"]] = child
        sse.append(float(leaf["sse"]))
    if (labels < 0).any():
        raise RuntimeError("recursive partition did not assign every cell")
    counts = np.bincount(labels, minlength=len(ordered))
    if (counts < min_leaf_size).any():
        raise RuntimeError("recursive partition produced an undersized leaf")
    return RecursivePartition(
        labels=labels,
        active_children=len(ordered),
        within_child_sse=np.asarray(sse, dtype=np.float64),
        split_gains=np.asarray(gains, dtype=np.float64),
        fallback_splits=int(fallback_splits),
        stopped_reason=stopped_reason,
    )


def _fixed_unicode(values) -> np.ndarray:
    flat = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    width = max((len(value) for value in flat), default=1)
    return np.asarray(flat, dtype=f"<U{width}").reshape(np.asarray(values).shape)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _line_summary(
    cell_line: str,
    counts: np.ndarray,
    silhouette: float,
    batch_nmi: float,
    partition: RecursivePartition,
) -> dict:
    occupancy = counts.astype(np.float64) / float(counts.sum())
    entropy = float(-(occupancy * np.log(occupancy)).sum())
    quantiles = np.quantile(counts, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "cell_line": str(cell_line),
        "num_cells": int(counts.sum()),
        "active_children": int(len(counts)),
        "size_min": int(counts.min()),
        "size_p05": float(quantiles[0]),
        "size_q1": float(quantiles[1]),
        "size_median": float(quantiles[2]),
        "size_q3": float(quantiles[3]),
        "size_p95": float(quantiles[4]),
        "size_max": int(counts.max()),
        "size_cv": float(counts.std() / counts.mean()),
        "occupancy_min": float(occupancy.min()),
        "occupancy_max": float(occupancy.max()),
        "occupancy_entropy": entropy,
        "effective_children": float(math.exp(entropy)),
        "latent_silhouette": float(silhouette),
        "batch_nmi": float(batch_nmi),
        "mean_within_child_squared_distance": float(
            partition.within_child_sse.sum() / counts.sum()
        ),
        "split_gain_min": float(partition.split_gains.min(initial=0.0)),
        "split_gain_median": float(
            np.median(partition.split_gains)
            if partition.split_gains.size
            else 0.0
        ),
        "fallback_splits": int(partition.fallback_splits),
        "stopped_reason": partition.stopped_reason,
    }


def build_recursive_context_artifact(
    *,
    h5ad_path: Path,
    feature_key: str = "X_hvg",
    perturbation_key: str = "gene",
    control_label: str = "non-targeting",
    cell_line_key: str = "cell_line",
    batch_key: str = "gem_group",
    anchor_capacity: int = 128,
    pca_dim: int = 64,
    min_leaf_size: int = 20,
    kmeans_n_init: int = 5,
    silhouette_sample_size: int = 10000,
    seed: int = 42,
    source_h5ad_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build arrays without writing, enabling focused unit tests."""

    if source_h5ad_sha256 not in (None, "", "null"):
        actual = sha256_file(h5ad_path)
        if actual != str(source_h5ad_sha256).lower():
            raise ValueError(
                f"source H5AD SHA256 mismatch: expected {source_h5ad_sha256}, got {actual}"
            )
        resolved_source_sha = actual
    else:
        resolved_source_sha = ""

    adata = ad.read_h5ad(h5ad_path, backed="r")
    line_results = []
    all_obs_indices = []
    all_line_ids = []
    all_child_ids = []
    try:
        obs = adata.obs
        perturbations = obs[perturbation_key].astype(str)
        cell_lines_obs = obs[cell_line_key].astype(str)
        control_global = perturbations.eq(str(control_label))
        cellline_names = sorted(cell_lines_obs[control_global].unique().tolist())
        if not cellline_names:
            raise ValueError("no control cell lines were found")
        for line_index, cell_line in enumerate(cellline_names):
            mask = control_global & cell_lines_obs.eq(cell_line)
            positions = np.flatnonzero(mask.to_numpy()).astype(np.int64)
            # Leakage boundary: expression is accessed only after the global
            # non-targeting mask and local cell-line mask are resolved.
            raw = np.asarray(adata.obsm[feature_key][positions], dtype=np.float32)
            if raw.ndim != 2 or not np.isfinite(raw).all():
                raise ValueError(f"invalid control expression for {cell_line}")
            batches = obs.iloc[positions][batch_key].astype(str).to_numpy()
            discovery = batch_center(raw, batches)
            components = min(int(pca_dim), len(raw) - 1, raw.shape[1])
            latent = PCA(
                n_components=max(2, components),
                svd_solver="randomized",
                random_state=int(seed) + line_index,
            ).fit_transform(discovery).astype(np.float32)
            partition = recursive_sse_partition(
                latent,
                max_children=anchor_capacity,
                min_leaf_size=min_leaf_size,
                seed=int(seed) + line_index,
                n_init=kmeans_n_init,
            )
            active = partition.active_children
            counts = np.bincount(partition.labels, minlength=active).astype(np.int64)
            means = np.stack(
                [raw[partition.labels == child].mean(axis=0) for child in range(active)]
            ).astype(np.float32)
            stds = np.stack(
                [raw[partition.labels == child].std(axis=0) for child in range(active)]
            ).astype(np.float32)
            weights = counts.astype(np.float32) / np.float32(len(raw))
            sampled = np.arange(len(raw), dtype=np.int64)
            if len(sampled) > int(silhouette_sample_size):
                sampled = np.random.default_rng(int(seed) + line_index).choice(
                    sampled, size=int(silhouette_sample_size), replace=False
                )
            silhouette = float(
                silhouette_score(latent[sampled], partition.labels[sampled])
            )
            batch_nmi = float(
                normalized_mutual_info_score(batches, partition.labels)
            )
            summary = _line_summary(
                cell_line, counts, silhouette, batch_nmi, partition
            )
            line_results.append(
                {
                    "name": str(cell_line),
                    "positions": positions,
                    "labels": partition.labels,
                    "means": means,
                    "stds": stds,
                    "weights": weights,
                    "counts": counts,
                    "partition": partition,
                    "silhouette": silhouette,
                    "batch_nmi": batch_nmi,
                    "summary": summary,
                }
            )
            all_obs_indices.append(positions)
            all_line_ids.append(
                np.full(len(positions), line_index, dtype=np.int16)
            )
            all_child_ids.append(partition.labels.astype(np.int16))
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    lines = len(line_results)
    genes = line_results[0]["means"].shape[1]
    shape = (lines, int(anchor_capacity), genes)
    means = np.zeros(shape, dtype=np.float32)
    stds = np.zeros(shape, dtype=np.float32)
    weights = np.zeros((lines, int(anchor_capacity)), dtype=np.float32)
    counts = np.zeros((lines, int(anchor_capacity)), dtype=np.int64)
    child_mask = np.zeros((lines, int(anchor_capacity)), dtype=bool)
    within_sse = np.zeros((lines, int(anchor_capacity)), dtype=np.float64)
    weighted_mean = np.zeros((lines, genes), dtype=np.float32)
    silhouettes = np.zeros(lines, dtype=np.float64)
    batch_nmis = np.zeros(lines, dtype=np.float64)
    for line, result in enumerate(line_results):
        active = len(result["counts"])
        means[line, :active] = result["means"]
        stds[line, :active] = result["stds"]
        weights[line, :active] = result["weights"]
        counts[line, :active] = result["counts"]
        child_mask[line, :active] = True
        within_sse[line, :active] = result["partition"].within_child_sse
        weighted_mean[line] = np.average(
            result["means"], axis=0, weights=result["weights"]
        ).astype(np.float32)
        silhouettes[line] = result["silhouette"]
        batch_nmis[line] = result["batch_nmi"]

    metadata = {
        "source_h5ad": str(Path(h5ad_path).resolve()),
        "source_h5ad_sha256": resolved_source_sha,
        "feature_key": str(feature_key),
        "perturbation_key": str(perturbation_key),
        "control_label": str(control_label),
        "cell_line_key": str(cell_line_key),
        "batch_key": str(batch_key),
        "anchor_capacity": int(anchor_capacity),
        "pca_dim": int(pca_dim),
        "min_leaf_size": int(min_leaf_size),
        "kmeans_n_init": int(kmeans_n_init),
        "silhouette_sample_size": int(silhouette_sample_size),
        "seed": int(seed),
        "clustering_space": "per-cell-line batch-centered PCA",
        "prototype_space": "original X_hvg",
        "assignment": "recursive constrained binary k-means",
        "split_priority": "SSE * (1 + log1p(size/min_leaf_size))",
    }
    arrays = {
        "schema_version": np.asarray(RECURSIVE_CONTEXT_SCHEMA),
        "control_only": np.asarray(True, dtype=bool),
        "treated_expression_used": np.asarray(False, dtype=bool),
        "cellline_names": _fixed_unicode([result["name"] for result in line_results]),
        "prototype_means": means,
        "prototype_stds": stds,
        "weights": weights,
        "cluster_cell_counts": counts,
        "child_mask": child_mask,
        "weighted_mean": weighted_mean,
        "within_child_sse": within_sse,
        "latent_silhouette": silhouettes,
        "batch_nmi": batch_nmis,
        "control_obs_indices": np.concatenate(all_obs_indices),
        "control_cellline_ids": np.concatenate(all_line_ids),
        "control_child_ids": np.concatenate(all_child_ids),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"builder produced unsafe object array {key!r}")
    return arrays, [result["summary"] for result in line_results]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a pickle-free recursive non-uniform control Child bank"
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", default="X_hvg")
    parser.add_argument("--perturbation-key", default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--cell-line-key", default="cell_line")
    parser.add_argument("--batch-key", default="gem_group")
    parser.add_argument("--anchor-capacity", type=int, default=128)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--min-leaf-size", type=int, default=20)
    parser.add_argument("--kmeans-n-init", type=int, default=5)
    parser.add_argument("--silhouette-sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-h5ad-sha256", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    arrays, summaries = build_recursive_context_artifact(
        h5ad_path=args.h5ad.expanduser().resolve(),
        feature_key=args.feature_key,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        cell_line_key=args.cell_line_key,
        batch_key=args.batch_key,
        anchor_capacity=args.anchor_capacity,
        pca_dim=args.pca_dim,
        min_leaf_size=args.min_leaf_size,
        kmeans_n_init=args.kmeans_n_init,
        silhouette_sample_size=args.silhouette_sample_size,
        seed=args.seed,
        source_h5ad_sha256=args.source_h5ad_sha256,
    )
    _atomic_savez(output, arrays)
    report = {
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "schema_version": RECURSIVE_CONTEXT_SCHEMA,
        "control_only": True,
        "treated_expression_used": False,
        "shape": list(arrays["prototype_means"].shape),
        "cell_lines": summaries,
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecursivePartition",
    "build_recursive_context_artifact",
    "recursive_sse_partition",
]

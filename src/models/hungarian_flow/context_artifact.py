"""Safe fixed-capacity control-context artifact helpers."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np


RECURSIVE_CONTEXT_SCHEMA = "hungarian_flow_recursive_fixed_anchor_bank_v1"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest for one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _required(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"control context artifact is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in control context key {key!r}")
    return value


def _scalar_text(loaded, key: str) -> str:
    value = _required(loaded, key)
    if value.size != 1:
        raise ValueError(f"control context {key!r} must be scalar")
    return str(value.reshape(-1)[0])


def _metadata(loaded) -> dict:
    raw = _scalar_text(loaded, "metadata_json")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("control context metadata_json is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("control context metadata_json must encode a mapping")
    return value


def load_recursive_context_bank(path: str | Path) -> dict:
    """Load and validate a pickle-free recursive Child bank.

    The returned mapping deliberately matches the legacy in-memory structure
    consumed by ``cell_detr_control.py``.  Padded slots remain represented by
    the artifact mask but are omitted from each cell-line context; the runtime
    bank builder pads them back to the configured fixed capacity.
    """

    artifact = Path(path).expanduser().resolve()
    with np.load(artifact, allow_pickle=False) as loaded:
        for key in loaded.files:
            if loaded[key].dtype == object:
                raise TypeError(f"unsafe object dtype in control context key {key!r}")
        schema = _scalar_text(loaded, "schema_version")
        if schema != RECURSIVE_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported control context schema {schema!r}")
        control_only = bool(_required(loaded, "control_only").reshape(-1)[0])
        treated_used = bool(
            _required(loaded, "treated_expression_used").reshape(-1)[0]
        )
        if not control_only or treated_used:
            raise ValueError(
                "control context leakage audit failed: only controls are allowed"
            )

        names = _required(loaded, "cellline_names").astype(str)
        means = _required(loaded, "prototype_means").astype(np.float32)
        stds = _required(loaded, "prototype_stds").astype(np.float32)
        weights = _required(loaded, "weights").astype(np.float32)
        counts = _required(loaded, "cluster_cell_counts").astype(np.int64)
        child_mask = _required(loaded, "child_mask").astype(bool)
        weighted_mean = _required(loaded, "weighted_mean").astype(np.float32)
        within_sse = _required(loaded, "within_child_sse").astype(np.float64)
        silhouette = _required(loaded, "latent_silhouette").astype(np.float64)
        batch_nmi = _required(loaded, "batch_nmi").astype(np.float64)
        auxiliary_nmi = (
            _required(loaded, "auxiliary_nmi").astype(np.float64)
            if "auxiliary_nmi" in loaded.files
            else np.zeros_like(batch_nmi)
        )
        rejected_keys = (
            "rejected_control_obs_indices",
            "rejected_control_source_ids",
            "rejected_control_source_rows",
            "rejected_control_group_ids",
            "rejected_control_child_ids",
        )
        rejected_present = [key in loaded.files for key in rejected_keys]
        if any(rejected_present) and not all(rejected_present):
            raise ValueError("control context has incomplete rejected-control arrays")
        rejected_arrays = (
            {
                key: _required(loaded, key).astype(np.int64)
                for key in rejected_keys
            }
            if all(rejected_present)
            else None
        )
        retained_obs = (
            _required(loaded, "control_obs_indices").astype(np.int64)
            if "control_obs_indices" in loaded.files
            else None
        )
        count_keys = (
            "input_control_count_by_group",
            "retained_control_count_by_group",
            "rejected_control_count_by_group",
            "retained_control_count",
            "rejected_control_count",
        )
        count_present = [key in loaded.files for key in count_keys]
        if any(count_present) and not all(count_present):
            raise ValueError("control context has incomplete retained/rejected counts")
        audit_counts = (
            {key: _required(loaded, key).astype(np.int64) for key in count_keys}
            if all(count_present)
            else None
        )
        has_auxiliary_names = "auxiliary_label_names" in loaded.files
        has_auxiliary_weights = "child_auxiliary_weights" in loaded.files
        if has_auxiliary_names != has_auxiliary_weights:
            raise ValueError("control context has incomplete auxiliary label arrays")
        auxiliary_names = (
            _required(loaded, "auxiliary_label_names").astype(str)
            if has_auxiliary_names else None
        )
        auxiliary_weights = (
            _required(loaded, "child_auxiliary_weights").astype(np.float32)
            if has_auxiliary_weights else None
        )
        metadata = _metadata(loaded)

    if names.ndim != 1 or len(names) < 1 or len(set(names.tolist())) != len(names):
        raise ValueError("cellline_names must be a non-empty unique vector")
    if means.ndim != 3:
        raise ValueError("prototype_means must have shape [L,K,G]")
    lines, capacity, genes = means.shape
    if lines != len(names) or min(capacity, genes) < 1:
        raise ValueError("control context L,K,G dimensions do not align")
    if stds.shape != means.shape:
        raise ValueError("prototype_stds must match prototype_means")
    expected_lk = (lines, capacity)
    for name, value in (
        ("weights", weights),
        ("cluster_cell_counts", counts),
        ("child_mask", child_mask),
        ("within_child_sse", within_sse),
    ):
        if value.shape != expected_lk:
            raise ValueError(f"{name} must have shape [L,K]")
    if weighted_mean.shape != (lines, genes):
        raise ValueError("weighted_mean must have shape [L,G]")
    if (
        silhouette.shape != (lines,)
        or batch_nmi.shape != (lines,)
        or auxiliary_nmi.shape != (lines,)
    ):
        raise ValueError("line diagnostics must have shape [L]")
    if not child_mask.any(axis=1).all():
        raise ValueError("every cell line needs at least one active Child")
    if not np.isfinite(means[child_mask]).all():
        raise ValueError("active prototypes contain non-finite values")
    if not np.isfinite(stds[child_mask]).all() or (stds[child_mask] < 0).any():
        raise ValueError("active prototype stds must be finite and non-negative")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if (weights[~child_mask] != 0).any() or (counts[~child_mask] != 0).any():
        raise ValueError("padded Children must have zero weight and count")
    if (counts[child_mask] <= 0).any():
        raise ValueError("active Children must have positive support")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-6, rtol=2e-6):
        raise ValueError("Child weights must sum to one per cell line")
    reconstructed = np.einsum("lk,lkg->lg", weights, means)
    if not np.allclose(reconstructed, weighted_mean, atol=2e-5, rtol=2e-4):
        raise ValueError("weighted_mean does not match Child prototypes/weights")
    configured_capacity = int(metadata.get("anchor_capacity", capacity))
    if configured_capacity != capacity:
        raise ValueError("metadata anchor_capacity does not match artifact K")
    clustering_mode = metadata.get("clustering_mode")
    if clustering_mode is not None and clustering_mode not in {
        "constrained_recursive",
        "postfilter_minibatch_kmeans",
    }:
        raise ValueError("metadata clustering_mode is unsupported")
    control_grouping = metadata.get("control_grouping")
    if control_grouping is not None and control_grouping not in {
        "celltype",
        "donor",
        "donor_celltype",
    }:
        raise ValueError("metadata control_grouping is unsupported")
    minimum_support = metadata.get("minimum_support")
    if minimum_support is not None:
        minimum_support = int(minimum_support)
        if minimum_support < 1:
            raise ValueError("metadata minimum_support must be positive")
        if (counts[child_mask] < minimum_support).any():
            raise ValueError("active Child violates metadata minimum_support")
    if clustering_mode == "postfilter_minibatch_kmeans" and minimum_support != 20:
        raise ValueError("postfilter_minibatch_kmeans requires minimum_support=20")

    if clustering_mode == "postfilter_minibatch_kmeans" and (
        rejected_arrays is None or audit_counts is None
    ):
        raise ValueError(
            "postfilter context requires rejected references and audit counts"
        )
    if rejected_arrays is not None:
        rejected_obs = rejected_arrays["rejected_control_obs_indices"]
        rejected_source_ids = rejected_arrays["rejected_control_source_ids"]
        rejected_source_rows = rejected_arrays["rejected_control_source_rows"]
        rejected_group_ids = rejected_arrays["rejected_control_group_ids"]
        rejected_child_ids = rejected_arrays["rejected_control_child_ids"]
        rejected_vectors = (
            rejected_source_ids,
            rejected_source_rows,
            rejected_group_ids,
            rejected_child_ids,
        )
        if rejected_obs.ndim != 1 or any(
            value.shape != rejected_obs.shape for value in rejected_vectors
        ):
            raise ValueError("rejected-control reference vectors do not align")
        if (rejected_obs < 0).any() or len(np.unique(rejected_obs)) != len(rejected_obs):
            raise ValueError("rejected logical control references must be unique")
        if (rejected_source_ids < 0).any() or (rejected_source_rows < 0).any():
            raise ValueError("rejected source references must be non-negative")
        if ((rejected_group_ids < 0) | (rejected_group_ids >= lines)).any():
            raise ValueError("rejected control group ID is out of range")
        if (rejected_child_ids != -1).any():
            raise ValueError("rejected control Child labels must all equal -1")
        if retained_obs is not None:
            if retained_obs.ndim != 1 or len(np.unique(retained_obs)) != len(retained_obs):
                raise ValueError("retained logical control references must be unique")
            if np.intersect1d(retained_obs, rejected_obs).size:
                raise ValueError("retained and rejected control references overlap")

    if audit_counts is not None:
        input_by_group = audit_counts["input_control_count_by_group"]
        retained_by_group = audit_counts["retained_control_count_by_group"]
        rejected_by_group = audit_counts["rejected_control_count_by_group"]
        if any(
            value.shape != (lines,)
            for value in (input_by_group, retained_by_group, rejected_by_group)
        ):
            raise ValueError("per-group retained/rejected counts must have shape [L]")
        if any(
            (value < 0).any()
            for value in (input_by_group, retained_by_group, rejected_by_group)
        ) or not np.array_equal(input_by_group, retained_by_group + rejected_by_group):
            raise ValueError("per-group retained/rejected counts are inconsistent")
        if not np.array_equal(retained_by_group, counts.sum(axis=1)):
            raise ValueError("retained counts do not match active Child support")
        retained_total = audit_counts["retained_control_count"]
        rejected_total = audit_counts["rejected_control_count"]
        if retained_total.size != 1 or rejected_total.size != 1:
            raise ValueError("total retained/rejected counts must be scalar")
        if int(retained_total.item()) != int(retained_by_group.sum()):
            raise ValueError("total retained count is inconsistent")
        if int(rejected_total.item()) != int(rejected_by_group.sum()):
            raise ValueError("total rejected count is inconsistent")
        if rejected_arrays is None and int(rejected_total.item()) != 0:
            raise ValueError("nonzero rejected count requires rejected references")
        if (
            rejected_arrays is not None
            and len(rejected_obs) != int(rejected_total.item())
        ):
            raise ValueError("rejected references do not match rejected count")
        if retained_obs is not None and len(retained_obs) != int(retained_total.item()):
            raise ValueError("retained references do not match retained count")
        for key, expected in (
            ("input_control_count", int(input_by_group.sum())),
            ("retained_control_count", int(retained_total.item())),
            ("rejected_control_count", int(rejected_total.item())),
        ):
            if key in metadata and int(metadata[key]) != expected:
                raise ValueError(f"metadata {key} is inconsistent")
        for key, expected in (
            ("retained_control_count_by_group", retained_by_group),
            ("rejected_control_count_by_group", rejected_by_group),
        ):
            if key in metadata and not np.array_equal(
                np.asarray(metadata[key], dtype=np.int64), expected
            ):
                raise ValueError(f"metadata {key} is inconsistent")
    if auxiliary_names is not None:
        if (
            auxiliary_names.ndim != 1
            or not len(auxiliary_names)
            or len(set(auxiliary_names.tolist())) != len(auxiliary_names)
        ):
            raise ValueError("auxiliary label names must be unique and non-empty")
        if auxiliary_weights.shape != (lines, capacity, len(auxiliary_names)):
            raise ValueError("child auxiliary weights have incompatible shape")
        if not np.isfinite(auxiliary_weights).all() or (auxiliary_weights < 0).any():
            raise ValueError("child auxiliary weights must be finite and non-negative")
        if not np.allclose(auxiliary_weights[child_mask].sum(axis=1), 1.0):
            raise ValueError("active Child auxiliary weights must sum to one")

    contexts = {}
    for line, cell_line in enumerate(names.tolist()):
        active = child_mask[line]
        local_counts = counts[line, active]
        local_sse = within_sse[line, active]
        contexts[str(cell_line)] = {
            "child_capacity": configured_capacity,
            "states": [
                f"recursive_{index:03d}" for index in range(int(active.sum()))
            ],
            "prototype_means": means[line, active].copy(),
            "prototype_stds": stds[line, active].copy(),
            "weights": weights[line, active].copy(),
            "weighted_mean": weighted_mean[line].copy(),
            "cluster_cell_counts": local_counts.copy(),
            "diagnostics": {
                "num_cells": int(local_counts.sum()),
                "num_requested_anchors": configured_capacity,
                "num_active_anchors": int(active.sum()),
                "mean_cells_per_anchor": float(local_counts.mean()),
                "min_cells_per_anchor": int(local_counts.min()),
                "max_cells_per_anchor": int(local_counts.max()),
                "latent_silhouette": float(silhouette[line]),
                "batch_nmi": float(batch_nmi[line]),
                "auxiliary_nmi": float(auxiliary_nmi[line]),
                "mean_within_anchor_squared_distance": float(
                    local_sse.sum() / max(int(local_counts.sum()), 1)
                ),
            },
        }
        if audit_counts is not None:
            contexts[str(cell_line)]["diagnostics"].update({
                "num_input_cells": int(audit_counts["input_control_count_by_group"][line]),
                "num_retained_cells": int(audit_counts["retained_control_count_by_group"][line]),
                "num_rejected_cells": int(audit_counts["rejected_control_count_by_group"][line]),
            })
        if auxiliary_names is not None:
            contexts[str(cell_line)].update({
                "auxiliary_label_names": auxiliary_names.copy(),
                "child_auxiliary_weights": auxiliary_weights[line, active].copy(),
            })
    return {
        "schema": schema,
        "anchor_capacity": configured_capacity,
        "context_by_cell_line": contexts,
        "metadata": metadata,
        "rejected_controls": (
            None
            if rejected_arrays is None
            else {key: value.copy() for key, value in rejected_arrays.items()}
        ),
        "control_audit_counts": (
            None
            if audit_counts is None
            else {key: value.copy() for key, value in audit_counts.items()}
        ),
    }


def load_control_context_bank(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Load a safe NPZ bank or the backward-compatible trusted pickle bank."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    actual = sha256_file(artifact)
    if expected_sha256 not in (None, "", "null"):
        if actual != str(expected_sha256).lower():
            raise ValueError(
                "control context SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )
    if artifact.suffix.lower() == ".npz":
        return load_recursive_context_bank(artifact)
    # Compatibility boundary: old frozen experiments still depend on the
    # historical local pickle. New artifacts must use the NPZ path above.
    with artifact.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("legacy control context pickle must contain a mapping")
    return dict(value)


__all__ = [
    "RECURSIVE_CONTEXT_SCHEMA",
    "load_control_context_bank",
    "load_recursive_context_bank",
    "sha256_file",
]

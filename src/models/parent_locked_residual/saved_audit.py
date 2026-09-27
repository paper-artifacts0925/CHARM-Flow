"""Independent on-disk audit for Parent-locked nonnegative sampling outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import anndata
import numpy as np
from scipy import sparse


GATE_KEY = "parent_locked_sampling_gate"
REFERENCE_KEY = "parent_locked_feasible_parent_reference"
GROUP_OBS_KEY = "parent_locked_projection_group"


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_gate_record(
    gate: Mapping,
    *,
    max_singleton_fraction: float,
    max_endpoint_mean_drift: float,
) -> dict:
    required = {
        "schema_version",
        "singleton_policy",
        "projection_policy",
        "parent_mean_policy",
        "valid_cells",
        "singleton_cells",
        "singleton_fraction",
        "endpoint_mean_drift_max_abs",
        "minimum_expression",
        "negative_values_after_projection",
        "saved_h5ad_audit_required",
        "passed",
    }
    missing = sorted(required.difference(gate))
    if missing:
        raise AssertionError(f"missing Parent-lock gate fields: {missing}")
    record = {str(key): _python_scalar(value) for key, value in gate.items()}
    if int(record["schema_version"]) != 2:
        raise AssertionError("formal nonnegative sampling requires gate schema 2")
    if str(record["singleton_policy"]) != "effective_parent_zero_residual":
        raise AssertionError("formal sampling did not use the effective-Parent singleton")
    if str(record["projection_policy"]) != (
        "euclidean_nonnegative_fixed_effective_parent_mean"
    ):
        raise AssertionError("formal sampling did not use the fixed-mean simplex")
    supported_mean_policies = {
        "clamp_condition_parent_mean_at_zero",
        "clamp_condition_integrated_child_mean_at_zero",
    }
    if str(record["parent_mean_policy"]) not in supported_mean_policies:
        raise AssertionError("formal sampling has an unknown endpoint-mean policy")
    expected_endpoint_policy = {
        "clamp_condition_parent_mean_at_zero": "parent",
        "clamp_condition_integrated_child_mean_at_zero": (
            "integrated_child_mean"
        ),
    }[str(record["parent_mean_policy"])]
    serialized_endpoint_policy = record.get("endpoint_mean_policy")
    if (
        serialized_endpoint_policy is not None
        and str(serialized_endpoint_policy) != expected_endpoint_policy
    ):
        raise AssertionError(
            "serialized endpoint_mean_policy disagrees with parent_mean_policy"
        )
    if (
        expected_endpoint_policy == "integrated_child_mean"
        and serialized_endpoint_policy is None
    ):
        raise AssertionError(
            "Child endpoint sampling must serialize endpoint_mean_policy"
        )
    if not bool(record["saved_h5ad_audit_required"]) or not bool(record["passed"]):
        raise AssertionError("serialized Parent-lock scientific gate did not pass")

    valid_cells = int(record["valid_cells"])
    singleton_cells = int(record["singleton_cells"])
    singleton_fraction = float(record["singleton_fraction"])
    endpoint_drift = float(record["endpoint_mean_drift_max_abs"])
    serialized_minimum = float(record["minimum_expression"])
    negative_after = int(record["negative_values_after_projection"])
    maximum_fraction = float(max_singleton_fraction)
    maximum_drift = float(max_endpoint_mean_drift)
    if valid_cells < 1 or not 0 <= singleton_cells <= valid_cells:
        raise AssertionError("invalid serialized singleton counts")
    if not math.isclose(
        singleton_fraction, singleton_cells / valid_cells, abs_tol=1.0e-12
    ):
        raise AssertionError("serialized singleton_fraction disagrees with counts")
    if not 0.0 <= singleton_fraction <= maximum_fraction:
        raise AssertionError("serialized singleton fraction exceeds the formal limit")
    if not math.isfinite(endpoint_drift) or not 0.0 <= endpoint_drift <= maximum_drift:
        raise AssertionError("serialized feasible Parent mean drift exceeds the limit")
    if not math.isfinite(serialized_minimum) or serialized_minimum < 0.0:
        raise AssertionError("serialized projection minimum is negative")
    if negative_after != 0:
        raise AssertionError("serialized projection retained negative values")
    return record


def _dense(block) -> np.ndarray:
    if sparse.issparse(block):
        block = block.toarray()
    value = np.asarray(block)
    if value.ndim != 2:
        raise AssertionError("H5AD expression matrix must be two-dimensional")
    return value


def audit_saved_parent_locked_h5ad(
    path: str | Path,
    *,
    max_singleton_fraction: float,
    max_endpoint_mean_drift: float,
    chunk_size: int = 2048,
) -> dict:
    """Reopen a saved H5AD and recompute nonnegativity and group mean drift."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    adata = anndata.read_h5ad(path, backed="r")
    try:
        if GATE_KEY not in adata.uns:
            raise AssertionError(f"saved H5AD has no {GATE_KEY!r}")
        record = validate_gate_record(
            adata.uns[GATE_KEY],
            max_singleton_fraction=max_singleton_fraction,
            max_endpoint_mean_drift=max_endpoint_mean_drift,
        )
        if REFERENCE_KEY not in adata.uns or GROUP_OBS_KEY not in adata.obs:
            raise AssertionError("saved H5AD has no auditable effective-Parent reference")
        reference_record = adata.uns[REFERENCE_KEY]
        if int(reference_record.get("schema_version", -1)) != 1:
            raise AssertionError("unsupported effective-Parent reference schema")
        reference_ids = np.asarray(reference_record["projection_group_ids"], dtype=np.int64)
        reference = np.asarray(reference_record["effective_parent_mean"], dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != adata.n_vars:
            raise AssertionError("effective-Parent reference has an invalid shape")
        if not np.array_equal(reference_ids, np.arange(reference.shape[0])):
            raise AssertionError("effective-Parent reference IDs must be contiguous")
        group_ids = np.asarray(adata.obs[GROUP_OBS_KEY], dtype=np.int64)
        generated = group_ids >= 0
        if int(generated.sum()) != int(reference_record["generated_cells"]):
            raise AssertionError("saved generated-cell count disagrees with reference")
        if not generated.any() or group_ids[generated].max() >= reference.shape[0]:
            raise AssertionError("saved projection group IDs are invalid")

        sums = np.zeros(reference.shape, dtype=np.float64)
        counts = np.zeros(reference.shape[0], dtype=np.int64)
        minimum = math.inf
        negative_values = 0
        for start in range(0, adata.n_obs, int(chunk_size)):
            stop = min(start + int(chunk_size), adata.n_obs)
            block = _dense(adata.X[start:stop])
            if not np.isfinite(block).all():
                raise AssertionError("saved H5AD contains non-finite expression")
            if block.size:
                minimum = min(minimum, float(block.min()))
                negative_values += int((block < 0).sum())
            block_ids = group_ids[start:stop]
            for group_id in np.unique(block_ids[block_ids >= 0]):
                selected = block_ids == group_id
                sums[group_id] += block[selected].sum(axis=0, dtype=np.float64)
                counts[group_id] += int(selected.sum())
        if minimum < 0.0 or negative_values:
            raise AssertionError(
                f"saved H5AD is negative: min={minimum:.8g}, count={negative_values}"
            )
        if (counts < 1).any():
            raise AssertionError("an effective-Parent reference group has no saved cells")
        means = sums / counts[:, None]
        saved_drift = float(np.max(np.abs(means - reference)))
        if not math.isfinite(saved_drift) or saved_drift > float(max_endpoint_mean_drift):
            raise AssertionError(
                "saved H5AD feasible Parent mean drift exceeds the gate: "
                f"{saved_drift:.8g} > {float(max_endpoint_mean_drift):.8g}"
            )
        record.update(
            {
                "saved_h5ad": str(path.resolve()),
                "saved_h5ad_minimum_expression": float(minimum),
                "saved_h5ad_negative_values": int(negative_values),
                "saved_h5ad_endpoint_mean_drift_max_abs": saved_drift,
                "saved_h5ad_projection_groups": int(reference.shape[0]),
                "saved_h5ad_audit_passed": True,
            }
        )
        return record
    finally:
        adata.file.close()


__all__ = [
    "GATE_KEY",
    "GROUP_OBS_KEY",
    "REFERENCE_KEY",
    "audit_saved_parent_locked_h5ad",
    "validate_gate_record",
]

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import anndata
import numpy as np
from scipy import sparse

from src.models.parent_locked_residual.saved_audit import (
    GATE_KEY,
    GROUP_OBS_KEY,
    REFERENCE_KEY,
    audit_saved_parent_locked_h5ad,
)


POSTPROCESS_KEY = "parent_locked_celleval_bounded_postprocess"
ALGORITHM = "euclidean_box_projection_fixed_effective_parent_mean"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    value = np.asarray(matrix)
    if value.ndim != 2:
        raise ValueError("expression must be a two-dimensional matrix")
    return value


def _dense_float32(matrix) -> np.ndarray:
    value = np.asarray(_dense(matrix), dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("input expression must be finite")
    return value.copy()


def _default_output(path: Path) -> Path:
    return path.with_name(path.stem + "_celleval_bounded_fixedmean.h5ad")


def _effective_upper_bound(threshold: float) -> float:
    threshold32 = np.float32(threshold)
    if not math.isfinite(float(threshold32)) or threshold32 <= 0:
        raise ValueError("Cell-Eval threshold must be finite and positive")
    upper = np.nextafter(threshold32, np.float32(-np.inf))
    if not 0 < upper < threshold32:
        raise AssertionError("failed to construct a strict float32 upper bound")
    return float(upper)


def project_box_fixed_sum(
    values: np.ndarray,
    *,
    target_sum: float,
    upper_bound: float,
) -> np.ndarray:
    """Euclidean projection onto ``[0, upper_bound]^N`` at fixed sum.

    The KKT solution is ``clip(values - lambda, 0, upper_bound)``.  Monotone
    bisection finds ``lambda`` and a final sub-ulp-scale correction removes
    accumulated floating-point sum error without leaving the box.
    """

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
        raise ValueError("box projection values must be a finite vector")
    upper = float(upper_bound)
    target = float(target_sum)
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("upper_bound must be finite and positive")
    if not math.isfinite(target):
        raise ValueError("target_sum must be finite")
    capacity = vector.size * upper
    feasibility_tolerance = 64 * np.finfo(np.float64).eps * max(1.0, capacity)
    if target < -feasibility_tolerance or target > capacity + feasibility_tolerance:
        raise ValueError(
            "fixed Parent sum is infeasible under the Cell-Eval upper bound: "
            f"target={target:.12g}, capacity={capacity:.12g}"
        )
    target = min(max(target, 0.0), capacity)
    if target == 0.0:
        return np.zeros_like(vector)
    if target == capacity:
        return np.full_like(vector, upper)

    lower_lambda = float(np.min(vector - upper))
    upper_lambda = float(np.max(vector))
    for _ in range(160):
        midpoint = lower_lambda + (upper_lambda - lower_lambda) * 0.5
        projected_sum = float(
            np.clip(vector - midpoint, 0.0, upper).sum(dtype=np.float64)
        )
        if projected_sum > target:
            lower_lambda = midpoint
        else:
            upper_lambda = midpoint
    result = np.clip(
        vector - (lower_lambda + upper_lambda) * 0.5,
        0.0,
        upper,
    )

    correction = target - float(result.sum(dtype=np.float64))
    correction_tolerance = 128 * np.finfo(np.float64).eps * max(1.0, capacity)
    if abs(correction) > correction_tolerance:
        raise AssertionError(
            f"box projection failed fixed-sum convergence: {correction:.12g}"
        )
    if correction > 0:
        candidates = np.flatnonzero(result < upper)
        if not len(candidates):
            raise AssertionError("box projection has no upper slack for correction")
        index = int(candidates[np.argmax(upper - result[candidates])])
    elif correction < 0:
        candidates = np.flatnonzero(result > 0)
        if not len(candidates):
            raise AssertionError("box projection has no positive value for correction")
        index = int(candidates[np.argmax(result[candidates])])
    else:
        index = -1
    if index >= 0:
        result[index] += correction
    if result.min() < 0 or result.max() > upper:
        raise AssertionError("box projection left its feasible interval")
    return result


def audit_celleval_bounded_h5ad(
    path: str | Path,
    *,
    threshold: float,
    expected_upper_bound: float | None = None,
    chunk_size: int = 2048,
) -> dict:
    """Reopen a derived H5AD and independently verify its strict range."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    expected_upper = (
        _effective_upper_bound(threshold)
        if expected_upper_bound is None
        else float(expected_upper_bound)
    )
    adata = anndata.read_h5ad(path, backed="r")
    try:
        if POSTPROCESS_KEY not in adata.uns:
            raise AssertionError(f"bounded H5AD has no {POSTPROCESS_KEY!r}")
        record = dict(adata.uns[POSTPROCESS_KEY])
        if int(record.get("schema_version", -1)) != 1:
            raise AssertionError("unsupported bounded postprocess schema")
        if str(record.get("algorithm")) != ALGORITHM:
            raise AssertionError("bounded postprocess algorithm is unknown")
        if not bool(record.get("passed", False)):
            raise AssertionError("serialized bounded postprocess did not pass")
        if bool(record.get("source_overwritten", True)):
            raise AssertionError("bounded provenance does not prove no-overwrite")
        if not math.isclose(
            float(record.get("cell_eval_threshold", math.nan)),
            float(threshold),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AssertionError("serialized Cell-Eval threshold disagrees")
        if not math.isclose(
            float(record.get("effective_upper_bound", math.nan)),
            expected_upper,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AssertionError("serialized effective upper bound disagrees")

        minimum = math.inf
        maximum = -math.inf
        nonfinite = 0
        negative = 0
        at_or_above_threshold = 0
        above_effective_bound = 0
        for start in range(0, adata.n_obs, int(chunk_size)):
            stop = min(start + int(chunk_size), adata.n_obs)
            block = _dense(adata.X[start:stop])
            finite = np.isfinite(block)
            nonfinite += int((~finite).sum())
            if finite.any():
                minimum = min(minimum, float(block[finite].min()))
                maximum = max(maximum, float(block[finite].max()))
            negative += int((block < 0).sum())
            at_or_above_threshold += int((block >= float(threshold)).sum())
            above_effective_bound += int((block > expected_upper).sum())
        if nonfinite or negative or at_or_above_threshold or above_effective_bound:
            raise AssertionError(
                "bounded H5AD failed strict range audit: "
                f"nonfinite={nonfinite}, negative={negative}, "
                f">=threshold={at_or_above_threshold}, >bound={above_effective_bound}"
            )
        return {
            "bounded_h5ad": str(path.resolve()),
            "cell_eval_threshold": float(threshold),
            "effective_upper_bound": expected_upper,
            "minimum_expression": float(minimum),
            "maximum_expression": float(maximum),
            "nonfinite_values": int(nonfinite),
            "negative_values": int(negative),
            "values_at_or_above_threshold": int(at_or_above_threshold),
            "values_above_effective_bound": int(above_effective_bound),
            "bounded_h5ad_audit_passed": True,
        }
    finally:
        adata.file.close()


def postprocess(
    *,
    input_path: Path,
    output_path: Path,
    cell_eval_threshold: float,
    max_singleton_fraction: float,
    max_endpoint_mean_drift: float,
) -> dict:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    sidecar = Path(str(output_path) + ".projection.json")
    if input_path == output_path:
        raise ValueError("postprocessing must never overwrite the source H5AD")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(
            f"refusing to overwrite existing bounded artifact: {output_path}"
        )

    threshold = float(cell_eval_threshold)
    upper = _effective_upper_bound(threshold)
    input_sha256 = _sha256(input_path)
    source_saved_audit = audit_saved_parent_locked_h5ad(
        input_path,
        max_singleton_fraction=max_singleton_fraction,
        max_endpoint_mean_drift=max_endpoint_mean_drift,
    )
    adata = anndata.read_h5ad(input_path)
    if GATE_KEY not in adata.uns or GROUP_OBS_KEY not in adata.obs:
        raise AssertionError("input is not an auditable Parent-locked H5AD")
    if REFERENCE_KEY not in adata.uns:
        raise AssertionError("input has no effective Parent reference")
    if POSTPROCESS_KEY in adata.uns:
        raise AssertionError("input has already received a Cell-Eval bound repair")

    source = _dense_float32(adata.X)
    if source.min() < 0:
        raise AssertionError("formal Parent-locked input unexpectedly contains negatives")
    repaired = source.copy()
    group_ids = np.asarray(adata.obs[GROUP_OBS_KEY], dtype=np.int64)
    generated = group_ids >= 0
    if not generated.any() or generated.shape != (adata.n_obs,):
        raise AssertionError("Parent projection group IDs are malformed")
    reference = np.asarray(
        adata.uns[REFERENCE_KEY]["effective_parent_mean"], dtype=np.float64
    )
    if reference.ndim != 2 or reference.shape[1] != adata.n_vars:
        raise AssertionError("effective Parent reference is malformed")
    if reference.min() < 0 or reference.max() > upper:
        raise ValueError(
            "an effective Parent mean is infeasible under the strict Cell-Eval bound"
        )

    control_violations = int((source[~generated] >= threshold).sum())
    if control_violations:
        raise ValueError(
            "control rows violate Cell-Eval's threshold; refusing to alter controls"
        )
    affected_pairs: set[tuple[int, int]] = set()
    offending_values = 0
    offending_rows: set[int] = set()
    for start in range(0, adata.n_obs, 2048):
        stop = min(start + 2048, adata.n_obs)
        bad = np.argwhere(source[start:stop] >= threshold)
        for local_row, gene_index in bad:
            row = start + int(local_row)
            if not generated[row]:
                continue
            offending_values += 1
            offending_rows.add(row)
            affected_pairs.add((int(group_ids[row]), int(gene_index)))
    if not affected_pairs:
        raise ValueError("input has no generated values requiring Cell-Eval repair")

    pair_records = []
    changed_values = 0
    maximum_adjustment = 0.0
    sum_absolute_adjustment = 0.0
    touched_mean_drift = 0.0
    for group_id, gene_index in sorted(affected_pairs):
        members = np.flatnonzero(group_ids == group_id)
        if not len(members):
            raise AssertionError("affected Parent group has no generated cells")
        original = source[members, gene_index].astype(np.float64)
        target_sum = float(reference[group_id, gene_index]) * len(members)
        projected64 = project_box_fixed_sum(
            original,
            target_sum=target_sum,
            upper_bound=upper,
        )
        projected = projected64.astype(np.float32)
        if projected.min() < 0 or projected.max() > upper:
            raise AssertionError("float32 bounded projection left its interval")
        drift = abs(
            float(projected.mean(dtype=np.float64))
            - float(reference[group_id, gene_index])
        )
        if drift > float(max_endpoint_mean_drift):
            raise AssertionError(
                "float32 bounded projection violated the Parent mean gate: "
                f"group={group_id}, gene={gene_index}, drift={drift:.12g}"
            )
        delta = projected.astype(np.float64) - original
        repaired[members, gene_index] = projected
        changed = int((projected != source[members, gene_index]).sum())
        changed_values += changed
        maximum_adjustment = max(maximum_adjustment, float(np.max(np.abs(delta))))
        sum_absolute_adjustment += float(np.abs(delta).sum(dtype=np.float64))
        touched_mean_drift = max(touched_mean_drift, drift)
        pair_records.append(
            {
                "projection_group": int(group_id),
                "gene_index": int(gene_index),
                "group_size": int(len(members)),
                "reference_mean": float(reference[group_id, gene_index]),
                "maximum_before": float(original.max()),
                "maximum_after": float(projected.max()),
                "changed_values": changed,
                "mean_drift_abs": drift,
            }
        )

    if not np.array_equal(source[~generated], repaired[~generated]):
        raise AssertionError("bounded projection changed immutable controls")
    output_maximum = float(repaired.max())
    if not np.isfinite(repaired).all() or repaired.min() < 0 or output_maximum >= threshold:
        raise AssertionError("in-memory bounded projection failed the Cell-Eval range")
    if int((repaired >= threshold).sum()) != 0:
        raise AssertionError("in-memory bounded projection retained threshold violations")

    pair_group_ids = np.asarray([pair[0] for pair in sorted(affected_pairs)], dtype=np.int64)
    pair_gene_indices = np.asarray([pair[1] for pair in sorted(affected_pairs)], dtype=np.int64)
    postprocess_record = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "cell_eval_threshold": threshold,
        "effective_upper_bound": upper,
        "input_h5ad": str(input_path),
        "input_sha256": input_sha256,
        "source_overwritten": False,
        "controls_immutable": True,
        "fixed_mean_reference_key": REFERENCE_KEY,
        "group_obs_key": GROUP_OBS_KEY,
        "offending_values_before": int(offending_values),
        "offending_rows_before": int(len(offending_rows)),
        "affected_group_gene_pairs": int(len(affected_pairs)),
        "affected_projection_group_ids": pair_group_ids,
        "affected_gene_indices": pair_gene_indices,
        "changed_values": int(changed_values),
        "maximum_adjustment_abs": float(maximum_adjustment),
        "sum_absolute_adjustment": float(sum_absolute_adjustment),
        "source_maximum_expression": float(source.max()),
        "output_maximum_expression": output_maximum,
        "touched_parent_mean_drift_max_abs": float(touched_mean_drift),
        "parent_mean_drift_tolerance": float(max_endpoint_mean_drift),
        "passed": True,
    }
    adata.X = repaired
    adata.uns[POSTPROCESS_KEY] = postprocess_record

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp.h5ad",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        adata.write_h5ad(temporary_path)
        saved_parent_audit = audit_saved_parent_locked_h5ad(
            temporary_path,
            max_singleton_fraction=max_singleton_fraction,
            max_endpoint_mean_drift=max_endpoint_mean_drift,
        )
        bounded_audit = audit_celleval_bounded_h5ad(
            temporary_path,
            threshold=threshold,
            expected_upper_bound=upper,
        )
        # Atomic no-overwrite publication on the same filesystem.
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if _sha256(input_path) != input_sha256:
        raise AssertionError("source H5AD SHA changed during bounded projection")
    output_sha256 = _sha256(output_path)
    result = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "input_h5ad": str(input_path),
        "input_sha256": input_sha256,
        "output_h5ad": str(output_path),
        "output_sha256": output_sha256,
        "source_overwritten": False,
        "cell_eval_threshold": threshold,
        "effective_upper_bound": upper,
        "offending_values_before": int(offending_values),
        "offending_rows_before": int(len(offending_rows)),
        "affected_group_gene_pairs": int(len(affected_pairs)),
        "affected_pairs": pair_records,
        "changed_values": int(changed_values),
        "maximum_adjustment_abs": float(maximum_adjustment),
        "sum_absolute_adjustment": float(sum_absolute_adjustment),
        "source_maximum_expression": float(source.max()),
        "output_maximum_expression": output_maximum,
        "touched_parent_mean_drift_max_abs": float(touched_mean_drift),
        "source_saved_parent_audit": source_saved_audit,
        "saved_parent_audit": saved_parent_audit,
        "bounded_audit": bounded_audit,
    }
    try:
        with sidecar.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        # The H5AD remains valid and auditable even if sidecar publication fails;
        # never remove or overwrite an independently created path here.
        raise
    result["sidecar"] = str(sidecar)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell-eval-threshold", type=float, default=15.0)
    parser.add_argument("--max-singleton-fraction", type=float, default=0.01)
    parser.add_argument("--max-endpoint-mean-drift", type=float, default=1.0e-5)
    args = parser.parse_args()
    output = args.output or _default_output(args.input)
    result = postprocess(
        input_path=args.input,
        output_path=output,
        cell_eval_threshold=args.cell_eval_threshold,
        max_singleton_fraction=args.max_singleton_fraction,
        max_endpoint_mean_drift=args.max_endpoint_mean_drift,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

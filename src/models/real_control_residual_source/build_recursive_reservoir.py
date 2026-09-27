"""Build a real-control reservoir from a safe recursive Child artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np

from src.models.hungarian_flow.context_artifact import (
    RECURSIVE_CONTEXT_SCHEMA,
    load_recursive_context_bank,
    sha256_file,
)

from .build_reservoir import (
    normalized_control_means,
    pack_assigned_controls,
    validate_control_assignment_partition,
)
from .reservoir import REAL_CONTROL_RESERVOIR_SCHEMA


def _required(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"recursive context is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in recursive context key {key!r}")
    return value


def _optional(loaded, key: str) -> np.ndarray | None:
    if key not in loaded.files:
        return None
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in recursive context key {key!r}")
    return value


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def build_recursive_real_control_reservoir(
    *,
    h5ad_path: Path,
    context_path: Path,
    normalization_divisor: float = 10.0,
    max_members_per_child: int = 64,
    seed: int = 42,
    prototype_atol: float = 2e-5,
    prototype_rtol: float = 2e-4,
) -> dict[str, np.ndarray]:
    """Pack exact persisted recursive assignments after strict control checks."""

    # Run the full public validation before using assignment-only arrays.
    context_bank = load_recursive_context_bank(context_path)
    metadata = context_bank["metadata"]
    with np.load(context_path, allow_pickle=False) as loaded:
        schema = str(_required(loaded, "schema_version").reshape(-1)[0])
        if schema != RECURSIVE_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported recursive context schema {schema!r}")
        names = _required(loaded, "cellline_names").astype(str)
        prototypes = _required(loaded, "prototype_means").astype(np.float32)
        weights = _required(loaded, "weights").astype(np.float32)
        counts = _required(loaded, "cluster_cell_counts").astype(np.int64)
        child_mask = _required(loaded, "child_mask").astype(bool)
        obs_indices = _required(loaded, "control_obs_indices").astype(np.int64)
        line_ids = _required(loaded, "control_cellline_ids").astype(np.int64)
        child_ids = _required(loaded, "control_child_ids").astype(np.int64)
        rejected_obs_indices = _optional(
            loaded, "rejected_control_obs_indices"
        )
        rejected_source_ids = _optional(loaded, "rejected_control_source_ids")
        rejected_source_rows = _optional(loaded, "rejected_control_source_rows")
        rejected_child_ids = _optional(loaded, "rejected_control_child_ids")
        rejected_group_ids = _optional(loaded, "rejected_control_group_ids")

    rejected_vectors = (
        rejected_source_ids,
        rejected_source_rows,
        rejected_child_ids,
        rejected_group_ids,
    )
    if rejected_obs_indices is None:
        if any(value is not None for value in rejected_vectors):
            raise ValueError("recursive context has incomplete rejected controls")
    else:
        rejected_obs_indices = np.asarray(
            rejected_obs_indices, dtype=np.int64
        )
        if any(value is None for value in rejected_vectors):
            raise ValueError("recursive context has incomplete rejected controls")
        rejected_source_ids = np.asarray(rejected_source_ids, dtype=np.int64)
        rejected_source_rows = np.asarray(rejected_source_rows, dtype=np.int64)
        rejected_child_ids = np.asarray(rejected_child_ids, dtype=np.int64)
        rejected_group_ids = np.asarray(rejected_group_ids, dtype=np.int64)
        if any(
            value.shape != rejected_obs_indices.shape
            for value in rejected_vectors
        ):
            raise ValueError("rejected control vectors do not align")
        if (rejected_child_ids != -1).any():
            raise ValueError("rejected control Child IDs must all be -1")

    if obs_indices.ndim != 1 or len(obs_indices) < 1:
        raise ValueError("control_obs_indices must be a non-empty vector")
    if line_ids.shape != obs_indices.shape or child_ids.shape != obs_indices.shape:
        raise ValueError("persisted control assignments do not align")
    if (obs_indices < 0).any() or len(np.unique(obs_indices)) != len(obs_indices):
        raise ValueError("control_obs_indices must be unique non-negative rows")
    lines, capacity, genes = prototypes.shape
    if names.shape != (lines,) or weights.shape != (lines, capacity):
        raise ValueError("recursive context line/Child axes do not align")
    if counts.shape != (lines, capacity) or child_mask.shape != (lines, capacity):
        raise ValueError("recursive context count/mask axes do not align")
    if ((line_ids < 0) | (line_ids >= lines)).any():
        raise ValueError("persisted control line ID is out of range")
    if ((child_ids < 0) | (child_ids >= capacity)).any():
        raise ValueError("persisted control Child ID is out of range")
    if not child_mask[line_ids, child_ids].all():
        raise ValueError("persisted control assignment selects a padded Child")

    feature_key = str(metadata.get("feature_key", "X_hvg"))
    perturbation_key = str(metadata.get("perturbation_key", "gene"))
    control_label = str(metadata.get("control_label", "non-targeting"))
    cell_line_key = str(metadata.get("cell_line_key", "cell_line"))

    adata = ad.read_h5ad(h5ad_path, backed="r")
    raw_parts = []
    line_parts = []
    runtime_child_parts = []
    source_index_parts = []
    runtime_prototypes = np.zeros_like(prototypes)
    runtime_masks = np.zeros_like(child_mask)
    try:
        obs = adata.obs
        perturbations = obs[perturbation_key].astype(str)
        observed_lines = obs[cell_line_key].astype(str)
        if int(obs_indices.max()) >= len(obs):
            raise ValueError("persisted control observation index is out of range")
        all_expected = np.flatnonzero(
            perturbations.eq(control_label).to_numpy()
            & observed_lines.isin(names.tolist()).to_numpy()
        ).astype(np.int64)
        _, rejected = validate_control_assignment_partition(
            all_control_obs_indices=all_expected,
            retained_control_obs_indices=obs_indices,
            rejected_control_obs_indices=rejected_obs_indices,
        )
        if len(rejected):
            if int(rejected.max()) >= len(obs):
                raise ValueError(
                    "rejected control observation index is out of range"
                )
            if not perturbations.iloc[rejected].eq(control_label).all():
                raise ValueError("treated row entered rejected controls")
            if ((rejected_group_ids < 0) | (rejected_group_ids >= lines)).any():
                raise ValueError("rejected control group ID is out of range")
            if not np.array_equal(
                names[rejected_group_ids],
                observed_lines.iloc[rejected].to_numpy(),
            ):
                raise ValueError("rejected control group assignment mismatch")
            if (rejected_source_ids != 0).any() or not np.array_equal(
                rejected_source_rows, rejected
            ):
                raise ValueError(
                    "rejected single-H5AD source references do not match source"
                )

        # Parent and reservoir share one exact baseline: the normalized mean
        # of every control in the group, including controls rejected from Child
        # matching. Only these already validated control positions are read.
        all_raw = np.asarray(
            adata.obsm[feature_key][all_expected], dtype=np.float32
        )
        group_by_name = {
            str(cell_line): line
            for line, cell_line in enumerate(names.tolist())
        }
        all_line_ids = np.asarray(
            [
                group_by_name[str(value)]
                for value in observed_lines.iloc[all_expected].tolist()
            ],
            dtype=np.int64,
        )
        full_control_mean, full_control_counts = normalized_control_means(
            control_expression=all_raw,
            control_line_ids=all_line_ids,
            cellline_names=names.tolist(),
            normalization_divisor=float(normalization_divisor),
        )
        rejected_counts = np.bincount(
            rejected_group_ids
            if rejected_obs_indices is not None
            else np.empty((0,), dtype=np.int64),
            minlength=lines,
        ).astype(np.int64)
        for line, cell_line in enumerate(names.tolist()):
            assigned_rows = np.flatnonzero(line_ids == line)
            positions = obs_indices[assigned_rows]
            if not perturbations.iloc[positions].eq(control_label).all():
                raise ValueError("treated row entered recursive control assignments")
            if not observed_lines.iloc[positions].eq(cell_line).all():
                raise ValueError("recursive control cell-line assignment mismatch")
            # Leakage boundary: only the validated control positions are read.
            raw = np.asarray(adata.obsm[feature_key][positions], dtype=np.float32)
            original_child = child_ids[assigned_rows]
            active_original = np.flatnonzero(child_mask[line])
            rebuilt_counts = np.bincount(original_child, minlength=capacity)
            if not np.array_equal(rebuilt_counts, counts[line]):
                raise ValueError("persisted recursive Child counts do not match")
            rebuilt_means = np.zeros((capacity, genes), dtype=np.float32)
            for child in active_original:
                rebuilt_means[child] = raw[original_child == child].mean(axis=0)
            if not np.allclose(
                rebuilt_means[active_original],
                prototypes[line, active_original],
                atol=float(prototype_atol),
                rtol=float(prototype_rtol),
            ):
                maximum = float(
                    np.max(
                        np.abs(
                            rebuilt_means[active_original]
                            - prototypes[line, active_original]
                        )
                    )
                )
                raise ValueError(
                    "recursive prototypes do not match persisted assignments; "
                    f"maximum absolute difference={maximum:.7g}"
                )

            # Match build_unique_cell_line_control_bank exactly. Runtime Child
            # IDs are descending occupancy, independently inside each line.
            order = np.argsort(-weights[line, active_original])
            ordered_original = active_original[order]
            inverse = np.full(capacity, -1, dtype=np.int64)
            inverse[ordered_original] = np.arange(len(ordered_original))
            runtime_child = inverse[original_child]
            if (runtime_child < 0).any():
                raise RuntimeError("could not map recursive Child to runtime order")
            active_count = len(ordered_original)
            runtime_prototypes[line, :active_count] = prototypes[
                line, ordered_original
            ]
            runtime_masks[line, :active_count] = True
            raw_parts.append(raw)
            line_parts.append(np.full(len(raw), line, dtype=np.int64))
            runtime_child_parts.append(runtime_child)
            source_index_parts.append(positions)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    arrays = pack_assigned_controls(
        control_expression=np.concatenate(raw_parts, axis=0),
        control_line_ids=np.concatenate(line_parts, axis=0),
        control_child_ids=np.concatenate(runtime_child_parts, axis=0),
        cellline_names=names.tolist(),
        child_prototypes=runtime_prototypes,
        child_mask=runtime_masks,
        normalization_divisor=float(normalization_divisor),
        max_members_per_child=int(max_members_per_child),
        seed=int(seed),
        source_obs_indices=np.concatenate(source_index_parts, axis=0),
        control_mean_override=full_control_mean,
    )
    arrays.update(
        {
            "context_bank_sha256": np.asarray(sha256_file(context_path)),
            "context_bank_schema": np.asarray(RECURSIVE_CONTEXT_SCHEMA),
            "control_label": np.asarray(control_label),
            "expression_key": np.asarray(feature_key),
            "perturbation_key": np.asarray(perturbation_key),
            "cell_line_key": np.asarray(cell_line_key),
            "control_grouping": np.asarray(
                str(metadata.get("control_grouping", "celltype"))
            ),
            "assignment_method": np.asarray(
                "persisted retained recursive control-only assignments "
                "with rejected-partition and prototype audit"
            ),
            "control_mean_policy": np.asarray(
                "complete control-group normalized mean"
            ),
            "control_mean_source_count": full_control_counts,
            "rejected_control_count": rejected_counts,
        }
    )
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"builder produced unsafe object array {key!r}")
    return arrays


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a real-control reservoir for a recursive Child bank"
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--context-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-divisor", type=float, default=10.0)
    parser.add_argument("--max-members-per-child", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prototype-atol", type=float, default=2e-5)
    parser.add_argument("--prototype-rtol", type=float, default=2e-4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    context = args.context_bank.expanduser().resolve()
    output = args.output.expanduser().resolve()
    arrays = build_recursive_real_control_reservoir(
        h5ad_path=args.h5ad.expanduser().resolve(),
        context_path=context,
        normalization_divisor=args.normalization_divisor,
        max_members_per_child=args.max_members_per_child,
        seed=args.seed,
        prototype_atol=args.prototype_atol,
        prototype_rtol=args.prototype_rtol,
    )
    _atomic_savez(output, arrays)
    active_counts = arrays["full_child_counts"][arrays["child_mask"]]
    stored_counts = arrays["stored_child_counts"][arrays["child_mask"]]
    report = {
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "context_bank": str(context),
        "context_bank_sha256": sha256_file(context),
        "schema_version": REAL_CONTROL_RESERVOIR_SCHEMA,
        "control_only": True,
        "treated_expression_used": False,
        "shape": list(arrays["member_residuals"].shape),
        "active_children_per_line": arrays["child_mask"].sum(axis=1).tolist(),
        "source_control_count": arrays["source_control_count"].tolist(),
        "retained_control_count": int(arrays["source_control_count"].sum()),
        "control_mean_source_count": arrays[
            "control_mean_source_count"
        ].tolist(),
        "rejected_control_count": arrays["rejected_control_count"].tolist(),
        "control_mean_policy": str(arrays["control_mean_policy"]),
        "control_grouping": str(arrays["control_grouping"]),
        "full_active_child_count_range": [
            int(active_counts.min()),
            int(active_counts.max()),
        ],
        "stored_active_child_count_range": [
            int(stored_counts.min()),
            int(stored_counts.max()),
        ],
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_recursive_real_control_reservoir"]

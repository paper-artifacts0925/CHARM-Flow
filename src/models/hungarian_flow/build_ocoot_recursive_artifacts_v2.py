#!/usr/bin/env python
"""Correct OCOO-T recursive artifact entry for processed PBMC/Tahoe data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np

from src.models.hungarian_flow.build_ocoot_recursive_artifacts import (
    CLUSTERING_MODES,
    CONTROL_GROUPINGS,
    OCOOT_DATASET_SPECS,
    OCOOT_GENE_DIM,
    OCOOT_MAX_CHILDREN,
    OCOOT_MIN_LEAF_SIZE,
    ControlRows,
    _atomic_savez,
    _atomic_write_json,
    _dense_float32,
    _feature_matrix,
    _resolve_feature_gene_names,
    build_ocoot_recursive_context_arrays,
    build_ocoot_recursive_reservoir_arrays,
    get_ocoot_dataset_spec,
    load_selected_genes,
    rekey_control_rows,
    resolve_ocoot_source_paths,
)
from src.models.hungarian_flow.context_artifact import (
    RECURSIVE_CONTEXT_SCHEMA,
    load_recursive_context_bank,
    sha256_file,
)
from src.models.real_control_residual_source.reservoir import (
    REAL_CONTROL_RESERVOIR_SCHEMA,
)


OCOOT_SCRATCH_NORMALIZATION_DIVISOR = 10.0


def load_ocoot_control_rows_v2(
    source_paths: Sequence[str | Path],
    *,
    dataset: str,
    feature_key: str = "X_hvg",
    selected_gene_file: str | Path | None = None,
) -> ControlRows:
    """Load only controls while proving the exact 2,000-feature gene order."""

    spec = get_ocoot_dataset_spec(dataset)
    # AnnData's backed mode does not keep dense matrices under ``obsm`` backed.
    # Tahoe100M stores the 2,000 model features in ``obsm/X_hvg`` across fourteen
    # very large shards, so opening them through AnnData can materialise treated
    # expression before the control mask is applied. Route Tahoe through the
    # bounded h5py reader; callers such as the PCA builder inherit the same safe
    # behavior without needing a second dataset-specific entry point.
    if spec.name == "tahoe100m":
        from src.models.hungarian_flow import (
            build_ocoot_recursive_artifacts_v2_lowmem as lowmem,
        )

        return lowmem.load_ocoot_control_rows_v2_lowmem(
            source_paths,
            dataset=dataset,
            feature_key=feature_key,
            selected_gene_file=selected_gene_file,
        )
    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if not paths:
        raise ValueError("at least one source H5AD is required")
    selected_genes = (
        None
        if selected_gene_file in (None, "", "null")
        else load_selected_genes(selected_gene_file)
    )

    expression_parts = []
    line_parts = []
    batch_parts = []
    logical_parts = []
    source_id_parts = []
    local_index_parts = []
    source_n_obs = []
    reference_genes = None
    logical_offset = 0
    for source_id, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            missing = [
                key
                for key in (
                    spec.perturbation_key,
                    spec.cell_line_key,
                    spec.batch_key,
                )
                if key not in adata.obs.columns
            ]
            if missing:
                raise ValueError(f"{path}: missing obs columns {missing}")
            feature = _feature_matrix(adata, feature_key, source=path)
            feature_shape = tuple(feature.shape)
            if feature_shape != (adata.n_obs, OCOOT_GENE_DIM):
                raise ValueError(
                    f"{path}: {feature_key} shape {feature_shape} does not match "
                    f"[n_obs,{OCOOT_GENE_DIM}]"
                )
            genes = _resolve_feature_gene_names(
                adata,
                source=path,
                feature_key=feature_key,
                expected_gene_dim=OCOOT_GENE_DIM,
                selected_genes=selected_genes,
            )
            if reference_genes is None:
                reference_genes = genes.copy()
            elif not np.array_equal(reference_genes, genes):
                raise ValueError(
                    f"{path}: 2,000-feature gene order differs from first shard"
                )

            perturbations = adata.obs[spec.perturbation_key].astype(str)
            control_positions = np.flatnonzero(
                perturbations.eq(spec.control_label).to_numpy()
            ).astype(np.int64)
            if not len(control_positions):
                raise ValueError(
                    f"{path}: no rows match control label {spec.control_label!r}"
                )
            # Leakage boundary: no treated expression is indexed or materialized.
            values = _dense_float32(
                feature[control_positions], source=path, feature_key=feature_key
            )
            expression_parts.append(values)
            line_parts.append(
                adata.obs.iloc[control_positions][spec.cell_line_key]
                .astype(str)
                .to_numpy()
            )
            batch_parts.append(
                adata.obs.iloc[control_positions][spec.batch_key]
                .astype(str)
                .to_numpy()
            )
            logical_parts.append(control_positions + int(logical_offset))
            source_id_parts.append(
                np.full(len(control_positions), source_id, dtype=np.int16)
            )
            local_index_parts.append(control_positions)
            source_n_obs.append(int(adata.n_obs))
            logical_offset += int(adata.n_obs)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    expression = np.concatenate(expression_parts).astype(np.float32, copy=False)
    cell_lines = np.concatenate(line_parts).astype(str, copy=False)
    batches = np.concatenate(batch_parts).astype(str, copy=False)
    logical = np.concatenate(logical_parts).astype(np.int64, copy=False)
    source_ids = np.concatenate(source_id_parts).astype(np.int16, copy=False)
    local_indices = np.concatenate(local_index_parts).astype(np.int64, copy=False)
    if len(np.unique(logical)) != len(logical):
        raise RuntimeError("logical control observation IDs are not unique")
    if not (
        len(expression)
        == len(cell_lines)
        == len(batches)
        == len(logical)
        == len(source_ids)
        == len(local_indices)
    ):
        raise RuntimeError("control-only source arrays do not align")
    return ControlRows(
        source_paths=paths,
        expression=expression,
        cell_lines=cell_lines,
        batches=batches,
        logical_obs_indices=logical,
        source_file_ids=source_ids,
        source_obs_indices=local_indices,
        source_n_obs=np.asarray(source_n_obs, dtype=np.int64),
        gene_names=np.asarray(reference_genes, dtype=str),
    )


def build_ocoot_recursive_artifacts_v2(
    *,
    dataset: str,
    input_path: str | Path,
    context_output: str | Path,
    reservoir_output: str | Path,
    selected_gene_file: str | Path | None = None,
    feature_key: str = "X_hvg",
    expected_source_files: int | None = None,
    anchor_capacity: int = OCOOT_MAX_CHILDREN,
    target_child_occupancy: int | None = None,
    pca_dim: int = 64,
    minimum_support: int = OCOOT_MIN_LEAF_SIZE,
    kmeans_n_init: int = 5,
    silhouette_sample_size: int = 10000,
    seed: int = 42,
    insufficient_group_policy: str = "adaptive",
    control_grouping: str = "celltype",
    clustering_mode: str = "constrained_recursive",
    normalization_divisor: float = OCOOT_SCRATCH_NORMALIZATION_DIVISOR,
    max_members_per_child: int = 64,
) -> dict:
    """Build context in raw X_hvg space and reservoir in scratch /10 space."""

    spec = get_ocoot_dataset_spec(dataset)
    sources = resolve_ocoot_source_paths(
        dataset, input_path, expected_source_files=expected_source_files
    )
    controls = load_ocoot_control_rows_v2(
        sources,
        dataset=dataset,
        feature_key=feature_key,
        selected_gene_file=selected_gene_file,
    )
    controls, discovery_center_by_batch = rekey_control_rows(
        controls, grouping=control_grouping
    )
    divisor = float(normalization_divisor)
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("normalization_divisor must be finite and positive")
    context_arrays, summaries = build_ocoot_recursive_context_arrays(
        controls,
        spec=spec,
        feature_key=feature_key,
        anchor_capacity=anchor_capacity,
        target_child_occupancy=target_child_occupancy,
        pca_dim=pca_dim,
        min_leaf_size=minimum_support,
        kmeans_n_init=kmeans_n_init,
        silhouette_sample_size=silhouette_sample_size,
        seed=seed,
        insufficient_group_policy=insufficient_group_policy,
        clustering_mode=clustering_mode,
        control_grouping=control_grouping,
        discovery_center_by_batch=discovery_center_by_batch,
    )
    context = Path(context_output).expanduser().resolve()
    reservoir = Path(reservoir_output).expanduser().resolve()
    if context == reservoir:
        raise ValueError("context_output and reservoir_output must differ")
    _atomic_savez(context, context_arrays)
    load_recursive_context_bank(context)
    reservoir_arrays = build_ocoot_recursive_reservoir_arrays(
        context_path=context,
        controls=controls,
        spec=spec,
        normalization_divisor=divisor,
        max_members_per_child=max_members_per_child,
        seed=seed,
    )
    _atomic_savez(reservoir, reservoir_arrays)

    active_counts = context_arrays["child_mask"].sum(axis=1)
    context_report = {
        "artifact": str(context),
        "artifact_sha256": sha256_file(context),
        "schema_version": RECURSIVE_CONTEXT_SCHEMA,
        "builder": "ocoot_recursive_v2",
        "dataset": spec.name,
        "source_h5ads": [str(path) for path in sources],
        "source_file_count": len(sources),
        "selected_gene_file": (
            None
            if selected_gene_file is None
            else str(Path(selected_gene_file).expanduser().resolve())
        ),
        "feature_key": feature_key,
        "gene_dim": OCOOT_GENE_DIM,
        "control_only": True,
        "treated_expression_used": False,
        "control_count": int(len(controls.expression)),
        "retained_control_count": int(context_arrays["retained_control_count"]),
        "rejected_control_count": int(context_arrays["rejected_control_count"]),
        "retained_control_count_by_group": context_arrays["retained_control_count_by_group"].tolist(),
        "rejected_control_count_by_group": context_arrays["rejected_control_count_by_group"].tolist(),
        "child_capacity": int(anchor_capacity),
        "target_child_occupancy": int(
            OCOOT_MIN_LEAF_SIZE
            if target_child_occupancy is None
            else target_child_occupancy
        ),
        "control_grouping": str(control_grouping),
        "clustering_mode": str(clustering_mode),
        "minimum_support": int(minimum_support),
        "min_leaf_size": int(minimum_support),
        "insufficient_group_policy": str(insufficient_group_policy),
        "active_children_per_group": active_counts.tolist(),
        "groups": summaries,
    }
    reservoir_report = {
        "artifact": str(reservoir),
        "artifact_sha256": sha256_file(reservoir),
        "schema_version": REAL_CONTROL_RESERVOIR_SCHEMA,
        "builder": "ocoot_recursive_v2",
        "dataset": spec.name,
        "context_bank": str(context),
        "context_bank_sha256": sha256_file(context),
        "control_only": True,
        "treated_expression_used": False,
        "normalization_divisor": divisor,
        "shape": list(reservoir_arrays["member_residuals"].shape),
        "source_control_count": reservoir_arrays["source_control_count"].tolist(),
        "retained_control_count": int(
            reservoir_arrays["source_control_count"].sum()
        ),
        "control_mean_source_count": reservoir_arrays[
            "control_mean_source_count"
        ].tolist(),
        "rejected_control_count": reservoir_arrays["rejected_control_count"].tolist(),
        "control_grouping": str(reservoir_arrays["control_grouping"]),
        "active_children_per_group": reservoir_arrays["child_mask"]
        .sum(axis=1)
        .tolist(),
    }
    _atomic_write_json(
        context.with_suffix(context.suffix + ".json"), context_report
    )
    _atomic_write_json(
        reservoir.with_suffix(reservoir.suffix + ".json"), reservoir_report
    )
    return {"context": context_report, "reservoir": reservoir_report}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build v2 PBMC/Tahoe100M control-only recursive context and "
            "real-control reservoir artifacts"
        )
    )
    parser.add_argument("--dataset", choices=sorted(OCOOT_DATASET_SPECS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selected-gene-file", type=Path, default=None)
    parser.add_argument("--feature-key", choices=("X", "X_hvg"), default="X_hvg")
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--reservoir-output", type=Path, required=True)
    parser.add_argument("--anchor-capacity", type=int, default=OCOOT_MAX_CHILDREN)
    parser.add_argument("--target-child-occupancy", type=int)
    parser.add_argument(
        "--minimum-support", type=int, default=OCOOT_MIN_LEAF_SIZE
    )
    parser.add_argument(
        "--clustering-mode", choices=CLUSTERING_MODES,
        default="constrained_recursive",
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--kmeans-n-init", type=int, default=5)
    parser.add_argument("--silhouette-sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control-grouping",
        choices=CONTROL_GROUPINGS,
        default="celltype",
    )
    parser.add_argument(
        "--insufficient-group-policy",
        choices=("adaptive", "fail"),
        default="adaptive",
    )
    parser.add_argument(
        "--normalization-divisor",
        type=float,
        default=OCOOT_SCRATCH_NORMALIZATION_DIVISOR,
        help="must match the active scratch training override (default: 10)",
    )
    parser.add_argument("--max-members-per-child", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_ocoot_recursive_artifacts_v2(
        dataset=args.dataset,
        input_path=args.input,
        selected_gene_file=args.selected_gene_file,
        feature_key=args.feature_key,
        context_output=args.context_output,
        reservoir_output=args.reservoir_output,
        anchor_capacity=args.anchor_capacity,
        target_child_occupancy=args.target_child_occupancy,
        minimum_support=args.minimum_support,
        pca_dim=args.pca_dim,
        kmeans_n_init=args.kmeans_n_init,
        silhouette_sample_size=args.silhouette_sample_size,
        seed=args.seed,
        insufficient_group_policy=args.insufficient_group_policy,
        control_grouping=args.control_grouping,
        clustering_mode=args.clustering_mode,
        normalization_divisor=args.normalization_divisor,
        max_members_per_child=args.max_members_per_child,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OCOOT_SCRATCH_NORMALIZATION_DIVISOR",
    "build_ocoot_recursive_artifacts_v2",
    "load_ocoot_control_rows_v2",
]

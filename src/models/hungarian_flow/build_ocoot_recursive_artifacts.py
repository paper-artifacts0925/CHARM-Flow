#!/usr/bin/env python
"""Build control-only recursive Child artifacts for OCOO-T datasets."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from src.models.hungarian_flow.build_anchor_bank import batch_center
from src.models.hungarian_flow.build_recursive_anchor_bank import (
    RecursivePartition,
    _atomic_savez,
    _fixed_unicode,
    _line_summary,
    recursive_sse_partition,
)
from src.models.hungarian_flow.context_artifact import (
    RECURSIVE_CONTEXT_SCHEMA,
    load_recursive_context_bank,
    sha256_file,
)
from src.models.real_control_residual_source.build_reservoir import (
    normalized_control_means,
    pack_assigned_controls,
    validate_control_assignment_partition,
)
from src.models.real_control_residual_source.reservoir import (
    REAL_CONTROL_RESERVOIR_SCHEMA,
)


OCOOT_GENE_DIM = 2000
OCOOT_MIN_LEAF_SIZE = 20
OCOOT_MAX_CHILDREN = 128
OCOOT_MAX_ANCHOR_CAPACITY = 1024


@dataclass(frozen=True)
class OcootDatasetSpec:
    name: str
    perturbation_key: str
    control_label: str
    cell_line_key: str
    batch_key: str
    expected_source_files: int
    normalization_divisor: float


OCOOT_DATASET_SPECS: Mapping[str, OcootDatasetSpec] = {
    "pbmc": OcootDatasetSpec(
        name="pbmc",
        perturbation_key="cytokine",
        control_label="PBS",
        cell_line_key="cell_type",
        batch_key="donor",
        expected_source_files=1,
        # The active OCOO-T scratch protocol explicitly overrides this to /10.
        normalization_divisor=10.0,
    ),
    "tahoe100m": OcootDatasetSpec(
        name="tahoe100m",
        perturbation_key="drugname_drugconc",
        control_label="[('DMSO_TF', 0.0, 'uM')]",
        cell_line_key="cell_line",
        batch_key="plate",
        expected_source_files=14,
        # The active OCOO-T scratch protocol explicitly overrides this to /10.
        normalization_divisor=10.0,
    ),
}


CONTROL_GROUPINGS = ("celltype", "donor", "donor_celltype")
CLUSTERING_MODES = ("constrained_recursive", "postfilter_minibatch_kmeans")
CONTROL_GROUP_SEPARATOR = "::"


@dataclass(frozen=True)
class ControlRows:
    """Control-only expression plus stable references into source H5ADs."""

    source_paths: tuple[Path, ...]
    expression: np.ndarray
    cell_lines: np.ndarray
    batches: np.ndarray
    logical_obs_indices: np.ndarray
    source_file_ids: np.ndarray
    source_obs_indices: np.ndarray
    source_n_obs: np.ndarray
    gene_names: np.ndarray
    auxiliary_labels: np.ndarray | None = None


def rekey_control_rows(
    controls: ControlRows,
    *,
    grouping: str,
) -> tuple[ControlRows, bool]:
    """Re-key PBMC controls without changing expression or source references.

    ``celltype`` preserves the historical per-cell-type, donor-centered bank.
    ``donor`` pools cell types inside each donor and clusters raw control
    geometry; the original cell type is retained as an auxiliary label for
    soft matching priors. ``donor_celltype`` creates exact composite banks.

    The boolean return value controls whether discovery features are centered
    by ``ControlRows.batches`` before PCA.
    """

    mode = str(grouping)
    if mode not in CONTROL_GROUPINGS:
        raise ValueError(
            f"grouping must be one of {CONTROL_GROUPINGS}; got {mode!r}"
        )
    if mode == "celltype":
        return controls, True

    celltypes = np.asarray(controls.cell_lines).astype(str, copy=False)
    donors = np.asarray(controls.batches).astype(str, copy=False)
    if celltypes.shape != donors.shape or celltypes.ndim != 1:
        raise ValueError("control cell-type and donor labels must align")
    if mode == "donor":
        groups = donors
    else:
        groups = np.char.add(
            np.char.add(donors, CONTROL_GROUP_SEPARATOR), celltypes
        )
    return (
        ControlRows(
            source_paths=controls.source_paths,
            expression=controls.expression,
            cell_lines=groups,
            batches=donors,
            logical_obs_indices=controls.logical_obs_indices,
            source_file_ids=controls.source_file_ids,
            source_obs_indices=controls.source_obs_indices,
            source_n_obs=controls.source_n_obs,
            gene_names=controls.gene_names,
            auxiliary_labels=celltypes,
        ),
        False,
    )

class _PrimitiveOnlyUnpickler(pickle.Unpickler):
    """Read the legacy primitive gene list without permitting imports."""

    def find_class(self, module: str, name: str):  # pragma: no cover - safety hook
        raise pickle.UnpicklingError(
            f"global object loading is forbidden ({module}.{name})"
        )


def load_selected_genes(path: str | Path) -> list[str]:
    """Load and validate the official primitive-list selected-gene artifact."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    value = _PrimitiveOnlyUnpickler(io.BytesIO(artifact.read_bytes())).load()
    if not isinstance(value, (list, tuple)):
        raise ValueError("selected gene artifact must be a list or tuple")
    genes = [str(item) for item in value]
    if len(genes) != OCOOT_GENE_DIM:
        raise ValueError(
            f"selected gene artifact must contain {OCOOT_GENE_DIM} genes"
        )
    if len(set(genes)) != len(genes) or any(not gene for gene in genes):
        raise ValueError("selected gene names must be non-empty and unique")
    return genes


def get_ocoot_dataset_spec(dataset: str) -> OcootDatasetSpec:
    key = str(dataset).strip().lower()
    try:
        return OCOOT_DATASET_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(OCOOT_DATASET_SPECS))
        raise ValueError(f"unsupported OCOO-T dataset {dataset!r}; choose {choices}") from exc


def resolve_ocoot_source_paths(
    dataset: str,
    input_path: str | Path,
    *,
    expected_source_files: int | None = None,
) -> tuple[Path, ...]:
    """Resolve the exact one-file PBMC or multi-file Tahoe source layout."""

    spec = get_ocoot_dataset_spec(dataset)
    source = Path(input_path).expanduser().resolve()
    expected = (
        spec.expected_source_files
        if expected_source_files is None
        else int(expected_source_files)
    )
    if expected < 1:
        raise ValueError("expected_source_files must be positive")
    if spec.name == "pbmc":
        if not source.is_file() or source.suffix.lower() != ".h5ad":
            raise ValueError(f"PBMC input must be one .h5ad file: {source}")
        paths = (source,)
    else:
        if not source.is_dir():
            raise ValueError(f"Tahoe100M input must be an H5AD directory: {source}")
        paths = tuple(sorted(path.resolve() for path in source.glob("*.h5ad")))
    if len(paths) != expected:
        raise ValueError(
            f"{spec.name} requires exactly {expected} source H5AD file(s); "
            f"found {len(paths)} under {source}"
        )
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("source H5AD basenames must be unique")
    return paths


def _dense_float32(value, *, source: Path, feature_key: str) -> np.ndarray:
    if sparse.issparse(value):
        value = value.toarray()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{source}: {feature_key} must have shape [N,G]")
    if not np.isfinite(result).all():
        raise ValueError(f"{source}: selected control expression is non-finite")
    return result


def _feature_matrix(adata, feature_key: str, *, source: Path):
    key = str(feature_key)
    if key == "X":
        return adata.X
    if key not in adata.obsm:
        raise ValueError(f"{source}: missing obsm/{key}")
    return adata.obsm[key]


def _resolve_feature_gene_names(
    adata,
    *,
    source: Path,
    feature_key: str,
    expected_gene_dim: int,
    selected_genes: Sequence[str] | None,
) -> np.ndarray:
    """Resolve the exact feature-column order without assuming n_vars=2,000."""

    var_names = np.asarray(adata.var_names.astype(str), dtype=str)
    if len(np.unique(var_names)) != len(var_names):
        raise ValueError(f"{source}: var_names must be unique")
    hvg_names = None
    if "highly_variable" in adata.var.columns:
        mask = np.asarray(adata.var["highly_variable"], dtype=bool)
        if mask.shape != (adata.n_vars,):
            raise ValueError(f"{source}: var.highly_variable does not align with var")
        if int(mask.sum()) == expected_gene_dim:
            hvg_names = var_names[mask]

    if str(feature_key) == "X":
        if int(adata.n_vars) != expected_gene_dim:
            raise ValueError(
                f"{source}: X has {adata.n_vars} genes; expected {expected_gene_dim}"
            )
        genes = var_names
    elif selected_genes is not None:
        genes = np.asarray([str(gene) for gene in selected_genes], dtype=str)
        if hvg_names is not None and not np.array_equal(genes, hvg_names):
            raise ValueError(
                f"{source}: selected gene order differs from var/highly_variable"
            )
    elif hvg_names is not None:
        genes = hvg_names
    elif int(adata.n_vars) == expected_gene_dim:
        genes = var_names
    else:
        raise ValueError(
            f"{source}: cannot prove obsm/{feature_key} gene order when "
            f"n_vars={adata.n_vars}; provide --selected-gene-file or a "
            f"var.highly_variable mask with {expected_gene_dim} true entries"
        )
    if genes.shape != (expected_gene_dim,) or len(np.unique(genes)) != len(genes):
        raise ValueError(
            f"{source}: resolved feature genes must be {expected_gene_dim} unique names"
        )
    return genes.astype(str, copy=False)


def load_ocoot_control_rows(
    source_paths: Sequence[str | Path],
    *,
    spec: OcootDatasetSpec,
    feature_key: str = "X_hvg",
    expected_gene_dim: int = OCOOT_GENE_DIM,
    selected_gene_file: str | Path | None = None,
) -> ControlRows:
    """Read only exact control rows and verify the 2,000-gene contract.

    Logical observation IDs are the local row index plus a cumulative source
    offset.  They are unique across Tahoe plates while remaining reproducible
    without materialising or rewriting a combined H5AD.
    """

    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if not paths:
        raise ValueError("at least one source H5AD is required")
    expected_gene_dim = int(expected_gene_dim)
    if expected_gene_dim != OCOOT_GENE_DIM:
        raise ValueError(
            f"OCOO-T artifacts require exactly {OCOOT_GENE_DIM} genes, "
            f"got expected_gene_dim={expected_gene_dim}"
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
            if feature_key not in adata.obsm:
                raise ValueError(f"{path}: missing obsm/{feature_key}")
            feature_shape = tuple(adata.obsm[feature_key].shape)
            if feature_shape != (adata.n_obs, expected_gene_dim):
                raise ValueError(
                    f"{path}: obsm/{feature_key} shape {feature_shape} does not "
                    f"match [n_obs,{expected_gene_dim}]"
                )
            if int(adata.n_vars) != expected_gene_dim:
                raise ValueError(
                    f"{path}: n_vars={adata.n_vars}; OCOO-T requires "
                    f"the processed {expected_gene_dim}-gene H5AD"
                )
            genes = np.asarray(adata.var_names.astype(str), dtype=str)
            if len(np.unique(genes)) != expected_gene_dim:
                raise ValueError(f"{path}: var_names must be unique")
            if reference_genes is None:
                reference_genes = genes.copy()
            elif not np.array_equal(reference_genes, genes):
                raise ValueError(
                    f"{path}: 2,000-gene order differs from the first source H5AD"
                )

            perturbations = adata.obs[spec.perturbation_key].astype(str)
            control_positions = np.flatnonzero(
                perturbations.eq(spec.control_label).to_numpy()
            ).astype(np.int64)
            if not len(control_positions):
                raise ValueError(
                    f"{path}: no rows match control label {spec.control_label!r}"
                )
            # Leakage boundary: expression is sliced only after the exact
            # control mask has been resolved from obs metadata.
            control_expression = _dense_float32(
                adata.obsm[feature_key][control_positions],
                source=path,
                feature_key=feature_key,
            )
            expression_parts.append(control_expression)
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

    expression = np.concatenate(expression_parts, axis=0).astype(
        np.float32, copy=False
    )
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


def _validate_build_contract(
    *,
    anchor_capacity: int,
    min_leaf_size: int,
    insufficient_group_policy: str,
    clustering_mode: str,
) -> tuple[int, str, str]:
    capacity = int(anchor_capacity)
    if capacity < 1 or capacity > OCOOT_MAX_ANCHOR_CAPACITY:
        raise ValueError(
            "anchor_capacity must be in "
            f"[1,{OCOOT_MAX_ANCHOR_CAPACITY}], got {capacity}"
        )
    if int(min_leaf_size) != OCOOT_MIN_LEAF_SIZE:
        raise ValueError(
            f"OCOO-T artifacts require minimum support={OCOOT_MIN_LEAF_SIZE}"
        )
    policy = str(insufficient_group_policy).strip().lower()
    if policy not in {"adaptive", "fail"}:
        raise ValueError("insufficient_group_policy must be 'adaptive' or 'fail'")
    mode = str(clustering_mode).strip().lower()
    if mode not in CLUSTERING_MODES:
        raise ValueError(
            f"clustering_mode must be one of {CLUSTERING_MODES}; got {mode!r}"
        )
    return capacity, policy, mode


def _postfilter_minibatch_partition(
    latent: np.ndarray,
    *,
    max_children: int,
    minimum_support: int,
    seed: int,
    n_init: int,
) -> tuple[RecursivePartition, np.ndarray, np.ndarray]:
    """Cluster without support constraints, then reject undersized clusters."""

    values = np.asarray(latent, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("latent must have non-empty shape [N,D]")
    if not np.isfinite(values).all():
        raise ValueError("latent contains non-finite values")
    minimum = int(minimum_support)
    if minimum < 2 or len(values) < minimum:
        raise ValueError("not enough controls for one supported Child")
    candidate_children = min(
        int(max_children), max(1, len(values) // minimum)
    )
    model = MiniBatchKMeans(
        n_clusters=candidate_children,
        n_init=int(n_init),
        max_iter=100,
        batch_size=min(len(values), max(1024, 4 * candidate_children)),
        max_no_improvement=20,
        random_state=int(seed),
        reassignment_ratio=0.01,
    )
    candidate_labels = np.asarray(model.fit_predict(values), dtype=np.int64)
    if candidate_labels.shape != (len(values),):
        raise RuntimeError("MiniBatchKMeans returned misaligned labels")
    if (
        (candidate_labels < 0).any()
        or (candidate_labels >= candidate_children).any()
    ):
        raise RuntimeError("MiniBatchKMeans returned an out-of-range label")
    candidate_counts = np.bincount(
        candidate_labels, minlength=candidate_children
    ).astype(np.int64)
    retained_candidates = np.flatnonzero(candidate_counts >= minimum).astype(
        np.int64
    )
    if not len(retained_candidates):
        raise ValueError(
            "post-filter clustering rejected every candidate Child; reduce "
            "anchor_capacity or minimum support"
        )
    remap = np.full(candidate_children, -1, dtype=np.int64)
    remap[retained_candidates] = np.arange(len(retained_candidates), dtype=np.int64)
    labels = remap[candidate_labels]
    within_sse = []
    for child in range(len(retained_candidates)):
        local = values[labels == child].astype(np.float64)
        center = local.mean(axis=0, dtype=np.float64)
        within_sse.append(float(np.square(local - center).sum()))
    partition = RecursivePartition(
        labels=labels,
        active_children=len(retained_candidates),
        within_child_sse=np.asarray(within_sse, dtype=np.float64),
        split_gains=np.empty(0, dtype=np.float64),
        fallback_splits=0,
        stopped_reason="postfilter_minibatch_kmeans",
    )
    return partition, candidate_counts, retained_candidates


def build_ocoot_recursive_context_arrays(
    controls: ControlRows,
    *,
    spec: OcootDatasetSpec,
    feature_key: str = "X_hvg",
    anchor_capacity: int = OCOOT_MAX_CHILDREN,
    target_child_occupancy: int | None = None,
    pca_dim: int = 64,
    min_leaf_size: int = OCOOT_MIN_LEAF_SIZE,
    kmeans_n_init: int = 5,
    silhouette_sample_size: int = 10000,
    seed: int = 42,
    insufficient_group_policy: str = "adaptive",
    clustering_mode: str = "constrained_recursive",
    control_grouping: str = "celltype",
    discovery_center_by_batch: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Cluster a control-only logical table into a fixed-capacity bank."""

    capacity, policy, mode = _validate_build_contract(
        anchor_capacity=anchor_capacity,
        min_leaf_size=min_leaf_size,
        insufficient_group_policy=insufficient_group_policy,
        clustering_mode=clustering_mode,
    )
    grouping = str(control_grouping).strip().lower()
    if grouping not in CONTROL_GROUPINGS:
        raise ValueError(f"control_grouping must be one of {CONTROL_GROUPINGS}")
    target_occupancy = (
        int(min_leaf_size)
        if target_child_occupancy is None
        else int(target_child_occupancy)
    )
    if target_occupancy < int(min_leaf_size):
        raise ValueError("target_child_occupancy must be at least min_leaf_size")
    expression = np.asarray(controls.expression, dtype=np.float32)
    if expression.ndim != 2 or expression.shape[1] != OCOOT_GENE_DIM:
        raise ValueError(
            f"control expression must have shape [N,{OCOOT_GENE_DIM}]"
        )
    if not np.isfinite(expression).all():
        raise ValueError("control expression contains non-finite values")
    line_names = sorted(np.unique(controls.cell_lines).tolist())
    if not line_names:
        raise ValueError("no control groups were found")

    auxiliary_labels = None
    if controls.auxiliary_labels is not None:
        auxiliary_labels = np.asarray(controls.auxiliary_labels).astype(
            str, copy=False
        )
        if auxiliary_labels.shape != controls.cell_lines.shape:
            raise ValueError("auxiliary control labels must align with controls")
    support = {
        name: int(np.count_nonzero(controls.cell_lines == name))
        for name in line_names
    }
    below_minimum = {
        name: count for name, count in support.items() if count < min_leaf_size
    }
    if below_minimum:
        allow_rare_composites = (
            grouping == "donor_celltype"
            and policy == "adaptive"
            and mode == "constrained_recursive"
        )
        if not allow_rare_composites:
            raise ValueError(
                "control group(s) cannot form one legal min_leaf=20 Child: "
                + ", ".join(
                    f"{name}={count}" for name, count in below_minimum.items()
                )
            )
    if policy == "fail":
        required = capacity * min_leaf_size
        insufficient = {
            name: count for name, count in support.items() if count < required
        }
        if insufficient:
            raise ValueError(
                f"strict {capacity}-Child policy requires at least {required} "
                "controls per group; insufficient: "
                + ", ".join(
                    f"{name}={count}" for name, count in insufficient.items()
                )
            )

    shared_latent_by_group: dict[str, np.ndarray] = {}
    if grouping == "donor_celltype":
        celltype_groups: dict[str, list[str]] = {}
        for name in line_names:
            if CONTROL_GROUP_SEPARATOR not in name:
                raise ValueError(f"invalid donor/cell-type control group {name!r}")
            _, celltype = name.split(CONTROL_GROUP_SEPARATOR, 1)
            celltype_groups.setdefault(celltype, []).append(name)
        for celltype_index, group_names in enumerate(
            celltype_groups.values()
        ):
            pooled_positions = np.flatnonzero(
                np.isin(controls.cell_lines, group_names)
            ).astype(np.int64)
            pooled_discovery = batch_center(
                expression[pooled_positions],
                controls.batches[pooled_positions],
            )
            components = min(
                int(pca_dim),
                len(pooled_discovery) - 1,
                pooled_discovery.shape[1],
            )
            if components < 2:
                raise ValueError(
                    f"{group_names[0]} cell-type pool needs two PCA components"
                )
            projection = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=int(seed) + celltype_index,
            ).fit(pooled_discovery)
            for name in group_names:
                positions = np.flatnonzero(
                    controls.cell_lines == name
                ).astype(np.int64)
                local_discovery = batch_center(
                    expression[positions], controls.batches[positions]
                )
                shared_latent_by_group[name] = projection.transform(
                    local_discovery
                ).astype(np.float32)

    line_results = []
    assignment_logical = []
    assignment_source_ids = []
    assignment_source_rows = []
    assignment_line_ids = []
    assignment_child_ids = []
    rejected_logical = []
    rejected_source_ids = []
    rejected_source_rows = []
    rejected_group_ids = []
    rejected_child_ids = []
    for line_index, cell_line in enumerate(line_names):
        positions = np.flatnonzero(controls.cell_lines == cell_line).astype(np.int64)
        raw = expression[positions]
        batches = controls.batches[positions]
        supported_children = min(
            capacity,
            max(1, len(raw) // target_occupancy),
        )
        if grouping == "donor_celltype":
            latent = shared_latent_by_group[cell_line]
        else:
            discovery = (
                batch_center(raw, batches)
                if discovery_center_by_batch
                else raw
            )
            components = min(int(pca_dim), len(raw) - 1, raw.shape[1])
            if components < 2:
                raise ValueError(f"{cell_line}: PCA needs at least two components")
            latent = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=int(seed) + line_index,
            ).fit_transform(discovery).astype(np.float32)
        if len(raw) < min_leaf_size:
            centered = latent.astype(np.float64) - latent.mean(
                axis=0, dtype=np.float64
            )
            partition = RecursivePartition(
                labels=np.zeros(len(raw), dtype=np.int64),
                active_children=1,
                within_child_sse=np.asarray(
                    [np.square(centered).sum()], dtype=np.float64
                ),
                split_gains=np.empty(0, dtype=np.float64),
                fallback_splits=0,
                stopped_reason="rare_group_single_child",
            )
            candidate_counts = np.asarray([len(raw)], dtype=np.int64)
        elif mode == "constrained_recursive":
            partition = recursive_sse_partition(
                latent,
                max_children=supported_children,
                min_leaf_size=min_leaf_size,
                seed=int(seed) + line_index,
                n_init=kmeans_n_init,
            )
            candidate_counts = np.bincount(
                partition.labels, minlength=partition.active_children
            ).astype(np.int64)
        else:
            partition, candidate_counts, _ = _postfilter_minibatch_partition(
                latent,
                max_children=supported_children,
                minimum_support=min_leaf_size,
                seed=int(seed) + line_index,
                n_init=kmeans_n_init,
            )
        if policy == "fail" and partition.active_children != capacity:
            raise RuntimeError(
                f"{cell_line}: strict policy requested {capacity} active Children "
                f"but clustering produced {partition.active_children} "
                f"({partition.stopped_reason})"
            )
        active = partition.active_children
        retained = partition.labels >= 0
        rejected = ~retained
        retained_labels = partition.labels[retained]
        retained_raw = raw[retained]
        retained_latent = latent[retained]
        retained_batches = batches[retained]
        counts = np.bincount(retained_labels, minlength=active).astype(np.int64)
        rare_single_child = bool(
            grouping == "donor_celltype"
            and len(raw) < min_leaf_size
            and active == 1
            and counts.tolist() == [len(raw)]
        )
        if (
            len(retained_raw) < min_leaf_size or (counts < min_leaf_size).any()
        ) and not rare_single_child:
            raise RuntimeError("retained Child violates minimum support")
        means = np.stack(
            [
                retained_raw[retained_labels == child].mean(axis=0)
                for child in range(active)
            ]
        ).astype(np.float32)
        stds = np.stack(
            [
                retained_raw[retained_labels == child].std(axis=0)
                for child in range(active)
            ]
        ).astype(np.float32)
        weights = counts.astype(np.float32) / np.float32(len(retained_raw))
        sampled = np.arange(len(retained_raw), dtype=np.int64)
        if len(sampled) > int(silhouette_sample_size):
            sampled = np.random.default_rng(int(seed) + line_index).choice(
                sampled, size=int(silhouette_sample_size), replace=False
            )
        if active > 1 and len(sampled) > active:
            silhouette = float(
                silhouette_score(retained_latent[sampled], retained_labels[sampled])
            )
        else:
            silhouette = 0.0
        batch_nmi = float(
            normalized_mutual_info_score(retained_batches, retained_labels)
        )
        auxiliary_nmi = (
            0.0
            if auxiliary_labels is None
            else float(
                normalized_mutual_info_score(
                    auxiliary_labels[positions][retained], retained_labels
                )
            )
        )
        summary = _line_summary(
            cell_line, counts, silhouette, batch_nmi, partition
        )
        rejected_candidate_children = (
            0
            if rare_single_child
            else int(
                np.count_nonzero(
                    (candidate_counts > 0) & (candidate_counts < min_leaf_size)
                )
            )
        )
        summary.update(
            {
                "child_capacity": capacity,
                "support_limited_child_ceiling": int(supported_children),
                "insufficient_group_policy": policy,
                "adapted_below_capacity": bool(active < capacity),
                "clustering_mode": mode,
                "auxiliary_nmi": auxiliary_nmi,
                "minimum_support": int(counts.min()),
                "split_minimum_support": int(min_leaf_size),
                "rare_single_child_exception": rare_single_child,
                "input_controls": int(len(raw)),
                "retained_controls": int(retained.sum()),
                "rejected_controls": int(rejected.sum()),
                "candidate_children": int(len(candidate_counts)),
                "occupied_candidate_children": int(
                    np.count_nonzero(candidate_counts)
                ),
                "rejected_candidate_children": rejected_candidate_children,
                "empty_candidate_children": int(
                    np.count_nonzero(candidate_counts == 0)
                ),
            }
        )
        line_results.append(
            {
                "name": cell_line,
                "positions": positions,
                "labels": partition.labels,
                "means": means,
                "stds": stds,
                "weights": weights,
                "counts": counts,
                "partition": partition,
                "silhouette": silhouette,
                "batch_nmi": batch_nmi,
                "auxiliary_nmi": auxiliary_nmi,
                "summary": summary,
            }
        )
        retained_positions = positions[retained]
        rejected_positions = positions[rejected]
        assignment_logical.append(controls.logical_obs_indices[retained_positions])
        assignment_source_ids.append(controls.source_file_ids[retained_positions])
        assignment_source_rows.append(controls.source_obs_indices[retained_positions])
        assignment_line_ids.append(
            np.full(len(retained_positions), line_index, dtype=np.int16)
        )
        assignment_child_ids.append(retained_labels.astype(np.int16))
        rejected_logical.append(controls.logical_obs_indices[rejected_positions])
        rejected_source_ids.append(controls.source_file_ids[rejected_positions])
        rejected_source_rows.append(controls.source_obs_indices[rejected_positions])
        rejected_group_ids.append(
            np.full(len(rejected_positions), line_index, dtype=np.int16)
        )
        rejected_child_ids.append(
            np.full(len(rejected_positions), -1, dtype=np.int16)
        )

    lines = len(line_results)
    genes = expression.shape[1]
    means = np.zeros((lines, capacity, genes), dtype=np.float32)
    stds = np.zeros_like(means)
    weights = np.zeros((lines, capacity), dtype=np.float32)
    counts = np.zeros((lines, capacity), dtype=np.int64)
    child_mask = np.zeros((lines, capacity), dtype=bool)
    within_sse = np.zeros((lines, capacity), dtype=np.float64)
    weighted_mean = np.zeros((lines, genes), dtype=np.float32)
    silhouettes = np.zeros(lines, dtype=np.float64)
    batch_nmis = np.zeros(lines, dtype=np.float64)
    auxiliary_nmis = np.zeros(lines, dtype=np.float64)
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
        auxiliary_nmis[line] = result["auxiliary_nmi"]

    auxiliary_names = None
    child_auxiliary_weights = None
    if auxiliary_labels is not None:
        auxiliary_names = sorted(np.unique(auxiliary_labels).tolist())
        auxiliary_to_id = {
            name: index for index, name in enumerate(auxiliary_names)
        }
        child_auxiliary_weights = np.zeros(
            (lines, capacity, len(auxiliary_names)), dtype=np.float32
        )
        for line, result in enumerate(line_results):
            positions = np.flatnonzero(
                controls.cell_lines == result["name"]
            ).astype(np.int64)
            local_auxiliary = auxiliary_labels[positions]
            labels = result["partition"].labels
            for child in range(len(result["counts"])):
                child_rows = labels == child
                for name in np.unique(local_auxiliary[child_rows]).tolist():
                    child_auxiliary_weights[line, child, auxiliary_to_id[name]] = (
                        np.count_nonzero(child_rows & (local_auxiliary == name))
                    )
                total = child_auxiliary_weights[line, child].sum()
                if total <= 0:
                    raise RuntimeError("active Child has no auxiliary labels")
                child_auxiliary_weights[line, child] /= total

    input_count_by_group = np.asarray(
        [result["summary"]["input_controls"] for result in line_results],
        dtype=np.int64,
    )
    retained_count_by_group = counts.sum(axis=1).astype(np.int64)
    rejected_count_by_group = input_count_by_group - retained_count_by_group
    retained_control_count = int(retained_count_by_group.sum())
    rejected_control_count = int(rejected_count_by_group.sum())
    effective_minimum_support = int(
        counts[child_mask].min()
        if below_minimum else min_leaf_size
    )
    metadata = {
        "dataset": spec.name,
        "source_h5ads": [str(path) for path in controls.source_paths],
        "source_h5ad_basenames": [path.name for path in controls.source_paths],
        "source_n_obs": controls.source_n_obs.tolist(),
        "feature_key": feature_key,
        "perturbation_key": spec.perturbation_key,
        "control_label": spec.control_label,
        "cell_line_key": spec.cell_line_key,
        "batch_key": spec.batch_key,
        "gene_dim": OCOOT_GENE_DIM,
        "anchor_capacity": capacity,
        "minimum_support": effective_minimum_support,
        "split_minimum_support": OCOOT_MIN_LEAF_SIZE,
        "target_child_occupancy": target_occupancy,
        "rare_single_child_contexts": sorted(below_minimum),
        "pca_dim": int(pca_dim),
        "min_leaf_size": OCOOT_MIN_LEAF_SIZE,
        "kmeans_n_init": int(kmeans_n_init),
        "silhouette_sample_size": int(silhouette_sample_size),
        "seed": int(seed),
        "control_grouping": grouping,
        "clustering_mode": mode,
        "input_control_count": int(len(expression)),
        "retained_control_count": retained_control_count,
        "rejected_control_count": rejected_control_count,
        "retained_control_count_by_group": retained_count_by_group.tolist(),
        "rejected_control_count_by_group": rejected_count_by_group.tolist(),
        "insufficient_group_policy": policy,
        "clustering_space": (
            "shared same-cell-type donor-centered PCA"
            if grouping == "donor_celltype"
            else (
                "per-control-group batch-centered PCA"
                if discovery_center_by_batch
                else "per-control-group raw PCA"
            )
        ),
        "prototype_space": f"original {feature_key}",
        "assignment": (
            "recursive constrained binary k-means"
            if mode == "constrained_recursive"
            else "MiniBatchKMeans followed by minimum-support filtering"
        ),
        "split_priority": (
            "SSE * (1 + log1p(size/min_leaf_size))"
            if mode == "constrained_recursive" else None
        ),
        "multi_h5ad_logical_index": "cumulative n_obs offset + source row",
    }
    arrays = {
        "schema_version": np.asarray(RECURSIVE_CONTEXT_SCHEMA),
        "control_only": np.asarray(True, dtype=bool),
        "treated_expression_used": np.asarray(False, dtype=bool),
        "cellline_names": _fixed_unicode(line_names),
        "prototype_means": means,
        "prototype_stds": stds,
        "weights": weights,
        "cluster_cell_counts": counts,
        "child_mask": child_mask,
        "weighted_mean": weighted_mean,
        "within_child_sse": within_sse,
        "latent_silhouette": silhouettes,
        "batch_nmi": batch_nmis,
        "auxiliary_nmi": auxiliary_nmis,
        "control_obs_indices": np.concatenate(assignment_logical),
        "control_cellline_ids": np.concatenate(assignment_line_ids),
        "control_child_ids": np.concatenate(assignment_child_ids),
        "control_source_file_ids": np.concatenate(assignment_source_ids),
        "control_source_obs_indices": np.concatenate(assignment_source_rows),
        "rejected_control_obs_indices": np.concatenate(rejected_logical),
        "rejected_control_source_ids": np.concatenate(rejected_source_ids),
        "rejected_control_source_rows": np.concatenate(rejected_source_rows),
        "rejected_control_group_ids": np.concatenate(rejected_group_ids),
        "rejected_control_child_ids": np.concatenate(rejected_child_ids),
        "input_control_count_by_group": input_count_by_group,
        "retained_control_count_by_group": retained_count_by_group,
        "rejected_control_count_by_group": rejected_count_by_group,
        "retained_control_count": np.asarray(retained_control_count, dtype=np.int64),
        "rejected_control_count": np.asarray(rejected_control_count, dtype=np.int64),
        "source_file_basenames": _fixed_unicode(
            [path.name for path in controls.source_paths]
        ),
        "source_n_obs": controls.source_n_obs.astype(np.int64),
        "gene_names": _fixed_unicode(controls.gene_names),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if auxiliary_names is not None:
        arrays["auxiliary_label_names"] = _fixed_unicode(auxiliary_names)
        arrays["child_auxiliary_weights"] = child_auxiliary_weights
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"builder produced unsafe object array {key!r}")
    return arrays, [result["summary"] for result in line_results]


def _required(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"OCOO-T recursive context is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in context key {key!r}")
    return value


def _optional(loaded, key: str) -> np.ndarray | None:
    if key not in loaded.files:
        return None
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in context key {key!r}")
    return value


def build_ocoot_recursive_reservoir_arrays(
    *,
    context_path: str | Path,
    controls: ControlRows,
    spec: OcootDatasetSpec,
    normalization_divisor: float | None = None,
    max_members_per_child: int = 64,
    seed: int = 42,
    prototype_atol: float = 2e-5,
    prototype_rtol: float = 2e-4,
) -> dict[str, np.ndarray]:
    """Audit persisted multi-file assignments and pack a runtime reservoir."""

    context_path = Path(context_path).expanduser().resolve()
    bank = load_recursive_context_bank(context_path)
    metadata = bank["metadata"]
    if str(metadata.get("dataset")) != spec.name:
        raise ValueError(
            f"context dataset {metadata.get('dataset')!r} does not match {spec.name!r}"
        )
    expected_basenames = [path.name for path in controls.source_paths]
    if metadata.get("source_h5ad_basenames") != expected_basenames:
        raise ValueError("context/source H5AD basenames or order differ")
    if int(metadata.get("gene_dim", -1)) != OCOOT_GENE_DIM:
        raise ValueError("context does not declare the 2,000-gene OCOO-T contract")

    with np.load(context_path, allow_pickle=False) as loaded:
        names = _required(loaded, "cellline_names").astype(str)
        prototypes = _required(loaded, "prototype_means").astype(np.float32)
        weights = _required(loaded, "weights").astype(np.float32)
        counts = _required(loaded, "cluster_cell_counts").astype(np.int64)
        child_mask = _required(loaded, "child_mask").astype(bool)
        logical = _required(loaded, "control_obs_indices").astype(np.int64)
        line_ids = _required(loaded, "control_cellline_ids").astype(np.int64)
        child_ids = _required(loaded, "control_child_ids").astype(np.int64)
        source_ids = _required(loaded, "control_source_file_ids").astype(np.int64)
        source_rows = _required(loaded, "control_source_obs_indices").astype(np.int64)
        genes = _required(loaded, "gene_names").astype(str)
        rejected_logical = _optional(loaded, "rejected_control_obs_indices")
        rejected_source_ids = _optional(loaded, "rejected_control_source_ids")
        rejected_source_rows = _optional(loaded, "rejected_control_source_rows")
        rejected_group_ids = _optional(loaded, "rejected_control_group_ids")
        rejected_child_ids = _optional(loaded, "rejected_control_child_ids")

    vectors = (line_ids, child_ids, source_ids, source_rows)
    if logical.ndim != 1 or not len(logical):
        raise ValueError("context control assignment must be non-empty")
    if any(value.shape != logical.shape for value in vectors):
        raise ValueError("context control assignment vectors do not align")
    if (logical < 0).any() or len(np.unique(logical)) != len(logical):
        raise ValueError("context logical observation IDs must be unique non-negative")
    if not np.array_equal(genes, controls.gene_names):
        raise ValueError("context/source 2,000-gene order differs")

    rejected_vectors = (
        rejected_source_ids,
        rejected_source_rows,
        rejected_group_ids,
        rejected_child_ids,
    )
    if rejected_logical is None:
        if any(value is not None for value in rejected_vectors):
            raise ValueError("context has incomplete rejected control vectors")
    else:
        rejected_logical = np.asarray(rejected_logical, dtype=np.int64)
        if any(value is None for value in rejected_vectors):
            raise ValueError("context has incomplete rejected control vectors")
        rejected_source_ids = np.asarray(rejected_source_ids, dtype=np.int64)
        rejected_source_rows = np.asarray(rejected_source_rows, dtype=np.int64)
        rejected_group_ids = np.asarray(rejected_group_ids, dtype=np.int64)
        rejected_child_ids = np.asarray(rejected_child_ids, dtype=np.int64)
        if any(
            value.shape != rejected_logical.shape for value in rejected_vectors
        ):
            raise ValueError("rejected control vectors do not align")
        if (rejected_child_ids != -1).any():
            raise ValueError("rejected control Child IDs must all be -1")

    lines, capacity, gene_dim = prototypes.shape
    if gene_dim != OCOOT_GENE_DIM or names.shape != (lines,):
        raise ValueError("context L/K/G dimensions do not satisfy OCOO-T contract")
    if ((line_ids < 0) | (line_ids >= lines)).any():
        raise ValueError("persisted control line ID is out of range")
    if ((child_ids < 0) | (child_ids >= capacity)).any():
        raise ValueError("persisted control Child ID is out of range")
    if not child_mask[line_ids, child_ids].all():
        raise ValueError("persisted control assignment selects a padded Child")

    all_logical = np.asarray(controls.logical_obs_indices, dtype=np.int64)
    validate_control_assignment_partition(
        all_control_obs_indices=all_logical,
        retained_control_obs_indices=logical,
        rejected_control_obs_indices=rejected_logical,
    )

    # Align retained and rejected rows independently to the freshly loaded
    # control-only source by stable logical IDs. The expression passed to the
    # reservoir packer is intentionally retained-only.
    source_order = np.argsort(controls.logical_obs_indices, kind="stable")
    sorted_source_logical = all_logical[source_order]

    def _source_positions(stable_ids: np.ndarray, label: str) -> np.ndarray:
        ids = np.asarray(stable_ids, dtype=np.int64)
        positions = np.searchsorted(sorted_source_logical, ids)
        if len(ids):
            in_range = positions < len(sorted_source_logical)
            safe = np.minimum(positions, len(sorted_source_logical) - 1)
            if not in_range.all() or not np.array_equal(
                sorted_source_logical[safe], ids
            ):
                raise ValueError(f"{label} logical control ID is not in source")
        return source_order[positions]

    retained_source_positions = _source_positions(logical, "retained")
    expression = np.asarray(
        controls.expression[retained_source_positions], dtype=np.float32
    )
    expected_lines = controls.cell_lines[retained_source_positions]
    if not np.array_equal(names[line_ids], expected_lines):
        raise ValueError("persisted control cell-line assignments do not match source")
    if not np.array_equal(
        source_ids, controls.source_file_ids[retained_source_positions]
    ) or not np.array_equal(
        source_rows, controls.source_obs_indices[retained_source_positions]
    ):
        raise ValueError("persisted source-file control references do not match source")

    if rejected_logical is not None:
        if ((rejected_group_ids < 0) | (rejected_group_ids >= lines)).any():
            raise ValueError("rejected control group ID is out of range")
        rejected_source_positions = _source_positions(
            rejected_logical, "rejected"
        )
        if not np.array_equal(
            names[rejected_group_ids],
            controls.cell_lines[rejected_source_positions],
        ):
            raise ValueError("rejected control group assignments do not match source")
        if not np.array_equal(
            rejected_source_ids,
            controls.source_file_ids[rejected_source_positions],
        ) or not np.array_equal(
            rejected_source_rows,
            controls.source_obs_indices[rejected_source_positions],
        ):
            raise ValueError(
                "rejected source-file control references do not match source"
            )

    divisor = (
        spec.normalization_divisor
        if normalization_divisor is None
        else float(normalization_divisor)
    )
    names_order = np.argsort(names, kind="stable")
    sorted_names = names[names_order]
    source_groups = np.asarray(controls.cell_lines).astype(str, copy=False)
    group_positions = np.searchsorted(sorted_names, source_groups)
    group_in_range = group_positions < len(sorted_names)
    safe_group_positions = np.minimum(group_positions, len(sorted_names) - 1)
    if not group_in_range.all() or not np.array_equal(
        sorted_names[safe_group_positions], source_groups
    ):
        raise ValueError("source contains a control group absent from context")
    full_line_ids = names_order[group_positions].astype(np.int64, copy=False)
    full_control_mean, full_control_counts = normalized_control_means(
        control_expression=controls.expression,
        control_line_ids=full_line_ids,
        cellline_names=names.tolist(),
        normalization_divisor=divisor,
    )
    rejected_counts = np.bincount(
        rejected_group_ids
        if rejected_logical is not None
        else np.empty((0,), dtype=np.int64),
        minlength=lines,
    ).astype(np.int64)

    runtime_prototypes = np.zeros_like(prototypes)
    runtime_masks = np.zeros_like(child_mask)
    runtime_child_ids = np.full_like(child_ids, -1)
    for line, cell_line in enumerate(names.tolist()):
        rows = np.flatnonzero(line_ids == line)
        original_child = child_ids[rows]
        active_original = np.flatnonzero(child_mask[line])
        rebuilt_counts = np.bincount(original_child, minlength=capacity)
        if not np.array_equal(rebuilt_counts, counts[line]):
            raise ValueError(f"{cell_line}: persisted recursive counts do not match")
        rebuilt_means = np.zeros((capacity, gene_dim), dtype=np.float32)
        for child in active_original:
            rebuilt_means[child] = expression[rows][original_child == child].mean(
                axis=0
            )
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
                f"{cell_line}: prototypes differ from control assignments; "
                f"max_abs={maximum:.7g}"
            )

        # Match build_unique_cell_line_control_bank: descending occupancy.
        order = np.argsort(-weights[line, active_original])
        ordered_original = active_original[order]
        inverse = np.full(capacity, -1, dtype=np.int64)
        inverse[ordered_original] = np.arange(len(ordered_original))
        runtime_child_ids[rows] = inverse[original_child]
        active_count = len(ordered_original)
        runtime_prototypes[line, :active_count] = prototypes[line, ordered_original]
        runtime_masks[line, :active_count] = True
    if (runtime_child_ids < 0).any():
        raise RuntimeError("could not map every recursive Child to runtime order")

    arrays = pack_assigned_controls(
        control_expression=expression,
        control_line_ids=line_ids,
        control_child_ids=runtime_child_ids,
        cellline_names=names.tolist(),
        child_prototypes=runtime_prototypes,
        child_mask=runtime_masks,
        normalization_divisor=divisor,
        max_members_per_child=int(max_members_per_child),
        seed=int(seed),
        source_obs_indices=logical,
        control_mean_override=full_control_mean,
    )
    arrays.update(
        {
            "context_bank_sha256": np.asarray(sha256_file(context_path)),
            "context_bank_schema": np.asarray(RECURSIVE_CONTEXT_SCHEMA),
            "dataset": np.asarray(spec.name),
            "control_label": np.asarray(spec.control_label),
            "expression_key": np.asarray(
                str(metadata.get("feature_key", "X_hvg"))
            ),
            "perturbation_key": np.asarray(spec.perturbation_key),
            "cell_line_key": np.asarray(spec.cell_line_key),
            "control_grouping": np.asarray(
                str(metadata.get("control_grouping", "celltype"))
            ),
            "source_h5ad_basenames": _fixed_unicode(expected_basenames),
            "assignment_method": np.asarray(
                "persisted retained multi-H5AD control-only assignments "
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


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_ocoot_recursive_artifacts(
    *,
    dataset: str,
    input_path: str | Path,
    context_output: str | Path,
    reservoir_output: str | Path,
    expected_source_files: int | None = None,
    target_child_occupancy: int | None = None,
    anchor_capacity: int = OCOOT_MAX_CHILDREN,
    pca_dim: int = 64,
    kmeans_n_init: int = 5,
    silhouette_sample_size: int = 10000,
    seed: int = 42,
    insufficient_group_policy: str = "adaptive",
    clustering_mode: str = "constrained_recursive",
    control_grouping: str = "celltype",
    normalization_divisor: float | None = None,
    max_members_per_child: int = 64,
) -> dict:
    """Build and atomically persist context plus real-control reservoir."""

    spec = get_ocoot_dataset_spec(dataset)
    sources = resolve_ocoot_source_paths(
        dataset,
        input_path,
        expected_source_files=expected_source_files,
    )
    controls = load_ocoot_control_rows(sources, spec=spec)
    controls, discovery_center_by_batch = rekey_control_rows(
        controls, grouping=control_grouping
    )
    context_arrays, summaries = build_ocoot_recursive_context_arrays(
        controls,
        target_child_occupancy=target_child_occupancy,
        spec=spec,
        anchor_capacity=anchor_capacity,
        pca_dim=pca_dim,
        kmeans_n_init=kmeans_n_init,
        silhouette_sample_size=silhouette_sample_size,
        clustering_mode=clustering_mode,
        control_grouping=control_grouping,
        discovery_center_by_batch=discovery_center_by_batch,
        seed=seed,
        insufficient_group_policy=insufficient_group_policy,
    )
    context = Path(context_output).expanduser().resolve()
    reservoir = Path(reservoir_output).expanduser().resolve()
    if context == reservoir:
        raise ValueError("context_output and reservoir_output must differ")
    _atomic_savez(context, context_arrays)
    # Validate the serialized public contract before deriving the reservoir.
    load_recursive_context_bank(context)
    reservoir_arrays = build_ocoot_recursive_reservoir_arrays(
        context_path=context,
        controls=controls,
        spec=spec,
        normalization_divisor=normalization_divisor,
        max_members_per_child=max_members_per_child,
        seed=seed,
    )
    _atomic_savez(reservoir, reservoir_arrays)

    active_counts = context_arrays["child_mask"].sum(axis=1)
    context_report = {
        "artifact": str(context),
        "artifact_sha256": sha256_file(context),
        "schema_version": RECURSIVE_CONTEXT_SCHEMA,
        "dataset": spec.name,
        "source_h5ads": [str(path) for path in sources],
        "source_file_count": len(sources),
        "gene_dim": OCOOT_GENE_DIM,
        "control_only": True,
        "target_child_occupancy": int(
            OCOOT_MIN_LEAF_SIZE if target_child_occupancy is None else target_child_occupancy
        ),
        "treated_expression_used": False,
        "control_count": int(len(controls.expression)),
        "child_capacity": int(anchor_capacity),
        "min_leaf_size": OCOOT_MIN_LEAF_SIZE,
        "insufficient_group_policy": str(insufficient_group_policy),
        "active_children_per_group": active_counts.tolist(),
        "groups": summaries,
    }
    reservoir_report = {
        "artifact": str(reservoir),
        "artifact_sha256": sha256_file(reservoir),
        "schema_version": REAL_CONTROL_RESERVOIR_SCHEMA,
        "dataset": spec.name,
        "context_bank": str(context),
        "context_bank_sha256": sha256_file(context),
        "control_only": True,
        "treated_expression_used": False,
        "normalization_divisor": float(
            reservoir_arrays["normalization_divisor"].reshape(-1)[0]
        ),
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
            "Build PBMC/Tahoe100M control-only recursive context and "
            "real-control reservoir artifacts"
        )
    )
    parser.add_argument("--dataset", choices=sorted(OCOOT_DATASET_SPECS), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="PBMC processed H5AD or Tahoe100M directory containing exactly 14 H5ADs",
    )
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--reservoir-output", type=Path, required=True)
    parser.add_argument("--anchor-capacity", type=int, default=OCOOT_MAX_CHILDREN)
    parser.add_argument("--target-child-occupancy", type=int)
    parser.add_argument(
        "--control-grouping",
        choices=CONTROL_GROUPINGS,
        default="celltype",
    )
    parser.add_argument(
        "--clustering-mode",
        choices=CLUSTERING_MODES,
        default="constrained_recursive",
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--kmeans-n-init", type=int, default=5)
    parser.add_argument("--silhouette-sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--insufficient-group-policy",
        choices=("adaptive", "fail"),
        default="adaptive",
        help=(
            "adaptive masks unsupported Child slots (but still requires >=20 "
            "controls); fail requires all requested Child slots"
        ),
    )
    parser.add_argument(
        "--normalization-divisor",
        type=float,
        default=None,
        help="defaults to 1.0, matching PBMC/Tahoe finetune normalize_counts=false",
    )
    parser.add_argument("--max-members-per-child", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_ocoot_recursive_artifacts(
        dataset=args.dataset,
        input_path=args.input,
        context_output=args.context_output,
        reservoir_output=args.reservoir_output,
        anchor_capacity=args.anchor_capacity,
        target_child_occupancy=args.target_child_occupancy,
        pca_dim=args.pca_dim,
        kmeans_n_init=args.kmeans_n_init,
        silhouette_sample_size=args.silhouette_sample_size,
        seed=args.seed,
        insufficient_group_policy=args.insufficient_group_policy,
        clustering_mode=args.clustering_mode,
        control_grouping=args.control_grouping,
        normalization_divisor=args.normalization_divisor,
        max_members_per_child=args.max_members_per_child,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ControlRows",
    "OCOOT_DATASET_SPECS",
    "OCOOT_GENE_DIM",
    "OCOOT_MAX_CHILDREN",
    "OCOOT_MIN_LEAF_SIZE",
    "OcootDatasetSpec",
    "build_ocoot_recursive_artifacts",
    "build_ocoot_recursive_context_arrays",
    "build_ocoot_recursive_reservoir_arrays",
    "get_ocoot_dataset_spec",
    "load_ocoot_control_rows",
    "resolve_ocoot_source_paths",
]

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import yaml

from scripts.parent_residual_gene_dit.build_replogle_artifact import (
    CONTROL,
    SCHEMA_VERSION,
    TEST,
    TRAIN,
    VALIDATION,
    _atomic_save_json,
    _atomic_save_npz,
    _canonical_json_sha256,
    _fixed_unicode,
    _normalization_divisor,
    _sha256_file,
    build_cross_line_priors,
    build_gene_modules,
    fit_gene_wise_ridge,
    load_selected_genes,
    read_categorical,
)


@dataclass(frozen=True)
class OCOOTSettings:
    """Resolved fields that determine the scientific artifact semantics."""

    dataset_kind: str
    shards: tuple[Path, ...]
    selected_gene_file: Path
    expression_key: str
    perturbation_key: str
    context_key: str
    celltype_key: str
    parent_grouping: str
    split_group_key: str
    control_label: str
    normalize_divisor: float
    holdout_groups: tuple[str, ...]
    validation_perturbations: tuple[str, ...]
    test_perturbations: tuple[str, ...]
    expected_gene_dim: int | None


@dataclass(frozen=True)
class ExpressionSpec:
    location: str
    shape: tuple[int, int]
    storage: str
    dtype: str


COMPOSITE_CONTEXT_SEPARATOR = "::"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_dataset_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(path)
    shards = tuple(sorted(item.resolve() for item in path.glob("*.h5ad")))
    if not shards:
        raise ValueError(f"dataset directory contains no .h5ad shards: {path}")
    return shards


def load_ocoot_settings(
    resolved_config: Path,
    *,
    dataset_kind: str = "auto",
    parent_grouping: str = "celltype",
    dataset_path_override: Path | None = None,
    selected_gene_override: Path | None = None,
) -> OCOOTSettings:
    """Resolve PBMC/Tahoe settings without invoking Hydra or loading AnnData."""

    with resolved_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or not isinstance(config.get("data"), Mapping):
        raise ValueError("resolved config must contain a resolved top-level data mapping")
    data = config["data"]
    dataset_value = dataset_path_override or data.get("dataset_path")
    selected_value = selected_gene_override or data.get("selected_gene_file")
    if dataset_value is None or selected_value is None:
        raise ValueError("data.dataset_path and data.selected_gene_file are required")
    dataset_path = Path(dataset_value).expanduser().resolve()
    selected_path = Path(selected_value).expanduser().resolve()
    shards = _resolve_dataset_files(dataset_path)

    kind = str(dataset_kind).lower()
    if kind not in {"auto", "pbmc", "tahoe100m"}:
        raise ValueError("dataset_kind must be auto, pbmc, or tahoe100m")
    holdout_batches = data.get("holdout_batches") or ()
    if isinstance(holdout_batches, Mapping):
        raise ValueError("PBMC holdout_batches must be one flat donor list")
    if kind == "auto":
        kind = "pbmc" if holdout_batches else "tahoe100m"
    parent_grouping = str(parent_grouping).lower()
    if parent_grouping not in {"celltype", "donor", "donor_celltype"}:
        raise ValueError(
            "parent_grouping must be celltype, donor, or donor_celltype"
        )
    if parent_grouping in {"donor", "donor_celltype"} and kind != "pbmc":
        raise ValueError("donor-based Parent grouping is currently PBMC-only")

    if bool(data.get("split_control", False)):
        raise ValueError("Parent artifact construction requires split_control=false")
    if bool(data.get("sort_gene_names", False)):
        raise ValueError("sort_gene_names=true would invalidate stored expression order")
    holdout = data.get("holdout_pert") or {}
    if not isinstance(holdout, Mapping):
        raise ValueError("data.holdout_pert must be a mapping")
    validation = tuple(str(item) for item in (holdout.get("validation") or ()))
    test = tuple(str(item) for item in (holdout.get("test") or ()))
    overlap = sorted(set(validation).intersection(test))
    if overlap:
        raise ValueError(f"validation/test perturbations overlap: {overlap[:10]}")

    celltype_key = str(data.get("cell_line_key", "cell_line"))
    context_key = celltype_key
    if kind == "pbmc":
        split_group_key = data.get("perturbseq_batch_col")
        if not split_group_key:
            raise ValueError("PBMC requires data.perturbseq_batch_col for donor splitting")
        if parent_grouping == "donor":
            context_key = str(split_group_key)
        elif parent_grouping == "donor_celltype":
            context_key = (
                f"{split_group_key}{COMPOSITE_CONTEXT_SEPARATOR}{celltype_key}"
            )
        holdout_groups = tuple(str(item) for item in holdout_batches)
        if not holdout_groups:
            raise ValueError("PBMC requires a non-empty data.holdout_batches donor list")
    else:
        split_group_key = context_key
        holdout_groups = tuple(str(item) for item in (data.get("holdout_celltype") or ()))
        if not holdout_groups:
            raise ValueError("Tahoe100M requires non-empty data.holdout_celltype")

    expected_dim = data.get("embed_shape")
    return OCOOTSettings(
        dataset_kind=kind,
        shards=shards,
        selected_gene_file=selected_path,
        expression_key=str(data.get("embed_key", "X_hvg")),
        perturbation_key=str(data.get("pert_col", "gene")),
        context_key=context_key,
        celltype_key=celltype_key,
        parent_grouping=parent_grouping,
        split_group_key=str(split_group_key),
        control_label=str(data.get("control_pert", "non-targeting")),
        normalize_divisor=_normalization_divisor(data.get("normalize_counts")),
        holdout_groups=holdout_groups,
        validation_perturbations=validation,
        test_perturbations=test,
        expected_gene_dim=int(expected_dim) if expected_dim is not None else None,
    )


def _sparse_shape(node: h5py.Group) -> tuple[int, int]:
    raw = node.attrs.get("shape")
    if raw is None:
        raise ValueError("sparse expression group has no shape attribute")
    shape = tuple(int(value) for value in np.asarray(raw).reshape(-1).tolist())
    if len(shape) != 2:
        raise ValueError(f"invalid sparse expression shape: {shape}")
    return shape


def inspect_expression(handle: h5py.File, expression_key: str) -> ExpressionSpec:
    """Select obsm/X_hvg or root X, accepting dense and CSR storage."""

    candidates = []
    if expression_key == "X":
        candidates.append("X")
    else:
        candidates.extend((f"obsm/{expression_key}", "X"))
    location = next((candidate for candidate in candidates if candidate in handle), None)
    if location is None:
        raise ValueError(f"H5AD has neither obsm/{expression_key} nor root X")
    node = handle[location]
    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            raise ValueError(f"{location} must be two-dimensional")
        return ExpressionSpec(
            location=location,
            shape=(int(node.shape[0]), int(node.shape[1])),
            storage="dense",
            dtype=str(node.dtype),
        )
    if not isinstance(node, h5py.Group):
        raise ValueError(f"unsupported expression object at {location}")
    required = {"data", "indices", "indptr"}
    if not required.issubset(node.keys()):
        raise ValueError(f"sparse {location} must contain {sorted(required)}")
    encoding = _decode(node.attrs.get("encoding-type", ""))
    if encoding and encoding not in {"csr_matrix", "csr"}:
        raise ValueError(f"only CSR expression is supported, got {encoding!r}")
    return ExpressionSpec(
        location=location,
        shape=_sparse_shape(node),
        storage="csr",
        dtype=str(node["data"].dtype),
    )


def read_expression_chunk(
    handle: h5py.File, spec: ExpressionSpec, start: int, stop: int
) -> np.ndarray:
    """Materialize one bounded dense row chunk from dense or CSR H5 storage."""

    node = handle[spec.location]
    if spec.storage == "dense":
        return np.asarray(node[start:stop], dtype=np.float32)
    assert isinstance(node, h5py.Group)
    pointers = np.asarray(node["indptr"][start : stop + 1], dtype=np.int64)
    base = int(pointers[0])
    end = int(pointers[-1])
    indices = np.asarray(node["indices"][base:end], dtype=np.int64)
    data = np.asarray(node["data"][base:end], dtype=np.float32)
    pointers = pointers - base
    values = np.zeros((stop - start, spec.shape[1]), dtype=np.float32)
    for row in range(stop - start):
        left, right = int(pointers[row]), int(pointers[row + 1])
        row_indices = indices[left:right]
        if len(row_indices) and (
            row_indices.min() < 0 or row_indices.max() >= spec.shape[1]
        ):
            raise ValueError(f"CSR column index outside expression width in row {start + row}")
        if len(row_indices) != len(np.unique(row_indices)):
            np.add.at(values[row], row_indices, data[left:right])
        else:
            values[row, row_indices] = data[left:right]
    return values


def _var_names(handle: h5py.File) -> list[str]:
    var = handle.get("var")
    if not isinstance(var, h5py.Group):
        raise ValueError("H5AD has no var group")
    index_name = _decode(var.attrs.get("_index", "_index"))
    if index_name not in var:
        raise ValueError(f"H5AD var index dataset {index_name!r} is absent")
    return [_decode(value) for value in var[index_name][:]]


def verify_expression_gene_order(
    handle: h5py.File, spec: ExpressionSpec, selected_genes: Sequence[str]
) -> str:
    """Fail closed unless every expression column has a proved selected-gene name."""

    if spec.shape[1] != len(selected_genes):
        raise ValueError(
            f"selected gene count {len(selected_genes)} != expression width {spec.shape[1]}"
        )
    names = _var_names(handle)
    var = handle["var"]
    if spec.location.startswith("obsm/"):
        if "highly_variable" in var:
            mask = np.asarray(var["highly_variable"][:], dtype=bool)
            if len(mask) != len(names):
                raise ValueError("var index and highly_variable lengths differ")
            projected = [name for name, keep in zip(names, mask) if keep]
            proof = "var/highly_variable"
        elif len(names) == spec.shape[1]:
            projected = names
            proof = "var index (already selected)"
        else:
            raise ValueError(
                "obsm expression requires var/highly_variable when n_vars != expression width"
            )
    else:
        if len(names) != spec.shape[1]:
            raise ValueError("root X width must equal the H5AD var index length")
        projected = names
        proof = "var index"
    if projected != list(selected_genes):
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(projected, selected_genes))
                if left != right
            ),
            min(len(projected), len(selected_genes)),
        )
        raise ValueError(
            f"selected gene order does not match {proof}; first mismatch at {mismatch}"
        )
    return proof


def _global_codes(
    obs: h5py.Group, key: str, lookup: Mapping[str, int]
) -> np.ndarray:
    if key not in obs:
        raise ValueError(f"H5AD obs has no {key!r}")
    names, codes = read_categorical(obs[key])
    missing = sorted(set(names).difference(lookup))
    if missing:
        raise RuntimeError(f"metadata pass omitted {key} values: {missing[:8]}")
    remap = np.asarray([lookup[name] for name in names], dtype=np.int64)
    return remap[codes]


def _composite_context_codes(
    obs: h5py.Group,
    settings: OCOOTSettings,
    lookup: Mapping[str, int],
) -> np.ndarray:
    """Encode donor/cell-type pairs without allocating one string per cell."""

    donor_names, donor_codes = read_categorical(obs[settings.split_group_key])
    celltype_names, celltype_codes = read_categorical(obs[settings.celltype_key])
    if len(donor_codes) != len(celltype_codes):
        raise ValueError("donor and cell-type covariate lengths differ")
    remap = np.empty((len(donor_names), len(celltype_names)), dtype=np.int64)
    for donor_id, donor in enumerate(donor_names):
        for celltype_id, celltype in enumerate(celltype_names):
            name = f"{donor}{COMPOSITE_CONTEXT_SEPARATOR}{celltype}"
            if name not in lookup:
                remap[donor_id, celltype_id] = -1
            else:
                remap[donor_id, celltype_id] = int(lookup[name])
    result = remap[donor_codes, celltype_codes]
    if np.any(result < 0):
        raise RuntimeError("metadata pass omitted an observed donor/cell-type pair")
    return result


def _read_shard_codes(
    handle: h5py.File,
    settings: OCOOTSettings,
    context_lookup: Mapping[str, int],
    group_lookup: Mapping[str, int],
    pert_lookup: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = handle.get("obs")
    if not isinstance(obs, h5py.Group):
        raise ValueError("H5AD has no obs group")
    if settings.parent_grouping == "donor_celltype":
        context = _composite_context_codes(obs, settings, context_lookup)
    else:
        context = _global_codes(obs, settings.context_key, context_lookup)
    if settings.split_group_key == settings.context_key:
        groups = context
    else:
        groups = _global_codes(obs, settings.split_group_key, group_lookup)
    perturbations = _global_codes(obs, settings.perturbation_key, pert_lookup)
    if len(context) != len(groups) or len(context) != len(perturbations):
        raise ValueError("obs covariate lengths differ")
    return context, groups, perturbations


def _assign_splits(
    group_codes: np.ndarray,
    perturbation_codes: np.ndarray,
    *,
    control_id: int,
    holdout_group_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> np.ndarray:
    control = perturbation_codes == int(control_id)
    held_group = np.isin(group_codes, holdout_group_ids)
    validation = ~control & held_group & np.isin(perturbation_codes, validation_ids)
    test = ~control & held_group & np.isin(perturbation_codes, test_ids)
    if np.any(validation & test):
        raise RuntimeError("validation and test cell masks overlap")
    result = np.full(len(group_codes), TRAIN, dtype=np.uint8)
    result[validation] = VALIDATION
    result[test] = TEST
    result[control] = CONTROL
    return result


def _inspect_all_shards(
    settings: OCOOTSettings, genes: Sequence[str]
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    contexts: set[str] = set()
    groups: set[str] = set()
    perturbations: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for shard in settings.shards:
        with h5py.File(shard, "r") as handle:
            spec = inspect_expression(handle, settings.expression_key)
            proof = verify_expression_gene_order(handle, spec, genes)
            if settings.expected_gene_dim is not None and spec.shape[1] != settings.expected_gene_dim:
                raise ValueError(
                    f"config embed_shape={settings.expected_gene_dim} != expression width={spec.shape[1]}"
                )
            obs = handle.get("obs")
            if not isinstance(obs, h5py.Group):
                raise ValueError(f"{shard} has no obs group")
            if settings.parent_grouping == "donor_celltype":
                donor_names, donor_codes = read_categorical(
                    obs[settings.split_group_key]
                )
                celltype_names, celltype_codes = read_categorical(
                    obs[settings.celltype_key]
                )
                if len(donor_codes) != len(celltype_codes):
                    raise ValueError("donor and cell-type covariate lengths differ")
                pair_ids = np.unique(
                    donor_codes.astype(np.int64) * len(celltype_names)
                    + celltype_codes.astype(np.int64)
                )
                context_names = [
                    f"{donor_names[pair_id // len(celltype_names)]}"
                    f"{COMPOSITE_CONTEXT_SEPARATOR}"
                    f"{celltype_names[pair_id % len(celltype_names)]}"
                    for pair_id in pair_ids.tolist()
                ]
                context_codes = donor_codes
                group_names, group_codes = donor_names, donor_codes
            else:
                context_names, context_codes = read_categorical(
                    obs[settings.context_key]
                )
            if settings.parent_grouping != "donor_celltype" and settings.split_group_key == settings.context_key:
                group_names, group_codes = context_names, context_codes
            elif settings.parent_grouping != "donor_celltype":
                group_names, group_codes = read_categorical(obs[settings.split_group_key])
            pert_names, pert_codes = read_categorical(obs[settings.perturbation_key])
            if not (
                len(context_codes) == len(group_codes) == len(pert_codes) == spec.shape[0]
            ):
                raise ValueError(f"obs/expression row mismatch in {shard}")
            contexts.update(context_names)
            groups.update(group_names)
            perturbations.update(pert_names)
            manifests.append(
                {
                    "path": str(shard),
                    "size_bytes": int(shard.stat().st_size),
                    "rows": int(spec.shape[0]),
                    "expression_location": spec.location,
                    "expression_storage": spec.storage,
                    "expression_dtype": spec.dtype,
                    "gene_order_proof": proof,
                }
            )
    return sorted(contexts), sorted(groups), sorted(perturbations), manifests


def _same_celltype_cross_context_priors(
    train_delta: np.ndarray,
    train_count: np.ndarray,
    train_mask: np.ndarray,
    context_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build leave-donor-out priors using only donors of the same cell type."""

    num_contexts, num_perts, gene_dim = train_delta.shape
    result = {
        "equal_line": np.zeros(
            (num_contexts, num_perts, gene_dim), dtype=np.float32
        ),
        "count_weighted": np.zeros(
            (num_contexts, num_perts, gene_dim), dtype=np.float32
        ),
        "mask": np.zeros((num_contexts, num_perts), dtype=bool),
        "cell_count": np.zeros((num_contexts, num_perts), dtype=np.int64),
        "source_count": np.zeros((num_contexts, num_perts), dtype=np.int16),
    }
    by_celltype: dict[str, list[int]] = {}
    for context_id, name in enumerate(context_names):
        if COMPOSITE_CONTEXT_SEPARATOR not in name:
            raise ValueError(f"invalid donor/cell-type context: {name!r}")
        _, celltype = name.split(COMPOSITE_CONTEXT_SEPARATOR, 1)
        by_celltype.setdefault(celltype, []).append(context_id)
    for context_ids in by_celltype.values():
        index = np.asarray(context_ids, dtype=np.int64)
        local = build_cross_line_priors(
            train_delta[index], train_count[index], train_mask[index]
        )
        for key in result:
            result[key][index] = local[key]
    return result


def _select_module_controls(
    settings: OCOOTSettings,
    context_lookup: Mapping[str, int],
    group_lookup: Mapping[str, int],
    pert_lookup: Mapping[str, int],
    *,
    maximum: int,
    excluded_contexts: Sequence[str],
    seed: int,
) -> list[tuple[int, int]]:
    """Uniformly sample controls using shard-local then global random priorities."""

    if maximum < 3:
        raise ValueError("module_max_controls must be at least 3")
    unknown = sorted(set(excluded_contexts).difference(context_lookup))
    if unknown:
        raise ValueError(f"module-excluded contexts are absent: {unknown}")
    excluded_ids = np.asarray(
        [context_lookup[name] for name in excluded_contexts], dtype=np.int64
    )
    control_id = pert_lookup[settings.control_label]
    rng = np.random.default_rng(int(seed))
    candidates: list[tuple[float, int, int]] = []
    for shard_id, shard in enumerate(settings.shards):
        with h5py.File(shard, "r") as handle:
            context, _, perturbations = _read_shard_codes(
                handle, settings, context_lookup, group_lookup, pert_lookup
            )
        eligible = perturbations == control_id
        if len(excluded_ids):
            eligible &= ~np.isin(context, excluded_ids)
        rows = np.flatnonzero(eligible).astype(np.int64)
        priorities = rng.random(len(rows))
        if len(rows) > maximum:
            keep = np.argpartition(priorities, maximum - 1)[:maximum]
            rows, priorities = rows[keep], priorities[keep]
        candidates.extend(
            (float(priority), int(shard_id), int(row))
            for priority, row in zip(priorities, rows)
        )
        if len(candidates) > maximum:
            candidates = sorted(candidates)[:maximum]
    if len(candidates) < 3:
        raise ValueError("fewer than three eligible controls remain for gene modules")
    return sorted((shard_id, row) for _, shard_id, row in sorted(candidates)[:maximum])


def _safe_dense_shape(
    contexts: int, perturbations: int, genes: int, maximum: int
) -> None:
    elements = int(contexts) * int(perturbations) * int(genes)
    if elements > int(maximum):
        gib = elements * 4 / 1024**3
        raise ValueError(
            "dense Parent bank would contain "
            f"{elements:,} float elements ({gib:.2f} GiB per float32 tensor), "
            f"above --max-dense-elements={int(maximum):,}; sparse consumer support "
            "is required before building this dataset"
        )


def build_ocoot_artifact(
    resolved_config: Path,
    output: Path,
    *,
    metadata_output: Path | None = None,
    dataset_kind: str = "auto",
    parent_grouping: str = "celltype",
    dataset_path_override: Path | None = None,
    selected_gene_override: Path | None = None,
    expected_shards: int | None = None,
    chunk_size: int = 4096,
    min_condition_cells: int = 1,
    module_max_controls: int = 8192,
    module_exclude_contexts: Sequence[str] = (),
    num_gene_modules: int = 128,
    module_embedding_dim: int = 64,
    ridge_alpha: float = 0.01,
    ridge_lambdas: Sequence[float] | str = (0.0001, 0.001, 0.01, 0.1, 1.0),
    ridge_default_lambda: float = 0.01,
    ridge_cv_folds: int = 5,
    max_dense_elements: int = 250_000_000,
    hash_source_files: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a safe, legacy-loader-compatible Parent core artifact."""

    resolved_config = Path(resolved_config).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    metadata_output = (
        Path(metadata_output).expanduser().resolve()
        if metadata_output is not None
        else output.with_suffix(".metadata.json")
    )
    if chunk_size < 1 or min_condition_cells < 1 or max_dense_elements < 1:
        raise ValueError("chunk/min-condition/max-dense arguments must be positive")
    settings = load_ocoot_settings(
        resolved_config,
        dataset_kind=dataset_kind,
        parent_grouping=parent_grouping,
        dataset_path_override=dataset_path_override,
        selected_gene_override=selected_gene_override,
    )
    if not settings.selected_gene_file.is_file():
        raise FileNotFoundError(settings.selected_gene_file)
    if expected_shards is not None and len(settings.shards) != int(expected_shards):
        raise ValueError(
            f"expected {int(expected_shards)} shards, found {len(settings.shards)}"
        )
    genes = load_selected_genes(settings.selected_gene_file)
    context_names, group_names, pert_names, manifests = _inspect_all_shards(
        settings, genes
    )
    context_lookup = {name: index for index, name in enumerate(context_names)}
    group_lookup = {name: index for index, name in enumerate(group_names)}
    pert_lookup = {name: index for index, name in enumerate(pert_names)}
    if settings.control_label not in pert_lookup:
        raise ValueError(f"control label {settings.control_label!r} is absent")
    missing_holdout_groups = sorted(set(settings.holdout_groups).difference(group_lookup))
    if missing_holdout_groups:
        raise ValueError(f"configured holdout groups are absent: {missing_holdout_groups}")
    missing_validation = sorted(set(settings.validation_perturbations).difference(pert_lookup))
    missing_test = sorted(set(settings.test_perturbations).difference(pert_lookup))
    if missing_validation or missing_test:
        raise ValueError(
            "configured held-out perturbations are absent; "
            f"validation={missing_validation[:8]}, test={missing_test[:8]}"
        )
    _safe_dense_shape(
        len(context_names), len(pert_names), len(genes), int(max_dense_elements)
    )

    module_samples = _select_module_controls(
        settings,
        context_lookup,
        group_lookup,
        pert_lookup,
        maximum=int(module_max_controls),
        excluded_contexts=tuple(module_exclude_contexts),
        seed=int(seed),
    )
    module_by_shard: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for position, (shard_id, row) in enumerate(module_samples):
        rows, positions = module_by_shard.get(
            shard_id,
            (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
        )
        module_by_shard[shard_id] = (
            np.append(rows, np.int64(row)),
            np.append(positions, np.int64(position)),
        )

    lines, perts, gene_dim = len(context_names), len(pert_names), len(genes)
    flat_conditions = lines * perts
    control_sum = np.zeros((lines, gene_dim), dtype=np.float64)
    control_count = np.zeros(lines, dtype=np.int64)
    condition_sum = np.zeros((flat_conditions, gene_dim), dtype=np.float64)
    condition_count = np.zeros(flat_conditions, dtype=np.int64)
    validation_cell_count = np.zeros(flat_conditions, dtype=np.int64)
    test_cell_count = np.zeros(flat_conditions, dtype=np.int64)
    module_values = np.empty((len(module_samples), gene_dim), dtype=np.float32)
    module_context_codes = np.empty(len(module_samples), dtype=np.int64)
    module_filled = np.zeros(len(module_samples), dtype=bool)
    digest = hashlib.sha256()
    split_cell_counts = {
        "train_treated": 0,
        "validation_treated": 0,
        "test_treated": 0,
        "training_controls": 0,
    }
    permitted_min, permitted_max = np.inf, -np.inf
    permitted_sum, permitted_elements = 0.0, 0
    control_id = pert_lookup[settings.control_label]
    holdout_group_ids = np.asarray(
        [group_lookup[name] for name in settings.holdout_groups], dtype=np.int64
    )
    validation_ids = np.asarray(
        [pert_lookup[name] for name in settings.validation_perturbations], dtype=np.int64
    )
    test_ids = np.asarray(
        [pert_lookup[name] for name in settings.test_perturbations], dtype=np.int64
    )

    for shard_id, shard in enumerate(settings.shards):
        with h5py.File(shard, "r") as handle:
            spec = inspect_expression(handle, settings.expression_key)
            context, groups, perturbations = _read_shard_codes(
                handle, settings, context_lookup, group_lookup, pert_lookup
            )
            split = _assign_splits(
                groups,
                perturbations,
                control_id=control_id,
                holdout_group_ids=holdout_group_ids,
                validation_ids=validation_ids,
                test_ids=test_ids,
            )
            split_cell_counts["train_treated"] += int(np.sum(split == TRAIN))
            split_cell_counts["validation_treated"] += int(np.sum(split == VALIDATION))
            split_cell_counts["test_treated"] += int(np.sum(split == TEST))
            split_cell_counts["training_controls"] += int(np.sum(split == CONTROL))
            flat = context * perts + perturbations
            validation_cell_count += np.bincount(
                flat[split == VALIDATION], minlength=flat_conditions
            ).astype(np.int64)
            test_cell_count += np.bincount(
                flat[split == TEST], minlength=flat_conditions
            ).astype(np.int64)
            sampled_rows, sampled_positions = module_by_shard.get(
                shard_id,
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
            )

            for start in range(0, spec.shape[0], int(chunk_size)):
                stop = min(start + int(chunk_size), spec.shape[0])
                raw = read_expression_chunk(handle, spec, start, stop)
                if not np.isfinite(raw).all():
                    raise ValueError(f"non-finite expression in {shard} rows [{start},{stop})")
                values = raw / np.float32(settings.normalize_divisor)
                local_split = split[start:stop]
                local_context = context[start:stop]
                local_perts = perturbations[start:stop]

                left = int(np.searchsorted(sampled_rows, start, side="left"))
                right = int(np.searchsorted(sampled_rows, stop, side="left"))
                if right > left:
                    relative = sampled_rows[left:right] - start
                    destinations = sampled_positions[left:right]
                    module_values[destinations] = values[relative]
                    module_context_codes[destinations] = local_context[relative]
                    module_filled[destinations] = True

                control_mask = local_split == CONTROL
                for context_id in np.unique(local_context[control_mask]):
                    chosen = values[control_mask & (local_context == context_id)]
                    control_sum[context_id] += chosen.sum(axis=0, dtype=np.float64)
                    control_count[context_id] += len(chosen)

                train_mask = local_split == TRAIN
                if train_mask.any():
                    train_values = values[train_mask]
                    train_flat = local_context[train_mask] * perts + local_perts[train_mask]
                    order = np.argsort(train_flat, kind="stable")
                    sorted_ids = train_flat[order]
                    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
                    unique_ids = sorted_ids[starts]
                    condition_sum[unique_ids] += np.add.reduceat(
                        train_values[order], starts, axis=0, dtype=np.float64
                    )
                    condition_count[unique_ids] += np.diff(np.r_[starts, len(sorted_ids)])

                permitted = control_mask | train_mask
                permitted_values = values[permitted]
                permitted_rows = np.flatnonzero(permitted).astype(np.int64) + start
                identities = np.empty((len(permitted_rows), 2), dtype="<i8")
                identities[:, 0] = shard_id
                identities[:, 1] = permitted_rows
                digest.update(identities.tobytes())
                digest.update(np.ascontiguousarray(permitted_values, dtype="<f4").tobytes())
                if permitted_values.size:
                    permitted_min = min(permitted_min, float(permitted_values.min()))
                    permitted_max = max(permitted_max, float(permitted_values.max()))
                    permitted_sum += float(permitted_values.sum(dtype=np.float64))
                    permitted_elements += int(permitted_values.size)

    if not module_filled.all():
        raise RuntimeError("not all sampled module controls were read")
    control_mean_source_count = control_count.copy()
    control_mean_fallback_mask = np.zeros(lines, dtype=bool)
    if np.any(control_count == 0):
        if settings.parent_grouping != "donor_celltype":
            missing = [context_names[index] for index in np.flatnonzero(control_count == 0)]
            raise ValueError(f"contexts without controls: {missing[:20]}")
        for context_id in np.flatnonzero(control_count == 0):
            _, celltype = context_names[context_id].split(
                COMPOSITE_CONTEXT_SEPARATOR, 1
            )
            peer_ids = [
                peer_id
                for peer_id, name in enumerate(context_names)
                if name.split(COMPOSITE_CONTEXT_SEPARATOR, 1)[1] == celltype
                and control_count[peer_id] > 0
            ]
            pooled_count = int(control_count[peer_ids].sum())
            if pooled_count <= 0:
                raise ValueError(f"cell type without any controls: {celltype!r}")
            control_sum[context_id] = control_sum[peer_ids].sum(axis=0)
            control_mean_source_count[context_id] = pooled_count
            control_mean_fallback_mask[context_id] = True
    control_mean = (
        control_sum / control_mean_source_count[:, None]
    ).astype(np.float32)
    condition_sum = condition_sum.reshape(lines, perts, gene_dim)
    train_count = condition_count.reshape(lines, perts)
    train_mask = train_count >= int(min_condition_cells)
    train_delta = np.zeros((lines, perts, gene_dim), dtype=np.float32)
    for line_id in range(lines):
        valid = train_mask[line_id]
        train_delta[line_id, valid] = (
            condition_sum[line_id, valid] / train_count[line_id, valid, None]
            - control_mean[line_id]
        ).astype(np.float32)
    train_count = np.where(train_mask, train_count, 0).astype(np.int64)
    train_mask[:, control_id] = False
    train_count[:, control_id] = 0
    train_delta[:, control_id] = 0.0

    validation_observed = validation_cell_count.reshape(lines, perts) > 0
    test_observed = test_cell_count.reshape(lines, perts) > 0
    # PBMC's donor-level split can have legal train support from non-held donors
    # for the same cell_type x cytokine.  Legacy consumer masks therefore mark
    # only fully held-out Parent conditions; observed masks preserve the audit.
    validation_condition_mask = validation_observed & ~train_mask
    test_condition_mask = test_observed & ~train_mask
    partial_validation = validation_observed & train_mask
    partial_test = test_observed & train_mask
    if settings.parent_grouping == "donor_celltype":
        cross_priors = _same_celltype_cross_context_priors(
            train_delta, train_count, train_mask, context_names
        )
    else:
        cross_priors = build_cross_line_priors(
            train_delta, train_count, train_mask
        )
    ridge = fit_gene_wise_ridge(
        cross_priors["equal_line"],
        cross_priors["mask"],
        train_delta,
        train_mask,
        ridge_alpha=float(ridge_alpha),
        ridge_lambdas=ridge_lambdas,
        ridge_default_lambda=float(ridge_default_lambda),
        ridge_cv_folds=int(ridge_cv_folds),
        seed=int(seed),
    )
    module_ids, module_diagnostics = build_gene_modules(
        module_values,
        module_context_codes,
        control_mean,
        num_modules=int(num_gene_modules),
        embedding_dim=int(module_embedding_dim),
        seed=int(seed),
    )
    module_sizes = np.bincount(
        module_ids.astype(np.int64), minlength=int(num_gene_modules)
    ).astype(np.int32)
    if module_exclude_contexts:
        module_diagnostics["excluded_control_contexts"] = list(module_exclude_contexts)

    if hash_source_files:
        for manifest, shard in zip(manifests, settings.shards):
            manifest["sha256"] = _sha256_file(shard)
    source_manifest_sha256 = hashlib.sha256(
        json.dumps(manifests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    permitted_scale = {
        "minimum": float(permitted_min),
        "maximum": float(permitted_max),
        "mean": float(permitted_sum / permitted_elements),
        "num_elements": int(permitted_elements),
    }
    metadata_base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema": "parent_residual_ocoot_v1",
        "dataset_kind": settings.dataset_kind,
        "source": {
            "resolved_config": str(resolved_config),
            "resolved_config_sha256": _sha256_file(resolved_config),
            "selected_gene_file": str(settings.selected_gene_file),
            "selected_gene_file_sha256": _sha256_file(settings.selected_gene_file),
            "shards": manifests,
            "source_manifest_sha256": source_manifest_sha256,
            "source_files_fully_hashed": bool(hash_source_files),
            "training_expression_sha256": digest.hexdigest(),
            "training_expression_hash_scope": (
                "normalized train-treated rows plus all controls, identified by shard and row; "
                "validation/test treated rows excluded"
            ),
        },
        "expression": {
            "key_requested": settings.expression_key,
            "gene_dim": gene_dim,
            "normalize_divisor": float(settings.normalize_divisor),
            "artifact_scale": "stored expression / normalize_divisor",
            "gene_order": "selected gene list, proved independently for every shard",
            "permitted_training_scale": permitted_scale,
        },
        "split": {
            "cell_level_semantics": (
                f"intersection of {settings.split_group_key} and held-out perturbations"
            ),
            "parent_context_key": settings.context_key,
            "parent_context_kind": settings.parent_grouping,
            "split_group_key": settings.split_group_key,
            "holdout_groups": list(settings.holdout_groups),
            "validation_perturbations": list(settings.validation_perturbations),
            "test_perturbations": list(settings.test_perturbations),
            "control_label": settings.control_label,
            "split_control": False,
            "cell_counts": split_cell_counts,
            "condition_counts": {
                "train": int(train_mask.sum()),
                "validation_observed": int(validation_observed.sum()),
                "test_observed": int(test_observed.sum()),
                "validation_fully_heldout": int(validation_condition_mask.sum()),
                "test_fully_heldout": int(test_condition_mask.sum()),
                "validation_partial_parent_context": int(partial_validation.sum()),
                "test_partial_parent_context": int(partial_test.sum()),
            },
            "leakage_audit": {
                "heldout_treated_expression_used": 0,
                "cell_level_split_applied_before_aggregation": True,
                "target_context_controls_allowed": True,
                "cross_context_prior_is_leave_target_context_out": True,
                "partial_parent_conditions_use_only_nonheldout_groups": True,
            },
        },
        "pseudobulk": {
            "minimum_condition_cells": int(min_condition_cells),
            "delta_definition": "mean(train treated condition) - mean(same-context controls)",
            "control_mean_fallback": (
                "pooled controls from other donors of the same cell type"
                if settings.parent_grouping == "donor_celltype"
                else "none"
            ),
            "control_mean_fallback_contexts": [
                context_names[index]
                for index in np.flatnonzero(control_mean_fallback_mask)
            ],
            "cross_context_prior_equal": (
                "same-cell-type leave-donor-out mean" if settings.parent_grouping == "donor_celltype"
                else "mean source-context train deltas excluding target context"
            ),
            "ridge_prior": "cross_line_delta_prior_equal_line",
        },
        "ridge": {
            "affine_penalty": float(ridge_alpha),
            "no_intercept_lambda_candidates": ridge["lambda_candidates"].tolist(),
            "no_intercept_default_lambda": float(ridge_default_lambda),
            "cv_folds": int(ridge_cv_folds),
            "support_count_by_context": {
                name: int(ridge["support_count"][index])
                for index, name in enumerate(context_names)
            },
        },
        "gene_modules": module_diagnostics,
        "mappings": {
            "context_to_id": context_lookup,
            "cellline_to_id": context_lookup,
            "donor_celltype_to_id": context_lookup if settings.parent_grouping == "donor_celltype" else {},
            "perturbation_to_id": pert_lookup,
            "control_perturbation_id": int(control_id),
        },
        "build": {
            "seed": int(seed),
            "chunk_size": int(chunk_size),
            "module_max_controls": int(module_max_controls),
            "max_dense_elements": int(max_dense_elements),
            "num_shards": len(settings.shards),
            "safe_npz": True,
            "pickle_required_to_load_output": False,
        },
    }
    metadata_payload_sha256 = _canonical_json_sha256(metadata_base)
    arrays = {
        "schema_version": _fixed_unicode(SCHEMA_VERSION),
        "dataset_schema": _fixed_unicode("parent_residual_ocoot_v1"),
        "parent_context_kind": _fixed_unicode(settings.parent_grouping),
        "metadata_payload_sha256": _fixed_unicode(metadata_payload_sha256),
        "gene_names": _fixed_unicode(genes),
        "gene_ids": np.arange(gene_dim, dtype=np.int32),
        "cellline_names": _fixed_unicode(context_names),
        "cellline_ids": np.arange(lines, dtype=np.int32),
        "perturbation_names": _fixed_unicode(pert_names),
        "perturbation_ids": np.arange(perts, dtype=np.int32),
        "control_perturbation_id": np.asarray(control_id, dtype=np.int32),
        "normalization_divisor": np.asarray(settings.normalize_divisor, dtype=np.float32),
        "control_mean": control_mean.astype(np.float32),
        "control_count": control_count.astype(np.int64),
        "train_condition_delta": train_delta.astype(np.float32),
        "control_mean_source_count": control_mean_source_count.astype(np.int64),
        "control_mean_fallback_mask": control_mean_fallback_mask.astype(bool),
        "train_condition_mask": train_mask.astype(bool),
        "train_condition_count": train_count.astype(np.int64),
        "validation_condition_mask": validation_condition_mask.astype(bool),
        "test_condition_mask": test_condition_mask.astype(bool),
        "validation_observed_condition_mask": validation_observed.astype(bool),
        "test_observed_condition_mask": test_observed.astype(bool),
        "validation_heldout_cell_count": validation_cell_count.reshape(lines, perts),
        "test_heldout_cell_count": test_cell_count.reshape(lines, perts),
        "cross_line_delta_prior_equal_line": cross_priors["equal_line"].astype(np.float32),
        "cross_line_equal_line_mask": cross_priors["mask"].astype(bool),
        "cross_line_equal_line_source_count": cross_priors["source_count"].astype(np.int16),
        "cross_line_delta_prior_count_weighted": cross_priors["count_weighted"].astype(np.float32),
        "cross_line_count_weighted_mask": cross_priors["mask"].astype(bool),
        "cross_line_count_weighted_cell_count": cross_priors["cell_count"].astype(np.int64),
        "cross_line_count_weighted_source_count": cross_priors["source_count"].astype(np.int16),
        "ridge_support_mask": ridge["support_mask"].astype(bool),
        "ridge_support_count": ridge["support_count"].astype(np.int32),
        "ridge_calibration_valid": ridge["valid"].astype(bool),
        "ridge_affine_scale": ridge["affine_scale"].astype(np.float32),
        "ridge_affine_intercept": ridge["affine_intercept"].astype(np.float32),
        "ridge_scale_no_intercept": ridge["no_intercept_scale"].astype(np.float32),
        "ridge_lambda_no_intercept": ridge["no_intercept_lambda"].astype(np.float32),
        "ridge_no_intercept_cv_mse": ridge["no_intercept_cv_mse"].astype(np.float32),
        "ridge_lambda_candidates": ridge["lambda_candidates"].astype(np.float32),
        "module_ids": module_ids.astype(np.int16),
        "module_cluster_sizes": module_sizes,
        "module_silhouette": np.asarray(module_diagnostics["silhouette"], dtype=np.float32),
    }
    _atomic_save_npz(output, arrays)
    artifact_sha256 = _sha256_file(output)
    metadata = {
        **metadata_base,
        "integrity": {
            "metadata_payload_sha256": metadata_payload_sha256,
            "artifact_sha256": artifact_sha256,
            "artifact": str(output),
        },
    }
    _atomic_save_json(metadata_output, metadata)
    print(
        json.dumps(
            {
                "artifact": str(output),
                "metadata": str(metadata_output),
                "artifact_sha256": artifact_sha256,
                "dataset_kind": settings.dataset_kind,
                "shards": len(settings.shards),
                "shape": {"contexts": lines, "perturbations": perts, "genes": gene_dim},
                "split_cell_counts": split_cell_counts,
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--dataset-kind", choices=("auto", "pbmc", "tahoe100m"), default="auto")
    parser.add_argument(
        "--parent-grouping",
        choices=("celltype", "donor", "donor_celltype"),
        default="celltype",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--selected-gene-file", type=Path)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--min-condition-cells", type=int, default=1)
    parser.add_argument("--module-max-controls", type=int, default=8192)
    parser.add_argument("--module-exclude-context", action="append", default=[])
    parser.add_argument("--num-gene-modules", type=int, default=128)
    parser.add_argument("--module-embedding-dim", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--ridge-lambdas", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--ridge-default-lambda", type=float, default=0.01)
    parser.add_argument("--ridge-cv-folds", type=int, default=5)
    parser.add_argument("--max-dense-elements", type=int, default=250_000_000)
    parser.add_argument("--hash-source-files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_ocoot_artifact(
        args.resolved_config,
        args.output,
        metadata_output=args.metadata_output,
        dataset_kind=args.dataset_kind,
        parent_grouping=args.parent_grouping,
        dataset_path_override=args.dataset_path,
        selected_gene_override=args.selected_gene_file,
        expected_shards=args.expected_shards,
        chunk_size=args.chunk_size,
        min_condition_cells=args.min_condition_cells,
        module_max_controls=args.module_max_controls,
        module_exclude_contexts=args.module_exclude_context,
        num_gene_modules=args.num_gene_modules,
        module_embedding_dim=args.module_embedding_dim,
        ridge_alpha=args.ridge_alpha,
        ridge_lambdas=args.ridge_lambdas,
        ridge_default_lambda=args.ridge_default_lambda,
        ridge_cv_folds=args.ridge_cv_folds,
        max_dense_elements=args.max_dense_elements,
        hash_source_files=args.hash_source_files,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

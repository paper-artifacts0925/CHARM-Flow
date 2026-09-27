from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd


SCHEMA_VERSION = "parent_residual_replogle_v1"
TRAIN = np.uint8(0)
VALIDATION = np.uint8(1)
TEST = np.uint8(2)
CONTROL = np.uint8(3)


@dataclass(frozen=True)
class DatasetSettings:
    """Resolved inputs that affect artifact semantics."""

    h5ad: Path
    selected_gene_file: Path
    expression_key: str
    perturbation_key: str
    cell_line_key: str
    control_label: str
    normalize_divisor: float
    holdout_cell_lines: tuple[str, ...]
    validation_perturbations: tuple[str, ...]
    test_perturbations: tuple[str, ...]
    split_control: bool
    expected_gene_dim: int | None


class _PrimitiveOnlyUnpickler(pickle.Unpickler):
    """Read the legacy gene-name list without permitting global imports."""

    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - safety hook
        raise pickle.UnpicklingError(
            f"global object loading is forbidden ({module}.{name})"
        )


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _normalization_divisor(value: Any) -> float:
    if value is None or value is False or value == 0:
        return 1.0
    if isinstance(value, bool):
        raise ValueError("normalize_counts=true is ambiguous; use a positive number")
    divisor = float(value)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("data.normalize_counts must be finite and positive")
    return divisor


def load_dataset_settings(
    resolved_config: Path,
    h5ad_override: Path | None = None,
    selected_gene_override: Path | None = None,
) -> DatasetSettings:
    """Read only the resolved data fields used by the builder."""

    with resolved_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("resolved config must contain a mapping")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("resolved config has no resolved top-level data mapping")

    h5ad_value = h5ad_override or data.get("dataset_path")
    gene_value = selected_gene_override or data.get("selected_gene_file")
    if h5ad_value is None or gene_value is None:
        raise ValueError("resolved config must define data.dataset_path and selected_gene_file")

    holdout = data.get("holdout_pert") or {}
    if not isinstance(holdout, Mapping):
        raise ValueError("data.holdout_pert must be a mapping")
    validation = tuple(str(x) for x in (holdout.get("validation") or ()))
    test = tuple(str(x) for x in (holdout.get("test") or ()))
    overlap = sorted(set(validation).intersection(test))
    if overlap:
        raise ValueError(f"validation/test perturbation lists overlap: {overlap[:10]}")

    split_control = bool(data.get("split_control", False))
    if split_control:
        raise ValueError(
            "this builder currently requires split_control=false so control support "
            "exactly matches the active Replogle protocol"
        )
    if bool(data.get("sort_gene_names", False)):
        raise ValueError(
            "data.sort_gene_names=true would break the original X_hvg column order"
        )

    expected_dim = data.get("embed_shape")
    return DatasetSettings(
        h5ad=Path(h5ad_value).expanduser().resolve(),
        selected_gene_file=Path(gene_value).expanduser().resolve(),
        expression_key=str(data.get("embed_key", "X_hvg")),
        perturbation_key=str(data.get("pert_col", "gene")),
        cell_line_key=str(data.get("cell_line_key", "cell_line")),
        control_label=str(data.get("control_pert", "non-targeting")),
        normalize_divisor=_normalization_divisor(data.get("normalize_counts")),
        holdout_cell_lines=tuple(str(x) for x in (data.get("holdout_celltype") or ())),
        validation_perturbations=validation,
        test_perturbations=test,
        split_control=split_control,
        expected_gene_dim=int(expected_dim) if expected_dim is not None else None,
    )


def load_selected_genes(path: Path) -> list[str]:
    """Safely read the legacy primitive-list pickle and validate its contents."""

    raw = path.read_bytes()
    value = _PrimitiveOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(value, (list, tuple)):
        raise ValueError("selected gene artifact must be a list or tuple")
    genes = [str(item) for item in value]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("selected gene names must be non-empty and unique")
    if any(not gene for gene in genes):
        raise ValueError("selected gene names must not be empty")
    return genes


def _read_string_array(dataset: h5py.Dataset) -> list[str]:
    return [_decode(value) for value in dataset[:]]


def read_categorical(group_or_dataset: h5py.Group | h5py.Dataset) -> tuple[list[str], np.ndarray]:
    """Read an AnnData categorical or a plain string dataset."""

    if isinstance(group_or_dataset, h5py.Group):
        if "categories" not in group_or_dataset or "codes" not in group_or_dataset:
            raise ValueError("categorical group must contain categories and codes")
        names = _read_string_array(group_or_dataset["categories"])
        codes = np.asarray(group_or_dataset["codes"][:], dtype=np.int64)
    else:
        values = np.asarray(group_or_dataset[:])
        decoded = np.asarray([_decode(value) for value in values], dtype=str)
        names = sorted(set(decoded.tolist()))
        mapping = {name: index for index, name in enumerate(names)}
        codes = np.asarray([mapping[value] for value in decoded], dtype=np.int64)
    if len(names) != len(set(names)):
        raise ValueError("categorical names are not unique")
    if len(codes) and (codes.min() < 0 or codes.max() >= len(names)):
        raise ValueError("categorical codes fall outside category range")
    return names, codes


def _stable_remap(names: Sequence[str], codes: np.ndarray) -> tuple[list[str], np.ndarray]:
    ordered = sorted(str(name) for name in names)
    new_id = {name: index for index, name in enumerate(ordered)}
    old_to_new = np.asarray([new_id[str(name)] for name in names], dtype=np.int64)
    return ordered, old_to_new[codes]


def verify_gene_order(handle: h5py.File, selected_genes: Sequence[str], gene_dim: int) -> None:
    """Prove X_hvg follows var order restricted by highly_variable."""

    if len(selected_genes) != gene_dim:
        raise ValueError(
            f"selected gene count {len(selected_genes)} != expression width {gene_dim}"
        )
    var = handle.get("var")
    if not isinstance(var, h5py.Group) or "highly_variable" not in var:
        raise ValueError("H5AD must contain var/highly_variable for order verification")
    index_name = _decode(var.attrs.get("_index", "_index"))
    if index_name not in var:
        raise ValueError(f"H5AD var index dataset {index_name!r} is absent")
    all_genes = _read_string_array(var[index_name])
    highly_variable = np.asarray(var["highly_variable"][:], dtype=bool)
    if len(all_genes) != len(highly_variable):
        raise ValueError("var index and highly_variable mask lengths differ")
    hvg_genes = [gene for gene, keep in zip(all_genes, highly_variable) if keep]
    if hvg_genes != list(selected_genes):
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(hvg_genes, selected_genes))
                if left != right
            ),
            min(len(hvg_genes), len(selected_genes)),
        )
        raise ValueError(
            "selected gene order does not match var/highly_variable order; "
            f"first mismatch at column {mismatch}"
        )


def assign_cell_splits(
    cell_line_names: Sequence[str],
    cell_line_codes: np.ndarray,
    perturbation_names: Sequence[str],
    perturbation_codes: np.ndarray,
    control_label: str,
    holdout_cell_lines: Iterable[str],
    validation_perturbations: Iterable[str],
    test_perturbations: Iterable[str],
) -> np.ndarray:
    """Return train/validation/test/control labels with no overlapping cells."""

    if control_label not in perturbation_names:
        raise ValueError(f"control label {control_label!r} is absent")
    if len(cell_line_codes) != len(perturbation_codes):
        raise ValueError("cell-line and perturbation code lengths differ")
    line_values = np.asarray(cell_line_names, dtype=str)[cell_line_codes]
    perturbation_values = np.asarray(perturbation_names, dtype=str)[perturbation_codes]
    control_mask = perturbation_values == control_label
    held_line_mask = np.isin(line_values, list(holdout_cell_lines))
    validation_mask = (
        ~control_mask
        & held_line_mask
        & np.isin(perturbation_values, list(validation_perturbations))
    )
    test_mask = (
        ~control_mask
        & held_line_mask
        & np.isin(perturbation_values, list(test_perturbations))
    )
    if np.any(validation_mask & test_mask):
        raise RuntimeError("validation and test cell masks overlap")
    split = np.full(len(cell_line_codes), TRAIN, dtype=np.uint8)
    split[validation_mask] = VALIDATION
    split[test_mask] = TEST
    split[control_mask] = CONTROL
    return split


def _choose_module_controls(
    control_indices: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    if maximum < 2:
        raise ValueError("module_max_controls must be at least 2")
    if len(control_indices) <= maximum:
        return np.asarray(control_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(control_indices, size=int(maximum), replace=False).astype(np.int64)
    )


def _normalize_module_exclude_cell_lines(
    values: Sequence[str] | str,
) -> tuple[str, ...]:
    """Canonicalize the optional module-only control-line exclusion."""

    if isinstance(values, str):
        values = values.split(",")
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError("module-excluded control cell lines must be unique")
    return normalized


def aggregate_training_expression(
    matrix: h5py.Dataset,
    cell_line_codes: np.ndarray,
    perturbation_codes: np.ndarray,
    split: np.ndarray,
    normalize_divisor: float,
    module_control_indices: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    """Stream X_hvg once and aggregate only controls plus train treated cells."""

    if matrix.ndim != 2 or matrix.shape[0] != len(split):
        raise ValueError("expression matrix shape is inconsistent with obs")
    num_lines = int(cell_line_codes.max()) + 1
    num_perturbations = int(perturbation_codes.max()) + 1
    gene_dim = int(matrix.shape[1])
    control_sum = np.zeros((num_lines, gene_dim), dtype=np.float64)
    control_count = np.zeros(num_lines, dtype=np.int64)
    condition_sum = np.zeros(
        (num_lines * num_perturbations, gene_dim), dtype=np.float64
    )
    condition_count = np.zeros(num_lines * num_perturbations, dtype=np.int64)
    module_values = np.empty(
        (len(module_control_indices), gene_dim), dtype=np.float32
    )
    digest = hashlib.sha256()
    permitted_min = np.inf
    permitted_max = -np.inf
    permitted_sum = 0.0
    permitted_elements = 0

    for start in range(0, matrix.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), matrix.shape[0])
        raw_values = np.asarray(matrix[start:stop], dtype=np.float32)
        if not np.isfinite(raw_values).all():
            raise ValueError(f"non-finite expression in rows [{start}, {stop})")
        values = raw_values / np.float32(normalize_divisor)
        local_split = split[start:stop]
        local_lines = cell_line_codes[start:stop]
        local_perts = perturbation_codes[start:stop]

        left = int(np.searchsorted(module_control_indices, start, side="left"))
        right = int(np.searchsorted(module_control_indices, stop, side="left"))
        if right > left:
            relative = module_control_indices[left:right] - start
            module_values[left:right] = values[relative]

        control_mask = local_split == CONTROL
        for line_id in np.unique(local_lines[control_mask]):
            selected = values[control_mask & (local_lines == line_id)]
            control_sum[line_id] += selected.sum(axis=0, dtype=np.float64)
            control_count[line_id] += len(selected)

        train_mask = local_split == TRAIN
        if train_mask.any():
            train_values = values[train_mask]
            flat_ids = (
                local_lines[train_mask] * num_perturbations + local_perts[train_mask]
            )
            order = np.argsort(flat_ids, kind="stable")
            sorted_ids = flat_ids[order]
            group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
            unique_ids = sorted_ids[group_starts]
            grouped_sum = np.add.reduceat(
                train_values[order], group_starts, axis=0, dtype=np.float64
            )
            grouped_count = np.diff(np.r_[group_starts, len(sorted_ids)])
            condition_sum[unique_ids] += grouped_sum
            condition_count[unique_ids] += grouped_count

        permitted_mask = control_mask | train_mask
        permitted_values = values[permitted_mask]
        permitted_indices = np.flatnonzero(permitted_mask).astype(np.int64) + start
        digest.update(permitted_indices.astype("<i8", copy=False).tobytes())
        digest.update(
            np.ascontiguousarray(permitted_values, dtype="<f4").tobytes()
        )
        if permitted_values.size:
            permitted_min = min(permitted_min, float(permitted_values.min()))
            permitted_max = max(permitted_max, float(permitted_values.max()))
            permitted_sum += float(permitted_values.sum(dtype=np.float64))
            permitted_elements += int(permitted_values.size)

    if (control_count == 0).any():
        missing = np.flatnonzero(control_count == 0).tolist()
        raise ValueError(f"cell lines without training controls: {missing}")
    control_mean = (control_sum / control_count[:, None]).astype(np.float32)
    condition_sum = condition_sum.reshape(num_lines, num_perturbations, gene_dim)
    condition_count = condition_count.reshape(num_lines, num_perturbations)
    condition_mean = np.zeros_like(condition_sum, dtype=np.float32)
    observed = condition_count > 0
    condition_mean[observed] = (
        condition_sum[observed] / condition_count[observed, None]
    ).astype(np.float32)
    train_delta = np.zeros_like(condition_mean, dtype=np.float32)
    train_delta[observed] = (
        condition_mean[observed]
        - np.broadcast_to(control_mean[:, None, :], condition_mean.shape)[observed]
    )
    return {
        "control_mean": control_mean,
        "control_count": control_count,
        "train_delta": train_delta,
        "train_count": condition_count,
        "train_mask": observed,
        "module_control_values": module_values,
        "training_expression_sha256": digest.hexdigest(),
        "permitted_scale": {
            "minimum": float(permitted_min),
            "maximum": float(permitted_max),
            "mean": float(permitted_sum / permitted_elements),
            "num_elements": int(permitted_elements),
        },
    }


def build_cross_line_priors(
    train_delta: np.ndarray,
    train_count: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Equal-line and cell-count-weighted priors excluding the target line."""

    weights = np.where(train_mask, train_count, 0).astype(np.float64)
    weighted_delta = train_delta.astype(np.float64) * weights[..., None]
    all_weighted = weighted_delta.sum(axis=0)
    all_counts = weights.sum(axis=0)
    masked_delta = train_delta.astype(np.float64) * train_mask[..., None]
    all_equal_line = masked_delta.sum(axis=0)
    all_sources = train_mask.sum(axis=0, dtype=np.int64)
    num_lines, num_perts, gene_dim = train_delta.shape
    equal_line_prior = np.zeros(
        (num_lines, num_perts, gene_dim), dtype=np.float32
    )
    count_weighted_prior = np.zeros_like(equal_line_prior)
    cell_count = np.zeros((num_lines, num_perts), dtype=np.int64)
    source_count = np.zeros((num_lines, num_perts), dtype=np.int16)
    prior_mask = np.zeros((num_lines, num_perts), dtype=bool)
    for target_line in range(num_lines):
        remaining_count = all_counts - weights[target_line]
        remaining_sources = all_sources - train_mask[target_line].astype(np.int64)
        valid = remaining_sources > 0
        weighted_numerator = all_weighted - weighted_delta[target_line]
        count_weighted_prior[target_line, valid] = (
            weighted_numerator[valid] / remaining_count[valid, None]
        ).astype(np.float32)
        equal_numerator = all_equal_line - masked_delta[target_line]
        equal_line_prior[target_line, valid] = (
            equal_numerator[valid] / remaining_sources[valid, None]
        ).astype(np.float32)
        cell_count[target_line] = remaining_count.astype(np.int64)
        source_count[target_line] = remaining_sources.astype(np.int16)
        prior_mask[target_line] = valid
    return {
        "equal_line": equal_line_prior,
        "count_weighted": count_weighted_prior,
        "mask": prior_mask,
        "cell_count": cell_count,
        "source_count": source_count,
    }


def _parse_ridge_lambdas(values: Sequence[float] | str) -> np.ndarray:
    if isinstance(values, str):
        parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    else:
        parsed = [float(value) for value in values]
    lambdas = np.asarray(sorted(set(parsed)), dtype=np.float64)
    if not len(lambdas) or not np.isfinite(lambdas).all() or (lambdas < 0).any():
        raise ValueError("ridge lambdas must be a non-empty list of finite non-negative values")
    return lambdas


def _no_intercept_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    default_lambda: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gene y=scale*x ridge with deterministic condition-level CV."""

    num_samples, gene_dim = x.shape
    total_x2 = np.square(x, dtype=np.float64).sum(axis=0)
    total_xy = (x * y).sum(axis=0, dtype=np.float64)
    if num_samples < 2 or folds < 2:
        selected = np.full(gene_dim, float(default_lambda), dtype=np.float64)
        scale = total_xy / np.maximum(total_x2 + selected, 1e-12)
        return np.clip(scale, 0.0, 2.0), selected, np.full(gene_dim, np.nan)

    n_splits = min(int(folds), num_samples)
    errors = np.zeros((len(lambdas), gene_dim), dtype=np.float64)
    validation_counts = 0
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for _, validation_indices in splitter.split(np.arange(num_samples)):
        x_val = x[validation_indices]
        y_val = y[validation_indices]
        val_x2 = np.square(x_val, dtype=np.float64).sum(axis=0)
        val_xy = (x_val * y_val).sum(axis=0, dtype=np.float64)
        train_x2 = total_x2 - val_x2
        train_xy = total_xy - val_xy
        for lambda_index, ridge_lambda in enumerate(lambdas):
            scale = train_xy / np.maximum(train_x2 + ridge_lambda, 1e-12)
            scale = np.clip(scale, 0.0, 2.0)
            residual = y_val - x_val * scale[None, :]
            errors[lambda_index] += np.square(residual, dtype=np.float64).sum(axis=0)
        validation_counts += len(validation_indices)
    errors /= max(validation_counts, 1)
    best_index = np.argmin(errors, axis=0)
    selected = lambdas[best_index]
    scale = total_xy / np.maximum(total_x2 + selected, 1e-12)
    scale = np.clip(scale, 0.0, 2.0)
    best_error = np.take_along_axis(errors, best_index[None, :], axis=0)[0]
    return scale, selected, best_error


def fit_gene_wise_ridge(
    cross_line_prior: np.ndarray,
    cross_line_mask: np.ndarray,
    train_delta: np.ndarray,
    train_mask: np.ndarray,
    ridge_alpha: float,
    ridge_lambdas: Sequence[float] | str,
    ridge_default_lambda: float,
    ridge_cv_folds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit affine and no-intercept calibrations using target-line train support only."""

    if ridge_alpha < 0 or ridge_default_lambda < 0:
        raise ValueError("ridge penalties must be non-negative")
    lambdas = _parse_ridge_lambdas(ridge_lambdas)
    num_lines, num_perts, gene_dim = train_delta.shape
    support_mask = train_mask & cross_line_mask
    affine_scale = np.ones((num_lines, gene_dim), dtype=np.float32)
    affine_intercept = np.zeros((num_lines, gene_dim), dtype=np.float32)
    no_intercept_scale = np.ones((num_lines, gene_dim), dtype=np.float32)
    selected_lambda = np.full(
        (num_lines, gene_dim), float(ridge_default_lambda), dtype=np.float32
    )
    cv_mse = np.full((num_lines, gene_dim), np.nan, dtype=np.float32)
    valid = np.zeros(num_lines, dtype=bool)
    support_count = support_mask.sum(axis=1, dtype=np.int32)

    for line_id in range(num_lines):
        selected = support_mask[line_id]
        if not selected.any():
            continue
        x = cross_line_prior[line_id, selected].astype(np.float64)
        y = train_delta[line_id, selected].astype(np.float64)
        if len(x) >= 2:
            x_mean = x.mean(axis=0)
            y_mean = y.mean(axis=0)
            centered_x = x - x_mean
            centered_y = y - y_mean
            slope = (centered_x * centered_y).sum(axis=0) / np.maximum(
                np.square(centered_x).sum(axis=0) + float(ridge_alpha), 1e-12
            )
            intercept = y_mean - slope * x_mean
            affine_scale[line_id] = slope.astype(np.float32)
            affine_intercept[line_id] = intercept.astype(np.float32)
            valid[line_id] = True
        scale, chosen, error = _no_intercept_ridge_cv(
            x,
            y,
            lambdas=lambdas,
            default_lambda=float(ridge_default_lambda),
            folds=int(ridge_cv_folds),
            seed=int(seed) + line_id,
        )
        no_intercept_scale[line_id] = scale.astype(np.float32)
        selected_lambda[line_id] = chosen.astype(np.float32)
        cv_mse[line_id] = error.astype(np.float32)

    return {
        "support_mask": support_mask,
        "support_count": support_count,
        "valid": valid,
        "affine_scale": affine_scale,
        "affine_intercept": affine_intercept,
        "no_intercept_scale": no_intercept_scale,
        "no_intercept_lambda": selected_lambda,
        "no_intercept_cv_mse": cv_mse,
        "lambda_candidates": lambdas.astype(np.float32),
    }


def build_gene_modules(
    control_values: np.ndarray,
    control_line_codes: np.ndarray,
    control_mean: np.ndarray,
    num_modules: int,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster genes by line-centered control coexpression (ordinary KMeans)."""

    num_controls, gene_dim = control_values.shape
    if not 1 < num_modules <= gene_dim:
        raise ValueError("num_gene_modules must satisfy 1 < K <= gene_dim")
    if len(control_line_codes) != num_controls:
        raise ValueError("module control values and line codes disagree")
    residual = control_values.astype(np.float64) - control_mean[control_line_codes]
    gene_std = residual.std(axis=0, ddof=0)
    informative = gene_std > 1e-8
    standardized = np.zeros_like(residual, dtype=np.float32)
    standardized[:, informative] = (
        residual[:, informative] / gene_std[informative]
    ).astype(np.float32)
    np.clip(standardized, -10.0, 10.0, out=standardized)
    components = min(int(embedding_dim), num_controls - 1, gene_dim - 1)
    if components < 2:
        raise ValueError("at least three sampled controls are required for gene modules")
    _, singular_values, right_vectors = randomized_svd(
        standardized,
        n_components=components,
        random_state=int(seed),
        n_iter=5,
    )
    gene_embedding = right_vectors.T * singular_values[None, :]
    norms = np.linalg.norm(gene_embedding, axis=1, keepdims=True)
    gene_embedding = gene_embedding / np.maximum(norms, 1e-12)
    fitted = KMeans(
        n_clusters=int(num_modules),
        n_init=20,
        max_iter=500,
        random_state=int(seed),
        algorithm="lloyd",
    ).fit(gene_embedding)
    raw_labels = fitted.labels_.astype(np.int64)

    # Canonicalize arbitrary KMeans IDs by the first gene in each module.
    order = sorted(
        range(int(num_modules)), key=lambda label: int(np.flatnonzero(raw_labels == label)[0])
    )
    remap = np.empty(int(num_modules), dtype=np.int64)
    remap[np.asarray(order, dtype=np.int64)] = np.arange(int(num_modules))
    labels = remap[raw_labels]
    sizes = np.bincount(labels, minlength=int(num_modules)).astype(np.int32)
    silhouette = float(silhouette_score(gene_embedding, labels, metric="cosine"))
    diagnostics = {
        "algorithm": "KMeans (unconstrained, non-exact-capacity)",
        "exact_capacity": False,
        "seed": int(seed),
        "num_modules": int(num_modules),
        "num_sampled_training_controls": int(num_controls),
        "embedding": "line-centered standardized control expression randomized-SVD gene loadings",
        "embedding_dim": int(components),
        "silhouette_metric": "cosine",
        "silhouette": silhouette,
        "cluster_sizes": sizes.tolist(),
        "min_cluster_size": int(sizes.min()),
        "median_cluster_size": float(np.median(sizes)),
        "max_cluster_size": int(sizes.max()),
        "num_zero_variance_genes": int((~informative).sum()),
        "inertia": float(fitted.inertia_),
    }
    return labels.astype(np.int16), diagnostics


def _fixed_unicode(values: Sequence[str] | str) -> np.ndarray:
    if isinstance(values, str):
        values = [values]
    maximum = max(1, *(len(str(value)) for value in values))
    return np.asarray([str(value) for value in values], dtype=f"<U{maximum}")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(temporary, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(f"unsafe object dtype in NPZ key {key!r}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_artifact(
    resolved_config: Path,
    output: Path,
    metadata_output: Path | None = None,
    h5ad_override: Path | None = None,
    selected_gene_override: Path | None = None,
    chunk_size: int = 4096,
    min_condition_cells: int = 1,
    module_max_controls: int = 8192,
    module_exclude_cell_lines: Sequence[str] | str = (),
    num_gene_modules: int = 128,
    module_embedding_dim: int = 64,
    ridge_alpha: float = 0.01,
    ridge_lambdas: Sequence[float] | str = (0.0001, 0.001, 0.01, 0.1, 1.0),
    ridge_default_lambda: float = 0.01,
    ridge_cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the NPZ and JSON sidecar, returning the final metadata."""

    resolved_config = Path(resolved_config).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    metadata_output = (
        Path(metadata_output).expanduser().resolve()
        if metadata_output is not None
        else output.with_suffix(".metadata.json")
    )
    if chunk_size < 1 or min_condition_cells < 1:
        raise ValueError("chunk_size and min_condition_cells must be positive")
    module_exclude_cell_lines = _normalize_module_exclude_cell_lines(
        module_exclude_cell_lines
    )
    settings = load_dataset_settings(
        resolved_config,
        h5ad_override=h5ad_override,
        selected_gene_override=selected_gene_override,
    )
    for required in (settings.h5ad, settings.selected_gene_file, resolved_config):
        if not required.is_file():
            raise FileNotFoundError(required)
    genes = load_selected_genes(settings.selected_gene_file)

    with h5py.File(settings.h5ad, "r") as handle:
        matrix_path = f"obsm/{settings.expression_key}"
        if matrix_path not in handle:
            raise ValueError(f"H5AD has no {matrix_path}")
        matrix = handle[matrix_path]
        if matrix.ndim != 2:
            raise ValueError("expression matrix must be two-dimensional")
        gene_dim = int(matrix.shape[1])
        if settings.expected_gene_dim is not None and settings.expected_gene_dim != gene_dim:
            raise ValueError(
                f"config embed_shape={settings.expected_gene_dim} != X_hvg width={gene_dim}"
            )
        verify_gene_order(handle, genes, gene_dim)
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group):
            raise ValueError("H5AD has no obs group")
        raw_line_names, raw_line_codes = read_categorical(obs[settings.cell_line_key])
        raw_pert_names, raw_pert_codes = read_categorical(obs[settings.perturbation_key])
        if len(raw_line_codes) != matrix.shape[0] or len(raw_pert_codes) != matrix.shape[0]:
            raise ValueError("obs lengths do not match expression rows")
        cell_line_names, cell_line_codes = _stable_remap(raw_line_names, raw_line_codes)
        perturbation_names, perturbation_codes = _stable_remap(
            raw_pert_names, raw_pert_codes
        )
        split = assign_cell_splits(
            cell_line_names,
            cell_line_codes,
            perturbation_names,
            perturbation_codes,
            settings.control_label,
            settings.holdout_cell_lines,
            settings.validation_perturbations,
            settings.test_perturbations,
        )
        unknown_module_lines = sorted(
            set(module_exclude_cell_lines).difference(cell_line_names)
        )
        if unknown_module_lines:
            raise ValueError(
                "module-excluded control cell lines are absent: "
                f"{unknown_module_lines}"
            )
        module_control_mask = split == CONTROL
        if module_exclude_cell_lines:
            excluded_line_ids = [
                cell_line_names.index(name) for name in module_exclude_cell_lines
            ]
            module_control_mask &= ~np.isin(cell_line_codes, excluded_line_ids)
        control_indices = np.flatnonzero(module_control_mask).astype(np.int64)
        if len(control_indices) < 3:
            raise ValueError(
                "module-only cell-line exclusion leaves fewer than three controls"
            )
        module_control_indices = _choose_module_controls(
            control_indices, int(module_max_controls), int(seed)
        )
        module_control_counts_by_line = {
            name: int(np.sum(cell_line_codes[module_control_indices] == line_id))
            for line_id, name in enumerate(cell_line_names)
            if name not in module_exclude_cell_lines
        }
        aggregates = aggregate_training_expression(
            matrix,
            cell_line_codes,
            perturbation_codes,
            split,
            settings.normalize_divisor,
            module_control_indices,
            int(chunk_size),
        )

    train_count = aggregates["train_count"]
    train_mask = aggregates["train_mask"] & (train_count >= int(min_condition_cells))
    train_delta = aggregates["train_delta"]
    train_delta[~train_mask] = 0.0
    train_count = np.where(train_mask, train_count, 0).astype(np.int64)
    control_id = perturbation_names.index(settings.control_label)
    train_mask[:, control_id] = False
    train_delta[:, control_id] = 0.0
    train_count[:, control_id] = 0

    line_lookup = {name: index for index, name in enumerate(cell_line_names)}
    pert_lookup = {name: index for index, name in enumerate(perturbation_names)}
    validation_condition_mask = np.zeros_like(train_mask)
    test_condition_mask = np.zeros_like(train_mask)
    for line_name in settings.holdout_cell_lines:
        if line_name not in line_lookup:
            continue
        line_id = line_lookup[line_name]
        for perturbation in settings.validation_perturbations:
            if perturbation in pert_lookup:
                validation_condition_mask[line_id, pert_lookup[perturbation]] = True
        for perturbation in settings.test_perturbations:
            if perturbation in pert_lookup:
                test_condition_mask[line_id, pert_lookup[perturbation]] = True
    if np.any(train_mask & (validation_condition_mask | test_condition_mask)):
        raise RuntimeError("held-out target-line condition entered training pseudobulk")

    cross_priors = build_cross_line_priors(
        train_delta, train_count, train_mask
    )
    # The published strong pseudobulk baseline gives each available source
    # cell line one vote, irrespective of its cell count.  Ridge calibration
    # therefore uses the equal-line prior; the count-weighted version remains
    # in the artifact as an explicit ablation.
    cross_prior = cross_priors["equal_line"]
    cross_mask = cross_priors["mask"]
    ridge = fit_gene_wise_ridge(
        cross_prior,
        cross_mask,
        train_delta,
        train_mask,
        ridge_alpha=float(ridge_alpha),
        ridge_lambdas=ridge_lambdas,
        ridge_default_lambda=float(ridge_default_lambda),
        ridge_cv_folds=int(ridge_cv_folds),
        seed=int(seed),
    )
    module_line_codes = cell_line_codes[module_control_indices]
    module_ids, module_diagnostics = build_gene_modules(
        aggregates["module_control_values"],
        module_line_codes,
        aggregates["control_mean"],
        num_modules=int(num_gene_modules),
        embedding_dim=int(module_embedding_dim),
        seed=int(seed),
    )
    if module_exclude_cell_lines:
        module_diagnostics.update(
            {
                "excluded_control_cell_lines": list(module_exclude_cell_lines),
                "eligible_training_controls": int(len(control_indices)),
                "sampled_control_counts_by_cell_line": module_control_counts_by_line,
            }
        )
    module_sizes = np.bincount(
        module_ids.astype(np.int64), minlength=int(num_gene_modules)
    ).astype(np.int32)

    split_cell_counts = {
        "train_treated": int((split == TRAIN).sum()),
        "validation_treated": int((split == VALIDATION).sum()),
        "test_treated": int((split == TEST).sum()),
        "training_controls": int((split == CONTROL).sum()),
    }
    heldout_train_overlap = int(
        np.sum(train_mask & (validation_condition_mask | test_condition_mask))
    )
    metadata_base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "resolved_config": str(resolved_config),
            "resolved_config_sha256": _sha256_file(resolved_config),
            "h5ad": str(settings.h5ad),
            "h5ad_size_bytes": int(settings.h5ad.stat().st_size),
            "selected_gene_file": str(settings.selected_gene_file),
            "selected_gene_file_sha256": _sha256_file(settings.selected_gene_file),
            "training_expression_sha256": aggregates["training_expression_sha256"],
            "training_expression_hash_scope": (
                "normalized train-treated rows plus all controls; target validation/test "
                "treated expression explicitly excluded"
            ),
        },
        "expression": {
            "key": settings.expression_key,
            "stored_dtype": "float32",
            "gene_dim": len(genes),
            "gene_order": "selected gene list, exactly matching var/highly_variable order",
            "input_scale": "existing X_hvg",
            "normalize_divisor": float(settings.normalize_divisor),
            "artifact_scale": "X_hvg / normalize_divisor",
            "permitted_training_scale": aggregates["permitted_scale"],
        },
        "split": {
            "semantics": (
                "validation/test are intersections of configured holdout cell lines "
                "and perturbation lists; all other treated conditions are train"
            ),
            "control_label": settings.control_label,
            "split_control": False,
            "holdout_cell_lines": list(settings.holdout_cell_lines),
            "validation_perturbations": list(settings.validation_perturbations),
            "test_perturbations": list(settings.test_perturbations),
            "cell_counts": split_cell_counts,
            "condition_counts": {
                "train": int(train_mask.sum()),
                "configured_validation_present": int(validation_condition_mask.sum()),
                "configured_test_present": int(test_condition_mask.sum()),
            },
            "leakage_audit": {
                "heldout_target_conditions_in_train": heldout_train_overlap,
                "heldout_treated_expression_used": 0,
                "target_line_controls_allowed": True,
                "cross_line_prior_is_leave_target_line_out": True,
                "ridge_support_is_target_line_train_treated_only": True,
            },
        },
        "pseudobulk": {
            "minimum_condition_cells": int(min_condition_cells),
            "delta_definition": "mean(train treated condition) - mean(same-line controls)",
            "cross_line_prior_equal_line": (
                "arithmetic mean of available source-line train deltas, excluding target line"
            ),
            "cross_line_prior_count_weighted": (
                "source-cell-count-weighted train deltas, excluding target line (ablation)"
            ),
            "ridge_prior": "cross_line_delta_prior_equal_line",
        },
        "ridge": {
            "affine_penalty": float(ridge_alpha),
            "affine_definition": "y = scale*x + intercept; unpenalized intercept",
            "no_intercept_definition": "y = scale*x; scale clipped to [0,2]",
            "no_intercept_lambda_candidates": ridge["lambda_candidates"].tolist(),
            "no_intercept_default_lambda": float(ridge_default_lambda),
            "cv_folds": int(ridge_cv_folds),
            "support_count_by_cell_line": {
                name: int(ridge["support_count"][line_id])
                for line_id, name in enumerate(cell_line_names)
            },
            "support_scope": "target-line train treated conditions with cross-line prior",
        },
        "gene_modules": module_diagnostics,
        "mappings": {
            "cellline_to_id": line_lookup,
            "perturbation_to_id": pert_lookup,
            "control_perturbation_id": int(control_id),
        },
        "build": {
            "seed": int(seed),
            "chunk_size": int(chunk_size),
            "module_max_controls": int(module_max_controls),
            "safe_npz": True,
            "pickle_required_to_load_output": False,
        },
    }
    if module_exclude_cell_lines:
        metadata_base["build"]["module_excluded_control_cell_lines"] = list(
            module_exclude_cell_lines
        )
    metadata_payload_sha256 = _canonical_json_sha256(metadata_base)

    arrays = {
        "schema_version": _fixed_unicode(SCHEMA_VERSION),
        "metadata_payload_sha256": _fixed_unicode(metadata_payload_sha256),
        "gene_names": _fixed_unicode(genes),
        "gene_ids": np.arange(len(genes), dtype=np.int32),
        "cellline_names": _fixed_unicode(cell_line_names),
        "cellline_ids": np.arange(len(cell_line_names), dtype=np.int16),
        "perturbation_names": _fixed_unicode(perturbation_names),
        "perturbation_ids": np.arange(len(perturbation_names), dtype=np.int32),
        "control_perturbation_id": np.asarray(control_id, dtype=np.int32),
        "normalization_divisor": np.asarray(settings.normalize_divisor, dtype=np.float32),
        "control_mean": aggregates["control_mean"].astype(np.float32),
        "control_count": aggregates["control_count"].astype(np.int64),
        "train_condition_delta": train_delta.astype(np.float32),
        "train_condition_mask": train_mask.astype(bool),
        "train_condition_count": train_count.astype(np.int64),
        "validation_condition_mask": validation_condition_mask.astype(bool),
        "test_condition_mask": test_condition_mask.astype(bool),
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
                "shape": {
                    "cell_lines": len(cell_line_names),
                    "perturbations": len(perturbation_names),
                    "genes": len(genes),
                    "gene_modules": int(num_gene_modules),
                },
                "split_cell_counts": split_cell_counts,
                "module_silhouette": module_diagnostics["silhouette"],
                "module_cluster_size_range": [
                    int(module_sizes.min()),
                    int(module_sizes.max()),
                ],
                "module_excluded_control_cell_lines": list(module_exclude_cell_lines),
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
    parser.add_argument("--h5ad", type=Path, help="optional override for tests/migrations")
    parser.add_argument("--selected-gene-file", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--min-condition-cells", type=int, default=1)
    parser.add_argument("--module-max-controls", type=int, default=8192)
    parser.add_argument(
        "--module-exclude-cell-line",
        action="append",
        default=[],
        help=(
            "exclude one cell line from gene-module controls only; repeat for "
            "multiple lines (Parent/control/prior statistics remain unchanged)"
        ),
    )
    parser.add_argument("--num-gene-modules", type=int, default=128)
    parser.add_argument("--module-embedding-dim", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument(
        "--ridge-lambdas", default="0.0001,0.001,0.01,0.1,1.0"
    )
    parser.add_argument("--ridge-default-lambda", type=float, default=0.01)
    parser.add_argument("--ridge-cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_artifact(
        resolved_config=args.resolved_config,
        output=args.output,
        metadata_output=args.metadata_output,
        h5ad_override=args.h5ad,
        selected_gene_override=args.selected_gene_file,
        chunk_size=args.chunk_size,
        min_condition_cells=args.min_condition_cells,
        module_max_controls=args.module_max_controls,
        module_exclude_cell_lines=args.module_exclude_cell_line,
        num_gene_modules=args.num_gene_modules,
        module_embedding_dim=args.module_embedding_dim,
        ridge_alpha=args.ridge_alpha,
        ridge_lambdas=args.ridge_lambdas,
        ridge_default_lambda=args.ridge_default_lambda,
        ridge_cv_folds=args.ridge_cv_folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from src.models.de_aware_residual.artifact import (
    SCHEMA_VERSION as DE_SCHEMA_VERSION,
    TrainDELabelBank,
    sha256_file,
)
from src.models.train_hurdle_bank.artifact import (
    SCHEMA_VERSION as HURDLE_SCHEMA_VERSION,
    TrainHurdleBank,
    ndarray_sha256,
    ordered_strings_sha256,
)
from src.models.train_hurdle_bank.build_artifact import (
    _expression_dataset,
    _map_codes_to_parent,
    _read_categorical,
    _verify_gene_order,
)


SOURCE_MANIFEST_KIND = "tahoe100m.14h5ad.identity_manifest.v1"
DE_LABEL_METHOD = "parent_train_delta_abs_topk.v1"


def _fixed_unicode(values: Sequence[str] | str) -> np.ndarray:
    if isinstance(values, str):
        values = [values]
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_split_sha256(
    mask: np.ndarray, lines: np.ndarray, perturbations: np.ndarray
) -> str:
    return _canonical_sha256(
        {
            "audit_condition_mask": np.asarray(mask, dtype=np.uint8).tolist(),
            "cellline_names": np.asarray(lines).astype(str).tolist(),
            "perturbation_names": np.asarray(perturbations).astype(str).tolist(),
        }
    )


def _calibration_support_sha256(
    mask: np.ndarray, lines: np.ndarray, perturbations: np.ndarray
) -> str:
    return _canonical_sha256(
        {
            "parent_calibration_support_mask": np.asarray(
                mask, dtype=np.uint8
            ).tolist(),
            "cellline_names": np.asarray(lines).astype(str).tolist(),
            "perturbation_names": np.asarray(perturbations).astype(str).tolist(),
        }
    )


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(f"unsafe object dtype in {key!r}")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_parent(path: Path) -> dict[str, np.ndarray]:
    required = (
        "gene_names",
        "cellline_names",
        "perturbation_names",
        "control_perturbation_id",
        "normalization_divisor",
        "control_count",
        "train_condition_mask",
        "train_condition_count",
        "validation_condition_mask",
        "test_condition_mask",
        "ridge_support_mask",
    )
    with np.load(path, allow_pickle=False) as parent:
        missing = [key for key in required if key not in parent.files]
        if missing:
            raise KeyError(f"Parent artifact is missing {missing}")
        for key in parent.files:
            if parent[key].dtype == object:
                raise TypeError(f"unsafe object dtype in Parent key {key!r}")
        arrays = {key: np.asarray(parent[key]).copy() for key in required}
    arrays["gene_names"] = arrays["gene_names"].astype(str)
    arrays["cellline_names"] = arrays["cellline_names"].astype(str)
    arrays["perturbation_names"] = arrays["perturbation_names"].astype(str)
    lines = arrays["cellline_names"]
    perturbations = arrays["perturbation_names"]
    genes = arrays["gene_names"]
    shape = (len(lines), len(perturbations))
    train = arrays["train_condition_mask"].astype(bool)
    validation = arrays["validation_condition_mask"].astype(bool)
    test = arrays["test_condition_mask"].astype(bool)
    counts = arrays["train_condition_count"].astype(np.int64)
    controls = arrays["control_count"].astype(np.int64)
    support = arrays["ridge_support_mask"].astype(bool)
    if train.shape != shape or validation.shape != shape or test.shape != shape:
        raise ValueError("Parent masks must have shape [L,P]")
    if counts.shape != shape or controls.shape != (len(lines),):
        raise ValueError("Parent count shapes are invalid")
    if support.shape != shape:
        raise ValueError("Parent ridge_support_mask must have shape [L,P]")
    if np.any(train & (validation | test)) or np.any(validation & test):
        raise ValueError("Parent split masks overlap")
    if np.any(support & (validation | test)):
        raise ValueError("Parent calibration support contains validation/test")
    if not np.array_equal(counts > 0, train):
        raise ValueError("Parent train count/mask provenance differs")
    if np.any(counts[train] < 1):
        raise ValueError("train supervision requires positive treated counts")
    if np.any(controls < 2):
        raise ValueError("DE/support supervision requires at least two controls")
    divisor = float(arrays["normalization_divisor"].reshape(-1)[0])
    if not np.isfinite(divisor) or not np.isclose(divisor, 10.0):
        raise ValueError("Tahoe Parent normalization divisor must be 10")
    control_id = int(arrays["control_perturbation_id"].reshape(-1)[0])
    if control_id < 0 or control_id >= len(perturbations):
        raise ValueError("Parent control_perturbation_id is invalid")
    if len(genes) != 2000:
        raise ValueError(f"Tahoe gene dimension must be 2000, got {len(genes)}")
    arrays.update(
        train_condition_mask=train,
        validation_condition_mask=validation,
        test_condition_mask=test,
        train_condition_count=counts,
        control_count=controls,
        ridge_support_mask=support,
        normalization_divisor=np.asarray(divisor, dtype=np.float32),
        control_perturbation_id=np.asarray(control_id, dtype=np.int32),
    )
    return arrays


def build_de_artifact(
    parent_path: Path, output_path: Path, *, top_k: int
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    parent_sha = sha256_file(parent_path)
    parent = _checked_parent(parent_path)
    genes = parent["gene_names"]
    lines = parent["cellline_names"]
    perturbations = parent["perturbation_names"]
    train = parent["train_condition_mask"]
    validation = parent["validation_condition_mask"]
    test = parent["test_condition_mask"]
    audit = np.zeros_like(train, dtype=bool)
    support = parent["ridge_support_mask"]
    audit_sha = _audit_split_sha256(audit, lines, perturbations)
    support_sha = _calibration_support_sha256(support, lines, perturbations)
    labels = np.zeros(train.shape + (len(genes),), dtype=bool)
    effective_k = min(int(top_k), len(genes))
    with np.load(parent_path, allow_pickle=False) as archive:
        delta = np.asarray(archive["train_condition_delta"], dtype=np.float32)
        if delta.shape != labels.shape:
            raise ValueError("Parent train_condition_delta must have shape [L,P,G]")
        for line_id in range(len(lines)):
            perturbation_ids = np.flatnonzero(train[line_id])
            for perturbation_id in perturbation_ids.tolist():
                magnitude = np.abs(delta[line_id, perturbation_id])
                selected = np.argpartition(magnitude, -effective_k)[-effective_k:]
                labels[line_id, perturbation_id, selected] = True
            print(
                json.dumps(
                    {
                        "stage": "de_labels",
                        "line": str(lines[line_id]),
                        "line_index": line_id,
                        "train_conditions": int(len(perturbation_ids)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if labels[~train].any():
        raise AssertionError("held-out/non-train DE labels were populated")
    metadata = {
        "schema_version": DE_SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path.resolve()),
        "source_parent_artifact_sha256": parent_sha,
        "source_de_csv_sha256": {},
        "label_method": DE_LABEL_METHOD,
        "label_top_k": effective_k,
        "fdr_threshold": None,
        "audit_split_sha256": audit_sha,
        "audit_conditions": [],
        "audit_treated_statistics_present": False,
        "validation_test_treated_statistics_present": False,
        "parent_calibration_mode": "no_intercept",
        "parent_calibration_support_sha256": support_sha,
        "parent_calibration_support_overlaps_audit": False,
    }
    payload = {
        "schema_version": _fixed_unicode(DE_SCHEMA_VERSION),
        "source_parent_artifact_sha256": _fixed_unicode(parent_sha),
        "audit_split_sha256": _fixed_unicode(audit_sha),
        "parent_calibration_mode": _fixed_unicode("no_intercept"),
        "parent_calibration_support_sha256": _fixed_unicode(support_sha),
        "parent_calibration_support_mask": support,
        "metadata_json": _fixed_unicode(json.dumps(metadata, sort_keys=True)),
        "gene_names": _fixed_unicode(genes),
        "cellline_names": _fixed_unicode(lines),
        "perturbation_names": _fixed_unicode(perturbations),
        "normalization_divisor": parent["normalization_divisor"],
        "train_condition_mask": train,
        "audit_condition_mask": audit,
        "validation_condition_mask": validation,
        "test_condition_mask": test,
        "label_available_mask": train.copy(),
        "train_de_label": labels,
        "train_condition_count": np.where(
            train, parent["train_condition_count"], 0
        ).astype(np.int64),
        "control_count": parent["control_count"].astype(np.int64),
    }
    _atomic_savez(output_path, payload)
    result = {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "parent_sha256": parent_sha,
        "label_method": DE_LABEL_METHOD,
        "label_top_k": effective_k,
        "train_conditions": int(train.sum()),
        "positive_labels": int(labels.sum()),
        "validation_test_treated_statistics_present": False,
    }
    _atomic_json(output_path.with_suffix(".metadata.json"), {**metadata, **result})
    return result


def _plate_number(path: Path) -> int:
    match = re.match(r"plate([0-9]+)_", path.name)
    if match is None:
        raise ValueError(f"unexpected Tahoe plate filename: {path.name}")
    return int(match.group(1))


def _plate_paths(source_dir: Path) -> list[Path]:
    paths = sorted(source_dir.glob("*.h5ad"), key=_plate_number)
    if len(paths) != 14 or [_plate_number(path) for path in paths] != list(
        range(1, 15)
    ):
        raise ValueError("Tahoe source directory must contain plates 1 through 14")
    return [path.resolve() for path in paths]


def _aggregate_authorized_chunks(
    matrix: h5py.Dataset,
    allowed: np.ndarray,
    row_lines: np.ndarray,
    row_perts: np.ndarray,
    *,
    control_id: int,
    num_lines: int,
    num_perturbations: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sequentially read HDF5 chunks but aggregate authorized rows only."""

    gene_dim = int(matrix.shape[1])
    control_zero = np.zeros((num_lines, gene_dim), dtype=np.uint64)
    control_total = np.zeros(num_lines, dtype=np.uint64)
    treated_zero = np.zeros(
        (num_lines * num_perturbations, gene_dim), dtype=np.uint64
    )
    treated_total = np.zeros(num_lines * num_perturbations, dtype=np.uint64)
    observed_min = np.inf
    observed_max = -np.inf
    observed_elements = 0
    chunks_read = 0
    for start in range(0, int(matrix.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(matrix.shape[0]))
        selected = np.asarray(allowed[start:stop], dtype=bool)
        if not selected.any():
            continue
        # Tahoe held-out rows are randomly interleaved. The physical HDF5
        # chunk may therefore contain them, but selection happens before any
        # statistic, comparison, or reduction is evaluated.
        values = np.asarray(matrix[start:stop, :], dtype=np.float32)[selected]
        chunks_read += 1
        if not np.isfinite(values).all():
            raise ValueError("non-finite expression in authorized rows")
        if (values < 0.0).any():
            raise ValueError("negative expression in authorized rows")
        observed_min = min(observed_min, float(values.min()))
        observed_max = max(observed_max, float(values.max()))
        observed_elements += int(values.size)
        zeros = values == np.float32(0.0)
        local_lines = row_lines[start:stop][selected]
        local_perts = row_perts[start:stop][selected]
        control = local_perts == int(control_id)
        for line in np.unique(local_lines[control]):
            line_mask = control & (local_lines == line)
            control_zero[line] += zeros[line_mask].sum(axis=0, dtype=np.uint64)
            control_total[line] += np.uint64(line_mask.sum())
        treated = ~control
        if treated.any():
            flat_ids = (
                local_lines[treated] * int(num_perturbations)
                + local_perts[treated]
            )
            order = np.argsort(flat_ids, kind="stable")
            sorted_ids = flat_ids[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
            unique_ids = sorted_ids[starts]
            grouped_zero = np.add.reduceat(
                zeros[treated][order], starts, axis=0, dtype=np.uint64
            )
            grouped_count = np.diff(
                np.r_[starts, len(sorted_ids)]
            ).astype(np.uint64)
            treated_zero[unique_ids] += grouped_zero
            treated_total[unique_ids] += grouped_count
    return (
        control_zero,
        control_total,
        treated_zero.reshape(num_lines, num_perturbations, gene_dim),
        treated_total.reshape(num_lines, num_perturbations),
        {
            "authorized_expression_minimum": float(observed_min),
            "authorized_expression_maximum": float(observed_max),
            "authorized_expression_elements": int(observed_elements),
            "physical_hdf5_chunks_read": int(chunks_read),
            "forbidden_rows_may_share_physical_hdf5_chunks": True,
        },
    )


def build_hurdle_artifact(
    parent_path: Path,
    source_dir: Path,
    output_path: Path,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    parent_sha = sha256_file(parent_path)
    parent = _checked_parent(parent_path)
    genes = parent["gene_names"]
    lines = parent["cellline_names"]
    perturbations = parent["perturbation_names"]
    train = parent["train_condition_mask"]
    validation = parent["validation_condition_mask"]
    test = parent["test_condition_mask"]
    audit = np.zeros_like(train, dtype=bool)
    control_id = int(parent["control_perturbation_id"].reshape(-1)[0])
    num_lines, num_perturbations = train.shape
    gene_dim = len(genes)
    control_zero = np.zeros((num_lines, gene_dim), dtype=np.uint64)
    control_total = np.zeros(num_lines, dtype=np.uint64)
    treated_zero = np.zeros(
        (num_lines, num_perturbations, gene_dim), dtype=np.uint64
    )
    treated_total = np.zeros((num_lines, num_perturbations), dtype=np.uint64)
    descriptors: list[dict[str, Any]] = []
    totals = {
        "control_rows_read": 0,
        "train_treated_rows_read": 0,
        "validation_treated_rows_skipped": 0,
        "test_treated_rows_skipped": 0,
        "audit_treated_rows_skipped": 0,
        "authorized_expression_elements": 0,
    }
    observed_min = np.inf
    observed_max = -np.inf
    total_rows = 0
    for plate_index, path in enumerate(_plate_paths(source_dir), start=1):
        stat_before = path.stat()
        with h5py.File(path, "r") as handle:
            expression_key, matrix = _expression_dataset(handle, "X_hvg")
            shape = (int(matrix.shape[0]), int(matrix.shape[1]))
            if shape[1] != gene_dim:
                raise ValueError(f"{path.name}: X_hvg/Parent width mismatch")
            _verify_gene_order(handle, genes, gene_dim)
            obs = handle["obs"]
            source_lines, source_line_codes = _read_categorical(obs["cell_line"])
            source_perts, source_pert_codes = _read_categorical(
                obs["drugname_drugconc"]
            )
            if len(source_line_codes) != shape[0] or len(source_pert_codes) != shape[0]:
                raise ValueError(f"{path.name}: obs/expression row count mismatch")
            row_lines = _map_codes_to_parent(
                source_lines, source_line_codes, lines, "cell-line"
            )
            row_perts = _map_codes_to_parent(
                source_perts, source_pert_codes, perturbations, "perturbation"
            )
            controls = row_perts == control_id
            treated = ~controls
            row_train = train[row_lines, row_perts] & treated
            row_validation = validation[row_lines, row_perts] & treated
            row_test = test[row_lines, row_perts] & treated
            classified = row_train | row_validation | row_test
            unclassified = treated & ~classified
            if unclassified.any():
                first = int(np.flatnonzero(unclassified)[0])
                raise ValueError(
                    f"{path.name}: treated row {first} is outside Parent splits"
                )
            if np.any(
                row_train.astype(np.uint8)
                + row_validation.astype(np.uint8)
                + row_test.astype(np.uint8)
                > 1
            ):
                raise ValueError(f"{path.name}: overlapping split classification")
            allowed = controls | row_train
            (
                plate_control_zero,
                plate_control_total,
                plate_treated_zero,
                plate_treated_total,
                expression_audit,
            ) = _aggregate_authorized_chunks(
                matrix,
                allowed,
                row_lines,
                row_perts,
                control_id=control_id,
                num_lines=num_lines,
                num_perturbations=num_perturbations,
                chunk_size=chunk_size,
            )
            control_zero += plate_control_zero
            control_total += plate_control_total
            treated_zero += plate_treated_zero
            treated_total += plate_treated_total
            totals["control_rows_read"] += int(controls.sum())
            totals["train_treated_rows_read"] += int(row_train.sum())
            totals["validation_treated_rows_skipped"] += int(row_validation.sum())
            totals["test_treated_rows_skipped"] += int(row_test.sum())
            totals["authorized_expression_elements"] += int(
                expression_audit["authorized_expression_elements"]
            )
            observed_min = min(
                observed_min,
                float(expression_audit["authorized_expression_minimum"]),
            )
            observed_max = max(
                observed_max,
                float(expression_audit["authorized_expression_maximum"]),
            )
            descriptors.append(
                {
                    "plate": plate_index,
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": int(stat_before.st_size),
                    "mtime_ns": int(stat_before.st_mtime_ns),
                    "expression_key": expression_key,
                    "expression_shape": list(shape),
                    "cell_line_categories_sha256": ordered_strings_sha256(
                        source_lines
                    ),
                    "perturbation_categories_sha256": ordered_strings_sha256(
                        source_perts
                    ),
                }
            )
            total_rows += shape[0]
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RuntimeError(f"{path.name} changed during aggregation")
        print(
            json.dumps(
                {
                    "stage": "hurdle_zero_counts",
                    "plate": plate_index,
                    "rows": shape[0],
                    "cumulative_rows": total_rows,
                    "control_rows": int(controls.sum()),
                    "train_rows": int(row_train.sum()),
                    "validation_rows_skipped": int(row_validation.sum()),
                    "test_rows_skipped": int(row_test.sum()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    expected_control = parent["control_count"].astype(np.uint64)
    expected_treated = np.where(
        train, parent["train_condition_count"], 0
    ).astype(np.uint64)
    if not np.array_equal(control_total, expected_control):
        raise ValueError("14-plate control counts differ from Parent provenance")
    if not np.array_equal(treated_total, expected_treated):
        difference = np.argwhere(treated_total != expected_treated)
        raise ValueError(
            "14-plate train counts differ from Parent provenance; "
            f"first_difference={difference[0].tolist() if len(difference) else None}"
        )
    if np.any(control_zero > control_total[:, None]):
        raise AssertionError("control zero count exceeds total")
    if np.any(treated_zero > treated_total[..., None]):
        raise AssertionError("treated zero count exceeds total")
    if np.any(treated_zero[~train] != 0):
        raise AssertionError("forbidden condition contains zero statistics")
    uint32_max = np.iinfo(np.uint32).max
    if (
        control_zero.max(initial=0) > uint32_max
        or treated_zero.max(initial=0) > uint32_max
    ):
        raise OverflowError("zero count exceeds uint32 artifact capacity")
    source_manifest = {
        "kind": SOURCE_MANIFEST_KIND,
        "plates": descriptors,
        "total_rows": total_rows,
        "gene_dim": gene_dim,
    }
    source_sha = _canonical_sha256(source_manifest)
    source_shape = (total_rows, gene_dim)
    gene_sha = ordered_strings_sha256(genes)
    line_sha = ordered_strings_sha256(lines)
    perturbation_sha = ordered_strings_sha256(perturbations)
    metadata = {
        "schema_version": HURDLE_SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path.resolve()),
        "source_parent_artifact_sha256": parent_sha,
        "source_h5ad": str(source_dir.resolve()),
        "source_h5ad_sha256": source_sha,
        "source_h5ad_shape": list(source_shape),
        "source_h5ad_size_bytes": int(sum(item["size_bytes"] for item in descriptors)),
        "source_expression_key": "obsm/X_hvg",
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_manifest": source_manifest,
        "normalization_divisor": 10.0,
        "zero_definition": "exact float equality X_hvg == 0.0; division by 10 invariant",
        "gene_order_sha256": gene_sha,
        "cellline_order_sha256": line_sha,
        "perturbation_order_sha256": perturbation_sha,
        "parent_train_conditions": int(train.sum()),
        "effective_train_conditions": int(train.sum()),
        "audit_conditions": 0,
        **totals,
        "authorized_expression_minimum": float(observed_min),
        "authorized_expression_maximum": float(observed_max),
        "validation_treated_expression_used": False,
        "test_treated_expression_used": False,
        "audit_treated_expression_used": False,
        "forbidden_rows_may_share_physical_hdf5_chunks": True,
    }
    payload = {
        "schema_version": _fixed_unicode(HURDLE_SCHEMA_VERSION),
        "source_parent_artifact_sha256": _fixed_unicode(parent_sha),
        "source_h5ad_sha256": _fixed_unicode(source_sha),
        "source_h5ad_shape": np.asarray(source_shape, dtype=np.uint64),
        "source_expression_key": _fixed_unicode("obsm/X_hvg"),
        "normalization_divisor": np.asarray(10.0, dtype=np.float32),
        "gene_names": _fixed_unicode(genes),
        "cellline_names": _fixed_unicode(lines),
        "perturbation_names": _fixed_unicode(perturbations),
        "gene_order_sha256": _fixed_unicode(gene_sha),
        "cellline_order_sha256": _fixed_unicode(line_sha),
        "perturbation_order_sha256": _fixed_unicode(perturbation_sha),
        "parent_train_condition_mask": train.copy(),
        "train_condition_mask": train.copy(),
        "audit_condition_mask": audit,
        "validation_condition_mask": validation,
        "test_condition_mask": test,
        "parent_train_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(train)
        ),
        "train_condition_mask_sha256": _fixed_unicode(ndarray_sha256(train)),
        "audit_condition_mask_sha256": _fixed_unicode(ndarray_sha256(audit)),
        "validation_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(validation)
        ),
        "test_condition_mask_sha256": _fixed_unicode(ndarray_sha256(test)),
        "control_zero_count": control_zero.astype(np.uint32),
        "control_total_count": control_total,
        "train_treated_zero_count": treated_zero.astype(np.uint32),
        "train_condition_count": treated_total,
        "metadata_json": _fixed_unicode(json.dumps(metadata, sort_keys=True)),
    }
    _atomic_savez(output_path, payload)
    result = {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "parent_sha256": parent_sha,
        "source_manifest_sha256": source_sha,
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_shape": list(source_shape),
        "train_conditions": int(train.sum()),
        "validation_treated_expression_used": False,
        "test_treated_expression_used": False,
    }
    _atomic_json(output_path.with_suffix(".metadata.json"), {**metadata, **result})
    return result


def verify_artifacts(
    parent_path: Path, de_path: Path, hurdle_path: Path
) -> dict[str, Any]:
    parent_sha = sha256_file(parent_path)
    de_sha = sha256_file(de_path)
    hurdle_sha = sha256_file(hurdle_path)
    with np.load(hurdle_path, allow_pickle=False) as loaded:
        source_sha = str(np.asarray(loaded["source_h5ad_sha256"]).reshape(-1)[0])
    de_bank = TrainDELabelBank(
        de_path,
        parent_path,
        expected_sha256=de_sha,
        expected_parent_sha256=parent_sha,
        expected_gene_dim=2000,
        expected_normalization_divisor=10.0,
        expected_parent_calibration="no_intercept",
    )
    hurdle_bank = TrainHurdleBank(
        artifact_path=hurdle_path,
        expected_artifact_sha256=hurdle_sha,
        parent_artifact_path=parent_path,
        expected_source_h5ad_sha256=source_sha,
        expected_gene_dim=2000,
        expected_normalization_divisor=10.0,
    )
    de_mask = de_bank.train_condition_mask.cpu().numpy()
    if not np.array_equal(de_mask, hurdle_bank.train_condition_mask):
        raise ValueError("DE and hurdle train masks differ")
    return {
        "parent": {"path": str(parent_path.resolve()), "sha256": parent_sha},
        "de": {
            "path": str(de_path.resolve()),
            "sha256": de_sha,
            "train_conditions": int(de_mask.sum()),
        },
        "hurdle": {
            "path": str(hurdle_path.resolve()),
            "sha256": hurdle_sha,
            "source_manifest_sha256": source_sha,
            "train_conditions": int(hurdle_bank.train_condition_mask.sum()),
        },
        "train_masks_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--de-output", type=Path, required=True)
    parser.add_argument("--hurdle-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "de", "hurdle", "verify"), default="all")
    parser.add_argument("--de-top-k", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    parent = args.parent.expanduser().resolve()
    source = args.source_dir.expanduser().resolve()
    de_output = args.de_output.expanduser().resolve()
    hurdle_output = args.hurdle_output.expanduser().resolve()
    if not parent.is_file() or not source.is_dir():
        raise FileNotFoundError("Parent artifact or Tahoe source directory is absent")
    if args.mode in {"all", "de"} and not de_output.exists():
        print(json.dumps({"stage": "build_de", "output": str(de_output)}), flush=True)
        build_de_artifact(parent, de_output, top_k=args.de_top_k)
    if args.mode in {"all", "hurdle"} and not hurdle_output.exists():
        print(
            json.dumps({"stage": "build_hurdle", "output": str(hurdle_output)}),
            flush=True,
        )
        build_hurdle_artifact(
            parent, source, hurdle_output, chunk_size=args.chunk_size
        )
    if args.mode in {"all", "verify"}:
        result = verify_artifacts(parent, de_output, hurdle_output)
        _atomic_json(args.manifest.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

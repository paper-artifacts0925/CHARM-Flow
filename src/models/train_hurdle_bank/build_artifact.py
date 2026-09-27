"""Stream a leakage-closed TrainHurdleBank artifact from Replogle X_hvg."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .artifact import (
    REPLOGLE_LINE_COUNT,
    REPLOGLE_NORMALIZATION_DIVISOR,
    SCHEMA_VERSION,
    ndarray_sha256,
    ordered_strings_sha256,
    sha256_file,
)


def _fixed_unicode(values: Sequence[str] | str) -> np.ndarray:
    if isinstance(values, str):
        values = [values]
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    raise TypeError(f"unsafe object dtype in NPZ key {key!r}")
        # Same-directory hard-link publication is atomic and never replaces
        # an existing destination; a racing writer therefore fails closed.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard-link publication is atomic and never replaces
        # an existing destination; a racing writer therefore fails closed.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_strings(dataset: h5py.Dataset) -> list[str]:
    return [_decode(value) for value in dataset[:]]


def _read_categorical(
    group_or_dataset: h5py.Group | h5py.Dataset,
) -> tuple[list[str], np.ndarray]:
    if isinstance(group_or_dataset, h5py.Group):
        if "categories" not in group_or_dataset or "codes" not in group_or_dataset:
            raise ValueError("categorical obs group must contain categories and codes")
        names = _read_strings(group_or_dataset["categories"])
        codes = np.asarray(group_or_dataset["codes"][:], dtype=np.int64)
    else:
        decoded = np.asarray(_read_strings(group_or_dataset), dtype=str)
        names = sorted(set(decoded.tolist()))
        lookup = {name: index for index, name in enumerate(names)}
        codes = np.asarray([lookup[value] for value in decoded], dtype=np.int64)
    if not names or len(set(names)) != len(names):
        raise ValueError("categorical names must be non-empty and unique")
    if len(codes) and (codes.min() < 0 or codes.max() >= len(names)):
        raise ValueError("categorical codes fall outside the category range")
    return names, codes


def _map_codes_to_parent(
    names: Sequence[str], codes: np.ndarray, parent_names: np.ndarray, label: str
) -> np.ndarray:
    parent_lookup = {str(name): index for index, name in enumerate(parent_names)}
    unknown = sorted(set(str(name) for name in names) - set(parent_lookup))
    if unknown:
        raise ValueError(f"source H5AD has unknown {label} values: {unknown[:10]}")
    remap = np.asarray([parent_lookup[str(name)] for name in names], dtype=np.int64)
    return remap[codes]


def _expression_dataset(
    handle: h5py.File, expression_key: str
) -> tuple[str, h5py.Dataset]:
    candidate = str(expression_key).strip("/")
    if candidate in handle:
        matrix = handle[candidate]
        canonical = candidate
    elif "/" not in candidate and "obsm" in handle and candidate in handle["obsm"]:
        matrix = handle["obsm"][candidate]
        canonical = f"obsm/{candidate}"
    else:
        raise KeyError(f"source H5AD has no expression dataset {expression_key!r}")
    if not isinstance(matrix, h5py.Dataset) or matrix.ndim != 2:
        raise ValueError("source expression dataset must be rank two")
    return canonical, matrix


def _verify_gene_order(
    handle: h5py.File, parent_genes: np.ndarray, gene_dim: int
) -> None:
    if len(parent_genes) != int(gene_dim):
        raise ValueError("Parent gene count differs from X_hvg width")
    if "var" not in handle or not isinstance(handle["var"], h5py.Group):
        raise ValueError("source H5AD must contain var metadata")
    var = handle["var"]
    if "highly_variable" not in var:
        raise ValueError("source H5AD must contain var/highly_variable")
    index_name = _decode(var.attrs.get("_index", "_index"))
    if index_name not in var:
        raise ValueError(f"source H5AD var index {index_name!r} is absent")
    all_genes = _read_strings(var[index_name])
    selected = np.asarray(var["highly_variable"][:], dtype=bool)
    if len(all_genes) != len(selected):
        raise ValueError("var index and highly_variable lengths differ")
    source_genes = np.asarray(
        [gene for gene, keep in zip(all_genes, selected) if keep], dtype=str
    )
    if not np.array_equal(source_genes, parent_genes):
        raise ValueError("source X_hvg/Parent gene order mismatch")


def _audit_mask(
    conditions: Iterable[tuple[str, str]],
    lines: np.ndarray,
    perturbations: np.ndarray,
    parent_train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    line_index = {str(name): index for index, name in enumerate(lines)}
    pert_index = {str(name): index for index, name in enumerate(perturbations)}
    pairs = [(str(line), str(perturbation)) for line, perturbation in conditions]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate audit condition")
    mask = np.zeros(parent_train.shape, dtype=bool)
    for line_name, perturbation_name in pairs:
        if line_name not in line_index:
            raise ValueError(f"unknown audit cell line {line_name!r}")
        if perturbation_name not in pert_index:
            raise ValueError(f"unknown audit perturbation {perturbation_name!r}")
        line = line_index[line_name]
        perturbation = pert_index[perturbation_name]
        if validation[line, perturbation] or test[line, perturbation]:
            raise ValueError("audit condition overlaps validation/test")
        if not parent_train[line, perturbation]:
            raise ValueError("audit condition is not in Parent train_condition_mask")
        mask[line, perturbation] = True
    return mask


def _checked_parent_arrays(
    parent_path: Path, expected_normalization_divisor: float
) -> dict[str, np.ndarray]:
    with np.load(parent_path, allow_pickle=False) as parent:
        for key in parent.files:
            if parent[key].dtype == object:
                raise TypeError(f"unsafe object dtype in Parent artifact key {key!r}")
        required = (
            "gene_names",
            "cellline_names",
            "perturbation_names",
            "normalization_divisor",
            "train_condition_mask",
            "validation_condition_mask",
            "test_condition_mask",
            "train_condition_count",
            "control_count",
        )
        missing = [key for key in required if key not in parent.files]
        if missing:
            raise KeyError(f"Parent artifact is missing {missing}")
        arrays = {key: np.asarray(parent[key]).copy() for key in required}
    genes = arrays["gene_names"].astype(str)
    lines = arrays["cellline_names"].astype(str)
    perturbations = arrays["perturbation_names"].astype(str)
    for name, values in (
        ("gene_names", genes),
        ("cellline_names", lines),
        ("perturbation_names", perturbations),
    ):
        if values.ndim != 1 or not len(values) or len(set(values.tolist())) != len(values):
            raise ValueError(f"Parent {name} must be non-empty, rank one, and unique")
    if len(lines) != REPLOGLE_LINE_COUNT:
        raise ValueError(
            f"Replogle hurdle builder requires {REPLOGLE_LINE_COUNT} cell lines"
        )
    divisor = float(arrays["normalization_divisor"].reshape(-1)[0])
    if not (
        np.isfinite(divisor)
        and np.isclose(divisor, REPLOGLE_NORMALIZATION_DIVISOR)
        and np.isclose(divisor, float(expected_normalization_divisor))
    ):
        raise ValueError(
            "Parent normalization divisor must be the pinned Replogle value 10"
        )
    shape = (len(lines), len(perturbations))
    train = arrays["train_condition_mask"].astype(bool)
    validation = arrays["validation_condition_mask"].astype(bool)
    test = arrays["test_condition_mask"].astype(bool)
    train_count = arrays["train_condition_count"]
    control_count = arrays["control_count"]
    if train.shape != shape or validation.shape != shape or test.shape != shape:
        raise ValueError("Parent condition masks must have shape [L,P]")
    if np.any(train & (validation | test)) or np.any(validation & test):
        raise ValueError("Parent condition masks overlap")
    if not np.issubdtype(train_count.dtype, np.integer) or train_count.shape != shape:
        raise ValueError("Parent train_condition_count must be integer [L,P]")
    if not np.issubdtype(control_count.dtype, np.integer) or control_count.shape != (
        len(lines),
    ):
        raise ValueError("Parent control_count must be integer [L]")
    if (train_count < 0).any() or (control_count <= 0).any():
        raise ValueError("Parent counts are invalid")
    if not np.array_equal(train_count > 0, train):
        raise ValueError("Parent train mask/count provenance is inconsistent")
    arrays.update(
        gene_names=genes,
        cellline_names=lines,
        perturbation_names=perturbations,
        train_condition_mask=train,
        validation_condition_mask=validation,
        test_condition_mask=test,
        train_condition_count=train_count.astype(np.uint64),
        control_count=control_count.astype(np.uint64),
        normalization_divisor=np.asarray(divisor, dtype=np.float32),
    )
    return arrays


def _authorized_row_batches(
    allowed_rows: np.ndarray, chunk_size: int
):
    """Yield contiguous authorized slices without spanning a forbidden row."""

    breaks = np.flatnonzero(np.diff(allowed_rows) != 1) + 1
    run_starts = np.r_[0, breaks]
    run_stops = np.r_[breaks, len(allowed_rows)]
    for run_start, run_stop in zip(run_starts, run_stops):
        first = int(allowed_rows[run_start])
        limit = int(allowed_rows[run_stop - 1]) + 1
        for start in range(first, limit, int(chunk_size)):
            stop = min(start + int(chunk_size), limit)
            yield np.arange(start, stop, dtype=np.int64), start, stop


def _aggregate_selected_rows(
    matrix: h5py.Dataset,
    allowed_rows: np.ndarray,
    row_line_ids: np.ndarray,
    row_perturbation_ids: np.ndarray,
    control_perturbation_id: int,
    num_lines: int,
    num_perturbations: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Read only pre-authorized row indices and accumulate exact-zero counts."""

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if allowed_rows.ndim != 1 or len(allowed_rows) == 0:
        raise ValueError("there are no authorized control/train rows")
    if np.any(np.diff(allowed_rows) <= 0):
        raise ValueError("authorized row indices must be strictly increasing")
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

    for rows, start, stop in _authorized_row_batches(allowed_rows, int(chunk_size)):
        # Every HDF5 lookup is a contiguous slice wholly contained in one
        # authorized run; no slice can span an audit/validation/test row.
        values = np.asarray(matrix[start:stop, :], dtype=np.float32)
        if values.shape != (len(rows), gene_dim):
            raise ValueError("source expression lookup returned an unexpected shape")
        if not np.isfinite(values).all():
            raise ValueError("non-finite expression in authorized control/train rows")
        if (values < 0.0).any():
            raise ValueError("negative expression in authorized control/train rows")
        observed_min = min(observed_min, float(values.min()))
        observed_max = max(observed_max, float(values.max()))
        observed_elements += int(values.size)
        zeros = values == np.float32(0.0)
        local_lines = row_line_ids[rows]
        local_perts = row_perturbation_ids[rows]
        control = local_perts == int(control_perturbation_id)
        for line in np.unique(local_lines[control]):
            selected = control & (local_lines == line)
            control_zero[line] += zeros[selected].sum(axis=0, dtype=np.uint64)
            control_total[line] += np.uint64(selected.sum())

        treated = ~control
        if treated.any():
            flat_ids = (
                local_lines[treated] * int(num_perturbations) + local_perts[treated]
            )
            order = np.argsort(flat_ids, kind="stable")
            sorted_ids = flat_ids[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
            unique_ids = sorted_ids[starts]
            grouped_zero = np.add.reduceat(
                zeros[treated][order], starts, axis=0, dtype=np.uint64
            )
            grouped_count = np.diff(np.r_[starts, len(sorted_ids)]).astype(np.uint64)
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
        },
    )


def build_train_hurdle_artifact(
    *,
    parent_artifact_path: str | Path,
    source_h5ad_path: str | Path,
    output_path: str | Path,
    metadata_output_path: str | Path | None = None,
    expected_parent_sha256: str | None = None,
    expected_source_h5ad_sha256: str | None = None,
    expression_key: str = "X_hvg",
    perturbation_key: str = "gene",
    cell_line_key: str = "cell_line",
    control_label: str = "non-targeting",
    audit_conditions: Iterable[tuple[str, str]] = (),
    normalization_divisor: float = REPLOGLE_NORMALIZATION_DIVISOR,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Build counts without ever looking up held-out treated expression rows."""

    parent_path = Path(parent_artifact_path).expanduser().resolve()
    source_path = Path(source_h5ad_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    metadata_path = (
        Path(metadata_output_path).expanduser().resolve()
        if metadata_output_path is not None
        else output_path.with_suffix(".metadata.json")
    )
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path in {parent_path, source_path} or metadata_path in {parent_path, source_path}:
        raise ValueError("output/metadata paths must not overwrite an input artifact")
    if output_path == metadata_path:
        raise ValueError("output_path and metadata_output_path must differ")
    # This preflight intentionally precedes Parent and 20+ GiB source hashing.
    # The hard-link publication below repeats the protection atomically.
    for target, label in ((output_path, "output NPZ"), (metadata_path, "metadata sidecar")):
        if os.path.lexists(target):
            raise FileExistsError(f"{label} already exists; refusing to overwrite: {target}")
    if not np.isclose(
        float(normalization_divisor), REPLOGLE_NORMALIZATION_DIVISOR
    ):
        raise ValueError("TrainHurdleBank is pinned to normalization divisor 10")

    parent_sha = sha256_file(parent_path)
    if expected_parent_sha256 is not None and parent_sha != str(
        expected_parent_sha256
    ).lower():
        raise ValueError(
            f"Parent artifact SHA256 mismatch: expected {expected_parent_sha256}, got {parent_sha}"
        )
    parent_stat = parent_path.stat()
    parent = _checked_parent_arrays(parent_path, normalization_divisor)
    genes = parent["gene_names"]
    lines = parent["cellline_names"]
    perturbations = parent["perturbation_names"]
    parent_train = parent["train_condition_mask"]
    validation = parent["validation_condition_mask"]
    test = parent["test_condition_mask"]
    audit = _audit_mask(
        audit_conditions, lines, perturbations, parent_train, validation, test
    )
    train = parent_train & ~audit

    source_stat_before = source_path.stat()
    source_sha = sha256_file(source_path)
    if expected_source_h5ad_sha256 is not None and source_sha != str(
        expected_source_h5ad_sha256
    ).lower():
        raise ValueError(
            "source H5AD SHA256 mismatch: "
            f"expected {expected_source_h5ad_sha256}, got {source_sha}"
        )

    with h5py.File(source_path, "r") as handle:
        canonical_expression_key, matrix = _expression_dataset(handle, expression_key)
        source_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
        _verify_gene_order(handle, genes, source_shape[1])
        if "obs" not in handle or not isinstance(handle["obs"], h5py.Group):
            raise ValueError("source H5AD must contain obs metadata")
        obs = handle["obs"]
        if cell_line_key not in obs or perturbation_key not in obs:
            raise KeyError("source H5AD is missing required obs fields")
        source_line_names, source_line_codes = _read_categorical(obs[cell_line_key])
        source_pert_names, source_pert_codes = _read_categorical(obs[perturbation_key])
        if len(source_line_codes) != source_shape[0] or len(source_pert_codes) != source_shape[0]:
            raise ValueError("source obs metadata and expression row counts differ")
        row_lines = _map_codes_to_parent(
            source_line_names, source_line_codes, lines, "cell-line"
        )
        row_perts = _map_codes_to_parent(
            source_pert_names, source_pert_codes, perturbations, "perturbation"
        )
        pert_lookup = {str(name): index for index, name in enumerate(perturbations)}
        if str(control_label) not in pert_lookup:
            raise ValueError(f"control label {control_label!r} is absent from Parent order")
        control_id = int(pert_lookup[str(control_label)])
        controls = row_perts == control_id
        treated = ~controls
        row_train = train[row_lines, row_perts] & treated
        row_audit = audit[row_lines, row_perts] & treated
        row_validation = validation[row_lines, row_perts] & treated
        row_test = test[row_lines, row_perts] & treated
        classified_treated = row_train | row_audit | row_validation | row_test
        unclassified = treated & ~classified_treated
        if unclassified.any():
            first = int(np.flatnonzero(unclassified)[0])
            raise ValueError(
                "source contains a treated row outside train/audit/validation/test; "
                "expression lookup refused before reading X_hvg: "
                f"row={first}, line={lines[row_lines[first]]!r}, "
                f"perturbation={perturbations[row_perts[first]]!r}"
            )
        if np.any(
            row_train.astype(np.uint8)
            + row_audit.astype(np.uint8)
            + row_validation.astype(np.uint8)
            + row_test.astype(np.uint8)
            > 1
        ):
            raise ValueError("a treated source row belongs to overlapping split masks")

        allowed = controls | row_train
        forbidden = row_audit | row_validation | row_test
        if np.any(allowed & forbidden):
            raise AssertionError("internal split error: an allowed row is forbidden")
        allowed_rows = np.flatnonzero(allowed).astype(np.int64)
        (
            control_zero,
            control_total,
            treated_zero,
            treated_total,
            expression_audit,
        ) = _aggregate_selected_rows(
            matrix,
            allowed_rows,
            row_lines,
            row_perts,
            control_perturbation_id=control_id,
            num_lines=len(lines),
            num_perturbations=len(perturbations),
            chunk_size=int(chunk_size),
        )

    source_stat_after = source_path.stat()
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    ):
        raise RuntimeError("source H5AD changed while TrainHurdleBank was built")
    parent_stat_after = parent_path.stat()
    if (
        parent_stat.st_size != parent_stat_after.st_size
        or parent_stat.st_mtime_ns != parent_stat_after.st_mtime_ns
    ):
        raise RuntimeError("Parent artifact changed while TrainHurdleBank was built")

    expected_control_total = parent["control_count"].astype(np.uint64)
    expected_treated_total = np.where(
        train, parent["train_condition_count"], 0
    ).astype(np.uint64)
    if not np.array_equal(control_total, expected_control_total):
        raise ValueError(
            "source control counts differ from Parent artifact provenance"
        )
    if not np.array_equal(treated_total, expected_treated_total):
        raise ValueError(
            "source train-treated counts differ from Parent artifact provenance"
        )
    if np.any(control_zero > control_total[:, None]):
        raise AssertionError("internal error: control zero count exceeds total")
    if np.any(treated_zero > treated_total[..., None]):
        raise AssertionError("internal error: treated zero count exceeds total")
    if np.any(treated_zero[~train] != 0):
        raise AssertionError("internal leakage error: forbidden treated statistics exist")
    uint32_max = np.iinfo(np.uint32).max
    if control_zero.max(initial=0) > uint32_max or treated_zero.max(initial=0) > uint32_max:
        raise OverflowError("exact-zero count exceeds uint32 artifact capacity")

    gene_hash = ordered_strings_sha256(genes)
    line_hash = ordered_strings_sha256(lines)
    perturbation_hash = ordered_strings_sha256(perturbations)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path),
        "source_parent_artifact_sha256": parent_sha,
        "source_h5ad": str(source_path),
        "source_h5ad_sha256": source_sha,
        "source_h5ad_shape": list(source_shape),
        "source_h5ad_size_bytes": int(source_stat_before.st_size),
        "source_expression_key": canonical_expression_key,
        "normalization_divisor": float(normalization_divisor),
        "zero_definition": "exact float equality X_hvg == 0.0; invariant under division by 10",
        "gene_order_sha256": gene_hash,
        "cellline_order_sha256": line_hash,
        "perturbation_order_sha256": perturbation_hash,
        "parent_train_conditions": int(parent_train.sum()),
        "effective_train_conditions": int(train.sum()),
        "audit_conditions": int(audit.sum()),
        "control_rows_read": int(controls.sum()),
        "train_treated_rows_read": int(row_train.sum()),
        "validation_treated_rows_skipped": int(row_validation.sum()),
        "test_treated_rows_skipped": int(row_test.sum()),
        "audit_treated_rows_skipped": int(row_audit.sum()),
        "validation_treated_expression_used": False,
        "test_treated_expression_used": False,
        "audit_treated_expression_used": False,
        **expression_audit,
    }
    arrays = {
        "schema_version": _fixed_unicode(SCHEMA_VERSION),
        "source_parent_artifact_sha256": _fixed_unicode(parent_sha),
        "source_h5ad_sha256": _fixed_unicode(source_sha),
        "source_h5ad_shape": np.asarray(source_shape, dtype=np.uint64),
        "source_expression_key": _fixed_unicode(canonical_expression_key),
        "normalization_divisor": np.asarray(
            float(normalization_divisor), dtype=np.float32
        ),
        "gene_names": _fixed_unicode(genes),
        "cellline_names": _fixed_unicode(lines),
        "perturbation_names": _fixed_unicode(perturbations),
        "gene_order_sha256": _fixed_unicode(gene_hash),
        "cellline_order_sha256": _fixed_unicode(line_hash),
        "perturbation_order_sha256": _fixed_unicode(perturbation_hash),
        "parent_train_condition_mask": parent_train,
        "train_condition_mask": train,
        "audit_condition_mask": audit,
        "validation_condition_mask": validation,
        "test_condition_mask": test,
        "parent_train_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(parent_train)
        ),
        "train_condition_mask_sha256": _fixed_unicode(ndarray_sha256(train)),
        "audit_condition_mask_sha256": _fixed_unicode(ndarray_sha256(audit)),
        "validation_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(validation)
        ),
        "test_condition_mask_sha256": _fixed_unicode(ndarray_sha256(test)),
        "control_zero_count": control_zero.astype(np.uint32),
        "control_total_count": control_total.astype(np.uint64),
        "train_treated_zero_count": treated_zero.astype(np.uint32),
        "train_condition_count": treated_total.astype(np.uint64),
        "metadata_json": _fixed_unicode(json.dumps(metadata, sort_keys=True)),
    }
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"builder produced unsafe object array {key!r}")
    _atomic_savez(output_path, arrays)
    artifact_sha = sha256_file(output_path)
    sidecar = dict(metadata)
    sidecar.update(
        artifact_path=str(output_path),
        artifact_sha256=artifact_sha,
        artifact_size_bytes=int(output_path.stat().st_size),
    )
    _atomic_json(metadata_path, sidecar)
    return sidecar


def _condition(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("audit condition must be CELL_LINE:PERTURBATION")
    line, perturbation = value.split(":", 1)
    if not line or not perturbation:
        raise argparse.ArgumentTypeError("audit condition must be CELL_LINE:PERTURBATION")
    return line, perturbation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--source-h5ad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--expected-parent-sha256")
    parser.add_argument("--expected-source-h5ad-sha256")
    parser.add_argument("--expression-key", default="X_hvg")
    parser.add_argument("--perturbation-key", default="gene")
    parser.add_argument("--cell-line-key", default="cell_line")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--audit-condition", action="append", type=_condition, default=[])
    parser.add_argument("--chunk-size", type=int, default=4096)
    arguments = parser.parse_args()
    result = build_train_hurdle_artifact(
        parent_artifact_path=arguments.parent_artifact,
        source_h5ad_path=arguments.source_h5ad,
        output_path=arguments.output,
        metadata_output_path=arguments.metadata_output,
        expected_parent_sha256=arguments.expected_parent_sha256,
        expected_source_h5ad_sha256=arguments.expected_source_h5ad_sha256,
        expression_key=arguments.expression_key,
        perturbation_key=arguments.perturbation_key,
        cell_line_key=arguments.cell_line_key,
        control_label=arguments.control_label,
        audit_conditions=arguments.audit_condition,
        chunk_size=arguments.chunk_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_train_hurdle_artifact"]

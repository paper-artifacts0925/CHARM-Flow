from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.de_aware_residual.artifact import (  # noqa: E402
    SCHEMA_VERSION as DE_SCHEMA_VERSION,
    TrainDELabelBank,
    sha256_file,
)
from src.models.train_hurdle_bank.artifact import (  # noqa: E402
    SCHEMA_VERSION as HURDLE_SCHEMA_VERSION,
    TrainHurdleBank,
    ndarray_sha256,
    ordered_strings_sha256,
)


EXPECTED_SOURCE_ROWS = 9_697_974
EXPECTED_GENE_DIM = 2_000
EXPECTED_LINES = 18
EXPECTED_DONOR_CELLTYPE_LINES = 216
EXPECTED_PERTURBATIONS = 91
EXPECTED_CONTROL_ROWS = 629_701
EXPECTED_TRAIN_TREATED_ROWS = 6_657_336
EXPECTED_VALIDATION_TREATED_ROWS = 150_484
EXPECTED_TEST_TREATED_ROWS = 2_260_453
EXPECTED_HOLDOUT_DONORS = ("Donor1", "Donor4", "Donor9", "Donor12")
EXPECTED_EXPRESSION_ELEMENTS = (
    EXPECTED_CONTROL_ROWS + EXPECTED_TRAIN_TREATED_ROWS
) * EXPECTED_GENE_DIM
EXPRESSION_KEY = "obsm/X_hvg"
CONTROL_LABEL = "PBS"
CELL_LINE_KEY = "cell_type"
PERTURBATION_KEY = "cytokine"
DONOR_KEY = "donor"
DE_LABEL_METHOD = "parent_train_delta_abs_topk_gene_index_tiebreak.v1"
SOURCE_MANIFEST_KIND = "parent_residual_ocoot.source_manifest.v1"
PARENT_GROUPING_CELLTYPE = "celltype"
PARENT_GROUPING_DONOR_CELLTYPE = "donor_celltype"
DONOR_CELLTYPE_SEPARATOR = "::"
EXPECTED_DONOR_CELLTYPE_TRAIN_CONDITIONS = 14_505


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


def _valid_sha256(value: Any, label: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} is not a lowercase SHA256: {value!r}")
    return result


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


def _atomic_json(
    path: Path, payload: Mapping[str, Any], *, accept_identical: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if path.exists():
        if accept_identical and json.loads(path.read_text(encoding="utf-8")) == normalized:
            return
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray([_decode(value) for value in dataset[:]], dtype=str)


def _categorical(
    obs: h5py.Group, key: str
) -> tuple[np.ndarray, h5py.Dataset]:
    if key not in obs or not isinstance(obs[key], h5py.Group):
        raise ValueError(f"obs/{key} must be a categorical group")
    group = obs[key]
    if "categories" not in group or "codes" not in group:
        raise ValueError(f"obs/{key} must contain categories and codes")
    categories = _read_strings(group["categories"])
    codes = group["codes"]
    if categories.ndim != 1 or not len(categories):
        raise ValueError(f"obs/{key}/categories is empty or non-vector")
    if len(set(categories.tolist())) != len(categories):
        raise ValueError(f"obs/{key}/categories contains duplicates")
    if not isinstance(codes, h5py.Dataset) or codes.ndim != 1:
        raise ValueError(f"obs/{key}/codes must be a rank-one dataset")
    return categories, codes


def _category_remap(
    source_names: np.ndarray, target_names: np.ndarray, label: str
) -> np.ndarray:
    target = {str(name): index for index, name in enumerate(target_names)}
    missing = sorted(set(source_names.tolist()).difference(target))
    if missing:
        raise ValueError(f"source has unknown {label} values: {missing[:10]}")
    return np.asarray([target[str(name)] for name in source_names], dtype=np.int32)


def _checked_codes(codes: np.ndarray, categories: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int64)
    if len(values) and (values.min() < 0 or values.max() >= len(categories)):
        raise ValueError(f"obs/{label}/codes falls outside its category range")
    return values


def _scalar_string(value: np.ndarray, label: str) -> str:
    flat = np.asarray(value).reshape(-1)
    if flat.shape != (1,):
        raise ValueError(f"{label} must contain exactly one string")
    return str(flat[0])


def _parent_grouping(parent: Mapping[str, np.ndarray]) -> str:
    """Return explicit Parent row semantics, preserving legacy artifacts."""

    lines = np.asarray(parent["cellline_names"]).astype(str)
    raw = parent.get("parent_context_kind")
    if raw is None:
        if len(lines) != EXPECTED_LINES:
            raise ValueError(
                "PBMC Parent without parent_context_kind must use the legacy "
                f"{EXPECTED_LINES}-row celltype layout"
            )
        return PARENT_GROUPING_CELLTYPE
    grouping = _scalar_string(np.asarray(raw), "parent_context_kind").strip().lower()
    if grouping not in {
        PARENT_GROUPING_CELLTYPE,
        PARENT_GROUPING_DONOR_CELLTYPE,
    }:
        raise ValueError(f"unsupported PBMC Parent grouping {grouping!r}")
    return grouping


def _source_group_remap(
    source_donors: np.ndarray,
    source_lines: np.ndarray,
    parent_lines: np.ndarray,
    *,
    parent_grouping: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map source categorical IDs to exact ordered Parent row IDs."""

    parent_lines = np.asarray(parent_lines).astype(str)
    source_donors = np.asarray(source_donors).astype(str)
    source_lines = np.asarray(source_lines).astype(str)
    if parent_grouping == PARENT_GROUPING_CELLTYPE:
        remap = _category_remap(source_lines, parent_lines, "cell type")
        return (
            remap,
            np.full(len(parent_lines), "", dtype="<U1"),
            parent_lines.copy(),
        )
    if parent_grouping != PARENT_GROUPING_DONOR_CELLTYPE:
        raise ValueError(f"unsupported Parent grouping {parent_grouping!r}")
    if any(DONOR_CELLTYPE_SEPARATOR in name for name in source_donors.tolist()):
        raise ValueError("source donor name contains the composite separator")
    if any(DONOR_CELLTYPE_SEPARATOR in name for name in source_lines.tolist()):
        raise ValueError("source cell-type name contains the composite separator")
    lookup = {str(name): index for index, name in enumerate(parent_lines)}
    expected_names = [
        f"{donor}{DONOR_CELLTYPE_SEPARATOR}{line}"
        for donor in source_donors.tolist()
        for line in source_lines.tolist()
    ]
    missing = sorted(set(expected_names).difference(lookup))
    extra = sorted(set(parent_lines.tolist()).difference(expected_names))
    if missing or extra or len(expected_names) != len(parent_lines):
        raise ValueError(
            "donor-celltype Parent/source context vocabulary mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    remap = np.empty((len(source_donors), len(source_lines)), dtype=np.int32)
    parent_donors = np.empty(len(parent_lines), dtype=object)
    parent_celltypes = np.empty(len(parent_lines), dtype=object)
    for donor_id, donor in enumerate(source_donors.tolist()):
        for line_id, line in enumerate(source_lines.tolist()):
            name = f"{donor}{DONOR_CELLTYPE_SEPARATOR}{line}"
            parent_id = int(lookup[name])
            remap[donor_id, line_id] = parent_id
            parent_donors[parent_id] = donor
            parent_celltypes[parent_id] = line
    return remap, parent_donors.astype(str), parent_celltypes.astype(str)


def _source_group_ids(
    group_remap: np.ndarray,
    donor_codes: np.ndarray,
    line_codes: np.ndarray,
    *,
    parent_grouping: str,
) -> np.ndarray:
    if parent_grouping == PARENT_GROUPING_CELLTYPE:
        return np.asarray(group_remap[line_codes], dtype=np.int32)
    if parent_grouping == PARENT_GROUPING_DONOR_CELLTYPE:
        return np.asarray(group_remap[donor_codes, line_codes], dtype=np.int32)
    raise ValueError(f"unsupported Parent grouping {parent_grouping!r}")


def _effective_control_statistics(
    direct_count: np.ndarray,
    direct_statistic: np.ndarray | None,
    *,
    parent: Mapping[str, np.ndarray],
    parent_grouping: str,
    parent_group_donors: np.ndarray,
    parent_group_celltypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply only the train-only control fallback already pinned by Parent."""

    direct_count = np.asarray(direct_count, dtype=np.uint64)
    expected_direct = np.asarray(parent["control_count"], dtype=np.uint64)
    if not np.array_equal(direct_count, expected_direct):
        raise ValueError("PBMC direct control counts differ from Parent provenance")
    fallback = np.asarray(parent["control_mean_fallback_mask"], dtype=bool)
    expected_source = np.asarray(parent["control_mean_source_count"], dtype=np.uint64)
    effective_count = direct_count.copy()
    effective_statistic = (
        None if direct_statistic is None else np.asarray(direct_statistic).copy()
    )
    if fallback.any() and parent_grouping != PARENT_GROUPING_DONOR_CELLTYPE:
        raise ValueError("control fallback is only supported for donor_celltype Parent")
    for target in np.flatnonzero(fallback).tolist():
        same_celltype = parent_group_celltypes == parent_group_celltypes[target]
        other_donor = parent_group_donors != parent_group_donors[target]
        sources = same_celltype & other_donor & ~fallback
        if not sources.any():
            raise ValueError(
                f"no legal same-celltype control fallback for Parent row {target}"
            )
        effective_count[target] = direct_count[sources].sum(dtype=np.uint64)
        if effective_statistic is not None:
            effective_statistic[target] = np.asarray(direct_statistic)[sources].sum(
                axis=0, dtype=np.uint64
            )
    if not np.array_equal(effective_count, expected_source):
        where = np.flatnonzero(effective_count != expected_source)
        raise ValueError(
            "PBMC effective control-source counts differ from Parent provenance; "
            f"first_difference={int(where[0]) if len(where) else None}"
        )
    return effective_count, effective_statistic


def _checked_parent(path: Path) -> dict[str, np.ndarray]:
    required = (
        "gene_names",
        "cellline_names",
        "perturbation_names",
        "control_perturbation_id",
        "normalization_divisor",
        "control_count",
        "train_condition_delta",
        "train_condition_mask",
        "train_condition_count",
        "validation_condition_mask",
        "test_condition_mask",
        "validation_heldout_cell_count",
        "test_heldout_cell_count",
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
        for key in (
            "parent_context_kind",
            "control_mean_source_count",
            "control_mean_fallback_mask",
        ):
            if key in parent.files:
                arrays[key] = np.asarray(parent[key]).copy()
    arrays["gene_names"] = arrays["gene_names"].astype(str)
    arrays["cellline_names"] = arrays["cellline_names"].astype(str)
    arrays["perturbation_names"] = arrays["perturbation_names"].astype(str)
    genes = arrays["gene_names"]
    lines = arrays["cellline_names"]
    perturbations = arrays["perturbation_names"]
    grouping = _parent_grouping(arrays)
    if len(genes) != EXPECTED_GENE_DIM:
        raise ValueError(f"PBMC Parent must contain {EXPECTED_GENE_DIM} genes")
    expected_lines = (
        EXPECTED_LINES
        if grouping == PARENT_GROUPING_CELLTYPE
        else EXPECTED_DONOR_CELLTYPE_LINES
    )
    if len(lines) != expected_lines or len(perturbations) != EXPECTED_PERTURBATIONS:
        raise ValueError("PBMC Parent line/perturbation vocabulary size changed")
    for name, values in (
        ("gene_names", genes),
        ("cellline_names", lines),
        ("perturbation_names", perturbations),
    ):
        if values.ndim != 1 or len(set(values.tolist())) != len(values):
            raise ValueError(f"Parent {name} must be rank-one and unique")
    shape = (len(lines), len(perturbations))
    train = arrays["train_condition_mask"].astype(bool)
    validation = arrays["validation_condition_mask"].astype(bool)
    test = arrays["test_condition_mask"].astype(bool)
    train_count = arrays["train_condition_count"].astype(np.int64)
    control_count = arrays["control_count"].astype(np.int64)
    validation_count = arrays["validation_heldout_cell_count"].astype(np.int64)
    test_count = arrays["test_heldout_cell_count"].astype(np.int64)
    support = arrays["ridge_support_mask"].astype(bool)
    delta = arrays["train_condition_delta"].astype(np.float32)
    for name, value in (
        ("train", train),
        ("validation", validation),
        ("test", test),
        ("train count", train_count),
        ("validation heldout count", validation_count),
        ("test heldout count", test_count),
        ("ridge support", support),
    ):
        if value.shape != shape:
            raise ValueError(f"Parent {name} must have shape [L,P]")
    if delta.shape != shape + (len(genes),):
        raise ValueError("Parent train_condition_delta must have shape [L,P,G]")
    if control_count.shape != (len(lines),):
        raise ValueError("Parent control_count must have shape [L]")
    if "control_mean_source_count" in arrays:
        if "control_mean_fallback_mask" not in arrays:
            raise ValueError("Parent control source counts lack fallback mask")
        control_source_count = arrays["control_mean_source_count"].astype(np.int64)
        fallback = arrays["control_mean_fallback_mask"].astype(bool)
        if control_source_count.shape != (len(lines),) or fallback.shape != (len(lines),):
            raise ValueError("Parent control fallback arrays must have shape [L]")
        if np.any(control_source_count < control_count):
            raise ValueError("Parent effective control-source count is below direct count")
        if np.any(~fallback & (control_source_count != control_count)):
            raise ValueError("Parent non-fallback control-source count differs")
        if np.any(fallback & (control_count != 0)):
            raise ValueError("PBMC control fallback must correspond to a zero-count row")
        if np.any(control_source_count < 2):
            raise ValueError("every Parent row needs at least two effective controls")
    else:
        if "control_mean_fallback_mask" in arrays:
            raise ValueError("Parent control fallback mask lacks source counts")
        control_source_count = control_count.copy()
        fallback = np.zeros(len(lines), dtype=bool)
        if np.any(control_source_count < 2):
            raise ValueError("every Parent row needs at least two controls")
    if np.any(train & (validation | test)) or np.any(validation & test):
        raise ValueError("Parent condition masks overlap")
    if np.any(support & (validation | test)):
        raise ValueError("Parent ridge support overlaps held-out condition masks")
    if not np.array_equal(train_count > 0, train):
        raise ValueError("Parent train count/mask provenance differs")
    divisor = float(arrays["normalization_divisor"].reshape(-1)[0])
    if not np.isfinite(divisor) or not np.isclose(divisor, 10.0):
        raise ValueError("PBMC Parent normalization divisor must be 10")
    control_id = int(arrays["control_perturbation_id"].reshape(-1)[0])
    if not 0 <= control_id < len(perturbations):
        raise ValueError("Parent control perturbation id is invalid")
    if str(perturbations[control_id]) != CONTROL_LABEL:
        raise ValueError("Parent control perturbation is not PBS")
    if int(control_count.sum()) != EXPECTED_CONTROL_ROWS:
        raise ValueError("Parent control total differs from the pinned PBMC protocol")
    if int(train_count.sum()) != EXPECTED_TRAIN_TREATED_ROWS:
        raise ValueError("Parent train total differs from the pinned PBMC protocol")
    if int(validation_count.sum()) != EXPECTED_VALIDATION_TREATED_ROWS:
        raise ValueError("Parent validation total differs from the pinned PBMC protocol")
    if int(test_count.sum()) != EXPECTED_TEST_TREATED_ROWS:
        raise ValueError("Parent test total differs from the pinned PBMC protocol")
    if grouping == PARENT_GROUPING_CELLTYPE:
        expected_train_conditions = EXPECTED_LINES * (EXPECTED_PERTURBATIONS - 1)
    else:
        expected_train_conditions = EXPECTED_DONOR_CELLTYPE_TRAIN_CONDITIONS
    if int(train.sum()) != expected_train_conditions:
        raise ValueError(
            "PBMC Parent train-condition count changed for "
            f"{grouping}: {int(train.sum())} != {expected_train_conditions}"
        )
    arrays.update(
        train_condition_mask=train,
        validation_condition_mask=validation,
        test_condition_mask=test,
        train_condition_count=train_count,
        control_count=control_count,
        control_mean_source_count=control_source_count,
        control_mean_fallback_mask=fallback,
        validation_heldout_cell_count=validation_count,
        test_heldout_cell_count=test_count,
        ridge_support_mask=support,
        train_condition_delta=delta,
        normalization_divisor=np.asarray(divisor, dtype=np.float32),
        control_perturbation_id=np.asarray(control_id, dtype=np.int32),
        parent_grouping=grouping,
    )
    return arrays


def _checked_parent_metadata(
    path: Path, source_h5ad: Path, *, parent_grouping: str
) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset_kind") != "pbmc":
        raise ValueError("Parent metadata is not PBMC")
    split = metadata.get("split")
    source = metadata.get("source")
    if not isinstance(split, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Parent metadata lacks split/source provenance")
    if split.get("split_group_key") != DONOR_KEY:
        raise ValueError("PBMC split_group_key must be donor")
    expected_context_key = (
        CELL_LINE_KEY
        if parent_grouping == PARENT_GROUPING_CELLTYPE
        else f"{DONOR_KEY}{DONOR_CELLTYPE_SEPARATOR}{CELL_LINE_KEY}"
    )
    if split.get("parent_context_key") != expected_context_key:
        raise ValueError(
            "PBMC Parent context key/grouping mismatch: "
            f"{split.get('parent_context_key')!r} != {expected_context_key!r}"
        )
    declared_grouping = split.get("parent_context_kind")
    if parent_grouping == PARENT_GROUPING_DONOR_CELLTYPE:
        if declared_grouping != PARENT_GROUPING_DONOR_CELLTYPE:
            raise ValueError("donor-celltype Parent metadata lacks matching context kind")
    elif declared_grouping not in (None, PARENT_GROUPING_CELLTYPE):
        raise ValueError("legacy celltype Parent metadata has a conflicting context kind")
    if split.get("control_label") != CONTROL_LABEL:
        raise ValueError("PBMC control label must be PBS")
    if split.get("split_control") is not False:
        raise ValueError("PBMC protocol must expose all control cells")
    if tuple(split.get("holdout_groups", ())) != EXPECTED_HOLDOUT_DONORS:
        raise ValueError("PBMC held-out donor order/content changed")
    leakage = split.get("leakage_audit", {})
    if leakage.get("cell_level_split_applied_before_aggregation") is not True:
        raise ValueError("Parent metadata does not prove cell-level split ordering")
    if int(leakage.get("heldout_treated_expression_used", -1)) != 0:
        raise ValueError("Parent metadata reports held-out treated-expression use")
    if leakage.get("target_context_controls_allowed") is not True:
        raise ValueError("PBMC protocol no longer allows target-context controls")
    counts = split.get("cell_counts", {})
    expected_counts = {
        "training_controls": EXPECTED_CONTROL_ROWS,
        "train_treated": EXPECTED_TRAIN_TREATED_ROWS,
        "validation_treated": EXPECTED_VALIDATION_TREATED_ROWS,
        "test_treated": EXPECTED_TEST_TREATED_ROWS,
    }
    if {key: int(counts.get(key, -1)) for key in expected_counts} != expected_counts:
        raise ValueError("Parent metadata split counts changed")
    source_sha = _valid_sha256(
        source.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    training_sha = _valid_sha256(
        source.get("training_expression_sha256"), "training_expression_sha256"
    )
    if source.get("source_files_fully_hashed") is not False:
        raise ValueError("expected manifest-anchored, not full-file-hashed, PBMC source")
    shards = source.get("shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise ValueError("PBMC Parent source must contain exactly one H5AD")
    descriptor = shards[0]
    if int(descriptor.get("rows", -1)) != EXPECTED_SOURCE_ROWS:
        raise ValueError("Parent source row count changed")
    source_stat = source_h5ad.stat()
    if int(descriptor.get("size_bytes", -1)) != int(source_stat.st_size):
        raise ValueError("runtime PBMC H5AD size differs from Parent source manifest")
    if descriptor.get("expression_location") != EXPRESSION_KEY:
        raise ValueError("Parent source expression location is not obsm/X_hvg")
    if Path(str(descriptor.get("path", ""))).name != source_h5ad.name:
        raise ValueError("runtime PBMC H5AD basename differs from Parent source manifest")
    return {
        "metadata": metadata,
        "source_manifest_sha256": source_sha,
        "training_expression_sha256": training_sha,
        "validation_perturbations": tuple(split.get("validation_perturbations", ())),
        "test_perturbations": tuple(split.get("test_perturbations", ())),
        "source_descriptor": descriptor,
    }


def _verify_gene_order(
    handle: h5py.File, parent_genes: np.ndarray, gene_dim: int
) -> str:
    if "var" not in handle or not isinstance(handle["var"], h5py.Group):
        raise ValueError("PBMC H5AD has no var metadata")
    var = handle["var"]
    if "highly_variable" not in var:
        raise ValueError("PBMC H5AD lacks var/highly_variable")
    index_key = _decode(var.attrs.get("_index", "_index"))
    if index_key not in var:
        raise ValueError(f"PBMC H5AD lacks var index {index_key!r}")
    all_genes = _read_strings(var[index_key])
    selected = np.asarray(var["highly_variable"][:], dtype=bool)
    if len(all_genes) != len(selected):
        raise ValueError("var index/highly_variable lengths differ")
    source_genes = all_genes[selected]
    if source_genes.shape != (gene_dim,) or not np.array_equal(
        source_genes, parent_genes
    ):
        raise ValueError("PBMC X_hvg/Parent gene order mismatch")
    return ordered_strings_sha256(source_genes)


def _split_masks(
    donor_codes: np.ndarray,
    perturbation_ids: np.ndarray,
    *,
    heldout_donor_categories: np.ndarray,
    validation_perturbations: np.ndarray,
    test_perturbations: np.ndarray,
    control_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control = perturbation_ids == int(control_id)
    heldout_donor = heldout_donor_categories[donor_codes]
    validation = (
        ~control & heldout_donor & validation_perturbations[perturbation_ids]
    )
    test = ~control & heldout_donor & test_perturbations[perturbation_ids]
    if np.any(validation & test):
        raise ValueError("PBMC validation/test split overlap")
    train = ~(control | validation | test)
    return control, train, validation, test


def _source_plan(
    source_h5ad: Path,
    parent: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    chunk_size: int,
    count_rows: bool,
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    stat_before = source_h5ad.stat()
    lines = parent["cellline_names"]
    perturbations = parent["perturbation_names"]
    control_id = int(parent["control_perturbation_id"].reshape(-1)[0])
    validation_names = tuple(metadata["validation_perturbations"])
    test_names = tuple(metadata["test_perturbations"])
    pert_lookup = {str(name): index for index, name in enumerate(perturbations)}
    if sorted(set(validation_names).intersection(test_names)):
        raise ValueError("metadata validation/test cytokines overlap")
    missing_validation = sorted(set(validation_names).difference(pert_lookup))
    missing_test = sorted(set(test_names).difference(pert_lookup))
    if missing_validation or missing_test:
        raise ValueError(
            "metadata held-out cytokines are absent from Parent: "
            f"validation={missing_validation}, test={missing_test}"
        )
    validation_by_perturbation = np.zeros(len(perturbations), dtype=bool)
    test_by_perturbation = np.zeros(len(perturbations), dtype=bool)
    validation_by_perturbation[[pert_lookup[name] for name in validation_names]] = True
    test_by_perturbation[[pert_lookup[name] for name in test_names]] = True
    with h5py.File(source_h5ad, "r") as handle:
        if EXPRESSION_KEY not in handle:
            raise ValueError(f"PBMC H5AD lacks {EXPRESSION_KEY}")
        matrix = handle[EXPRESSION_KEY]
        if not isinstance(matrix, h5py.Dataset) or matrix.ndim != 2:
            raise ValueError("PBMC obsm/X_hvg must be a rank-two dense dataset")
        shape = (int(matrix.shape[0]), int(matrix.shape[1]))
        if shape != (EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM):
            raise ValueError(f"PBMC X_hvg shape changed: {shape}")
        if matrix.dtype != np.dtype(np.float32):
            raise ValueError(f"PBMC X_hvg dtype changed: {matrix.dtype}")
        gene_sha = _verify_gene_order(handle, parent["gene_names"], shape[1])
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group):
            raise ValueError("PBMC H5AD lacks obs")
        source_lines, line_codes_ds = _categorical(obs, CELL_LINE_KEY)
        source_perts, pert_codes_ds = _categorical(obs, PERTURBATION_KEY)
        source_donors, donor_codes_ds = _categorical(obs, DONOR_KEY)
        for label, dataset in (
            (CELL_LINE_KEY, line_codes_ds),
            (PERTURBATION_KEY, pert_codes_ds),
            (DONOR_KEY, donor_codes_ds),
        ):
            if len(dataset) != shape[0]:
                raise ValueError(f"obs/{label} row count differs from X_hvg")
        group_remap, parent_group_donors, parent_group_celltypes = (
            _source_group_remap(
                source_donors,
                source_lines,
                lines,
                parent_grouping=str(parent["parent_grouping"]),
            )
        )
        pert_remap = _category_remap(source_perts, perturbations, "cytokine")
        heldout_donor_categories = np.asarray(
            [name in EXPECTED_HOLDOUT_DONORS for name in source_donors], dtype=bool
        )
        observed_holdouts = tuple(
            name for name, held in zip(source_donors.tolist(), heldout_donor_categories)
            if held
        )
        if observed_holdouts != EXPECTED_HOLDOUT_DONORS:
            raise ValueError(
                "source donor categories do not contain pinned holdouts in order: "
                f"{observed_holdouts}"
            )
        flat_size = len(lines) * len(perturbations)
        split_condition_counts = {
            name: np.zeros(flat_size, dtype=np.uint64)
            for name in ("control", "train", "validation", "test")
        }
        split_totals = {name: 0 for name in split_condition_counts}
        if count_rows:
            for start in range(0, shape[0], int(chunk_size)):
                stop = min(start + int(chunk_size), shape[0])
                line_codes = _checked_codes(
                    line_codes_ds[start:stop], source_lines, CELL_LINE_KEY
                )
                pert_codes = _checked_codes(
                    pert_codes_ds[start:stop], source_perts, PERTURBATION_KEY
                )
                donor_codes = _checked_codes(
                    donor_codes_ds[start:stop], source_donors, DONOR_KEY
                )
                line_ids = _source_group_ids(
                    group_remap,
                    donor_codes,
                    line_codes,
                    parent_grouping=str(parent["parent_grouping"]),
                )
                perturbation_ids = pert_remap[pert_codes]
                masks = _split_masks(
                    donor_codes,
                    perturbation_ids,
                    heldout_donor_categories=heldout_donor_categories,
                    validation_perturbations=validation_by_perturbation,
                    test_perturbations=test_by_perturbation,
                    control_id=control_id,
                )
                flat_ids = line_ids * len(perturbations) + perturbation_ids
                for name, mask in zip(
                    ("control", "train", "validation", "test"), masks
                ):
                    split_totals[name] += int(mask.sum())
                    split_condition_counts[name] += np.bincount(
                        flat_ids[mask], minlength=flat_size
                    ).astype(np.uint64)
    stat_after = source_h5ad.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError("PBMC H5AD changed during preflight")
    return {
        "shape": shape,
        "dtype": str(np.dtype(np.float32)),
        "size_bytes": int(stat_before.st_size),
        "mtime_ns": int(stat_before.st_mtime_ns),
        "gene_order_sha256": gene_sha,
        "cellline_categories_sha256": ordered_strings_sha256(source_lines),
        "perturbation_categories_sha256": ordered_strings_sha256(source_perts),
        "donor_categories_sha256": ordered_strings_sha256(source_donors),
        "source_lines": source_lines,
        "source_perts": source_perts,
        "source_donors": source_donors,
        "group_remap": group_remap,
        "parent_group_donors": parent_group_donors,
        "parent_group_celltypes": parent_group_celltypes,
        "pert_remap": pert_remap,
        "heldout_donor_categories": heldout_donor_categories,
        "validation_by_perturbation": validation_by_perturbation,
        "test_by_perturbation": test_by_perturbation,
        "split_condition_counts": split_condition_counts,
        "split_totals": split_totals,
    }


def preflight(
    parent_path: Path,
    parent_metadata_path: Path,
    source_h5ad: Path,
    *,
    chunk_size: int,
    expected_source_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for label, path in (
        ("Parent artifact", parent_path),
        ("Parent metadata", parent_metadata_path),
        ("PBMC H5AD", source_h5ad),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    parent = _checked_parent(parent_path)
    metadata = _checked_parent_metadata(
        parent_metadata_path,
        source_h5ad,
        parent_grouping=str(parent["parent_grouping"]),
    )
    if expected_source_manifest_sha256 is not None:
        expected_sha = _valid_sha256(
            expected_source_manifest_sha256, "expected_source_manifest_sha256"
        )
        if metadata["source_manifest_sha256"] != expected_sha:
            raise ValueError("Parent source-manifest SHA differs from CLI pin")
    source = _source_plan(
        source_h5ad, parent, metadata, chunk_size=chunk_size, count_rows=True
    )
    actual = source["split_totals"]
    expected = {
        "control": EXPECTED_CONTROL_ROWS,
        "train": EXPECTED_TRAIN_TREATED_ROWS,
        "validation": EXPECTED_VALIDATION_TREATED_ROWS,
        "test": EXPECTED_TEST_TREATED_ROWS,
    }
    if actual != expected:
        raise ValueError(f"PBMC source split totals changed: {actual} != {expected}")
    shape = (len(parent["cellline_names"]), len(parent["perturbation_names"]))
    control_matrix = source["split_condition_counts"]["control"].reshape(shape)
    train_matrix = source["split_condition_counts"]["train"].reshape(shape)
    validation_matrix = source["split_condition_counts"]["validation"].reshape(shape)
    test_matrix = source["split_condition_counts"]["test"].reshape(shape)
    direct_control_by_line = control_matrix.sum(axis=1)
    effective_control_by_line, _ = _effective_control_statistics(
        direct_control_by_line,
        None,
        parent=parent,
        parent_grouping=str(parent["parent_grouping"]),
        parent_group_donors=source["parent_group_donors"],
        parent_group_celltypes=source["parent_group_celltypes"],
    )
    if not np.array_equal(
        effective_control_by_line,
        parent["control_mean_source_count"].astype(np.uint64),
    ):
        raise AssertionError("effective Parent control-source verification failed")
    expected_train = np.where(
        parent["train_condition_mask"], parent["train_condition_count"], 0
    ).astype(np.uint64)
    if not np.array_equal(train_matrix, expected_train):
        where = np.argwhere(train_matrix != expected_train)
        raise ValueError(
            "PBMC source train counts differ from Parent provenance; "
            f"first_difference={where[0].tolist() if len(where) else None}"
        )
    if not np.array_equal(
        validation_matrix,
        parent["validation_heldout_cell_count"].astype(np.uint64),
    ):
        raise ValueError("PBMC validation donor×cytokine counts differ from Parent")
    if not np.array_equal(
        test_matrix, parent["test_heldout_cell_count"].astype(np.uint64)
    ):
        raise ValueError("PBMC test donor×cytokine counts differ from Parent")
    report = {
        "schema_version": "cell_detr.pbmc_train_supervision_preflight.v1",
        "passed": True,
        "split_rule": "CONTROL=PBS; VALIDATION/TEST=heldout_donor intersection heldout_cytokine; remaining treated=TRAIN",
        "source_h5ad": str(source_h5ad),
        "source_expression_key": EXPRESSION_KEY,
        "source_shape": list(source["shape"]),
        "source_size_bytes": source["size_bytes"],
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "training_expression_sha256": metadata["training_expression_sha256"],
        "parent_grouping": str(parent["parent_grouping"]),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "metadata": str(parent_metadata_path),
            "metadata_sha256": sha256_file(parent_metadata_path),
        },
        "holdout_donors": list(EXPECTED_HOLDOUT_DONORS),
        "validation_cytokines": list(metadata["validation_perturbations"]),
        "test_cytokines": list(metadata["test_perturbations"]),
        "counts": actual,
        "condition_counts": {
            "train": int(parent["train_condition_mask"].sum()),
            "validation_fully_heldout": int(
                parent["validation_condition_mask"].sum()
            ),
            "test_fully_heldout": int(parent["test_condition_mask"].sum()),
            "validation_observed": int((validation_matrix > 0).sum()),
            "test_observed": int((test_matrix > 0).sum()),
        },
        "gene_order_sha256": source["gene_order_sha256"],
        "forbidden_treated_expression_read": False,
        "expression_matrix_scanned": False,
    }
    context = {
        "parent": parent,
        "metadata": metadata,
        "source": source,
        "report": report,
    }
    return report, context


def build_de_artifact(
    parent_path: Path, output_path: Path, *, top_k: int
) -> dict[str, Any]:
    if int(top_k) != 256:
        raise ValueError("the controlled PBMC experiment pins DE top-k to 256")
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
    labels = np.zeros(train.shape + (len(genes),), dtype=bool)
    gene_ids = np.arange(len(genes), dtype=np.int64)
    for line_id in range(len(lines)):
        for perturbation_id in np.flatnonzero(train[line_id]).tolist():
            magnitude = np.abs(
                parent["train_condition_delta"][line_id, perturbation_id]
            )
            selected = np.lexsort((gene_ids, -magnitude))[: int(top_k)]
            labels[line_id, perturbation_id, selected] = True
    if labels[~train].any():
        raise AssertionError("non-train DE labels were populated")
    audit_sha = _audit_split_sha256(audit, lines, perturbations)
    support_sha = _calibration_support_sha256(support, lines, perturbations)
    metadata = {
        "schema_version": DE_SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path),
        "source_parent_artifact_sha256": parent_sha,
        "source_de_csv_sha256": {},
        "label_method": DE_LABEL_METHOD,
        "label_top_k": int(top_k),
        "fdr_threshold": None,
        "audit_split_sha256": audit_sha,
        "audit_conditions": [],
        "audit_treated_statistics_present": False,
        "validation_test_treated_statistics_present": False,
        "parent_calibration_mode": "no_intercept",
        "parent_calibration_support_sha256": support_sha,
        "parent_calibration_support_overlaps_audit": False,
        "pbmc_split_unit": "donor x cytokine cell-level intersection",
        "parent_grouping": str(parent["parent_grouping"]),
        "control_count_semantics": "Parent control_mean_source_count with pinned train-only fallback",
        "control_fallback_contexts": lines[
            parent["control_mean_fallback_mask"]
        ].tolist(),
        "heldout_donor_treated_statistics_present": False,
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
        "control_count": parent["control_mean_source_count"].astype(np.int64),
    }
    _atomic_savez(output_path, payload)
    result = {
        **metadata,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "parent_sha256": parent_sha,
        "train_conditions": int(train.sum()),
        "positive_labels": int(labels.sum()),
    }
    _atomic_json(output_path.with_suffix(".metadata.json"), result)
    return result


def _aggregate_zero_counts(
    zeros: np.ndarray,
    line_ids: np.ndarray,
    perturbation_ids: np.ndarray,
    *,
    control_id: int,
    num_lines: int,
    num_perturbations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gene_dim = int(zeros.shape[1])
    control_zero = np.zeros((num_lines, gene_dim), dtype=np.uint64)
    control_total = np.zeros(num_lines, dtype=np.uint64)
    treated_zero = np.zeros(
        (num_lines * num_perturbations, gene_dim), dtype=np.uint64
    )
    treated_total = np.zeros(num_lines * num_perturbations, dtype=np.uint64)
    control = perturbation_ids == int(control_id)
    for line_id in np.unique(line_ids[control]):
        chosen = control & (line_ids == line_id)
        control_zero[line_id] += zeros[chosen].sum(axis=0, dtype=np.uint64)
        control_total[line_id] += np.uint64(chosen.sum())
    treated = ~control
    if treated.any():
        flat_ids = (
            line_ids[treated] * int(num_perturbations)
            + perturbation_ids[treated]
        )
        order = np.argsort(flat_ids, kind="stable")
        sorted_ids = flat_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
        unique_ids = sorted_ids[starts]
        treated_zero[unique_ids] += np.add.reduceat(
            zeros[treated][order], starts, axis=0, dtype=np.uint64
        )
        treated_total[unique_ids] += np.diff(
            np.r_[starts, len(sorted_ids)]
        ).astype(np.uint64)
    return (
        control_zero,
        control_total,
        treated_zero.reshape(num_lines, num_perturbations, gene_dim),
        treated_total.reshape(num_lines, num_perturbations),
    )


def build_hurdle_artifact(
    parent_path: Path,
    parent_metadata_path: Path,
    source_h5ad: Path,
    output_path: Path,
    *,
    chunk_size: int,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    preflight_report, context = preflight(
        parent_path,
        parent_metadata_path,
        source_h5ad,
        chunk_size=max(int(chunk_size), 262_144),
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    parent = context["parent"]
    metadata = context["metadata"]
    source_plan = context["source"]
    lines = parent["cellline_names"]
    perturbations = parent["perturbation_names"]
    train_mask = parent["train_condition_mask"]
    validation_mask = parent["validation_condition_mask"]
    test_mask = parent["test_condition_mask"]
    audit_mask = np.zeros_like(train_mask, dtype=bool)
    control_id = int(parent["control_perturbation_id"].reshape(-1)[0])
    num_lines, num_perturbations = train_mask.shape
    control_zero = np.zeros((num_lines, EXPECTED_GENE_DIM), dtype=np.uint64)
    control_total = np.zeros(num_lines, dtype=np.uint64)
    treated_zero = np.zeros(
        (num_lines, num_perturbations, EXPECTED_GENE_DIM), dtype=np.uint64
    )
    treated_total = np.zeros((num_lines, num_perturbations), dtype=np.uint64)
    totals = {"control": 0, "train": 0, "validation": 0, "test": 0}
    observed_min = np.inf
    observed_max = -np.inf
    authorized_elements = 0
    chunks_read = 0
    stat_before = source_h5ad.stat()
    with h5py.File(source_h5ad, "r") as handle:
        matrix = handle[EXPRESSION_KEY]
        obs = handle["obs"]
        source_lines, line_codes_ds = _categorical(obs, CELL_LINE_KEY)
        source_perts, pert_codes_ds = _categorical(obs, PERTURBATION_KEY)
        source_donors, donor_codes_ds = _categorical(obs, DONOR_KEY)
        group_remap = source_plan["group_remap"]
        pert_remap = source_plan["pert_remap"]
        heldout_donor_categories = source_plan["heldout_donor_categories"]
        validation_by_perturbation = source_plan["validation_by_perturbation"]
        test_by_perturbation = source_plan["test_by_perturbation"]
        for start in range(0, EXPECTED_SOURCE_ROWS, int(chunk_size)):
            stop = min(start + int(chunk_size), EXPECTED_SOURCE_ROWS)
            line_codes = _checked_codes(
                line_codes_ds[start:stop], source_lines, CELL_LINE_KEY
            )
            pert_codes = _checked_codes(
                pert_codes_ds[start:stop], source_perts, PERTURBATION_KEY
            )
            donor_codes = _checked_codes(
                donor_codes_ds[start:stop], source_donors, DONOR_KEY
            )
            line_ids = _source_group_ids(
                group_remap,
                donor_codes,
                line_codes,
                parent_grouping=str(parent["parent_grouping"]),
            )
            perturbation_ids = pert_remap[pert_codes]
            control, train, validation, test = _split_masks(
                donor_codes,
                perturbation_ids,
                heldout_donor_categories=heldout_donor_categories,
                validation_perturbations=validation_by_perturbation,
                test_perturbations=test_by_perturbation,
                control_id=control_id,
            )
            for name, mask in (
                ("control", control),
                ("train", train),
                ("validation", validation),
                ("test", test),
            ):
                totals[name] += int(mask.sum())
            allowed = control | train
            # HDF5 performs a physical contiguous read.  Held-out rows that
            # share that physical block are selected out before any numerical
            # check, comparison, reduction, or statistic is computed.
            values = np.asarray(matrix[start:stop, :], dtype=np.float32)[allowed]
            chunks_read += 1
            if not np.isfinite(values).all() or (values < 0.0).any():
                raise ValueError("authorized PBMC expression is non-finite or negative")
            observed_min = min(observed_min, float(values.min()))
            observed_max = max(observed_max, float(values.max()))
            authorized_elements += int(values.size)
            local = _aggregate_zero_counts(
                values == np.float32(0.0),
                line_ids[allowed],
                perturbation_ids[allowed],
                control_id=control_id,
                num_lines=num_lines,
                num_perturbations=num_perturbations,
            )
            control_zero += local[0]
            control_total += local[1]
            treated_zero += local[2]
            treated_total += local[3]
            if (chunks_read % 100) == 0 or stop == EXPECTED_SOURCE_ROWS:
                print(
                    json.dumps(
                        {
                            "stage": "hurdle_zero_counts",
                            "chunks_read": chunks_read,
                            "rows_complete": stop,
                            "rows_total": EXPECTED_SOURCE_ROWS,
                            "control_rows": totals["control"],
                            "train_rows": totals["train"],
                            "validation_rows_skipped": totals["validation"],
                            "test_rows_skipped": totals["test"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    stat_after = source_h5ad.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError("PBMC H5AD changed during hurdle aggregation")
    expected_totals = {
        "control": EXPECTED_CONTROL_ROWS,
        "train": EXPECTED_TRAIN_TREATED_ROWS,
        "validation": EXPECTED_VALIDATION_TREATED_ROWS,
        "test": EXPECTED_TEST_TREATED_ROWS,
    }
    if totals != expected_totals:
        raise ValueError(f"hurdle split totals changed: {totals}")
    control_total, effective_control_zero = _effective_control_statistics(
        control_total,
        control_zero,
        parent=parent,
        parent_grouping=str(parent["parent_grouping"]),
        parent_group_donors=source_plan["parent_group_donors"],
        parent_group_celltypes=source_plan["parent_group_celltypes"],
    )
    if effective_control_zero is None:
        raise AssertionError("effective control-zero aggregation returned None")
    control_zero = effective_control_zero
    expected_control = parent["control_mean_source_count"].astype(np.uint64)
    expected_treated = np.where(
        train_mask, parent["train_condition_count"], 0
    ).astype(np.uint64)
    if not np.array_equal(control_total, expected_control):
        raise ValueError("hurdle effective control totals differ from Parent")
    if not np.array_equal(treated_total, expected_treated):
        where = np.argwhere(treated_total != expected_treated)
        raise ValueError(
            "hurdle train totals differ from Parent; "
            f"first_difference={where[0].tolist() if len(where) else None}"
        )
    if authorized_elements != EXPECTED_EXPRESSION_ELEMENTS:
        raise ValueError("authorized expression element count changed")
    if np.any(control_zero > control_total[:, None]):
        raise AssertionError("control zero count exceeds total")
    if np.any(treated_zero > treated_total[..., None]):
        raise AssertionError("treated zero count exceeds total")
    if np.any(treated_zero[~train_mask] != 0):
        raise AssertionError("non-train condition contains hurdle statistics")
    if (
        control_zero.max(initial=0) > np.iinfo(np.uint32).max
        or treated_zero.max(initial=0) > np.iinfo(np.uint32).max
    ):
        raise OverflowError("hurdle zero count exceeds uint32 capacity")
    parent_sha = sha256_file(parent_path)
    source_sha = metadata["source_manifest_sha256"]
    gene_sha = ordered_strings_sha256(parent["gene_names"])
    line_sha = ordered_strings_sha256(lines)
    perturbation_sha = ordered_strings_sha256(perturbations)
    artifact_metadata = {
        "schema_version": HURDLE_SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path),
        "source_parent_artifact_sha256": parent_sha,
        "source_h5ad": str(source_h5ad),
        "source_h5ad_sha256": source_sha,
        "source_h5ad_shape": [EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM],
        "source_h5ad_size_bytes": int(stat_before.st_size),
        "source_expression_key": EXPRESSION_KEY,
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_manifest_sha256": source_sha,
        "source_training_expression_sha256": metadata[
            "training_expression_sha256"
        ],
        "source_descriptor": {
            "name": source_h5ad.name,
            "size_bytes": int(stat_before.st_size),
            "mtime_ns": int(stat_before.st_mtime_ns),
            "expression_key": EXPRESSION_KEY,
            "expression_shape": [EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM],
            "cellline_categories_sha256": source_plan[
                "cellline_categories_sha256"
            ],
            "perturbation_categories_sha256": source_plan[
                "perturbation_categories_sha256"
            ],
            "donor_categories_sha256": source_plan["donor_categories_sha256"],
        },
        "normalization_divisor": 10.0,
        "parent_grouping": str(parent["parent_grouping"]),
        "control_count_semantics": "Parent control_mean_source_count with pinned train-only fallback",
        "control_fallback_contexts": lines[
            parent["control_mean_fallback_mask"]
        ].tolist(),
        "zero_definition": "exact float equality X_hvg == 0.0; division by 10 invariant",
        "gene_order_sha256": gene_sha,
        "cellline_order_sha256": line_sha,
        "perturbation_order_sha256": perturbation_sha,
        "parent_train_conditions": int(train_mask.sum()),
        "effective_train_conditions": int(train_mask.sum()),
        "audit_conditions": 0,
        "control_rows_read": totals["control"],
        "train_treated_rows_read": totals["train"],
        "validation_treated_rows_skipped": totals["validation"],
        "test_treated_rows_skipped": totals["test"],
        "audit_treated_rows_skipped": 0,
        "authorized_expression_elements": authorized_elements,
        "authorized_expression_minimum": float(observed_min),
        "authorized_expression_maximum": float(observed_max),
        "physical_hdf5_chunks_read": chunks_read,
        "validation_treated_expression_used": False,
        "test_treated_expression_used": False,
        "audit_treated_expression_used": False,
        "forbidden_rows_may_share_physical_hdf5_chunks": True,
        "split_rule": preflight_report["split_rule"],
        "holdout_donors": list(EXPECTED_HOLDOUT_DONORS),
        "validation_cytokines": preflight_report["validation_cytokines"],
        "test_cytokines": preflight_report["test_cytokines"],
    }
    payload = {
        "schema_version": _fixed_unicode(HURDLE_SCHEMA_VERSION),
        "source_parent_artifact_sha256": _fixed_unicode(parent_sha),
        "source_h5ad_sha256": _fixed_unicode(source_sha),
        "source_h5ad_shape": np.asarray(
            [EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM], dtype=np.uint64
        ),
        "source_expression_key": _fixed_unicode(EXPRESSION_KEY),
        "normalization_divisor": np.asarray(10.0, dtype=np.float32),
        "gene_names": _fixed_unicode(parent["gene_names"]),
        "cellline_names": _fixed_unicode(lines),
        "perturbation_names": _fixed_unicode(perturbations),
        "gene_order_sha256": _fixed_unicode(gene_sha),
        "cellline_order_sha256": _fixed_unicode(line_sha),
        "perturbation_order_sha256": _fixed_unicode(perturbation_sha),
        "parent_train_condition_mask": train_mask.copy(),
        "train_condition_mask": train_mask.copy(),
        "audit_condition_mask": audit_mask,
        "validation_condition_mask": validation_mask,
        "test_condition_mask": test_mask,
        "parent_train_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(train_mask)
        ),
        "train_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(train_mask)
        ),
        "audit_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(audit_mask)
        ),
        "validation_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(validation_mask)
        ),
        "test_condition_mask_sha256": _fixed_unicode(
            ndarray_sha256(test_mask)
        ),
        "control_zero_count": control_zero.astype(np.uint32),
        "control_total_count": control_total,
        "train_treated_zero_count": treated_zero.astype(np.uint32),
        "train_condition_count": treated_total,
        "metadata_json": _fixed_unicode(json.dumps(artifact_metadata, sort_keys=True)),
    }
    _atomic_savez(output_path, payload)
    result = {
        **artifact_metadata,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "parent_sha256": parent_sha,
        "source_shape": [EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM],
        "train_conditions": int(train_mask.sum()),
    }
    _atomic_json(output_path.with_suffix(".metadata.json"), result)
    return result


def verify_artifacts(
    parent_path: Path,
    parent_metadata_path: Path,
    de_path: Path,
    hurdle_path: Path,
) -> dict[str, Any]:
    for label, path in (
        ("Parent", parent_path),
        ("Parent metadata", parent_metadata_path),
        ("DE", de_path),
        ("hurdle", hurdle_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    parent = _checked_parent(parent_path)
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    source_sha = _valid_sha256(
        parent_metadata["source"]["source_manifest_sha256"],
        "source_manifest_sha256",
    )
    parent_sha = sha256_file(parent_path)
    de_sha = sha256_file(de_path)
    hurdle_sha = sha256_file(hurdle_path)
    de_bank = TrainDELabelBank(
        de_path,
        parent_path,
        expected_sha256=de_sha,
        expected_parent_sha256=parent_sha,
        expected_gene_dim=EXPECTED_GENE_DIM,
        expected_normalization_divisor=10.0,
        expected_parent_calibration="no_intercept",
    )
    hurdle_bank = TrainHurdleBank(
        artifact_path=hurdle_path,
        expected_artifact_sha256=hurdle_sha,
        parent_artifact_path=parent_path,
        expected_source_h5ad_sha256=source_sha,
        expected_source_h5ad_shape=(EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM),
        expected_gene_dim=EXPECTED_GENE_DIM,
        expected_normalization_divisor=10.0,
    )
    de_mask = de_bank.train_condition_mask.cpu().numpy()
    if not np.array_equal(de_mask, hurdle_bank.train_condition_mask):
        raise ValueError("DE and hurdle train masks differ")
    if not np.array_equal(
        hurdle_bank.control_total_count,
        parent["control_mean_source_count"].astype(np.uint64),
    ):
        raise ValueError("verified hurdle effective control counts differ from Parent")
    expected_treated = np.where(
        parent["train_condition_mask"], parent["train_condition_count"], 0
    ).astype(np.uint64)
    if not np.array_equal(hurdle_bank.train_condition_count, expected_treated):
        raise ValueError("verified hurdle train counts differ from Parent")
    with np.load(de_path, allow_pickle=False) as loaded:
        positive_labels = int(np.asarray(loaded["train_de_label"], dtype=bool).sum())
    expected_positive_labels = int(de_mask.sum()) * 256
    if positive_labels != expected_positive_labels:
        raise ValueError("DE artifact does not contain exactly top-256 labels")
    hurdle_metadata_path = hurdle_path.with_suffix(".metadata.json")
    if not hurdle_metadata_path.is_file():
        raise FileNotFoundError(hurdle_metadata_path)
    hurdle_metadata = json.loads(hurdle_metadata_path.read_text(encoding="utf-8"))
    expected_rows = {
        "control_rows_read": EXPECTED_CONTROL_ROWS,
        "train_treated_rows_read": EXPECTED_TRAIN_TREATED_ROWS,
        "validation_treated_rows_skipped": EXPECTED_VALIDATION_TREATED_ROWS,
        "test_treated_rows_skipped": EXPECTED_TEST_TREATED_ROWS,
    }
    for key, expected in expected_rows.items():
        if int(hurdle_metadata.get(key, -1)) != expected:
            raise ValueError(f"hurdle metadata {key} changed")
    return {
        "schema_version": "cell_detr.pbmc_train_supervision_manifest.v1",
        "passed": True,
        "parent": {"path": str(parent_path), "sha256": parent_sha},
        "parent_metadata": {
            "path": str(parent_metadata_path),
            "sha256": sha256_file(parent_metadata_path),
        },
        "de": {
            "path": str(de_path),
            "sha256": de_sha,
            "label_method": DE_LABEL_METHOD,
            "label_top_k": 256,
            "train_conditions": int(de_mask.sum()),
            "positive_labels": positive_labels,
        },
        "hurdle": {
            "path": str(hurdle_path),
            "sha256": hurdle_sha,
            "source_manifest_sha256": source_sha,
            "source_manifest_kind": SOURCE_MANIFEST_KIND,
            "source_shape": [EXPECTED_SOURCE_ROWS, EXPECTED_GENE_DIM],
            "train_conditions": int(hurdle_bank.train_condition_mask.sum()),
            "counts": {
                "control": EXPECTED_CONTROL_ROWS,
                "train": EXPECTED_TRAIN_TREATED_ROWS,
                "validation_skipped": EXPECTED_VALIDATION_TREATED_ROWS,
                "test_skipped": EXPECTED_TEST_TREATED_ROWS,
            },
        },
        "train_masks_equal": True,
        "parent_grouping": str(parent["parent_grouping"]),
        "control_fallback_contexts": parent["cellline_names"][
            parent["control_mean_fallback_mask"]
        ].tolist(),
        "heldout_treated_expression_used": False,
    }


def _required_output(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for this mode")
    return value.expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-metadata", type=Path, required=True)
    parser.add_argument("--source-h5ad", type=Path)
    parser.add_argument("--de-output", type=Path)
    parser.add_argument("--hurdle-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--mode", choices=("preflight", "all", "verify"), default="preflight"
    )
    parser.add_argument("--de-top-k", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--preflight-chunk-size", type=int, default=262_144)
    parser.add_argument("--expected-source-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    parent = args.parent.expanduser().resolve()
    parent_metadata = args.parent_metadata.expanduser().resolve()
    if args.mode == "preflight":
        source = _required_output(args.source_h5ad, "--source-h5ad")
        report, _ = preflight(
            parent,
            parent_metadata,
            source,
            chunk_size=args.preflight_chunk_size,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    de_output = _required_output(args.de_output, "--de-output")
    hurdle_output = _required_output(args.hurdle_output, "--hurdle-output")
    manifest = _required_output(args.manifest, "--manifest")
    if args.mode == "all":
        source = _required_output(args.source_h5ad, "--source-h5ad")
        report, _ = preflight(
            parent,
            parent_metadata,
            source,
            chunk_size=args.preflight_chunk_size,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        if not de_output.exists():
            print(json.dumps({"stage": "build_de", "output": str(de_output)}), flush=True)
            build_de_artifact(parent, de_output, top_k=args.de_top_k)
        if not hurdle_output.exists():
            print(
                json.dumps(
                    {"stage": "build_hurdle", "output": str(hurdle_output)}
                ),
                flush=True,
            )
            build_hurdle_artifact(
                parent,
                parent_metadata,
                source,
                hurdle_output,
                chunk_size=args.chunk_size,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            )
    result = verify_artifacts(parent, parent_metadata, de_output, hurdle_output)
    _atomic_json(manifest, result, accept_identical=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict, train-only exact-zero statistics for a future hurdle head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "perturbdiff.train_hurdle_bank.v1"
REPLOGLE_LINE_COUNT = 4
REPLOGLE_NORMALIZATION_DIVISOR = 10.0


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA256 of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(block_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ordered_strings_sha256(values: Sequence[str] | np.ndarray) -> str:
    """Hash an ordered string sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    flat = np.asarray(values).astype(str).reshape(-1)
    digest.update(int(len(flat)).to_bytes(8, byteorder="little", signed=False))
    for value in flat.tolist():
        encoded = str(value).encode("utf-8")
        digest.update(int(len(encoded)).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def ndarray_sha256(value: np.ndarray) -> str:
    """Hash shape, canonical dtype, and contiguous array bytes."""

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _required(archive: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in archive:
        raise KeyError(f"TrainHurdleBank artifact is missing {key!r}")
    return np.asarray(archive[key])


def _scalar_string(value: np.ndarray, name: str) -> str:
    flat = np.asarray(value).reshape(-1)
    if flat.shape != (1,):
        raise ValueError(f"{name} must contain exactly one string")
    return str(flat[0])


def _validate_names(values: np.ndarray, name: str) -> np.ndarray:
    names = np.asarray(values).astype(str)
    if names.ndim != 1 or not len(names):
        raise ValueError(f"{name} must be a non-empty rank-one array")
    if any(not value for value in names.tolist()):
        raise ValueError(f"{name} contains an empty value")
    if len(set(names.tolist())) != len(names):
        raise ValueError(f"{name} must be unique")
    return names


def _require_integer_counts(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    if np.issubdtype(array.dtype, np.signedinteger) and (array < 0).any():
        raise ValueError(f"{name} contains a negative count")
    return array.astype(np.uint64, copy=False)


def _parent_effective_control_count(
    parent: Mapping[str, np.ndarray], expected_lines: int
) -> np.ndarray:
    """Resolve direct or explicitly pinned fallback Parent control counts."""

    direct = _require_integer_counts(
        _required(parent, "control_count"), "Parent control_count"
    )
    if direct.shape != (int(expected_lines),):
        raise ValueError("Parent control_count must have shape [L]")
    files = set(getattr(parent, "files", parent.keys()))
    has_source = "control_mean_source_count" in files
    has_fallback = "control_mean_fallback_mask" in files
    if has_source != has_fallback:
        raise ValueError("Parent control fallback provenance is incomplete")
    if not has_source:
        return direct
    source = _require_integer_counts(
        _required(parent, "control_mean_source_count"),
        "Parent control_mean_source_count",
    )
    fallback = np.asarray(
        _required(parent, "control_mean_fallback_mask"), dtype=bool
    )
    if source.shape != direct.shape or fallback.shape != direct.shape:
        raise ValueError("Parent control fallback arrays must have shape [L]")
    if np.any(source < direct):
        raise ValueError("Parent effective control count is below direct count")
    if np.any(~fallback & (source != direct)):
        raise ValueError("Parent non-fallback control count differs from direct count")
    if np.any(fallback & (direct != 0)):
        raise ValueError("Parent fallback mask must identify zero direct controls")
    return source


def _validate_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} is not a lowercase SHA256")
    return normalized


def _source_expression_shape(path: Path, expression_key: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        candidate = expression_key.strip("/")
        if candidate in handle:
            matrix = handle[candidate]
        elif "/" not in candidate and "obsm" in handle and candidate in handle["obsm"]:
            matrix = handle["obsm"][candidate]
        else:
            raise KeyError(f"source H5AD has no expression dataset {expression_key!r}")
        if not isinstance(matrix, h5py.Dataset) or matrix.ndim != 2:
            raise ValueError("source expression dataset must be rank two")
        return int(matrix.shape[0]), int(matrix.shape[1])


@dataclass(frozen=True)
class TrainHurdleLookup:
    """Exact-zero sufficient statistics for one legal training condition."""

    line_id: int
    perturbation_id: int
    cell_line: str
    perturbation: str
    control_zero_count: np.ndarray
    control_total_count: int
    treated_zero_count: np.ndarray
    treated_total_count: int
    control_zero_rate: np.ndarray
    treated_zero_rate: np.ndarray


class TrainHurdleBank:
    """Validated, immutable loader for train-only hurdle sufficient statistics."""

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        expected_artifact_sha256: str,
        parent_artifact_path: str | Path,
        source_h5ad_path: str | Path | None = None,
        expected_source_h5ad_sha256: str | None = None,
        expected_source_h5ad_shape: Sequence[int] | None = None,
        expected_gene_dim: int | None = None,
        expected_normalization_divisor: float = REPLOGLE_NORMALIZATION_DIVISOR,
    ) -> None:
        artifact_path = Path(artifact_path).expanduser().resolve()
        parent_path = Path(parent_artifact_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)

        expected_artifact = _validate_sha256(
            expected_artifact_sha256, "expected_artifact_sha256"
        )
        artifact_sha = sha256_file(artifact_path)
        if artifact_sha != expected_artifact:
            raise ValueError(
                "TrainHurdleBank artifact SHA256 mismatch: "
                f"expected {expected_artifact}, got {artifact_sha}"
            )
        parent_sha = sha256_file(parent_path)

        with np.load(parent_path, allow_pickle=False) as parent, np.load(
            artifact_path, allow_pickle=False
        ) as loaded:
            for archive_name, archive in (("Parent", parent), ("TrainHurdleBank", loaded)):
                for key in archive.files:
                    if archive[key].dtype == object:
                        raise TypeError(
                            f"unsafe object dtype in {archive_name} artifact key {key!r}"
                        )

            schema = _scalar_string(_required(loaded, "schema_version"), "schema_version")
            if schema != SCHEMA_VERSION:
                raise ValueError(f"unsupported TrainHurdleBank schema {schema!r}")
            recorded_parent_sha = _validate_sha256(
                _scalar_string(
                    _required(loaded, "source_parent_artifact_sha256"),
                    "source_parent_artifact_sha256",
                ),
                "source_parent_artifact_sha256",
            )
            if recorded_parent_sha != parent_sha:
                raise ValueError(
                    "TrainHurdleBank was built from a different Parent artifact: "
                    f"recorded {recorded_parent_sha}, runtime {parent_sha}"
                )

            genes = _validate_names(_required(loaded, "gene_names"), "gene_names")
            lines = _validate_names(
                _required(loaded, "cellline_names"), "cellline_names"
            )
            perturbations = _validate_names(
                _required(loaded, "perturbation_names"), "perturbation_names"
            )
            parent_genes = _validate_names(
                _required(parent, "gene_names"), "Parent gene_names"
            )
            parent_lines = _validate_names(
                _required(parent, "cellline_names"), "Parent cellline_names"
            )
            parent_perturbations = _validate_names(
                _required(parent, "perturbation_names"),
                "Parent perturbation_names",
            )
            if not np.array_equal(genes, parent_genes):
                raise ValueError("TrainHurdleBank/Parent gene order mismatch")
            if not np.array_equal(lines, parent_lines):
                raise ValueError("TrainHurdleBank/Parent cell-line order mismatch")
            if not np.array_equal(perturbations, parent_perturbations):
                raise ValueError("TrainHurdleBank/Parent perturbation order mismatch")
            if expected_gene_dim is not None and len(genes) != int(expected_gene_dim):
                raise ValueError(
                    f"TrainHurdleBank gene dimension {len(genes)} != {int(expected_gene_dim)}"
                )

            for key, values in (
                ("gene_order_sha256", genes),
                ("cellline_order_sha256", lines),
                ("perturbation_order_sha256", perturbations),
            ):
                recorded = _validate_sha256(
                    _scalar_string(_required(loaded, key), key), key
                )
                computed = ordered_strings_sha256(values)
                if recorded != computed:
                    raise ValueError(
                        f"{key} mismatch: recorded {recorded}, computed {computed}"
                    )

            divisor = float(
                np.asarray(_required(loaded, "normalization_divisor")).reshape(-1)[0]
            )
            parent_divisor = float(
                np.asarray(_required(parent, "normalization_divisor")).reshape(-1)[0]
            )
            expected_divisor = float(expected_normalization_divisor)
            if not (
                np.isfinite(divisor)
                and np.isclose(divisor, REPLOGLE_NORMALIZATION_DIVISOR)
                and np.isclose(divisor, parent_divisor)
                and np.isclose(divisor, expected_divisor)
            ):
                raise ValueError(
                    "TrainHurdleBank/Parent/runtime normalization mismatch: "
                    f"{divisor}, {parent_divisor}, {expected_divisor}"
                )

            condition_shape = (len(lines), len(perturbations))
            parent_train = np.asarray(
                _required(parent, "train_condition_mask"), dtype=bool
            )
            parent_validation = np.asarray(
                _required(parent, "validation_condition_mask"), dtype=bool
            )
            parent_test = np.asarray(
                _required(parent, "test_condition_mask"), dtype=bool
            )
            recorded_parent_train = np.asarray(
                _required(loaded, "parent_train_condition_mask"), dtype=bool
            )
            train_mask = np.asarray(
                _required(loaded, "train_condition_mask"), dtype=bool
            )
            audit_mask = np.asarray(
                _required(loaded, "audit_condition_mask"), dtype=bool
            )
            validation_mask = np.asarray(
                _required(loaded, "validation_condition_mask"), dtype=bool
            )
            test_mask = np.asarray(
                _required(loaded, "test_condition_mask"), dtype=bool
            )
            for name, mask in (
                ("Parent train", parent_train),
                ("Parent validation", parent_validation),
                ("Parent test", parent_test),
                ("recorded Parent train", recorded_parent_train),
                ("train", train_mask),
                ("audit", audit_mask),
                ("validation", validation_mask),
                ("test", test_mask),
            ):
                if mask.shape != condition_shape:
                    raise ValueError(f"{name} mask must have shape [L,P]")
            if np.any(parent_train & (parent_validation | parent_test)):
                raise ValueError("Parent train and validation/test masks overlap")
            if np.any(parent_validation & parent_test):
                raise ValueError("Parent validation and test masks overlap")
            if not np.array_equal(recorded_parent_train, parent_train):
                raise ValueError("recorded Parent train mask differs from Parent artifact")
            if not np.array_equal(validation_mask, parent_validation):
                raise ValueError("validation mask differs from Parent artifact")
            if not np.array_equal(test_mask, parent_test):
                raise ValueError("test mask differs from Parent artifact")
            if np.any(audit_mask & ~parent_train):
                raise ValueError("audit mask must be a subset of Parent train")
            if np.any(audit_mask & (validation_mask | test_mask)):
                raise ValueError("audit and validation/test masks overlap")
            if not np.array_equal(train_mask, parent_train & ~audit_mask):
                raise ValueError("effective train mask must equal Parent train minus audit")
            if np.any(train_mask & (audit_mask | validation_mask | test_mask)):
                raise ValueError("effective train mask contains a forbidden condition")

            for key, mask in (
                ("parent_train_condition_mask_sha256", parent_train),
                ("train_condition_mask_sha256", train_mask),
                ("audit_condition_mask_sha256", audit_mask),
                ("validation_condition_mask_sha256", validation_mask),
                ("test_condition_mask_sha256", test_mask),
            ):
                recorded = _validate_sha256(
                    _scalar_string(_required(loaded, key), key), key
                )
                computed = ndarray_sha256(mask)
                if recorded != computed:
                    raise ValueError(
                        f"{key} mismatch: recorded {recorded}, computed {computed}"
                    )

            control_total = _require_integer_counts(
                _required(loaded, "control_total_count"), "control_total_count"
            )
            control_zero = _require_integer_counts(
                _required(loaded, "control_zero_count"), "control_zero_count"
            )
            treated_total = _require_integer_counts(
                _required(loaded, "train_condition_count"),
                "train_condition_count",
            )
            treated_zero = _require_integer_counts(
                _required(loaded, "train_treated_zero_count"),
                "train_treated_zero_count",
            )
            if control_total.shape != (len(lines),):
                raise ValueError("control_total_count must have shape [L]")
            if control_zero.shape != (len(lines), len(genes)):
                raise ValueError("control_zero_count must have shape [L,G]")
            if treated_total.shape != condition_shape:
                raise ValueError("train_condition_count must have shape [L,P]")
            if treated_zero.shape != condition_shape + (len(genes),):
                raise ValueError("train_treated_zero_count must have shape [L,P,G]")
            if (control_total == 0).any():
                raise ValueError("every Parent cell line must have control cells")
            if np.any(control_zero > control_total[:, None]):
                raise ValueError("control zero count exceeds the line total")
            if np.any(treated_zero > treated_total[..., None]):
                raise ValueError("treated zero count exceeds the condition total")
            if np.any(treated_total[~train_mask] != 0):
                raise ValueError("forbidden condition contains a treated total")
            if np.any(treated_zero[~train_mask] != 0):
                raise ValueError("forbidden condition contains treated zero statistics")
            if np.any(treated_total[train_mask] == 0):
                raise ValueError("effective training condition has no treated cells")
            if not np.array_equal(treated_total > 0, train_mask):
                raise ValueError("treated count availability differs from train mask")

            parent_control_total = _parent_effective_control_count(
                parent, len(lines)
            )
            parent_treated_total = _require_integer_counts(
                _required(parent, "train_condition_count"),
                "Parent train_condition_count",
            )
            if not np.array_equal(control_total, parent_control_total):
                raise ValueError(
                    "effective control-source totals differ from the Parent artifact"
                )
            expected_treated_total = np.where(train_mask, parent_treated_total, 0)
            if not np.array_equal(treated_total, expected_treated_total):
                raise ValueError("treated totals differ from Parent train provenance")

            control_rate = control_zero.astype(np.float64) / control_total[:, None]
            treated_rate = np.zeros(treated_zero.shape, dtype=np.float64)
            np.divide(
                treated_zero,
                treated_total[..., None],
                out=treated_rate,
                where=train_mask[..., None],
            )
            if not np.isfinite(control_rate).all() or not np.isfinite(
                treated_rate[train_mask]
            ).all():
                raise ValueError("derived exact-zero rates must be finite")
            if np.any((control_rate < 0.0) | (control_rate > 1.0)):
                raise ValueError("derived control exact-zero rate falls outside [0,1]")
            if np.any(
                (treated_rate[train_mask] < 0.0)
                | (treated_rate[train_mask] > 1.0)
            ):
                raise ValueError("derived treated exact-zero rate falls outside [0,1]")

            source_sha = _validate_sha256(
                _scalar_string(
                    _required(loaded, "source_h5ad_sha256"), "source_h5ad_sha256"
                ),
                "source_h5ad_sha256",
            )
            source_shape_array = _require_integer_counts(
                _required(loaded, "source_h5ad_shape"), "source_h5ad_shape"
            )
            if source_shape_array.shape != (2,) or (source_shape_array == 0).any():
                raise ValueError("source_h5ad_shape must contain positive [N,G]")
            source_shape = tuple(int(value) for value in source_shape_array.tolist())
            if source_shape[1] != len(genes):
                raise ValueError("source H5AD width differs from Parent gene dimension")
            expression_key = _scalar_string(
                _required(loaded, "source_expression_key"), "source_expression_key"
            )
            metadata_raw = _scalar_string(
                _required(loaded, "metadata_json"), "metadata_json"
            )
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError as error:
                raise ValueError("metadata_json is invalid") from error
            if not isinstance(metadata, Mapping):
                raise ValueError("metadata_json must decode to a mapping")
            expected_metadata = {
                "schema_version": schema,
                "source_parent_artifact_sha256": parent_sha,
                "source_h5ad_sha256": source_sha,
                "source_h5ad_shape": list(source_shape),
                "normalization_divisor": divisor,
                "gene_order_sha256": ordered_strings_sha256(genes),
                "cellline_order_sha256": ordered_strings_sha256(lines),
                "perturbation_order_sha256": ordered_strings_sha256(perturbations),
            }
            for key, expected_value in expected_metadata.items():
                if metadata.get(key) != expected_value:
                    raise ValueError(f"metadata provenance mismatch for {key!r}")
            for key in (
                "validation_treated_expression_used",
                "test_treated_expression_used",
                "audit_treated_expression_used",
            ):
                if metadata.get(key) is not False:
                    raise ValueError(f"leakage provenance flag {key!r} is not false")

            # Copy while the NPZ handles are open.  Stored arrays remain integer
            # counts; rates are derived only after all safety checks pass.
            control_total_copy = np.asarray(control_total, dtype=np.uint64).copy()
            control_zero_copy = np.asarray(control_zero, dtype=np.uint64).copy()
            treated_total_copy = np.asarray(treated_total, dtype=np.uint64).copy()
            treated_zero_copy = np.asarray(treated_zero, dtype=np.uint64).copy()

        configured_source_sha = (
            _validate_sha256(
                expected_source_h5ad_sha256, "expected_source_h5ad_sha256"
            )
            if expected_source_h5ad_sha256 is not None
            else None
        )
        if configured_source_sha is not None and configured_source_sha != source_sha:
            raise ValueError(
                "source H5AD SHA256 mismatch: "
                f"expected {configured_source_sha}, artifact records {source_sha}"
            )
        if expected_source_h5ad_shape is not None:
            configured_shape = tuple(int(value) for value in expected_source_h5ad_shape)
            if configured_shape != source_shape:
                raise ValueError(
                    f"source H5AD shape mismatch: {configured_shape} vs {source_shape}"
                )

        resolved_source_path: Path | None = None
        if source_h5ad_path is not None:
            resolved_source_path = Path(source_h5ad_path).expanduser().resolve()
            if not resolved_source_path.is_file():
                raise FileNotFoundError(resolved_source_path)
            actual_source_sha = sha256_file(resolved_source_path)
            if actual_source_sha != source_sha:
                raise ValueError(
                    "runtime source H5AD differs from artifact provenance: "
                    f"recorded {source_sha}, actual {actual_source_sha}"
                )
            actual_source_shape = _source_expression_shape(
                resolved_source_path, expression_key
            )
            if actual_source_shape != source_shape:
                raise ValueError(
                    "runtime source H5AD expression shape differs from provenance: "
                    f"recorded {source_shape}, actual {actual_source_shape}"
                )
        elif configured_source_sha is None:
            raise ValueError(
                "strict provenance requires source_h5ad_path or "
                "expected_source_h5ad_sha256"
            )

        for array in (
            genes,
            lines,
            perturbations,
            train_mask,
            audit_mask,
            validation_mask,
            test_mask,
            control_total_copy,
            control_zero_copy,
            treated_total_copy,
            treated_zero_copy,
            control_rate,
            treated_rate,
        ):
            array.setflags(write=False)

        self.artifact_path = str(artifact_path)
        self.artifact_sha256 = artifact_sha
        self.parent_artifact_path = str(parent_path)
        self.parent_artifact_sha256 = parent_sha
        self.source_h5ad_path = (
            str(resolved_source_path) if resolved_source_path is not None else None
        )
        self.source_h5ad_sha256 = source_sha
        self.source_h5ad_shape = source_shape
        self.source_expression_key = expression_key
        self.normalization_divisor = divisor
        self.gene_names = genes
        self.cellline_names = lines
        self.perturbation_names = perturbations
        self.train_condition_mask = train_mask
        self.audit_condition_mask = audit_mask
        self.validation_condition_mask = validation_mask
        self.test_condition_mask = test_mask
        self.control_total_count = control_total_copy
        self.control_zero_count = control_zero_copy
        self.train_condition_count = treated_total_copy
        self.train_treated_zero_count = treated_zero_copy
        self.control_zero_rate = control_rate
        self.train_treated_zero_rate = treated_rate
        self.metadata = dict(metadata)

    @staticmethod
    def _checked_index(value: int, size: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer scalar")
        index = int(value)
        if index < 0 or index >= int(size):
            raise IndexError(f"{name}={index} falls outside [0,{int(size)})")
        return index

    def lookup(self, line_id: int, perturbation_id: int) -> TrainHurdleLookup:
        """Return statistics only for an effective training condition.

        The availability mask is checked before indexing either treated count
        array.  This is intentionally stricter than returning an all-zero row:
        held-out and unavailable conditions are programmer errors.
        """

        line = self._checked_index(line_id, len(self.cellline_names), "line_id")
        perturbation = self._checked_index(
            perturbation_id,
            len(self.perturbation_names),
            "perturbation_id",
        )
        if not bool(self.train_condition_mask[line, perturbation]):
            if self.audit_condition_mask[line, perturbation]:
                reason = "audit"
            elif self.validation_condition_mask[line, perturbation]:
                reason = "validation"
            elif self.test_condition_mask[line, perturbation]:
                reason = "test"
            else:
                reason = "non-train"
            raise PermissionError(
                "TrainHurdleBank lookup denied for "
                f"{reason} condition ({self.cellline_names[line]!r}, "
                f"{self.perturbation_names[perturbation]!r})"
            )

        control_zero = self.control_zero_count[line].copy()
        treated_zero = self.train_treated_zero_count[line, perturbation].copy()
        control_rate = self.control_zero_rate[line].copy()
        treated_rate = self.train_treated_zero_rate[line, perturbation].copy()
        for array in (control_zero, treated_zero, control_rate, treated_rate):
            array.setflags(write=False)
        return TrainHurdleLookup(
            line_id=line,
            perturbation_id=perturbation,
            cell_line=str(self.cellline_names[line]),
            perturbation=str(self.perturbation_names[perturbation]),
            control_zero_count=control_zero,
            control_total_count=int(self.control_total_count[line]),
            treated_zero_count=treated_zero,
            treated_total_count=int(self.train_condition_count[line, perturbation]),
            control_zero_rate=control_rate,
            treated_zero_rate=treated_rate,
        )


__all__ = [
    "REPLOGLE_LINE_COUNT",
    "REPLOGLE_NORMALIZATION_DIVISOR",
    "SCHEMA_VERSION",
    "TrainHurdleBank",
    "TrainHurdleLookup",
    "ndarray_sha256",
    "ordered_strings_sha256",
    "sha256_file",
]

"""Strict physical-source-row identity utilities for PBMC evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd


PROVENANCE_SCHEMA_VERSION = "pbmc-physical-row-provenance/v1"
STAGE_IDENTITY_SCHEMA_VERSION = "pbmc-stage-physical-identity/v1"
IDENTITY_SEMANTICS = "physical_source_h5_row"
IDENTITY_COLUMNS = (
    "source_namespace",
    "source_dataset_name",
    "source_physical_row",
    "source_obs_name",
    "source_role",
)
SOURCE_FILE_COLUMN = "source_file"
TREATED_ROLE = "treated_target"
REFERENCE_ROLE = "reference_control"
VALID_ROLES = frozenset({TREATED_ROLE, REFERENCE_ROLE})
LOCATOR_COLUMNS = (
    "source_namespace",
    "source_dataset_name",
    "source_physical_row",
)


def _decode_h5_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _take_h5_vector(
    node: h5py.Dataset | h5py.Group, rows: np.ndarray
) -> np.ndarray:
    """Read arbitrary rows from an AnnData vector without loading all rows."""

    unique_rows, inverse = np.unique(
        rows.astype(np.int64, copy=False), return_inverse=True
    )
    if isinstance(node, h5py.Dataset):
        values = node[unique_rows.tolist()]
    elif isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][unique_rows.tolist()], dtype=np.int64)
        if np.any(codes < 0):
            raise ValueError("source obs index contains missing categorical codes")
        values = np.asarray(node["categories"])[codes]
    else:
        raise ValueError("unsupported source obs-index encoding")
    return np.asarray(
        [_decode_h5_text(value) for value in np.asarray(values)[inverse]],
        dtype=object,
    )


def read_source_obs_names(
    handle: h5py.File, rows: Sequence[int]
) -> np.ndarray:
    """Read actual source AnnData obs names for physical H5 row indices."""

    physical_rows = np.asarray(rows, dtype=np.int64)
    if physical_rows.ndim != 1 or np.any(physical_rows < 0):
        raise ValueError("source physical rows must be a non-negative 1-D array")
    if "obs" not in handle:
        raise ValueError("source file has no AnnData obs group")
    obs = handle["obs"]
    index_key = _decode_h5_text(obs.attrs.get("_index", "_index"))
    if index_key not in obs:
        raise ValueError(f"source file has no declared obs index {index_key!r}")
    node = obs[index_key]
    size = len(node) if isinstance(node, h5py.Dataset) else len(node["codes"])
    if physical_rows.size and int(physical_rows.max()) >= size:
        raise ValueError("source physical row is outside the source H5AD")
    return _take_h5_vector(node, physical_rows)


class PhysicalIdentityError(ValueError):
    """Raised when physical-row provenance cannot be established exactly."""


class _HashingTextSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self.digest.update(encoded)
        return len(value)


def dataframe_sha256(frame: pd.DataFrame) -> str:
    sink = _HashingTextSink()
    frame.to_csv(sink, index=False, lineterminator="\n")
    return sink.digest.hexdigest()


def multiset_frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Duplicate-sensitive and row-order-independent lineage fingerprint."""

    modulus = 1 << 256
    total = 0
    total_squared = 0
    xor_value = 0
    columns = tuple(map(str, frame.columns))
    for row in frame.astype("string").itertuples(index=False, name=None):
        digest = hashlib.sha256()
        for column, value in zip(columns, row):
            for item in (column, str(value)):
                encoded = item.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little", signed=False))
                digest.update(encoded)
        integer = int.from_bytes(digest.digest(), "big", signed=False)
        total = (total + integer) % modulus
        total_squared = (total_squared + integer * integer) % modulus
        xor_value ^= integer
    payload = {
        "columns": list(columns),
        "rows": int(len(frame)),
        "sum_mod_2_256": f"{total:064x}",
        "sum_squares_mod_2_256": f"{total_squared:064x}",
        "xor": f"{xor_value:064x}",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def combine_multiset_fingerprints(
    fingerprints: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not fingerprints:
        return None
    columns = list(fingerprints[0]["columns"])
    modulus = 1 << 256
    total = 0
    total_squared = 0
    xor_value = 0
    rows = 0
    for fingerprint in fingerprints:
        if list(fingerprint.get("columns", [])) != columns:
            raise PhysicalIdentityError(
                "cannot combine lineage fingerprints with different columns"
            )
        rows += int(fingerprint["rows"])
        total = (total + int(str(fingerprint["sum_mod_2_256"]), 16)) % modulus
        total_squared = (
            total_squared
            + int(str(fingerprint["sum_squares_mod_2_256"]), 16)
        ) % modulus
        xor_value ^= int(str(fingerprint["xor"]), 16)
    payload = {
        "columns": columns,
        "rows": rows,
        "sum_mod_2_256": f"{total:064x}",
        "sum_squares_mod_2_256": f"{total_squared:064x}",
        "xor": f"{xor_value:064x}",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def canonical_source_namespace(
    *, kind: str, dataset_name: str, source_file: str | Path
) -> str:
    resolved = str(Path(source_file).expanduser().resolve(strict=True))
    return f"{kind}://{dataset_name}?path={resolved}"


def _nonempty_strings(series: pd.Series, *, column: str, context: str) -> pd.Series:
    if bool(series.isna().any()):
        raise PhysicalIdentityError(f"{context}: {column} contains missing values")
    result = series.astype("string")
    empty = result.str.len().fillna(0) == 0
    if bool(empty.any()):
        raise PhysicalIdentityError(
            f"{context}: {column} contains {int(empty.sum())} empty values"
        )
    return result


def normalized_identity_frame(
    obs: pd.DataFrame,
    *,
    group_key: str,
    reference: str,
    context: str,
    expected_role: str | None = None,
    require_source_file: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalize one H5AD ``obs`` physical identity table."""

    if expected_role is not None and expected_role not in VALID_ROLES:
        raise ValueError(f"unsupported expected PBMC source role: {expected_role!r}")
    required = [*IDENTITY_COLUMNS, group_key]
    if require_source_file:
        required.append(SOURCE_FILE_COLUMN)
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise PhysicalIdentityError(
            f"{context}: missing physical identity columns {missing}"
        )

    frame = obs.loc[:, required].copy()
    for column in (
        "source_namespace",
        "source_dataset_name",
        "source_obs_name",
        "source_role",
        group_key,
    ):
        frame[column] = _nonempty_strings(
            frame[column], column=column, context=context
        )
    if require_source_file:
        frame[SOURCE_FILE_COLUMN] = _nonempty_strings(
            frame[SOURCE_FILE_COLUMN], column=SOURCE_FILE_COLUMN, context=context
        )

    numeric = pd.to_numeric(frame["source_physical_row"], errors="coerce")
    if bool(numeric.isna().any()):
        raise PhysicalIdentityError(
            f"{context}: source_physical_row contains non-numeric values"
        )
    as_float = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(as_float).all():
        raise PhysicalIdentityError(
            f"{context}: source_physical_row contains non-finite values"
        )
    rounded = np.rint(as_float)
    if not np.array_equal(as_float, rounded) or np.any(rounded < 0):
        raise PhysicalIdentityError(
            f"{context}: source_physical_row must contain non-negative integers"
        )
    frame["source_physical_row"] = rounded.astype(np.int64)

    roles = set(map(str, frame["source_role"].unique()))
    unexpected_roles = roles - VALID_ROLES
    if unexpected_roles:
        raise PhysicalIdentityError(
            f"{context}: non-canonical source_role values {sorted(unexpected_roles)}; "
            f"expected {sorted(VALID_ROLES)}"
        )
    if expected_role is not None and roles != {expected_role}:
        raise PhysicalIdentityError(
            f"{context}: expected only source_role={expected_role!r}, "
            f"observed={sorted(roles)}"
        )

    is_reference = frame[group_key].astype(str).to_numpy() == str(reference)
    is_reference_role = (
        frame["source_role"].astype(str).to_numpy() == REFERENCE_ROLE
    )
    if not np.array_equal(is_reference, is_reference_role):
        mismatches = int(np.count_nonzero(is_reference != is_reference_role))
        raise PhysicalIdentityError(
            f"{context}: source_role/group relation is invalid for {mismatches} rows; "
            f"{reference!r} must be {REFERENCE_ROLE!r} and every target must be "
            f"{TREATED_ROLE!r}"
        )

    locator = frame.loc[:, list(LOCATOR_COLUMNS)]
    duplicate_locator = locator.duplicated(keep=False)
    if bool(duplicate_locator.any()):
        duplicate_keys = int(len(locator) - len(locator.drop_duplicates()))
        raise PhysicalIdentityError(
            f"{context}: {duplicate_keys} physical source locator keys are duplicated "
            f"across {int(duplicate_locator.sum())} rows"
        )

    binding_columns = ["source_namespace", "source_dataset_name"]
    if require_source_file:
        binding_columns.append(SOURCE_FILE_COLUMN)
    bindings = frame.loc[:, binding_columns].drop_duplicates()
    namespace_counts = bindings.groupby("source_namespace", observed=True).size()
    if bool((namespace_counts != 1).any()):
        bad = sorted(map(str, namespace_counts.index[namespace_counts != 1]))
        raise PhysicalIdentityError(
            f"{context}: source namespace has conflicting dataset/file bindings: {bad}"
        )

    identity = frame.loc[:, list(IDENTITY_COLUMNS)].astype("string")
    identity_group = identity.copy()
    identity_group[group_key] = frame[group_key].astype("string").to_numpy()
    report = {
        "schema_version": STAGE_IDENTITY_SCHEMA_VERSION,
        "passed": True,
        "identity_semantics": IDENTITY_SEMANTICS,
        "identity_columns": list(IDENTITY_COLUMNS),
        "source_file_column": SOURCE_FILE_COLUMN if require_source_file else None,
        "rows": int(len(frame)),
        "physical_locator_unique": True,
        "role_group_relation_verified": True,
        "role_counts": {
            str(key): int(value)
            for key, value in frame["source_role"].value_counts().sort_index().items()
        },
        "ordered_identity_sha256": dataframe_sha256(identity),
        "ordered_identity_group_sha256": dataframe_sha256(identity_group),
        "identity_group_multiset": multiset_frame_fingerprint(identity_group),
        "source_bindings": bindings.sort_values(
            ["source_namespace", "source_dataset_name"], kind="stable"
        ).to_dict(orient="records"),
    }
    return frame, report


def assert_paired_identity_equal(
    real_obs: pd.DataFrame,
    pred_obs: pd.DataFrame,
    *,
    group_key: str,
    reference: str,
    context: str,
    expected_role: str | None = None,
) -> dict[str, Any]:
    real, real_report = normalized_identity_frame(
        real_obs,
        group_key=group_key,
        reference=reference,
        context=f"{context} real",
        expected_role=expected_role,
    )
    pred, pred_report = normalized_identity_frame(
        pred_obs,
        group_key=group_key,
        reference=reference,
        context=f"{context} pred",
        expected_role=expected_role,
    )
    compare_columns = [*IDENTITY_COLUMNS, SOURCE_FILE_COLUMN, group_key]
    if len(real) != len(pred) or not real.loc[:, compare_columns].equals(
        pred.loc[:, compare_columns]
    ):
        raise PhysicalIdentityError(
            f"{context}: ordered real/pred physical identities or group labels differ"
        )
    return {
        "schema_version": STAGE_IDENTITY_SCHEMA_VERSION,
        "passed": True,
        "paired_order_identical": True,
        "real": real_report,
        "pred": pred_report,
    }


def attach_control_source_identity(
    obs: pd.DataFrame,
    *,
    source_path: str | Path,
    dataset_name: str,
    reference: str,
) -> None:
    """Attach identities while ``obs`` still has the source H5AD row order."""

    resolved = str(Path(source_path).expanduser().resolve(strict=True))
    names = np.asarray(obs.index.astype(str), dtype=object)
    rows = np.arange(len(obs), dtype=np.int64)
    namespace = canonical_source_namespace(
        kind="control-h5ad", dataset_name=str(dataset_name), source_file=resolved
    )
    obs["source_namespace"] = np.repeat(namespace, len(obs))
    obs["source_dataset_name"] = np.repeat(str(dataset_name), len(obs))
    obs["source_physical_row"] = rows
    obs["source_obs_name"] = names
    obs["source_role"] = np.repeat(REFERENCE_ROLE, len(obs))
    obs[SOURCE_FILE_COLUMN] = np.repeat(resolved, len(obs))
    if "cytokine" in obs.columns:
        labels = obs["cytokine"].astype(str).to_numpy()
        if not np.all(labels == str(reference)):
            raise PhysicalIdentityError(
                "canonical control source contains a non-reference cytokine"
            )


def provenance_uns_payload(source_bindings: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": STAGE_IDENTITY_SCHEMA_VERSION,
        "identity_semantics": IDENTITY_SEMANTICS,
        "identity_columns": list(IDENTITY_COLUMNS),
        "source_file_column": SOURCE_FILE_COLUMN,
        "source_bindings": [dict(value) for value in source_bindings],
        "historical_output_order_inference_allowed": False,
        "canonical_roles": {
            "treated": TREATED_ROLE,
            "reference": REFERENCE_ROLE,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

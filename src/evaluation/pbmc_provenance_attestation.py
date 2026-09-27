"""Verification of PBMC pooled physical-row mapping attestations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.pbmc_physical_identity import (
    IDENTITY_COLUMNS,
    REFERENCE_ROLE,
    SOURCE_FILE_COLUMN,
    TREATED_ROLE,
    PhysicalIdentityError,
    normalized_identity_frame,
)


MAPPING_SCHEMA_VERSION = "pbmc-physical-source-row-mapping/v1"
MAPPING_KIND = "pbmc_physical_source_row_mapping"
BUILDER_SCHEMA_VERSION = "pbmc-pooled-input-builder/v1"


class _HashingTextSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self.digest.update(encoded)
        return len(value)


def _sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mapping_and_source_bindings(
    *,
    payload: Mapping[str, Any],
    mapping: Mapping[str, Any],
    group_key: str,
    reference: str,
    expected_rows: int,
    expected_identity_sha256: str | None,
    expected_identity_group_sha256: str | None,
    stage_records: Sequence[Mapping[str, Any]],
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Any], list[str]]:
    """Audit mapping content and all exact source-file SHA bindings."""

    issues: list[str] = []
    report = dict(mapping)
    try:
        mapping_path = Path(str(mapping["path"])).expanduser().resolve(strict=True)
        mapping_sha = _sha256_file(mapping_path)
        report["resolved_path"] = str(mapping_path)
        report["observed_sha256"] = mapping_sha
        if mapping.get("sha256") != mapping_sha:
            issues.append("physical source-row mapping SHA differs")
        if mapping.get("schema_version") != MAPPING_SCHEMA_VERSION:
            issues.append("physical source-row mapping schema is invalid")
        if mapping.get("kind") != MAPPING_KIND:
            issues.append("physical source-row mapping kind is invalid")
        if int(mapping.get("rows", -1)) != int(expected_rows):
            issues.append("physical source-row mapping declared row count differs")

        expected_columns = [
            "pooled_row",
            *IDENTITY_COLUMNS,
            SOURCE_FILE_COLUMN,
            group_key,
            "cell_type",
            "stage_manifest_sha256",
        ]
        if list(mapping.get("columns", [])) != expected_columns:
            issues.append("physical source-row mapping declared columns differ")

        identity_sink = _HashingTextSink()
        identity_group_sink = _HashingTextSink()
        observed_rows = 0
        header = True
        source_counts: dict[tuple[str, str, str, str], int] = {}
        stage_counts: dict[str, int] = {}
        for chunk in pd.read_csv(
            mapping_path,
            dtype={column: "string" for column in expected_columns if column != "pooled_row"},
            chunksize=int(chunk_rows),
            keep_default_na=False,
        ):
            if list(chunk.columns) != expected_columns:
                issues.append("physical source-row mapping CSV columns differ")
                break
            pooled_rows = pd.to_numeric(chunk["pooled_row"], errors="coerce")
            expected_range = np.arange(
                observed_rows, observed_rows + len(chunk), dtype=np.int64
            )
            if pooled_rows.isna().any() or not np.array_equal(
                pooled_rows.to_numpy(dtype=np.int64), expected_range
            ):
                issues.append("physical source-row mapping pooled_row is not contiguous")
                break
            try:
                normalized, _ = normalized_identity_frame(
                    chunk,
                    group_key=group_key,
                    reference=reference,
                    context="physical source-row mapping",
                )
            except PhysicalIdentityError as error:
                issues.append(str(error))
                break

            identity = normalized.loc[:, list(IDENTITY_COLUMNS)].astype("string")
            identity_group = identity.copy()
            identity_group[group_key] = normalized[group_key].astype("string").to_numpy()
            identity.to_csv(
                identity_sink, index=False, header=header, lineterminator="\n"
            )
            identity_group.to_csv(
                identity_group_sink, index=False, header=header, lineterminator="\n"
            )
            header = False
            for row in normalized.itertuples(index=False):
                key = (
                    str(row.source_namespace),
                    str(row.source_dataset_name),
                    str(getattr(row, SOURCE_FILE_COLUMN)),
                    str(row.source_role),
                )
                source_counts[key] = source_counts.get(key, 0) + 1
            for value, count in chunk["stage_manifest_sha256"].value_counts().items():
                stage_counts[str(value)] = stage_counts.get(str(value), 0) + int(count)
            if bool((chunk["cell_type"].astype("string").str.len() == 0).any()):
                issues.append("physical source-row mapping has an empty cell_type")
                break
            observed_rows += len(chunk)

        report["observed_rows"] = int(observed_rows)
        observed_identity_sha = identity_sink.digest.hexdigest()
        observed_identity_group_sha = identity_group_sink.digest.hexdigest()
        report["observed_ordered_identity_sha256"] = observed_identity_sha
        report["observed_ordered_identity_group_sha256"] = observed_identity_group_sha
        if observed_rows != int(expected_rows):
            issues.append("physical source-row mapping observed row count differs")
        for label, observed, expected in (
            (
                "ordered_identity_sha256",
                observed_identity_sha,
                expected_identity_sha256,
            ),
            (
                "ordered_identity_group_sha256",
                observed_identity_group_sha,
                expected_identity_group_sha256,
            ),
        ):
            if mapping.get(label) != observed or observed != expected:
                issues.append(f"physical source-row mapping {label} differs")

        expected_stage_counts = {
            str(record.get("sha256")): int(record.get("total_cells", -1))
            for record in stage_records
        }
        if stage_counts != expected_stage_counts:
            issues.append("physical source-row mapping stage row counts differ")

        declared_bindings = payload.get("source_bindings")
        if not isinstance(declared_bindings, list) or not declared_bindings:
            issues.append("physical provenance source_bindings is empty or invalid")
        else:
            observed_binding_counts: dict[tuple[str, str, str, str], int] = {}
            source_sha_cache: dict[Path, str] = {}
            for binding in declared_bindings:
                if not isinstance(binding, dict):
                    issues.append("physical provenance source binding is malformed")
                    continue
                try:
                    source_path = Path(str(binding["source_file"])).expanduser().resolve(
                        strict=True
                    )
                    key = (
                        str(binding["source_namespace"]),
                        str(binding["source_dataset_name"]),
                        str(source_path),
                        str(binding["source_role"]),
                    )
                    if str(binding["source_file"]) != str(source_path):
                        issues.append("physical provenance source_file is not canonical")
                    if key[3] not in {TREATED_ROLE, REFERENCE_ROLE}:
                        issues.append("physical provenance source role is non-canonical")
                    if key in observed_binding_counts:
                        issues.append("physical provenance has duplicate source bindings")
                    observed_binding_counts[key] = int(binding["rows"])
                    source_sha = source_sha_cache.get(source_path)
                    if source_sha is None:
                        source_sha = _sha256_file(source_path)
                        source_sha_cache[source_path] = source_sha
                    if binding.get("source_file_sha256") != source_sha:
                        issues.append(f"physical source file SHA differs: {source_path}")
                    if binding.get("physical_rows_verified_against_obs_index") is not True:
                        issues.append(
                            "source binding lacks physical-row/obs-index verification"
                        )
                except (KeyError, OSError, TypeError, ValueError) as error:
                    issues.append(
                        "cannot verify physical source binding: "
                        f"{type(error).__name__}: {error}"
                    )
            if observed_binding_counts != source_counts:
                issues.append("physical source binding row counts differ from mapping")
    except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as error:
        issues.append(
            "cannot verify physical source-row mapping: "
            f"{type(error).__name__}: {error}"
        )
    report["passed"] = not issues
    report["issues"] = list(issues)
    return report, issues

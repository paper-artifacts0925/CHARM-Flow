"""Build identity-sealed PBMC pooled real/pred H5AD inputs on disk."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from src.evaluation.pbmc_physical_identity import (
    IDENTITY_COLUMNS,
    IDENTITY_SEMANTICS,
    LOCATOR_COLUMNS,
    PROVENANCE_SCHEMA_VERSION,
    REFERENCE_ROLE,
    SOURCE_FILE_COLUMN,
    read_source_obs_names,
    STAGE_IDENTITY_SCHEMA_VERSION,
    TREATED_ROLE,
    assert_paired_identity_equal,
    normalized_identity_frame,
)


BUILDER_SCHEMA_VERSION = "pbmc-pooled-input-builder/v1"
MAPPING_SCHEMA_VERSION = "pbmc-physical-source-row-mapping/v1"
MAPPING_KIND = "pbmc_physical_source_row_mapping"


class PooledBuilderError(RuntimeError):
    """Raised before publication when a pooled identity contract fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PooledBuilderError(f"JSON root is not an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_manifest(stage: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    pred_path = Path(str(stage.get("pred", ""))).expanduser().resolve(strict=True)
    manifest_path = pred_path.parent / "stage_manifest.json"
    payload = _load_json(manifest_path)
    if payload != dict(stage):
        raise PooledBuilderError(
            f"embedded merge stage differs from its stage manifest: {manifest_path}"
        )
    return manifest_path, payload


def _verify_source_rows(frame: pd.DataFrame, *, context: str) -> None:
    """Verify physical row bounds and obs names against exact source files."""

    for source_file, block in frame.groupby(SOURCE_FILE_COLUMN, sort=False):
        path = Path(str(source_file)).expanduser().resolve(strict=True)
        if str(path) != str(source_file):
            raise PooledBuilderError(
                f"{context}: source_file is not canonical: {source_file!r}"
            )
        rows = block["source_physical_row"].to_numpy(dtype=np.int64)
        expected_names = block["source_obs_name"].astype(str).to_numpy()
        with h5py.File(path, "r") as handle:
            observed_names = read_source_obs_names(handle, rows)
        if not np.array_equal(observed_names.astype(str), expected_names):
            mismatches = int(np.count_nonzero(observed_names.astype(str) != expected_names))
            raise PooledBuilderError(
                f"{context}: {mismatches} source_obs_name values do not match "
                f"physical rows in {path}"
            )


def _insert_global_identities(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    stage_cell_type: str,
) -> None:
    records = (
        (
            str(row.source_namespace),
            str(row.source_dataset_name),
            int(row.source_physical_row),
            str(row.source_obs_name),
            str(row.source_role),
            str(getattr(row, SOURCE_FILE_COLUMN)),
            str(row.cytokine),
            str(stage_cell_type),
        )
        for row in frame.itertuples(index=False)
    )
    try:
        connection.executemany(
            """
            INSERT INTO physical_rows (
                source_namespace, source_dataset_name, source_physical_row,
                source_obs_name, source_role, source_file, group_label,
                stage_cell_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        raise PooledBuilderError(
            "physical source locator is duplicated across PBMC stages"
        ) from error


def _open_identity_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE physical_rows (
            source_namespace TEXT NOT NULL,
            source_dataset_name TEXT NOT NULL,
            source_physical_row INTEGER NOT NULL,
            source_obs_name TEXT NOT NULL,
            source_role TEXT NOT NULL,
            source_file TEXT NOT NULL,
            group_label TEXT NOT NULL,
            stage_cell_type TEXT NOT NULL,
            PRIMARY KEY (
                source_namespace, source_dataset_name, source_physical_row
            )
        ) WITHOUT ROWID
        """
    )
    return connection


def _assert_ordered_identity_rows_equal(
    observed_obs: pd.DataFrame,
    expected_obs: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Compare canonical row identities without categorical-dtype metadata.

    AnnData on-disk concatenation unifies categorical dictionaries across
    inputs. A stage that lacks one otherwise valid cytokine therefore has
    different categorical metadata from its pooled slice even when every row
    value and its order are identical. Physical identity is defined by the
    canonical values below, not by unused category levels.
    """

    observed, _ = normalized_identity_frame(
        observed_obs,
        group_key="cytokine",
        reference="PBS",
        context=f"{context} observed",
    )
    expected, _ = normalized_identity_frame(
        expected_obs,
        group_key="cytokine",
        reference="PBS",
        context=f"{context} expected",
    )
    columns = [*IDENTITY_COLUMNS, SOURCE_FILE_COLUMN, "cytokine"]
    if len(observed) != len(expected):
        raise PooledBuilderError(
            f"{context}: row count differs: {len(observed)} != {len(expected)}"
        )
    for column in columns:
        observed_values = observed[column].to_numpy()
        expected_values = expected[column].to_numpy()
        if np.array_equal(observed_values, expected_values):
            continue
        mismatches = np.flatnonzero(observed_values != expected_values)
        first = int(mismatches[0]) if mismatches.size else -1
        raise PooledBuilderError(
            f"{context}: ordered {column} values differ at "
            f"{int(mismatches.size)} rows; first mismatch={first}"
        )


def _inspect_pooled_pair(
    real_path: Path,
    pred_path: Path,
    *,
    expected_rows: int,
    expected_genes: Sequence[str],
    stage_paths: Sequence[tuple[Path, Path]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    real = ad.read_h5ad(real_path, backed="r")
    pred = ad.read_h5ad(pred_path, backed="r")
    try:
        if real.shape != pred.shape or real.shape != (
            int(expected_rows),
            len(expected_genes),
        ):
            raise PooledBuilderError(
                f"pooled shape differs: real={real.shape}, pred={pred.shape}"
            )
        if list(map(str, real.var_names)) != list(map(str, expected_genes)):
            raise PooledBuilderError("pooled real gene order differs from stages")
        if list(map(str, pred.var_names)) != list(map(str, expected_genes)):
            raise PooledBuilderError("pooled pred gene order differs from stages")
        identity = assert_paired_identity_equal(
            real.obs,
            pred.obs,
            group_key="cytokine",
            reference="PBS",
            context="pooled PBMC inputs",
        )
        if list(map(str, real.obs_names)) != list(map(str, pred.obs_names)):
            raise PooledBuilderError("pooled real/pred obs order differs")

        offset = 0
        for stage_real_path, stage_pred_path in stage_paths:
            stage_real = ad.read_h5ad(stage_real_path, backed="r")
            stage_pred = ad.read_h5ad(stage_pred_path, backed="r")
            try:
                stop = offset + int(stage_real.n_obs)
                _assert_ordered_identity_rows_equal(
                    real.obs.iloc[offset:stop],
                    stage_real.obs,
                    context=f"pooled real rows versus stage {stage_real_path}",
                )
                _assert_ordered_identity_rows_equal(
                    pred.obs.iloc[offset:stop],
                    stage_pred.obs,
                    context=f"pooled pred rows versus stage {stage_pred_path}",
                )
                offset = stop
            finally:
                stage_real.file.close()
                stage_pred.file.close()
        if offset != int(expected_rows):
            raise PooledBuilderError("stage row order verification missed pooled rows")
        return identity["real"], identity["pred"]
    finally:
        real.file.close()
        pred.file.close()


def build_pooled_inputs(
    *,
    merge_manifest_path: Path,
    output_dir: Path,
    max_loaded_elems: int = 25_000_000,
) -> dict[str, Any]:
    """Build and atomically publish pooled H5ADs and identity attestation."""

    merge_path = merge_manifest_path.expanduser().resolve(strict=True)
    destination = output_dir.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pooled output: {destination}")
    merge = _load_json(merge_path)
    if merge.get("schema_version") != 2 or merge.get("passed") is not True:
        raise PooledBuilderError("merge manifest is stale or failed")
    if merge.get("kind") != "pbmc_u2_attempt2_celltype_merge":
        raise PooledBuilderError("wrong PBMC merge manifest kind")
    merge_identity = merge.get("physical_identity")
    if not isinstance(merge_identity, dict) or merge_identity.get("passed") is not True:
        raise PooledBuilderError("merge manifest lacks physical identity evidence")
    if merge_identity.get("schema_version") != STAGE_IDENTITY_SCHEMA_VERSION:
        raise PooledBuilderError("merge physical identity schema is stale")
    if merge_identity.get("identity_semantics") != IDENTITY_SEMANTICS:
        raise PooledBuilderError("merge physical identity semantics differ")
    if list(merge_identity.get("identity_columns", [])) != list(IDENTITY_COLUMNS):
        raise PooledBuilderError("merge physical identity columns differ")

    stages = merge.get("stages")
    if not isinstance(stages, list) or not stages:
        raise PooledBuilderError("merge manifest has no stages")
    cell_types = [str(stage.get("cell_type", "")) for stage in stages]
    if any(not value for value in cell_types) or len(set(cell_types)) != len(cell_types):
        raise PooledBuilderError("stage cell types are empty or duplicated")

    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build.", dir=destination.parent))
    connection: sqlite3.Connection | None = None
    try:
        mapping_path = work / "source_row_mapping.csv"
        database_path = work / "identity_uniqueness.sqlite"
        connection = _open_identity_database(database_path)
        stage_records: list[dict[str, Any]] = []
        stage_pairs: list[tuple[Path, Path]] = []
        stage_real_paths: list[Path] = []
        stage_pred_paths: list[Path] = []
        source_counts: dict[tuple[str, str, str, str], int] = {}
        genes: list[str] | None = None
        pooled_row = 0
        mapping_columns = [
            "pooled_row",
            *IDENTITY_COLUMNS,
            SOURCE_FILE_COLUMN,
            "cytokine",
            "cell_type",
            "stage_manifest_sha256",
        ]
        with mapping_path.open("x", encoding="utf-8", newline="") as mapping_handle:
            writer = csv.DictWriter(mapping_handle, fieldnames=mapping_columns)
            writer.writeheader()
            for embedded in stages:
                stage_manifest_path, stage = _stage_manifest(embedded)
                if stage.get("physical_identity", {}).get("passed") is not True:
                    raise PooledBuilderError(
                        f"stage lacks physical identity evidence: {stage_manifest_path}"
                    )
                if stage.get("physical_identity", {}).get("schema_version") != (
                    STAGE_IDENTITY_SCHEMA_VERSION
                ):
                    raise PooledBuilderError(
                        f"stage physical identity schema is stale: {stage_manifest_path}"
                    )
                real_path = Path(stage["real"]).expanduser().resolve(strict=True)
                pred_path = Path(stage["pred"]).expanduser().resolve(strict=True)
                if sha256_file(real_path) != stage.get("real_sha256"):
                    raise PooledBuilderError(f"stage real SHA changed: {real_path}")
                if sha256_file(pred_path) != stage.get("pred_sha256"):
                    raise PooledBuilderError(f"stage pred SHA changed: {pred_path}")

                real = ad.read_h5ad(real_path, backed="r")
                pred = ad.read_h5ad(pred_path, backed="r")
                try:
                    if int(real.n_obs) != int(stage.get("total_cells", -1)):
                        raise PooledBuilderError("stage real row count differs from manifest")
                    if real.shape != pred.shape:
                        raise PooledBuilderError("stage real/pred shapes differ")
                    stage_genes = list(map(str, real.var_names))
                    if stage_genes != list(map(str, pred.var_names)):
                        raise PooledBuilderError("stage real/pred gene order differs")
                    if genes is None:
                        genes = stage_genes
                    elif stage_genes != genes:
                        raise PooledBuilderError("gene order differs across stages")
                    identity = assert_paired_identity_equal(
                        real.obs,
                        pred.obs,
                        group_key="cytokine",
                        reference="PBS",
                        context=f"stage {stage['cell_type']}",
                    )
                    if identity != stage["physical_identity"]:
                        raise PooledBuilderError(
                            f"stage identity differs from manifest: {stage_manifest_path}"
                        )
                    frame, _ = normalized_identity_frame(
                        real.obs,
                        group_key="cytokine",
                        reference="PBS",
                        context=f"stage {stage['cell_type']}",
                    )
                    _verify_source_rows(frame, context=f"stage {stage['cell_type']}")
                    _insert_global_identities(
                        connection, frame, stage_cell_type=str(stage["cell_type"])
                    )
                    stage_sha = sha256_file(stage_manifest_path)
                    for row in frame.itertuples(index=False):
                        role = str(row.source_role)
                        source_file = str(getattr(row, SOURCE_FILE_COLUMN))
                        key = (
                            str(row.source_namespace),
                            str(row.source_dataset_name),
                            source_file,
                            role,
                        )
                        source_counts[key] = source_counts.get(key, 0) + 1
                        writer.writerow(
                            {
                                "pooled_row": pooled_row,
                                **{
                                    column: getattr(row, column)
                                    for column in IDENTITY_COLUMNS
                                },
                                SOURCE_FILE_COLUMN: source_file,
                                "cytokine": str(row.cytokine),
                                "cell_type": str(stage["cell_type"]),
                                "stage_manifest_sha256": stage_sha,
                            }
                        )
                        pooled_row += 1
                finally:
                    real.file.close()
                    pred.file.close()

                stage_records.append(
                    {
                        "path": str(stage_manifest_path),
                        "sha256": sha256_file(stage_manifest_path),
                        "cell_type": str(stage["cell_type"]),
                        "rows": int(stage["total_cells"]),
                        "real_sha256": str(stage["real_sha256"]),
                        "pred_sha256": str(stage["pred_sha256"]),
                    }
                )
                stage_pairs.append((real_path, pred_path))
                stage_real_paths.append(real_path)
                stage_pred_paths.append(pred_path)
            mapping_handle.flush()
            os.fsync(mapping_handle.fileno())

        expected_rows = int(merge.get("total_cells", -1))
        if pooled_row != expected_rows:
            raise PooledBuilderError(
                f"stage union row count differs: {pooled_row} != {expected_rows}"
            )
        database_rows = int(
            connection.execute("SELECT COUNT(*) FROM physical_rows").fetchone()[0]
        )
        if database_rows != expected_rows:
            raise PooledBuilderError("global physical locator uniqueness count differs")
        connection.close()
        connection = None
        database_path.unlink()
        wal_path = database_path.with_name(database_path.name + "-wal")
        shm_path = database_path.with_name(database_path.name + "-shm")
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()

        if genes is None:
            raise PooledBuilderError("no stage genes were observed")
        temporary_real = work / "real_pooled.h5ad"
        temporary_pred = work / "pred_pooled.h5ad"
        ad.experimental.concat_on_disk(
            stage_real_paths,
            temporary_real,
            axis=0,
            join="inner",
            merge="same",
            uns_merge=None,
            index_unique=None,
            max_loaded_elems=int(max_loaded_elems),
        )
        ad.experimental.concat_on_disk(
            stage_pred_paths,
            temporary_pred,
            axis=0,
            join="inner",
            merge="same",
            uns_merge=None,
            index_unique=None,
            max_loaded_elems=int(max_loaded_elems),
        )
        real_identity, pred_identity = _inspect_pooled_pair(
            temporary_real,
            temporary_pred,
            expected_rows=expected_rows,
            expected_genes=genes,
            stage_paths=stage_pairs,
        )

        source_bindings = []
        for (namespace, dataset_name, source_file, role), rows in sorted(
            source_counts.items()
        ):
            source_path = Path(source_file).expanduser().resolve(strict=True)
            source_bindings.append(
                {
                    "source_namespace": namespace,
                    "source_dataset_name": dataset_name,
                    "source_file": str(source_path),
                    "source_file_sha256": sha256_file(source_path),
                    "source_role": role,
                    "rows": int(rows),
                    "physical_rows_verified_against_obs_index": True,
                }
            )

        final_real = destination / temporary_real.name
        final_pred = destination / temporary_pred.name
        final_mapping = destination / mapping_path.name
        mapping_sha = sha256_file(mapping_path)
        real_sha = sha256_file(temporary_real)
        pred_sha = sha256_file(temporary_pred)
        attestation = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "builder_schema_version": BUILDER_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "passed": True,
            "identity_semantics": IDENTITY_SEMANTICS,
            "identity_columns": list(IDENTITY_COLUMNS),
            "source_file_column": SOURCE_FILE_COLUMN,
            "canonical_roles": {
                "treated": TREATED_ROLE,
                "reference": REFERENCE_ROLE,
            },
            "group_key": "cytokine",
            "reference": "PBS",
            "merge_manifest": {
                "path": str(merge_path),
                "sha256": sha256_file(merge_path),
            },
            "stage_manifests": [
                {"path": record["path"], "sha256": record["sha256"]}
                for record in stage_records
            ],
            "source_bindings": source_bindings,
            "source_row_mapping": {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "kind": MAPPING_KIND,
                "path": str(final_mapping),
                "sha256": mapping_sha,
                "rows": expected_rows,
                "columns": mapping_columns,
                "ordered_identity_sha256": real_identity[
                    "ordered_identity_sha256"
                ],
                "ordered_identity_group_sha256": real_identity[
                    "ordered_identity_group_sha256"
                ],
                "identity_group_multiset_sha256": real_identity[
                    "identity_group_multiset"
                ]["sha256"],
                "physical_locator_unique_across_stages": True,
                "physical_rows_verified_against_source_obs_index": True,
            },
            "real": {
                "path": str(final_real),
                "sha256": real_sha,
                "ordered_identity_sha256": real_identity[
                    "ordered_identity_sha256"
                ],
                "ordered_identity_group_sha256": real_identity[
                    "ordered_identity_group_sha256"
                ],
                "identity_group_multiset_sha256": real_identity[
                    "identity_group_multiset"
                ]["sha256"],
            },
            "pred": {
                "path": str(final_pred),
                "sha256": pred_sha,
                "ordered_identity_sha256": pred_identity[
                    "ordered_identity_sha256"
                ],
                "ordered_identity_group_sha256": pred_identity[
                    "ordered_identity_group_sha256"
                ],
                "identity_group_multiset_sha256": pred_identity[
                    "identity_group_multiset"
                ]["sha256"],
            },
            "coverage": {
                "rows": expected_rows,
                "treated_rows": int(merge.get("treated_cells", -1)),
                "reference_rows": int(merge.get("control_cells", -1)),
                "genes": len(genes),
                "stages": len(stage_records),
            },
        }
        _write_json(work / "physical_identity_provenance.json", attestation)
        _write_json(
            work / "builder_manifest.json",
            {
                "schema_version": BUILDER_SCHEMA_VERSION,
                "created_utc": attestation["created_utc"],
                "passed": True,
                "merge_manifest": attestation["merge_manifest"],
                "physical_identity_provenance": {
                    "path": str(destination / "physical_identity_provenance.json"),
                    "sha256": sha256_file(work / "physical_identity_provenance.json"),
                },
                "real": attestation["real"],
                "pred": attestation["pred"],
                "source_row_mapping": attestation["source_row_mapping"],
                "coverage": attestation["coverage"],
            },
        )
        os.replace(work, destination)
        _fsync_directory(destination.parent)
        return attestation
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(work, ignore_errors=True)
        raise

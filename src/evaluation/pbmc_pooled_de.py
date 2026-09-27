"""PBMC pooled differential-expression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control, mannwhitneyu

from src.evaluation.pbmc_physical_identity import (
    IDENTITY_COLUMNS as PHYSICAL_IDENTITY_COLUMNS,
    PROVENANCE_SCHEMA_VERSION,
    REFERENCE_ROLE,
    SOURCE_FILE_COLUMN,
    STAGE_IDENTITY_SCHEMA_VERSION,
    TREATED_ROLE,
    PhysicalIdentityError,
    normalized_identity_frame,
)
from src.evaluation.pbmc_provenance_attestation import (
    BUILDER_SCHEMA_VERSION,
    audit_mapping_and_source_bindings,
)


AUDIT_SCHEMA_VERSION = "pbmc-pooled-de-input-audit/v1"
CACHE_SCHEMA_VERSION = "pbmc-pooled-de-cache/v1"
DEGENERATE_POLICY_VERSION = "equal-constant-only/v1"

VALID_TEST_STATUSES = frozenset({"ok", "equal_constant", "different_constants"})
CELLEVAL_COLUMNS = (
    "target",
    "reference",
    "feature",
    "target_mean",
    "reference_mean",
    "percent_change",
    "fold_change",
    "p_value",
    "statistic",
    "fdr",
)
SUPPORT_COLUMNS = (
    "side",
    "target",
    "reference",
    "feature",
    "gene_index",
    "n_target",
    "n_reference",
    "target_finite_count",
    "reference_finite_count",
    "target_nonzero_count",
    "reference_nonzero_count",
    "target_zero_count",
    "reference_zero_count",
    "target_zero_rate",
    "reference_zero_rate",
    "target_unique_count",
    "reference_unique_count",
    "target_min",
    "target_max",
    "reference_min",
    "reference_max",
    "target_constant",
    "reference_constant",
    "status",
    "invalid_reason",
)


class PooledDEError(RuntimeError):
    """Base error for the audited pooled-DE pipeline."""


class InputAuditError(PooledDEError):
    """Raised when H5AD metadata, identities, or stage manifests fail audit."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        issues = self.report.get("issues", [])
        super().__init__("PBMC pooled-DE input audit failed: " + "; ".join(map(str, issues)))


class RawFamilyError(PooledDEError):
    """Raised before BH when a raw side-wide family is incomplete or invalid."""

    def __init__(
        self,
        side: str,
        diagnostics: pd.DataFrame,
        issues: Sequence[str],
    ) -> None:
        self.side = side
        self.diagnostics = diagnostics.copy()
        self.issues = tuple(issues)
        super().__init__(
            f"raw DE family for {side!r} failed closed: " + "; ".join(self.issues)
        )


@dataclass(frozen=True)
class MWUConfig:
    """Frozen statistical choices matching the historical pdex call."""

    alternative: str = "two-sided"
    method: str = "auto"
    use_continuity: bool = True
    is_log1p: bool = True
    exp_post_agg: bool = True
    fold_change_clip: float = 20.0

    def __post_init__(self) -> None:
        if self.alternative not in {"two-sided", "less", "greater"}:
            raise ValueError(f"unsupported MWU alternative: {self.alternative}")
        if self.method not in {"auto", "asymptotic", "exact"}:
            raise ValueError(f"unsupported MWU method: {self.method}")
        if not np.isfinite(self.fold_change_clip) or self.fold_change_clip <= 1:
            raise ValueError("fold_change_clip must be finite and greater than one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": "scipy.stats.mannwhitneyu",
            "alternative": self.alternative,
            "method": self.method,
            "use_continuity": self.use_continuity,
            "is_log1p": self.is_log1p,
            "exp_post_agg": self.exp_post_agg,
            "fold_change_clip": self.fold_change_clip,
            "degenerate_policy": DEGENERATE_POLICY_VERSION,
            "bh_method": "scipy.stats.false_discovery_control(method='bh')",
            "bh_family": "one complete target-by-gene family per side",
            "library_versions": {
                package: _package_version(package)
                for package in ("anndata", "numpy", "pandas", "scipy")
            },
        }


@dataclass(frozen=True)
class SideInputPlan:
    side: str
    path: Path
    genes: tuple[str, ...]
    group_rows: Mapping[str, np.ndarray]
    file_identity: tuple[int, int, int, int]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class PooledInputPlan:
    real: SideInputPlan
    pred: SideInputPlan
    targets: tuple[str, ...]
    reference: str
    report: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _stable_string_sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_axis_sha256(values: Iterable[str]) -> str:
    """Match the canonical gene-axis digest used by PBMC stage manifests."""

    payload = json.dumps(
        [str(value) for value in values], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _HashingTextSink:
    """Minimal text sink used by pandas to hash a canonical CSV stream."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self.digest.update(encoded)
        return len(value)


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    sink = _HashingTextSink()
    frame.to_csv(sink, index=False, lineterminator="\n")
    return sink.digest.hexdigest()


def _multiset_frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a row-order-independent, duplicate-sensitive lineage fingerprint.

    The two modular accumulators and XOR are computed from SHA-256 row digests.
    This avoids sorting millions of PBMC identity rows while still detecting
    omissions, duplicates, and group-label substitutions in stage unions.
    """

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


def _combine_multiset_fingerprints(
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
            raise ValueError("cannot combine lineage fingerprints with different columns")
        rows += int(fingerprint["rows"])
        total = (total + int(str(fingerprint["sum_mod_2_256"]), 16)) % modulus
        total_squared = (
            total_squared + int(str(fingerprint["sum_squares_mod_2_256"]), 16)
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


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _close_backed(adata: ad.AnnData) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def _inspect_side(
    side: str,
    path: Path,
    *,
    group_key: str,
    reference: str,
    identity_columns: Sequence[str],
    hash_input: bool,
) -> tuple[SideInputPlan, list[str]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    identity_before = _file_identity(resolved)
    issues: list[str] = []
    adata = ad.read_h5ad(resolved, backed="r")
    try:
        genes = tuple(map(str, adata.var_names.tolist()))
        if len(set(genes)) != len(genes):
            issues.append(f"{side}: gene names are not unique")
        if len(genes) != adata.n_vars:
            issues.append(f"{side}: var_names length does not match X columns")

        obs = adata.obs
        if group_key not in obs.columns:
            issues.append(f"{side}: missing group column {group_key!r}")
            group_rows: dict[str, np.ndarray] = {}
            group_counts: dict[str, int] = {}
        elif bool(obs[group_key].isna().any()):
            issues.append(f"{side}: group column {group_key!r} contains missing values")
            group_rows = {}
            group_counts = {}
        else:
            group_values = obs[group_key].astype("string")
            empty_labels = int((group_values.str.len() == 0).sum())
            if empty_labels:
                issues.append(f"{side}: group column contains {empty_labels} empty labels")
            group_rows = {
                str(label): np.asarray(rows, dtype=np.int64)
                for label, rows in obs.groupby(group_key, observed=True, sort=False).indices.items()
            }
            group_counts = {label: int(rows.size) for label, rows in group_rows.items()}

        missing_identity = [column for column in identity_columns if column not in obs.columns]
        identity_sha: str | None = None
        identity_group_sha: str | None = None
        identity_group_multiset: dict[str, Any] | None = None
        duplicate_identity_rows: int | None = None
        duplicate_identity_keys: int | None = None
        if not identity_columns:
            issues.append(f"{side}: no physical source identity columns were supplied")
        elif missing_identity:
            issues.append(f"{side}: missing identity columns {missing_identity}")
        else:
            identity_frame = obs.loc[:, list(identity_columns)].copy()
            if bool(identity_frame.isna().any(axis=None)):
                issues.append(f"{side}: physical source identity contains missing values")
            identity_frame = identity_frame.astype("string")
            duplicate_mask = identity_frame.duplicated(keep=False)
            duplicate_identity_rows = int(duplicate_mask.sum())
            duplicate_identity_keys = int(len(identity_frame) - len(identity_frame.drop_duplicates()))
            if duplicate_identity_keys:
                issues.append(
                    f"{side}: physical source identity has {duplicate_identity_keys} duplicate keys "
                    f"across {duplicate_identity_rows} rows"
                )
            identity_sha = _dataframe_sha256(identity_frame)
            if group_key in obs.columns:
                identity_group_frame = identity_frame.copy()
                identity_group_frame[group_key] = obs[group_key].astype("string").to_numpy()
                identity_group_sha = _dataframe_sha256(identity_group_frame)
                identity_group_multiset = _multiset_frame_fingerprint(
                    identity_group_frame
                )

        strict_identity_report: dict[str, Any] | None = None
        try:
            _, strict_identity_report = normalized_identity_frame(
                obs,
                group_key=group_key,
                reference=reference,
                context=f"{side} pooled input",
            )
        except PhysicalIdentityError as error:
            issues.append(str(error))

        matrix = adata.X
        matrix_type = f"{type(matrix).__module__}.{type(matrix).__name__}"
        matrix_dtype = str(getattr(matrix, "dtype", "unknown"))
        report: dict[str, Any] = {
            "side": side,
            "path": str(resolved),
            "bytes": identity_before[2],
            "mtime_ns": identity_before[3],
            "sha256": sha256_file(resolved) if hash_input else None,
            "sha256_computed": hash_input,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "matrix_type": matrix_type,
            "matrix_dtype": matrix_dtype,
            "gene_order_sha256": _stable_string_sequence_sha256(genes),
            "stage_compatible_gene_axis_sha256": _canonical_axis_sha256(genes),
            "genes_unique": len(set(genes)) == len(genes),
            "group_key": group_key,
            "group_counts": group_counts,
            "identity_columns": list(identity_columns),
            "ordered_identity_sha256": identity_sha,
            "ordered_identity_group_sha256": identity_group_sha,
            "identity_group_multiset": identity_group_multiset,
            "duplicate_identity_rows": duplicate_identity_rows,
            "duplicate_identity_keys": duplicate_identity_keys,
            "issues": list(issues),
            "strict_physical_identity": strict_identity_report,
        }
    finally:
        _close_backed(adata)

    identity_after = _file_identity(resolved)
    if identity_after != identity_before:
        issues.append(f"{side}: input file changed while it was audited")
        report["issues"] = list(issues)
    return (
        SideInputPlan(
            side=side,
            path=resolved,
            genes=genes,
            group_rows=group_rows,
            file_identity=identity_after,
            report=report,
        ),
        issues,
    )


def _inspect_stage_lineage(
    path: Path,
    *,
    side: str,
    identity_columns: Sequence[str],
    group_key: str,
    reference: str,
    expected_genes: Sequence[str],
    expected_rows: int,
    expected_cell_type: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    """Inspect stage metadata only; expression values remain backed and unread."""

    issues: list[str] = []
    adata = ad.read_h5ad(path, backed="r")
    try:
        genes = tuple(map(str, adata.var_names.tolist()))
        if genes != tuple(expected_genes):
            issues.append(f"{side} stage gene names/order differ: {path}")
        if int(adata.n_obs) != expected_rows:
            issues.append(
                f"{side} stage row count differs at {path}: "
                f"expected={expected_rows}, observed={adata.n_obs}"
            )
        required_columns = [*identity_columns, group_key]
        missing = [column for column in required_columns if column not in adata.obs.columns]
        fingerprint: dict[str, Any] | None = None
        duplicate_keys: int | None = None
        if missing:
            issues.append(f"{side} stage {path} misses lineage columns {missing}")
        else:
            frame = adata.obs.loc[:, required_columns].copy()
            if bool(frame.isna().any(axis=None)):
                issues.append(f"{side} stage lineage contains missing values: {path}")
            identity = frame.loc[:, list(identity_columns)].astype("string")
            duplicate_keys = int(len(identity) - len(identity.drop_duplicates()))
            if duplicate_keys:
                issues.append(
                    f"{side} stage has {duplicate_keys} duplicate source identities: {path}"
                )
            fingerprint = _multiset_frame_fingerprint(frame)
        strict_identity_report: dict[str, Any] | None = None
        try:
            _, strict_identity_report = normalized_identity_frame(
                adata.obs,
                group_key=group_key,
                reference=reference,
                context=f"{side} stage {path}",
            )
        except PhysicalIdentityError as error:
            issues.append(str(error))

        if "cell_type" not in adata.obs.columns:
            issues.append(f"{side} stage misses required cell_type label: {path}")
        else:
            observed_types = set(map(str, adata.obs["cell_type"].astype("string")))
            if observed_types != {expected_cell_type}:
                issues.append(
                    f"{side} stage cell_type labels differ at {path}: "
                    f"expected={expected_cell_type!r}, observed={sorted(observed_types)}"
                )
        report = {
            "path": str(path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "lineage_columns": required_columns,
            "identity_group_multiset": fingerprint,
            "duplicate_identity_keys": duplicate_keys,
            "strict_physical_identity": strict_identity_report,
        }
    finally:
        _close_backed(adata)
    return fingerprint, report, issues


def _stage_manifest_records(
    paths: Sequence[Path],
    *,
    targets: Sequence[str],
    gene_axis_sha256: str,
    pooled_genes: Sequence[str],
    genes: int,
    pooled_treated_cells: int,
    pooled_reference_cells: int,
    reference: str,
    pooled_total_cells: int,
    identity_columns: Sequence[str],
    group_key: str,
    pooled_lineage: Mapping[str, Mapping[str, Any] | None],
    verify_artifact_hashes: bool,
    verify_row_union: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    seen: set[Path] = set()
    payloads: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    lineage_by_side: dict[str, list[Mapping[str, Any]]] = {"real": [], "pred": []}
    required = {
        "passed",
        "kind",
        "schema_version",
        "input_fingerprint",
        "canonical_control_sha256",
        "cell_type",
        "cytokines",
        "gene_axis_sha256",
        "genes",
        "treated_cells",
        "control_cells",
        "total_cells",
        "canonical_controls_appended_once",
        "source_prediction_values_unchanged",
        "physical_identity",
        "real",
        "real_sha256",
        "pred",
        "pred_sha256",
    }
    for supplied in paths:
        try:
            path = supplied.expanduser().resolve(strict=True)
        except FileNotFoundError:
            issues.append(f"stage manifest does not exist: {supplied.expanduser().absolute()}")
            continue
        if path in seen:
            issues.append(f"duplicate stage manifest path: {path}")
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            issues.append(f"cannot read stage manifest {path}: {type(error).__name__}: {error}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"stage manifest is not a JSON object: {path}")
            continue
        missing = sorted(required - set(payload))
        if missing:
            issues.append(f"stage manifest {path} is missing required fields {missing}")
        passed = payload.get("passed") is True
        if not passed:
            issues.append(f"stage manifest is not passed: {path}")
        payloads.append(payload)
        record = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "passed": passed,
                "cell_type": payload.get("cell_type"),
                "cytokines": payload.get("cytokines"),
                "gene_axis_sha256": payload.get("gene_axis_sha256"),
                "genes": payload.get("genes"),
                "treated_cells": payload.get("treated_cells"),
                "control_cells": payload.get("control_cells"),
                "total_cells": payload.get("total_cells"),
                "sampling_seed": payload.get("sampling_seed"),
                "kind": payload.get("kind"),
                "schema_version": payload.get("schema_version"),
                "input_fingerprint": payload.get("input_fingerprint"),
                "canonical_control_sha256": payload.get("canonical_control_sha256"),
                "artifacts": {},
            }
        for side in ("real", "pred"):
            supplied_artifact = payload.get(side)
            declared_sha = payload.get(f"{side}_sha256")
            artifact_record: dict[str, Any] = {
                "declared_path": supplied_artifact,
                "declared_sha256": declared_sha,
            }
            if not isinstance(supplied_artifact, str) or not supplied_artifact:
                issues.append(f"stage manifest has invalid {side} path: {path}")
            else:
                try:
                    artifact = Path(supplied_artifact).expanduser().resolve(strict=True)
                    if not artifact.is_file():
                        raise FileNotFoundError(artifact)
                    artifact_record["path"] = str(artifact)
                    artifact_record["bytes"] = artifact.stat().st_size
                    if verify_artifact_hashes:
                        observed_sha = hash_cache.get(artifact)
                        if observed_sha is None:
                            observed_sha = sha256_file(artifact)
                            hash_cache[artifact] = observed_sha
                        artifact_record["observed_sha256"] = observed_sha
                        if declared_sha != observed_sha:
                            issues.append(
                                f"stage {side} SHA mismatch at {path}: "
                                f"declared={declared_sha}, observed={observed_sha}"
                            )
                    if verify_row_union:
                        fingerprint, lineage_report, lineage_issues = _inspect_stage_lineage(
                            artifact,
                            side=side,
                            identity_columns=identity_columns,
                            reference=reference,
                            group_key=group_key,
                            expected_genes=pooled_genes,
                            expected_rows=int(payload.get("total_cells", -1)),
                            expected_cell_type=str(payload.get("cell_type", "")),
                        )
                        artifact_record["lineage"] = lineage_report
                        issues.extend(lineage_issues)
                        if fingerprint is not None:
                            lineage_by_side[side].append(fingerprint)
                except (OSError, ValueError, KeyError) as error:
                    issues.append(
                        f"cannot verify stage {side} artifact for {path}: "
                        f"{type(error).__name__}: {error}"
                    )
            record["artifacts"][side] = artifact_record
        records.append(record)

    expected_target_set = set(map(str, targets))
    cell_types: list[str] = []
    treated_sum = 0
    reference_sum = 0
    total_sum = 0
    donor_signatures: set[tuple[str, ...]] = set()
    sampling_seeds: set[Any] = set()
    cytokine_union: set[str] = set()
    count_fields_valid = True
    input_fingerprints: set[str] = set()
    canonical_control_hashes: set[str] = set()
    for record, payload in zip(records, payloads):
        label = str(record["path"])
        cell_type = payload.get("cell_type")
        if payload.get("kind") != "pbmc_u2_attempt2_celltype_stage":
            issues.append(f"stage manifest kind differs at {label}: {payload.get('kind')!r}")
        if payload.get("schema_version") != 2:
            issues.append(
                f"stage manifest schema_version differs at {label}: "
                f"{payload.get('schema_version')!r}"
            )
        physical_identity = payload.get("physical_identity")
        if not isinstance(physical_identity, dict):
            issues.append(f"stage manifest lacks physical_identity: {label}")
        elif (
            physical_identity.get("schema_version")
            != STAGE_IDENTITY_SCHEMA_VERSION
            or physical_identity.get("passed") is not True
            or physical_identity.get("paired_order_identical") is not True
        ):
            issues.append(f"stage manifest physical_identity is stale or failed: {label}")
        else:
            for side in ("real", "pred"):
                observed = (
                    record.get("artifacts", {})
                    .get(side, {})
                    .get("lineage", {})
                    .get("strict_physical_identity")
                )
                declared = physical_identity.get(side)
                if observed is None or declared != observed:
                    issues.append(
                        f"stage {side} physical identity differs from manifest: {label}"
                    )
        input_fingerprint = payload.get("input_fingerprint")
        canonical_control_sha = payload.get("canonical_control_sha256")
        if not isinstance(input_fingerprint, str) or len(input_fingerprint) != 64:
            issues.append(f"stage manifest has invalid input_fingerprint: {label}")
        else:
            input_fingerprints.add(input_fingerprint)
        if not isinstance(canonical_control_sha, str) or len(canonical_control_sha) != 64:
            issues.append(f"stage manifest has invalid canonical_control_sha256: {label}")
        else:
            canonical_control_hashes.add(canonical_control_sha)
        if cell_type is None or str(cell_type) == "":
            issues.append(f"stage manifest has empty cell_type: {label}")
        else:
            cell_types.append(str(cell_type))
        cytokines = payload.get("cytokines")
        if not isinstance(cytokines, list):
            issues.append(f"stage manifest cytokines is not a list: {label}")
        else:
            stage_target_set = set(map(str, cytokines))
            cytokine_union.update(stage_target_set)
            unexpected = stage_target_set - expected_target_set
            if unexpected:
                issues.append(
                    f"stage manifest has targets outside the official set at {label}: "
                    f"unexpected={sorted(unexpected)}"
                )
        if payload.get("gene_axis_sha256") != gene_axis_sha256:
            issues.append(f"stage manifest gene axis differs from pooled H5AD: {label}")
        if payload.get("genes") != genes:
            issues.append(
                f"stage manifest gene count differs at {label}: "
                f"expected={genes}, observed={payload.get('genes')}"
            )
        if payload.get("canonical_controls_appended_once") is not True:
            issues.append(f"stage manifest did not append canonical controls exactly once: {label}")
        if payload.get("source_prediction_values_unchanged") is not True:
            issues.append(f"stage manifest does not certify unchanged predictions: {label}")
        if "parent_gate_passed" in payload and payload.get("parent_gate_passed") is not True:
            issues.append(f"stage manifest parent gate is not passed: {label}")
        try:
            treated = int(payload["treated_cells"])
            controls = int(payload["control_cells"])
            total = int(payload["total_cells"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"stage manifest counts are not integers: {label}")
            count_fields_valid = False
            continue
        if treated < 0 or controls < 0 or total < 0 or total != treated + controls:
            issues.append(
                f"stage manifest has invalid counts at {label}: "
                f"treated={treated}, controls={controls}, total={total}"
            )
            count_fields_valid = False
        treated_sum += treated
        reference_sum += controls
        total_sum += total
        donors = payload.get("donors")
        if isinstance(donors, list):
            donor_signatures.add(tuple(map(str, donors)))
        if "sampling_seed" in payload:
            sampling_seeds.add(payload.get("sampling_seed"))

    duplicate_cell_types = len(cell_types) - len(set(cell_types))
    if duplicate_cell_types:
        issues.append(f"stage manifests contain {duplicate_cell_types} duplicate cell_type entries")
    if len(donor_signatures) > 1:
        issues.append("stage manifests disagree on donor IDs or donor order")
    if len(sampling_seeds) > 1:
        issues.append("stage manifests disagree on sampling_seed")
    if len(input_fingerprints) > 1:
        issues.append("stage manifests disagree on input_fingerprint")
    if len(canonical_control_hashes) > 1:
        issues.append("stage manifests disagree on canonical_control_sha256")
    if records and cytokine_union != expected_target_set:
        issues.append(
            "stage manifest cytokine union differs from pooled target set: "
            f"missing={sorted(expected_target_set - cytokine_union)}, "
            f"unexpected={sorted(cytokine_union - expected_target_set)}"
        )
    if count_fields_valid and records:
        if treated_sum != pooled_treated_cells:
            issues.append(
                f"stage treated sum {treated_sum} != pooled treated cells {pooled_treated_cells}"
            )
        if reference_sum != pooled_reference_cells:
            issues.append(
                f"stage control sum {reference_sum} != pooled reference cells "
                f"{pooled_reference_cells}"
            )
        if total_sum != pooled_total_cells:
            issues.append(f"stage total sum {total_sum} != pooled total cells {pooled_total_cells}")
    stage_union: dict[str, Any] = {}
    if verify_row_union:
        for side in ("real", "pred"):
            combined = _combine_multiset_fingerprints(lineage_by_side[side])
            pooled = pooled_lineage.get(side)
            matches = bool(
                combined is not None
                and pooled is not None
                and combined.get("sha256") == pooled.get("sha256")
                and combined.get("rows") == pooled.get("rows")
            )
            stage_union[side] = {
                "combined": combined,
                "pooled": pooled,
                "matches_pooled": matches,
            }
            if not matches:
                issues.append(
                    f"{side} stage identity+group row union does not equal pooled H5AD"
                )
    summary = {
        "stage_count": len(records),
        "unique_cell_types": len(set(cell_types)),
        "treated_cells_sum": treated_sum,
        "control_cells_sum": reference_sum,
        "total_cells_sum": total_sum,
        "donor_signatures": [list(values) for values in sorted(donor_signatures)],
        "sampling_seeds": sorted(sampling_seeds, key=str),
        "cytokine_union": sorted(cytokine_union),
        "input_fingerprints": sorted(input_fingerprints),
        "canonical_control_sha256": sorted(canonical_control_hashes),
        "artifact_hashes_verified": verify_artifact_hashes,
        "row_union_verified": verify_row_union,
        "stage_union": stage_union,
    }
    return records, summary, issues


def _audit_physical_provenance(
    path: Path | None,
    *,
    required: bool,
    group_key: str,
    reference: str,
    identity_columns: Sequence[str],
    real: SideInputPlan,
    pred: SideInputPlan,
    stage_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Bind a builder attestation to the exact pooled/stage artifacts.

    Uniqueness of arbitrary H5AD columns cannot establish that they identify
    physical rows in the source PBMC dataset.  Production evaluation therefore
    requires a separate, hashed source-row mapping attestation.  In particular,
    historical ``source_obs_name`` values created from generated shard outputs
    are not accepted as physical source indices merely because they are unique.
    """

    issues: list[str] = []
    if path is None:
        status = "missing_required_attestation" if required else "unverified_test_override"
        report = {
            "required": required,
            "status": status,
            "passed": not required,
            "warning": (
                "identity columns are structurally audited but physical-source-row "
                "semantics are not established"
            ),
        }
        if required:
            issues.append(
                "physical source-row provenance is unverified: a bound provenance "
                "attestation is required before any pooled MWU computation"
            )
        return report, issues

    try:
        resolved = path.expanduser().resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return (
            {
                "required": required,
                "status": "unreadable",
                "path": str(path.expanduser().absolute()),
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            },
            [f"cannot read physical provenance attestation: {type(error).__name__}: {error}"],
        )
    if not isinstance(payload, dict):
        return (
            {
                "required": required,
                "status": "invalid_schema",
                "path": str(resolved),
                "passed": False,
            },
            ["physical provenance attestation is not a JSON object"],
        )

    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        issues.append(
            "physical provenance schema differs: "
            f"expected={PROVENANCE_SCHEMA_VERSION!r}, "
            f"observed={payload.get('schema_version')!r}"
        )
    if payload.get("passed") is not True:
        issues.append("physical provenance attestation is not passed")
    if payload.get("builder_schema_version") != BUILDER_SCHEMA_VERSION:
        issues.append("physical provenance was not produced by the pinned pooled builder")
    if payload.get("reference") != reference:
        issues.append("physical provenance reference differs from pooled audit")
    if payload.get("source_file_column") != SOURCE_FILE_COLUMN:
        issues.append("physical provenance source_file_column differs")
    if payload.get("canonical_roles") != {
        "treated": TREATED_ROLE,
        "reference": REFERENCE_ROLE,
    }:
        issues.append("physical provenance canonical_roles differ")
    if payload.get("identity_semantics") != "physical_source_h5_row":
        issues.append(
            "physical provenance identity_semantics must be 'physical_source_h5_row'"
        )
    if payload.get("group_key") != group_key:
        issues.append("physical provenance group_key differs from pooled audit")
    if list(payload.get("identity_columns", [])) != list(identity_columns):
        issues.append("physical provenance identity_columns differ from pooled audit")

    for side_plan in (real, pred):
        observed = payload.get(side_plan.side)
        if not isinstance(observed, dict):
            issues.append(f"physical provenance misses {side_plan.side} binding")
            continue
        expected = {
            "path": str(side_plan.path),
            "sha256": side_plan.report.get("sha256"),
            "ordered_identity_sha256": side_plan.report.get("ordered_identity_sha256"),
            "ordered_identity_group_sha256": side_plan.report.get(
                "ordered_identity_group_sha256"
            ),
            "identity_group_multiset_sha256": (
                side_plan.report.get("identity_group_multiset") or {}
            ).get("sha256"),
        }
        for field, expected_value in expected.items():
            if observed.get(field) != expected_value:
                issues.append(
                    f"physical provenance {side_plan.side}.{field} differs: "
                    f"expected={expected_value!r}, observed={observed.get(field)!r}"
                )

    declared_stages = payload.get("stage_manifests")
    expected_stages = sorted(
        (str(record.get("path")), str(record.get("sha256"))) for record in stage_records
    )
    if not isinstance(declared_stages, list):
        issues.append("physical provenance stage_manifests is not a list")
    else:
        try:
            observed_stages = sorted(
                (str(record["path"]), str(record["sha256"]))
                for record in declared_stages
            )
            if observed_stages != expected_stages:
                issues.append("physical provenance stage manifest path/SHA set differs")
        except (KeyError, TypeError):
            issues.append("physical provenance stage_manifests entries are malformed")

    mapping = payload.get("source_row_mapping")
    mapping_report: dict[str, Any] = {}
    if not isinstance(mapping, dict):
        issues.append("physical provenance misses source_row_mapping binding")
    else:
        mapping_report, mapping_issues = audit_mapping_and_source_bindings(
            payload=payload,
            mapping=mapping,
            group_key=group_key,
            reference=reference,
            expected_rows=int(real.report["n_obs"]),
            expected_identity_sha256=real.report.get("ordered_identity_sha256"),
            expected_identity_group_sha256=real.report.get(
                "ordered_identity_group_sha256"
            ),
            stage_records=stage_records,
        )
        issues.extend(mapping_issues)

    report = {
        "required": required,
        "status": "verified" if not issues else "invalid",
        "passed": not issues,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "schema_version": payload.get("schema_version"),
        "identity_semantics": payload.get("identity_semantics"),
        "source_row_mapping": mapping_report,
        "issues": issues,
    }
    return report, issues


def audit_pooled_inputs(
    real_path: Path,
    pred_path: Path,
    *,
    group_key: str,
    reference: str,
    identity_columns: Sequence[str],
    stage_manifests: Sequence[Path] = (),
    identity_provenance_manifest: Path | None = None,
    expected_targets: Sequence[str] | None = None,
    expected_target_count: int | None = None,
    expected_stage_count: int | None = None,
    expected_total_cells: int | None = None,
    expected_treated_cells: int | None = None,
    expected_reference_cells: int | None = None,
    expected_genes: int | None = None,
    require_aligned_identities: bool = True,
    require_physical_identity_attestation: bool = True,
    verify_stage_artifact_hashes: bool = True,
    verify_stage_row_union: bool = True,
    hash_inputs: bool = True,
) -> PooledInputPlan:
    """Audit metadata and construct immutable row plans without loading ``X``.

    ``identity_columns`` must identify physical source rows, not expression
    values.  Structural uniqueness never proves that semantic claim; production
    use therefore requires a separately hashed provenance attestation.
    """

    if not group_key:
        raise ValueError("group_key must be non-empty")
    if not reference:
        raise ValueError("reference must be non-empty")
    if require_physical_identity_attestation and not hash_inputs:
        raise ValueError("physical provenance binding requires hash_inputs=True")
    normalized_id_columns = tuple(dict.fromkeys(map(str, identity_columns)))
    provenance_preflight_issues: list[str] = []
    missing_physical_columns = [
        column for column in PHYSICAL_IDENTITY_COLUMNS if column not in normalized_id_columns
    ]
    if require_physical_identity_attestation and missing_physical_columns:
        provenance_preflight_issues.append(
            "production physical identity_columns miss explicit fields: "
            f"{missing_physical_columns}"
        )
    if require_physical_identity_attestation and identity_provenance_manifest is None:
        provenance_preflight_issues.append(
            "physical source-row provenance is unverified: a bound provenance "
            "attestation is required before input hashing, stage audit, or MWU"
        )
    if provenance_preflight_issues:
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "passed": False,
            "status": "blocked_before_input_scan",
            "group_key": group_key,
            "reference": reference,
            "identity_columns": list(normalized_id_columns),
            "identity_semantics": "unverified",
            "real_path": str(Path(real_path).expanduser().absolute()),
            "pred_path": str(Path(pred_path).expanduser().absolute()),
            "identity_provenance": {
                "required": True,
                "status": (
                    "missing_required_attestation"
                    if identity_provenance_manifest is None
                    else "invalid_identity_columns"
                ),
                "passed": False,
            },
            "expression_scan": {
                "passed": False,
                "status": "not_started_due_to_unverified_physical_identity",
            },
            "required_physical_identity_columns": list(PHYSICAL_IDENTITY_COLUMNS),
            "issues": provenance_preflight_issues,
        }
        raise InputAuditError(report)
    issues: list[str] = []
    real, real_issues = _inspect_side(
        "real",
        Path(real_path),
        group_key=group_key,
        reference=reference,
        identity_columns=normalized_id_columns,
        hash_input=hash_inputs,
    )
    pred, pred_issues = _inspect_side(
        "pred",
        Path(pred_path),
        group_key=group_key,
        reference=reference,
        identity_columns=normalized_id_columns,
        hash_input=hash_inputs,
    )
    issues.extend(real_issues)
    issues.extend(pred_issues)

    if real.genes != pred.genes:
        issues.append("real/pred gene names or gene order differ")
    if expected_genes is not None:
        for side in (real, pred):
            if len(side.genes) != expected_genes:
                issues.append(
                    f"{side.side}: expected {expected_genes} genes, observed {len(side.genes)}"
                )

    real_groups = set(real.group_rows)
    pred_groups = set(pred.group_rows)
    if real_groups != pred_groups:
        issues.append(
            "real/pred group sets differ: "
            f"real_only={sorted(real_groups - pred_groups)}, "
            f"pred_only={sorted(pred_groups - real_groups)}"
        )
    if reference not in real_groups:
        issues.append(f"real: reference group {reference!r} is absent")
    if reference not in pred_groups:
        issues.append(f"pred: reference group {reference!r} is absent")

    actual_targets = sorted(real_groups - {reference})
    if expected_targets is not None:
        normalized_targets = tuple(map(str, expected_targets))
        if len(set(normalized_targets)) != len(normalized_targets):
            issues.append("expected target list contains duplicates")
        expected_set = set(normalized_targets)
        if set(actual_targets) != expected_set:
            issues.append(
                "observed target groups differ from expected targets: "
                f"missing={sorted(expected_set - set(actual_targets))}, "
                f"unexpected={sorted(set(actual_targets) - expected_set)}"
            )
        targets = normalized_targets
    else:
        targets = tuple(actual_targets)
    if expected_target_count is not None and len(targets) != expected_target_count:
        issues.append(
            f"expected {expected_target_count} target groups, observed/declared {len(targets)}"
        )

    real_counts = real.report.get("group_counts", {})
    pred_counts = pred.report.get("group_counts", {})
    if real_counts != pred_counts:
        mismatched = {
            group: {"real": real_counts.get(group), "pred": pred_counts.get(group)}
            for group in sorted(set(real_counts) | set(pred_counts))
            if real_counts.get(group) != pred_counts.get(group)
        }
        issues.append(f"real/pred group counts differ: {mismatched}")

    for side in (real, pred):
        reference_cells = int(side.group_rows.get(reference, np.empty(0)).size)
        treated_cells = int(sum(side.group_rows.get(target, np.empty(0)).size for target in targets))
        if expected_total_cells is not None and side.report["n_obs"] != expected_total_cells:
            issues.append(
                f"{side.side}: expected {expected_total_cells} total cells, "
                f"observed {side.report['n_obs']}"
            )
        if expected_reference_cells is not None and reference_cells != expected_reference_cells:
            issues.append(
                f"{side.side}: expected {expected_reference_cells} reference cells, "
                f"observed {reference_cells}"
            )
        if expected_treated_cells is not None and treated_cells != expected_treated_cells:
            issues.append(
                f"{side.side}: expected {expected_treated_cells} treated cells, "
                f"observed {treated_cells}"
            )
        if reference_cells == 0:
            issues.append(f"{side.side}: reference group {reference!r} is empty")
        for target in targets:
            if side.group_rows.get(target, np.empty(0)).size == 0:
                issues.append(f"{side.side}: target group {target!r} is empty")

    if require_aligned_identities:
        real_identity = real.report.get("ordered_identity_sha256")
        pred_identity = pred.report.get("ordered_identity_sha256")
        if real_identity is None or pred_identity is None:
            issues.append("cannot verify ordered real/pred physical source identity alignment")
        elif real_identity != pred_identity:
            issues.append("ordered real/pred physical source identities differ")
        real_identity_group = real.report.get("ordered_identity_group_sha256")
        pred_identity_group = pred.report.get("ordered_identity_group_sha256")
        if real_identity_group is None or pred_identity_group is None:
            issues.append("cannot verify ordered real/pred identity+group alignment")
        elif real_identity_group != pred_identity_group:
            issues.append("ordered real/pred identity+group labels differ")

    pooled_reference_cells = int(real.group_rows.get(reference, np.empty(0)).size)
    pooled_treated_cells = int(
        sum(real.group_rows.get(target, np.empty(0)).size for target in targets)
    )
    stage_records, stage_summary, stage_issues = _stage_manifest_records(
        tuple(map(Path, stage_manifests)),
        targets=targets,
        gene_axis_sha256=str(real.report["stage_compatible_gene_axis_sha256"]),
        pooled_genes=real.genes,
        genes=len(real.genes),
        pooled_treated_cells=pooled_treated_cells,
        pooled_reference_cells=pooled_reference_cells,
        pooled_total_cells=int(real.report["n_obs"]),
        identity_columns=normalized_id_columns,
        group_key=group_key,
        reference=reference,
        pooled_lineage={
            "real": real.report.get("identity_group_multiset"),
            "pred": pred.report.get("identity_group_multiset"),
        },
        verify_artifact_hashes=verify_stage_artifact_hashes,
        verify_row_union=verify_stage_row_union,
    )
    issues.extend(stage_issues)
    if expected_stage_count is not None and len(stage_records) != expected_stage_count:
        issues.append(
            f"expected {expected_stage_count} unique stage manifests, observed {len(stage_records)}"
        )

    provenance_report, provenance_issues = _audit_physical_provenance(
        identity_provenance_manifest,
        required=require_physical_identity_attestation,
        group_key=group_key,
        reference=reference,
        identity_columns=normalized_id_columns,
        real=real,
        pred=pred,
        stage_records=stage_records,
    )
    issues.extend(provenance_issues)

    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "group_key": group_key,
        "reference": reference,
        "targets": list(targets),
        "target_count": len(targets),
        "identity_columns": list(normalized_id_columns),
        "identity_semantics": (
            "physical_source_h5_row"
            if provenance_report.get("status") == "verified"
            else "unverified"
        ),
        "require_aligned_identities": require_aligned_identities,
        "require_physical_identity_attestation": require_physical_identity_attestation,
        "identity_provenance": provenance_report,
        "real": dict(real.report),
        "pred": dict(pred.report),
        "stage_manifests": stage_records,
        "stage_summary": stage_summary,
        "expectations": {
            "targets": list(map(str, expected_targets)) if expected_targets is not None else None,
            "target_count": expected_target_count,
            "stage_count": expected_stage_count,
            "total_cells_per_side": expected_total_cells,
            "treated_cells_per_side": expected_treated_cells,
            "reference_cells_per_side": expected_reference_cells,
            "genes": expected_genes,
        },
        "stage_verification": {
            "artifact_hashes": verify_stage_artifact_hashes,
            "identity_group_row_union": verify_stage_row_union,
        },
        "issues": issues,
        "passed": not issues,
    }
    if issues:
        raise InputAuditError(report)
    return PooledInputPlan(
        real=real,
        pred=pred,
        targets=targets,
        reference=reference,
        report=report,
    )


def _read_dense_block(
    matrix: Any,
    rows: np.ndarray,
    gene_start: int,
    gene_stop: int,
    *,
    row_chunk_size: int,
) -> np.ndarray:
    """Read a 2-D bounded block even when backed CSR slices decode full rows.

    AnnData 0.10.x backed CSR applies a row selection before its column slice,
    so a single large ``matrix[rows, gene_slice]`` can transiently decode all
    genes for every selected row.  Splitting the rows here puts an explicit
    upper bound on that hidden allocation.  The returned group-by-gene block
    is still bounded by ``len(rows) * gene_chunk_size``.
    """

    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if rows.size == 0:
        return np.empty((0, gene_stop - gene_start), dtype=np.float64)
    width = gene_stop - gene_start
    result = np.empty((rows.size, width), dtype=np.float64)
    for row_start in range(0, rows.size, row_chunk_size):
        row_stop = min(rows.size, row_start + row_chunk_size)
        block = _read_backed_slice(
            matrix,
            rows[row_start:row_stop],
            gene_start,
            gene_stop,
        )
        result[row_start:row_stop] = block
    return result


def _read_backed_slice(
    matrix: Any,
    rows: np.ndarray,
    gene_start: int,
    gene_stop: int,
) -> np.ndarray:
    """Read one row-bounded backed slice; split out for peak-memory tests."""

    block = matrix[rows, slice(gene_start, gene_stop)]
    if sparse.issparse(block):
        block = block.toarray()
    array = np.asarray(block)
    if array.ndim == 1:
        array = array.reshape(rows.size, gene_stop - gene_start)
    expected = (rows.size, gene_stop - gene_start)
    if array.shape != expected:
        raise PooledDEError(f"expression block has shape {array.shape}, expected {expected}")
    return np.asarray(array, dtype=np.float64, order="C")


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "nonzero_count": 0,
            "unique_count": 0,
            "minimum": None,
            "maximum": None,
            "constant": False,
        }
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    return {
        "finite_count": int(finite.size),
        "nonzero_count": int(np.count_nonzero(finite)),
        "unique_count": int(np.unique(finite).size),
        "minimum": minimum,
        "maximum": maximum,
        "constant": bool(minimum == maximum and finite.size == values.size),
    }


def _sample_mean(values: np.ndarray, config: MWUConfig) -> float:
    if config.is_log1p:
        if config.exp_post_agg:
            return float(np.expm1(np.mean(values, dtype=np.float64)))
        return float(np.mean(np.expm1(values), dtype=np.float64))
    return float(np.mean(values, dtype=np.float64))


def _fold_change(target_mean: float, reference_mean: float, clip: float) -> float:
    if target_mean == 0 and reference_mean == 0:
        return 1.0
    if reference_mean == 0:
        return float(clip)
    if target_mean == 0:
        return float(1.0 / clip)
    return float(target_mean / reference_mean)


def compute_raw_mwu_test(
    target_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    side: str,
    target: str,
    reference: str,
    feature: str,
    gene_index: int,
    config: MWUConfig,
) -> dict[str, Any]:
    """Compute one raw test and retain enough evidence to diagnose failure."""

    target_array = np.asarray(target_values, dtype=np.float64).reshape(-1)
    reference_array = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    target_support = _finite_summary(target_array)
    reference_support = _finite_summary(reference_array)
    record: dict[str, Any] = {
        "side": side,
        "target": str(target),
        "reference": str(reference),
        "feature": str(feature),
        "gene_index": int(gene_index),
        "n_target": int(target_array.size),
        "n_reference": int(reference_array.size),
        "target_finite_count": target_support["finite_count"],
        "reference_finite_count": reference_support["finite_count"],
        "target_nonzero_count": target_support["nonzero_count"],
        "reference_nonzero_count": reference_support["nonzero_count"],
        "target_zero_count": target_support["finite_count"] - target_support["nonzero_count"],
        "reference_zero_count": (
            reference_support["finite_count"] - reference_support["nonzero_count"]
        ),
        "target_zero_rate": (
            (target_support["finite_count"] - target_support["nonzero_count"])
            / target_array.size
            if target_array.size
            else np.nan
        ),
        "reference_zero_rate": (
            (reference_support["finite_count"] - reference_support["nonzero_count"])
            / reference_array.size
            if reference_array.size
            else np.nan
        ),
        "target_unique_count": target_support["unique_count"],
        "reference_unique_count": reference_support["unique_count"],
        "target_min": target_support["minimum"],
        "target_max": target_support["maximum"],
        "reference_min": reference_support["minimum"],
        "reference_max": reference_support["maximum"],
        "target_constant": target_support["constant"],
        "reference_constant": reference_support["constant"],
        "target_mean": np.nan,
        "reference_mean": np.nan,
        "mean_difference": np.nan,
        "percent_change": np.nan,
        "percent_change_status": "not_computed",
        "fold_change": np.nan,
        "rank_biserial": np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
        "status": "invalid",
        "invalid_reason": "",
        "warnings": "",
    }

    if target_array.size == 0 or reference_array.size == 0:
        record["status"] = "empty_group"
        record["invalid_reason"] = (
            f"empty sample: n_target={target_array.size}, n_reference={reference_array.size}"
        )
        return record
    if target_support["finite_count"] != target_array.size or reference_support[
        "finite_count"
    ] != reference_array.size:
        record["status"] = "nonfinite_input"
        record["invalid_reason"] = (
            "input contains non-finite values: "
            f"target={target_array.size - target_support['finite_count']}, "
            f"reference={reference_array.size - reference_support['finite_count']}"
        )
        return record

    with np.errstate(over="ignore", invalid="ignore"):
        target_mean = _sample_mean(target_array, config)
        reference_mean = _sample_mean(reference_array, config)
    record["target_mean"] = target_mean
    record["reference_mean"] = reference_mean
    record["mean_difference"] = target_mean - reference_mean
    if reference_mean == 0:
        if target_mean == 0:
            record["percent_change"] = 0.0
            record["percent_change_status"] = "both_means_zero_defined_zero"
        else:
            record["percent_change_status"] = "undefined_reference_mean_zero"
    else:
        record["percent_change"] = (target_mean - reference_mean) / reference_mean
        record["percent_change_status"] = "defined"
    record["fold_change"] = _fold_change(
        target_mean, reference_mean, config.fold_change_clip
    )
    if not np.isfinite(target_mean) or not np.isfinite(reference_mean):
        record["status"] = "nonfinite_derived_mean"
        record["invalid_reason"] = "mean transformation produced a non-finite value"
        return record
    if not np.isfinite(record["fold_change"]):
        record["status"] = "nonfinite_fold_change"
        record["invalid_reason"] = "fold-change calculation produced a non-finite value"
        return record

    both_constant = bool(target_support["constant"] and reference_support["constant"])
    same_constant = bool(
        both_constant and target_array[0] == reference_array[0]
    )
    if same_constant:
        record.update(
            {
                "rank_biserial": 0.0,
                "statistic": float(target_array.size * reference_array.size / 2.0),
                "p_value": 1.0,
                "status": "equal_constant",
            }
        )
        return record

    caught_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            result = mannwhitneyu(
                target_array,
                reference_array,
                use_continuity=config.use_continuity,
                alternative=config.alternative,
                method=config.method,
            )
        caught_warnings = [
            f"{type(item.message).__name__}: {item.message}" for item in warning_records
        ]
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception as error:  # diagnostic capture; the family still fails closed
        record["status"] = "mwu_exception"
        record["invalid_reason"] = f"{type(error).__name__}: {error}"
        return record
    record["warnings"] = " | ".join(caught_warnings)
    max_statistic = float(target_array.size * reference_array.size)
    if not np.isfinite(statistic) or statistic < 0 or statistic > max_statistic:
        record["status"] = "invalid_mwu_statistic"
        record["invalid_reason"] = (
            f"MWU statistic is non-finite or outside [0,{max_statistic}]: {statistic}"
        )
        record["statistic"] = statistic
        record["p_value"] = p_value
        return record
    if not np.isfinite(p_value) or p_value < 0 or p_value > 1:
        record["status"] = "invalid_mwu_p_value"
        record["invalid_reason"] = f"MWU p-value is non-finite or outside [0,1]: {p_value}"
        record["statistic"] = statistic
        record["p_value"] = p_value
        return record

    record.update(
        {
            "rank_biserial": float(2.0 * statistic / max_statistic - 1.0),
            "statistic": statistic,
            "p_value": p_value,
            "status": "different_constants" if both_constant else "ok",
        }
    )
    return record


def iter_raw_mwu_chunks(
    plan: SideInputPlan,
    *,
    targets: Sequence[str],
    reference: str,
    config: MWUConfig,
    gene_chunk_size: int,
    row_chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield raw target-by-gene records while keeping expression memory bounded."""

    if gene_chunk_size <= 0:
        raise ValueError("gene_chunk_size must be positive")
    if _file_identity(plan.path) != plan.file_identity:
        raise PooledDEError(f"{plan.side} input changed after audit: {plan.path}")
    adata = ad.read_h5ad(plan.path, backed="r")
    try:
        if tuple(map(str, adata.var_names.tolist())) != plan.genes:
            raise PooledDEError(f"{plan.side} gene order changed after audit")
        matrix = adata.X
        reference_rows = plan.group_rows.get(reference, np.empty(0, dtype=np.int64))
        for gene_start in range(0, len(plan.genes), gene_chunk_size):
            gene_stop = min(len(plan.genes), gene_start + gene_chunk_size)
            reference_block = _read_dense_block(
                matrix,
                reference_rows,
                gene_start,
                gene_stop,
                row_chunk_size=row_chunk_size,
            )
            records: list[dict[str, Any]] = []
            for target in targets:
                target_rows = plan.group_rows.get(target, np.empty(0, dtype=np.int64))
                target_block = _read_dense_block(
                    matrix,
                    target_rows,
                    gene_start,
                    gene_stop,
                    row_chunk_size=row_chunk_size,
                )
                for offset, feature in enumerate(plan.genes[gene_start:gene_stop]):
                    records.append(
                        compute_raw_mwu_test(
                            target_block[:, offset],
                            reference_block[:, offset],
                            side=plan.side,
                            target=target,
                            reference=reference,
                            feature=feature,
                            gene_index=gene_start + offset,
                            config=config,
                        )
                    )
            yield pd.DataFrame.from_records(records)
    finally:
        _close_backed(adata)
    if _file_identity(plan.path) != plan.file_identity:
        raise PooledDEError(f"{plan.side} input changed while raw tests were computed")


def diagnose_raw_family(
    raw: pd.DataFrame,
    *,
    side: str,
    targets: Sequence[str],
    genes: Sequence[str],
    reference: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return concrete invalid/missing/duplicate tests without applying BH."""

    required = {
        "side",
        "target",
        "reference",
        "feature",
        "status",
        "p_value",
        "statistic",
        "invalid_reason",
    }
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        diagnostic = pd.DataFrame(
            [
                {
                    "side": side,
                    "status": "missing_columns",
                    "invalid_reason": f"missing raw columns: {missing_columns}",
                }
            ]
        )
        return diagnostic, [f"missing raw columns: {missing_columns}"]

    diagnostics: list[dict[str, Any]] = []
    issues: list[str] = []
    expected_targets = tuple(map(str, targets))
    expected_genes = tuple(map(str, genes))
    expected_pairs = {(target, gene) for target in expected_targets for gene in expected_genes}

    side_values = set(raw["side"].astype(str))
    if side_values != {side}:
        issues.append(f"side column is {sorted(side_values)}, expected only {side!r}")
    reference_values = set(raw["reference"].astype(str))
    if reference_values != {reference}:
        issues.append(
            f"reference column is {sorted(reference_values)}, expected only {reference!r}"
        )

    keys = list(zip(raw["target"].astype(str), raw["feature"].astype(str)))
    key_counts = pd.Series(keys, dtype="object").value_counts()
    duplicate_keys = key_counts[key_counts > 1]
    if not duplicate_keys.empty:
        issues.append(f"found {len(duplicate_keys)} duplicate target-feature keys")
        for (target, feature), count in duplicate_keys.items():
            diagnostics.append(
                {
                    "side": side,
                    "target": target,
                    "reference": reference,
                    "feature": feature,
                    "status": "duplicate_test",
                    "invalid_reason": f"observed {int(count)} copies",
                }
            )
    actual_pairs = set(keys)
    missing_pairs = sorted(expected_pairs - actual_pairs)
    unexpected_pairs = sorted(actual_pairs - expected_pairs)
    if missing_pairs:
        issues.append(f"missing {len(missing_pairs)} expected target-feature tests")
        diagnostics.extend(
            {
                "side": side,
                "target": target,
                "reference": reference,
                "feature": feature,
                "status": "missing_test",
                "invalid_reason": "expected target-feature test is absent",
            }
            for target, feature in missing_pairs
        )
    if unexpected_pairs:
        issues.append(f"found {len(unexpected_pairs)} unexpected target-feature tests")
        diagnostics.extend(
            {
                "side": side,
                "target": target,
                "reference": reference,
                "feature": feature,
                "status": "unexpected_test",
                "invalid_reason": "target-feature test is outside the declared family",
            }
            for target, feature in unexpected_pairs
        )

    status_invalid = ~raw["status"].isin(VALID_TEST_STATUSES)
    p_values = pd.to_numeric(raw["p_value"], errors="coerce")
    statistic = pd.to_numeric(raw["statistic"], errors="coerce")
    numeric_invalid = (~np.isfinite(p_values)) | (p_values < 0) | (p_values > 1)
    statistic_invalid = ~np.isfinite(statistic)
    invalid_rows = raw.loc[status_invalid | numeric_invalid | statistic_invalid]
    if not invalid_rows.empty:
        issues.append(f"found {len(invalid_rows)} invalid raw statistical tests")
        diagnostics.extend(invalid_rows.to_dict(orient="records"))

    return pd.DataFrame.from_records(diagnostics), issues


def apply_complete_family_bh(
    raw: pd.DataFrame,
    *,
    side: str,
    targets: Sequence[str],
    genes: Sequence[str],
    reference: str,
) -> pd.DataFrame:
    """Validate a complete side and perform exactly one side-wide BH call."""

    diagnostics, issues = diagnose_raw_family(
        raw, side=side, targets=targets, genes=genes, reference=reference
    )
    if issues:
        raise RawFamilyError(side, diagnostics, issues)
    target_order = {str(value): index for index, value in enumerate(targets)}
    gene_order = {str(value): index for index, value in enumerate(genes)}
    ordered = raw.copy()
    ordered["_target_order"] = ordered["target"].astype(str).map(target_order)
    ordered["_gene_order"] = ordered["feature"].astype(str).map(gene_order)
    ordered = ordered.sort_values(["_target_order", "_gene_order"], kind="stable")
    ordered = ordered.drop(columns=["_target_order", "_gene_order"]).reset_index(drop=True)
    p_values = ordered["p_value"].to_numpy(dtype=np.float64, copy=True)
    # This is intentionally the only BH call: chunks never receive correction.
    fdr = np.asarray(false_discovery_control(p_values, method="bh"), dtype=np.float64)
    if fdr.shape != p_values.shape or not np.isfinite(fdr).all():
        raise RawFamilyError(
            side,
            pd.DataFrame(
                [
                    {
                        "side": side,
                        "status": "invalid_bh_result",
                        "invalid_reason": "BH returned wrong shape or non-finite values",
                    }
                ]
            ),
            ["BH returned wrong shape or non-finite values"],
        )
    if bool(((fdr < 0) | (fdr > 1)).any()):
        raise RawFamilyError(
            side,
            pd.DataFrame(
                [
                    {
                        "side": side,
                        "status": "invalid_bh_result",
                        "invalid_reason": "BH returned values outside [0,1]",
                    }
                ]
            ),
            ["BH returned values outside [0,1]"],
        )
    ordered["fdr"] = fdr
    return ordered


def celleval_de_frame(adjusted: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CELLEVAL_COLUMNS if column not in adjusted.columns]
    if missing:
        raise ValueError(f"adjusted DE table lacks CellEval columns: {missing}")
    return adjusted.loc[:, list(CELLEVAL_COLUMNS)].copy()


def _finalized_expression_input_audit(
    plan: PooledInputPlan,
    combined: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Attach the all-values finite scan proven by complete group/gene traversal."""

    report = dict(plan.report)
    scan: dict[str, Any] = {
        "strategy": (
            "backed matrix traversed by disjoint cytokine row groups and gene chunks; "
            "implicit CSR zeros included after block densification"
        ),
        "sides": {},
    }
    issues = list(report.get("issues", []))
    for side_plan in (plan.real, plan.pred):
        raw = combined[side_plan.side]
        target_complete = raw["target_finite_count"].to_numpy() == raw[
            "n_target"
        ].to_numpy()
        reference_complete = raw["reference_finite_count"].to_numpy() == raw[
            "n_reference"
        ].to_numpy()
        all_values_finite = bool(target_complete.all() and reference_complete.all())
        nonfinite_records = int((~target_complete | ~reference_complete).sum())
        target_values_scanned = int(raw["n_target"].sum())
        target_finite_values = int(raw["target_finite_count"].sum())
        reference_once = (
            raw.sort_values(["gene_index", "target"], kind="stable")
            .drop_duplicates("gene_index", keep="first")
        )
        reference_values_scanned = int(reference_once["n_reference"].sum())
        reference_finite_values = int(reference_once["reference_finite_count"].sum())
        observed_values_scanned = target_values_scanned + reference_values_scanned
        observed_finite_values = target_finite_values + reference_finite_values
        observed_nonfinite_values = observed_values_scanned - observed_finite_values
        covered_groups = set(map(str, side_plan.group_rows))
        expected_groups = set(plan.targets) | {plan.reference}
        partitioned_rows = int(sum(rows.size for rows in side_plan.group_rows.values()))
        full_coverage = bool(
            covered_groups == expected_groups
            and partitioned_rows == int(side_plan.report["n_obs"])
            and len(raw) == len(plan.targets) * len(side_plan.genes)
            and observed_values_scanned
            == int(side_plan.report["n_obs"]) * len(side_plan.genes)
        )
        side_report = {
            "all_values_finite": all_values_finite,
            "nonfinite_test_records": nonfinite_records,
            "observed_values_scanned": observed_values_scanned,
            "observed_finite_values": observed_finite_values,
            "observed_nonfinite_values": observed_nonfinite_values,
            "reference_counts_deduplicated_across_target_comparisons": True,
            "all_rows_partitioned_once_by_group": (
                partitioned_rows == int(side_plan.report["n_obs"])
            ),
            "covered_groups": sorted(covered_groups),
            "covered_genes": len(side_plan.genes),
            "expected_matrix_values": int(side_plan.report["n_obs"])
            * len(side_plan.genes),
            "full_group_gene_coverage": full_coverage,
        }
        scan["sides"][side_plan.side] = side_report
        if not all_values_finite:
            issues.append(
                f"{side_plan.side}: expression scan found {observed_nonfinite_values} "
                f"non-finite matrix values across {nonfinite_records} comparison records"
            )
        if not full_coverage:
            issues.append(f"{side_plan.side}: expression finite scan did not cover the full matrix")
    scan["passed"] = all(
        value["all_values_finite"] and value["full_group_gene_coverage"]
        for value in scan["sides"].values()
    )
    report["expression_scan"] = scan
    report["issues"] = issues
    report["passed"] = not issues and bool(scan["passed"])
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in records:
                normalized = {str(key): _jsonable_scalar(value) for key, value in record.items()}
                handle.write(
                    json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def run_pooled_de_pipeline(
    *,
    real_path: Path,
    pred_path: Path,
    output_dir: Path,
    group_key: str,
    reference: str,
    identity_columns: Sequence[str],
    stage_manifests: Sequence[Path] = (),
    identity_provenance_manifest: Path | None = None,
    expected_targets: Sequence[str] | None = None,
    expected_target_count: int | None = None,
    expected_stage_count: int | None = None,
    expected_total_cells: int | None = None,
    expected_treated_cells: int | None = None,
    expected_reference_cells: int | None = None,
    expected_genes: int | None = None,
    require_aligned_identities: bool = True,
    require_physical_identity_attestation: bool = True,
    verify_stage_artifact_hashes: bool = True,
    verify_stage_row_union: bool = True,
    hash_inputs: bool = True,
    gene_chunk_size: int = 128,
    row_chunk_size: int = 4_096,
    config: MWUConfig | None = None,
) -> dict[str, Any]:
    """Run the audited raw-cache/BH pipeline into a new output directory."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.mkdir(parents=True)
    config = config or MWUConfig()
    input_audit_path = destination / "input_audit.json"
    try:
        plan = audit_pooled_inputs(
            real_path,
            pred_path,
            group_key=group_key,
            reference=reference,
            identity_columns=identity_columns,
            stage_manifests=stage_manifests,
            identity_provenance_manifest=identity_provenance_manifest,
            expected_targets=expected_targets,
            expected_target_count=expected_target_count,
            expected_stage_count=expected_stage_count,
            expected_total_cells=expected_total_cells,
            expected_treated_cells=expected_treated_cells,
            expected_reference_cells=expected_reference_cells,
            expected_genes=expected_genes,
            require_aligned_identities=require_aligned_identities,
            require_physical_identity_attestation=require_physical_identity_attestation,
            verify_stage_artifact_hashes=verify_stage_artifact_hashes,
            verify_stage_row_union=verify_stage_row_union,
            hash_inputs=hash_inputs,
        )
    except InputAuditError as error:
        _atomic_json(input_audit_path, error.report)
        _atomic_json(
            destination / "de_cache_manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "created_utc": utc_now(),
                "passed": False,
                "phase": "input_audit",
                "error": str(error),
                "input_audit": _artifact_record(input_audit_path),
            },
        )
        raise
    except Exception as error:
        failure_report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "passed": False,
            "status": "input_unreadable_or_uninspectable",
            "real_path": str(Path(real_path).expanduser().absolute()),
            "pred_path": str(Path(pred_path).expanduser().absolute()),
            "identity_provenance_manifest": (
                str(identity_provenance_manifest.expanduser().absolute())
                if identity_provenance_manifest is not None
                else None
            ),
            "issues": [f"{type(error).__name__}: {error}"],
        }
        _atomic_json(input_audit_path, failure_report)
        _atomic_json(
            destination / "de_cache_manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "created_utc": utc_now(),
                "passed": False,
                "phase": "input_audit",
                "error": f"{type(error).__name__}: {error}",
                "input_audit": _artifact_record(input_audit_path),
            },
        )
        raise
    raw_frames: dict[str, list[pd.DataFrame]] = {"real": [], "pred": []}
    raw_records: dict[str, list[dict[str, Any]]] = {"real": [], "pred": []}
    try:
        for side_plan in (plan.real, plan.pred):
            for chunk_index, frame in enumerate(
                iter_raw_mwu_chunks(
                    side_plan,
                    targets=plan.targets,
                    reference=plan.reference,
                    config=config,
                    gene_chunk_size=gene_chunk_size,
                    row_chunk_size=row_chunk_size,
                )
            ):
                if frame.empty:
                    raise PooledDEError(
                        f"{side_plan.side} raw chunk {chunk_index} unexpectedly contains no rows"
                    )
                gene_start = int(frame["gene_index"].min())
                gene_stop = int(frame["gene_index"].max()) + 1
                chunk_path = (
                    destination
                    / "raw_de_stats"
                    / side_plan.side
                    / f"genes_{gene_start:04d}_{gene_stop:04d}.parquet"
                )
                _atomic_parquet(chunk_path, frame)
                raw_frames[side_plan.side].append(frame)
                raw_records[side_plan.side].append(
                    _artifact_record(chunk_path, rows=len(frame))
                )

        combined = {
            side: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for side, frames in raw_frames.items()
        }
        finalized_input_audit = _finalized_expression_input_audit(plan, combined)
        _atomic_json(input_audit_path, finalized_input_audit)
        support = pd.concat(
            [combined["real"].loc[:, list(SUPPORT_COLUMNS)], combined["pred"].loc[:, list(SUPPORT_COLUMNS)]],
            ignore_index=True,
        )
        support_path = destination / "group_gene_support.parquet"
        _atomic_parquet(support_path, support)

        diagnostics_by_side: dict[str, pd.DataFrame] = {}
        issues_by_side: dict[str, list[str]] = {}
        for side in ("real", "pred"):
            diagnostics, issues = diagnose_raw_family(
                combined[side],
                side=side,
                targets=plan.targets,
                genes=plan.real.genes,
                reference=plan.reference,
            )
            diagnostics_by_side[side] = diagnostics
            issues_by_side[side] = issues
        diagnostic_records: list[dict[str, Any]] = []
        for side in ("real", "pred"):
            diagnostic_records.extend(diagnostics_by_side[side].to_dict(orient="records"))
        invalid_path = destination / "invalid_tests.jsonl"

        if any(issues_by_side.values()):
            _atomic_jsonl(invalid_path, diagnostic_records)
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "created_utc": utc_now(),
                "passed": False,
                "phase": "raw_family_validation",
                "statistical_config": config.as_dict(),
                "gene_chunk_size": gene_chunk_size,
                "row_chunk_size": row_chunk_size,
                "issues": issues_by_side,
                "raw_chunks": raw_records,
                "input_audit": _artifact_record(input_audit_path),
                "group_gene_support": _artifact_record(support_path, rows=len(support)),
                "invalid_tests": _artifact_record(invalid_path, rows=len(diagnostic_records)),
            }
            _atomic_json(destination / "de_cache_manifest.json", manifest)
            failing_side = next(side for side in ("real", "pred") if issues_by_side[side])
            raise RawFamilyError(
                failing_side,
                diagnostics_by_side[failing_side],
                issues_by_side[failing_side],
            )

        adjusted: dict[str, pd.DataFrame] = {}
        try:
            for side in ("real", "pred"):
                adjusted[side] = apply_complete_family_bh(
                    combined[side],
                    side=side,
                    targets=plan.targets,
                    genes=plan.real.genes,
                    reference=plan.reference,
                )
        except RawFamilyError as error:
            bh_records = error.diagnostics.to_dict(orient="records")
            _atomic_jsonl(invalid_path, bh_records)
            _atomic_json(
                destination / "de_cache_manifest.json",
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "created_utc": utc_now(),
                    "passed": False,
                    "phase": "bh_validation",
                    "error": str(error),
                    "issues": {error.side: list(error.issues)},
                    "statistical_config": config.as_dict(),
                    "gene_chunk_size": gene_chunk_size,
                    "row_chunk_size": row_chunk_size,
                    "raw_chunks": raw_records,
                    "input_audit": _artifact_record(input_audit_path),
                    "group_gene_support": _artifact_record(
                        support_path, rows=len(support)
                    ),
                    "invalid_tests": _artifact_record(
                        invalid_path, rows=len(bh_records)
                    ),
                },
            )
            raise
        _atomic_jsonl(invalid_path, [])
        de_paths: dict[str, Path] = {}
        for side in ("real", "pred"):
            de_path = destination / f"{side}_de.csv"
            _atomic_csv(de_path, celleval_de_frame(adjusted[side]))
            de_paths[side] = de_path

        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "passed": True,
            "phase": "complete",
            "statistical_config": config.as_dict(),
            "gene_chunk_size": gene_chunk_size,
            "row_chunk_size": row_chunk_size,
            "bh_scope": "separate complete family per side; never per chunk or across sides",
            "implementation": _artifact_record(Path(__file__)),
            "targets": list(plan.targets),
            "genes": len(plan.real.genes),
            "expected_tests_per_side": len(plan.targets) * len(plan.real.genes),
            "input_audit": _artifact_record(input_audit_path),
            "group_gene_support": _artifact_record(support_path, rows=len(support)),
            "invalid_tests": _artifact_record(invalid_path, rows=0),
            "raw_chunks": raw_records,
            "de": {
                side: _artifact_record(de_paths[side], rows=len(adjusted[side]))
                for side in ("real", "pred")
            },
            "status_counts": {
                side: {
                    str(status): int(count)
                    for status, count in combined[side]["status"].value_counts().items()
                }
                for side in ("real", "pred")
            },
        }
        manifest_path = destination / "de_cache_manifest.json"
        _atomic_json(manifest_path, manifest)
        return manifest
    except (InputAuditError, RawFamilyError):
        raise
    except Exception as error:
        manifest_path = destination / "de_cache_manifest.json"
        if not input_audit_path.exists():
            incomplete_audit = dict(plan.report)
            incomplete_audit["expression_scan"] = {
                "passed": False,
                "status": "aborted_before_complete_scan",
                "error": f"{type(error).__name__}: {error}",
            }
            incomplete_audit["issues"] = list(incomplete_audit.get("issues", [])) + [
                "expression scan aborted before complete coverage"
            ]
            incomplete_audit["passed"] = False
            _atomic_json(input_audit_path, incomplete_audit)
        if not manifest_path.exists():
            _atomic_json(
                manifest_path,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "created_utc": utc_now(),
                    "passed": False,
                    "phase": "raw_computation",
                    "error": f"{type(error).__name__}: {error}",
                    "statistical_config": config.as_dict(),
                    "gene_chunk_size": gene_chunk_size,
                    "row_chunk_size": row_chunk_size,
                    "raw_chunks": raw_records,
                    "input_audit": _artifact_record(input_audit_path),
                },
            )
        raise

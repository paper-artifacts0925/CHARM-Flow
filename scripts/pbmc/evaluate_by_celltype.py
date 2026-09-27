from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pbmc_physical_identity import (
    IDENTITY_COLUMNS as PHYSICAL_IDENTITY_COLUMNS,
    IDENTITY_SEMANTICS as PHYSICAL_IDENTITY_SEMANTICS,
    REFERENCE_ROLE,
    SOURCE_FILE_COLUMN,
    STAGE_IDENTITY_SCHEMA_VERSION,
    TREATED_ROLE,
    assert_paired_identity_equal,
    attach_control_source_identity,
    normalized_identity_frame,
    provenance_uns_payload,
)

_SAVED_AUDIT_PATH = ROOT / "src/models/parent_locked_residual/saved_audit.py"
_SAVED_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "_pbmc_parent_locked_saved_audit", _SAVED_AUDIT_PATH
)
if _SAVED_AUDIT_SPEC is None or _SAVED_AUDIT_SPEC.loader is None:
    raise ImportError(f"cannot load Parent audit module: {_SAVED_AUDIT_PATH}")
_SAVED_AUDIT = importlib.util.module_from_spec(_SAVED_AUDIT_SPEC)
_SAVED_AUDIT_SPEC.loader.exec_module(_SAVED_AUDIT)
GATE_KEY = _SAVED_AUDIT.GATE_KEY
GROUP_OBS_KEY = _SAVED_AUDIT.GROUP_OBS_KEY
REFERENCE_KEY = _SAVED_AUDIT.REFERENCE_KEY
audit_saved_parent_locked_h5ad = _SAVED_AUDIT.audit_saved_parent_locked_h5ad
validate_gate_record = _SAVED_AUDIT.validate_gate_record


SCHEMA_VERSION = 2
PARENT_MAX_SINGLETON_FRACTION = 0.01
PARENT_MAX_ENDPOINT_DRIFT = 1.0e-5
OBS_KEYS = ("cytokine", "cell_type", "donor")
FORMAL_EXPECTED = {
    "cytokines": 62,
    "cell_types": 18,
    "donors": 4,
    "shards": 7,
    "nonempty_cytokine_cell_type_pairs": 1_115,
    "treated_cells": 2_260_453,
    "control_cells": 235_478,
    "total_cells": 2_495_931,
    "genes": 2_000,
}
METRICS = (
    "R2", "PDCorr", "MAE", "MSE", "PDSL1", "PDSL2", "PDScos",
    "DEOver", "DEPrec", "DirAgr", "LFCSpear", "AUROC", "AUPRC", "ES",
)
SKIP_METRICS = [
    "mse_delta", "mae_delta", "pearson_edistance", "clustering_agreement",
    "de_sig_genes_recall", "de_nsig_counts", "overlap_at_50", "overlap_at_100",
    "overlap_at_200", "overlap_at_500", "precision_at_50", "precision_at_100",
    "precision_at_200", "precision_at_500",
]
PDEX_FDR_NAN_GUARD_POLICY = "degenerate_mannwhitney_nan_to_one_v1"
CELL_EVAL_UNDEFINED_DE_POLICY = "undefined_de_metric_to_zero_with_degeneracy_proof_v1"
CELL_EVAL_UNDEFINED_DE_METRICS = ("DirAgr", "LFCSpear", "AUROC", "AUPRC")
RENAME = {
    "overlap_at_N": "DEOver", "precision_at_N": "DEPrec",
    "de_spearman_sig": "ES", "de_direction_match": "DirAgr",
    "de_spearman_lfc_sig": "LFCSpear", "pr_auc": "AUPRC",
    "roc_auc": "AUROC", "pearson_delta": "PDCorr", "mse": "MSE",
    "mae": "MAE", "discrimination_score_l1": "PDSL1",
    "discrimination_score_l2": "PDSL2",
    "discrimination_score_cosine": "PDScos",
}


def install_pdex_fdr_nan_guard(pdex_single_cell: Any | None = None) -> Callable[..., Any]:
    """Make pdex FDR correction robust to undefined all-tie tests.

    SciPy Mann-Whitney can return NaN for degenerate groups whose pooled
    ranks have zero variance. Such a comparison contains no evidence against
    the null, so its p-value is conservatively treated as one for
    Benjamini-Hochberg correction. Infinite and finite out-of-range values
    remain hard failures.
    """

    if pdex_single_cell is None:
        from pdex import _single_cell as pdex_single_cell

    current = pdex_single_cell.false_discovery_control
    if getattr(current, "_pbmc_fdr_nan_guard_policy", None) == PDEX_FDR_NAN_GUARD_POLICY:
        return current

    counters = SimpleNamespace(calls=0, calls_with_nan=0, nan_pvalues_replaced=0)

    def guarded_false_discovery_control(ps: Any, *args: Any, **kwargs: Any) -> Any:
        values = np.asarray(ps, dtype=np.float64)
        invalid_non_nan = np.isinf(values) | (
            np.isfinite(values) & ((values < 0.0) | (values > 1.0))
        )
        if invalid_non_nan.any():
            raise ValueError(
                "pdex produced infinite or finite out-of-range p-values; "
                "the PBMC NaN guard only permits degenerate Mann-Whitney NaNs"
            )
        nan_mask = np.isnan(values)
        counters.calls += 1
        if nan_mask.any():
            counters.calls_with_nan += 1
            counters.nan_pvalues_replaced += int(nan_mask.sum())
            values = values.copy()
            values[nan_mask] = 1.0
        return current(values, *args, **kwargs)

    guarded_false_discovery_control._pbmc_fdr_nan_guard_policy = (  # type: ignore[attr-defined]
        PDEX_FDR_NAN_GUARD_POLICY
    )
    guarded_false_discovery_control._pbmc_fdr_nan_guard_counters = counters  # type: ignore[attr-defined]
    guarded_false_discovery_control._pbmc_fdr_nan_guard_original = current  # type: ignore[attr-defined]
    pdex_single_cell.false_discovery_control = guarded_false_discovery_control
    return guarded_false_discovery_control


def _pdex_guard_snapshot(guard: Callable[..., Any] | None) -> dict[str, int]:
    if guard is None:
        return {"calls": 0, "calls_with_nan": 0, "nan_pvalues_replaced": 0}
    counters = guard._pbmc_fdr_nan_guard_counters  # type: ignore[attr-defined]
    return {
        "calls": int(counters.calls),
        "calls_with_nan": int(counters.calls_with_nan),
        "nan_pvalues_replaced": int(counters.nan_pvalues_replaced),
    }


def _pdex_guard_delta(
    before: Mapping[str, int], after: Mapping[str, int], *, active: bool,
) -> dict[str, Any]:
    return {
        "policy": PDEX_FDR_NAN_GUARD_POLICY,
        "active": bool(active),
        "replacement_value": 1.0,
        "fdr_calls": int(after["calls"] - before["calls"]),
        "calls_with_nan": int(after["calls_with_nan"] - before["calls_with_nan"]),
        "nan_pvalues_replaced": int(
            after["nan_pvalues_replaced"] - before["nan_pvalues_replaced"]
        ),
    }


def _empty_undefined_de_audit() -> dict[str, Any]:
    return {
        "policy": CELL_EVAL_UNDEFINED_DE_POLICY,
        "replacement_value": 0.0,
        "fdr_threshold": 0.05,
        "values_replaced": 0,
        "metrics": {
            metric: {"values_replaced": 0, "perturbations": [], "evidence": []}
            for metric in CELL_EVAL_UNDEFINED_DE_METRICS
        },
    }


def repair_undefined_de_metrics(
    frame: pd.DataFrame, real_de_path: str | Path,
    pred_de_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Conservatively score only Cell-Eval's proven DE degeneracies as zero.

    Direction agreement is undefined with no real significant genes, LFC
    Spearman is undefined with fewer than two observations or a constant rank
    input, and ROC/PR AUC are undefined when the real significance labels
    contain only one class. Other NaNs and every infinite value remain hard failures.
    """

    result = frame.copy()
    audit = _empty_undefined_de_audit()
    numeric: dict[str, pd.Series] = {}
    needs_evidence = False
    for metric in CELL_EVAL_UNDEFINED_DE_METRICS:
        require(metric in result, f"Cell-Eval omitted metric {metric}")
        values = pd.to_numeric(result[metric], errors="coerce")
        require(
            (values.notna() | result[metric].isna()).all(),
            f"Cell-Eval metric {metric} contains a non-numeric value",
        )
        require(
            not np.isinf(values.to_numpy(dtype=float)).any(),
            f"Cell-Eval metric {metric} contains infinity",
        )
        numeric[metric] = values
        needs_evidence = needs_evidence or bool(values.isna().any())

    if not needs_evidence:
        for metric, values in numeric.items():
            result[metric] = values.astype(float)
        return result, audit

    require(pred_de_path is not None, "predicted DE output is required for NaN evidence")
    de_frames = {}
    required = {"target", "feature", "fold_change", "fdr"}
    for name, path in (("real", real_de_path), ("predicted", pred_de_path)):
        de = pd.read_csv(path)
        require(required.issubset(de.columns), f"{name} DE output lacks required columns")
        de = de[["target", "feature", "fold_change", "fdr"]].copy()
        de["target"] = de["target"].astype(str)
        de["feature"] = de["feature"].astype(str)
        for column in ("fold_change", "fdr"):
            de[column] = pd.to_numeric(de[column], errors="coerce")
            require(
                np.isfinite(de[column].to_numpy(dtype=float)).all(),
                f"{name} DE output contains non-finite {column}",
            )
        fdr = de["fdr"].to_numpy(dtype=float)
        require(((fdr >= 0.0) & (fdr <= 1.0)).all(), f"{name} DE output contains invalid FDR")
        require(
            not de.duplicated(["target", "feature"]).any(),
            f"{name} DE output contains duplicate target-feature rows",
        )
        de_frames[name] = de
    real_de, pred_de = de_frames["real"], de_frames["predicted"]
    real_de["significant"] = real_de["fdr"] < 0.05
    counts = real_de.groupby("target", sort=False).agg(
        features=("feature", "size"), real_significant=("significant", "sum")
    )
    significant_join = real_de[real_de["significant"]].merge(
        pred_de[["target", "feature", "fold_change"]],
        on=["target", "feature"], how="inner", suffixes=("_real", "_pred"),
        validate="one_to_one",
    )

    for metric, values in numeric.items():
        for index in values.index[values.isna()]:
            perturbation = str(result.at[index, "perturbation"])
            require(
                perturbation in counts.index,
                f"Cell-Eval metric {metric} is NaN without real DE rows for {perturbation}",
            )
            features = int(counts.at[perturbation, "features"])
            real_significant = int(counts.at[perturbation, "real_significant"])
            if metric == "DirAgr":
                allowed = real_significant == 0
                reason = "no_real_significant_genes"
            elif metric == "LFCSpear":
                lfc_rows = significant_join[significant_join["target"] == perturbation]
                joined = len(lfc_rows)
                real_unique = int(lfc_rows["fold_change_real"].nunique())
                pred_unique = int(lfc_rows["fold_change_pred"].nunique())
                if real_significant < 2:
                    allowed = True
                    reason = "fewer_than_two_real_significant_genes"
                else:
                    require(
                        joined == real_significant,
                        f"real/predicted DE feature mismatch for {perturbation}",
                    )
                    allowed = real_unique < 2 or pred_unique < 2
                    if real_unique < 2 and pred_unique < 2:
                        reason = "constant_real_and_predicted_fold_change"
                    elif real_unique < 2:
                        reason = "constant_real_fold_change"
                    else:
                        reason = "constant_predicted_fold_change"
            else:
                allowed = real_significant in (0, features)
                reason = "one_class_real_significance_labels"
            require(
                allowed,
                f"Cell-Eval metric {metric} is NaN without recognized DE degeneracy "
                f"for {perturbation}",
            )
            values.at[index] = 0.0
            metric_audit = audit["metrics"][metric]
            metric_audit["values_replaced"] += 1
            metric_audit["perturbations"].append(perturbation)
            evidence = {
                "perturbation": perturbation,
                "reason": reason,
                "real_significant_genes": real_significant,
                "real_tested_genes": features,
            }
            if metric == "LFCSpear":
                evidence.update(
                    {
                        "joined_significant_genes": joined,
                        "real_fold_change_unique": real_unique,
                        "predicted_fold_change_unique": pred_unique,
                    }
                )
            metric_audit["evidence"].append(evidence)
            audit["values_replaced"] += 1
        result[metric] = values.astype(float)
    return result, audit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def axis_digest(values: list[str] | np.ndarray) -> str:
    return canonical_digest([str(value) for value in values])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish once; never replace an existing different result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite different output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode())


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root must be a mapping: {path}")
    return value


def strings(adata: ad.AnnData, key: str) -> np.ndarray:
    require(key in adata.obs, f"missing PBMC obs key: {key}")
    return adata.obs[key].astype(str).to_numpy()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"cannot form safe slug from {value!r}")
    return slug


def load_protocol(path: Path) -> dict:
    value = load_json(path)
    expected = value.get("expected")
    require(value.get("schema_version") == 1, "unsupported PBMC protocol schema")
    require(
        value.get("kind") == "pbmc_u2_attempt2_official_full_test_shards",
        "wrong PBMC full-test protocol kind",
    )
    require(value.get("split") == "test", "PBMC protocol is not test-only")
    require(value.get("control") == "PBS", "PBMC protocol control changed")
    require(isinstance(expected, dict), "PBMC protocol has no expected contract")
    for key in (
        "cytokines", "cell_types", "nonempty_cytokine_cell_type_pairs",
        "treated_cells", "control_cells", "total_cells", "genes",
    ):
        require(
            int(expected.get(key, -1)) == int(FORMAL_EXPECTED[key]),
            f"PBMC expected {key} changed",
        )
    require(
        int(expected["treated_cells"]) + int(expected["control_cells"])
        == int(expected["total_cells"]),
        "PBMC expected total is inconsistent",
    )
    cytokines = list(map(str, value.get("official_test_cytokines", [])))
    cell_types = list(map(str, value.get("cell_types", [])))
    donors = list(map(str, value.get("heldout_donors", [])))
    require(
        len(cytokines) == len(set(cytokines)) == int(expected["cytokines"]),
        "PBMC official cytokines are not unique and complete",
    )
    require(
        len(cell_types) == len(set(cell_types)) == int(expected["cell_types"]),
        "PBMC cell types are not unique and complete",
    )
    require(
        len(donors) == len(set(donors)) == int(FORMAL_EXPECTED["donors"]),
        "PBMC heldout donors are not unique and complete",
    )
    require(
        len({safe_slug(item) for item in cell_types}) == len(cell_types),
        "PBMC cell-type slugs collide",
    )
    controls_by_type = value.get("control_cells_by_cell_type")
    require(isinstance(controls_by_type, dict), "missing PBMC control counts by cell type")
    require(set(map(str, controls_by_type)) == set(cell_types), "PBMC control cell types changed")
    require(
        sum(int(count) for count in controls_by_type.values())
        == int(expected["control_cells"]),
        "PBMC per-cell-type control counts do not sum to the official total",
    )
    shards = value.get("shards")
    require(isinstance(shards, list), "PBMC shards must be a list")
    require(len(shards) == int(FORMAL_EXPECTED["shards"]), "PBMC shard count changed")
    require(
        {int(shard.get("id", -1)) for shard in shards} == set(range(len(shards))),
        "PBMC shard IDs must be contiguous",
    )
    flattened = [str(item) for shard in shards for item in shard.get("cytokines", [])]
    require(
        len(flattened) == len(set(flattened)) == len(cytokines),
        "PBMC shards overlap or omit cytokines",
    )
    require(set(flattened) == set(cytokines), "PBMC shard union differs from official test")
    require(
        sum(int(shard["expected_treated_cells"]) for shard in shards)
        == int(expected["treated_cells"]),
        "PBMC shard treated counts do not sum to the official total",
    )
    require(
        sum(int(shard["expected_nonempty_pairs"]) for shard in shards)
        == int(expected["nonempty_cytokine_cell_type_pairs"]),
        "PBMC shard pair counts do not sum to the official total",
    )
    return value


def shard_spec(protocol: dict, shard_id: int) -> dict:
    matches = [item for item in protocol["shards"] if int(item["id"]) == int(shard_id)]
    require(len(matches) == 1, f"unknown PBMC shard ID: {shard_id}")
    return matches[0]


def _matrix_audit(adata: ad.AnnData, *, chunk_size: int = 2_048) -> dict:
    minimum, maximum, rows = math.inf, -math.inf, 0
    for start in range(0, adata.n_obs, int(chunk_size)):
        stop = min(start + int(chunk_size), adata.n_obs)
        block = adata.X[start:stop, :]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        require(np.isfinite(values).all(), "PBMC expression contains non-finite values")
        if values.size:
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
        rows += stop - start
    require(rows == adata.n_obs, "PBMC matrix chunk audit missed rows")
    require(minimum >= 0.0, f"PBMC expression contains negative values: {minimum}")
    return {"minimum": minimum, "maximum": maximum, "finite": True, "nonnegative": True}


def _json_scalars(value: Mapping[str, Any]) -> dict:
    result = {}
    for key, item in value.items():
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[str(key)] = item
    return result


def _resume_shard_record(
    path: Path,
    *,
    protocol_sha: str,
    shard: dict,
    seed: int,
    pred: Path,
    real: Path,
    pred_sha: str,
    real_sha: str,
) -> dict | None:
    if not path.exists():
        return None
    record = load_json(path)
    require(record.get("schema_version") == SCHEMA_VERSION, "stale PBMC shard audit schema")
    require(record.get("passed") is True, "existing PBMC shard audit did not pass")
    require(int(record.get("shard_id", -1)) == int(shard["id"]), "shard audit ID changed")
    require(int(record.get("sampling_seed", -1)) == int(seed), "shard audit seed changed")
    require(record.get("protocol_sha256") == protocol_sha, "shard audit protocol changed")
    require(Path(record.get("pred", "")).resolve() == pred.resolve(), "shard pred path changed")
    require(Path(record.get("real", "")).resolve() == real.resolve(), "shard real path changed")
    require(record.get("pred_sha256") == pred_sha, "shard pred changed after audit")
    require(record.get("real_sha256") == real_sha, "shard real changed after audit")
    require(record.get("parent_gate_passed") is True, "shard Parent gate was not audited")
    identity = record.get("physical_identity")
    require(isinstance(identity, dict), "shard audit lacks physical identity evidence")
    require(
        identity.get("schema_version") == STAGE_IDENTITY_SCHEMA_VERSION
        and identity.get("passed") is True,
        "shard physical identity evidence is stale or failed",
    )
    return record


def audit_shard(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol)
    shard = shard_spec(protocol, args.shard_id)
    protocol_sha = sha256(args.protocol)
    pred_path, real_path = args.pred.resolve(), args.real.resolve()
    require(pred_path.is_file(), f"missing prediction shard: {pred_path}")
    require(real_path.is_file(), f"missing truth shard: {real_path}")
    pred_sha, real_sha = sha256(pred_path), sha256(real_path)
    resumed = _resume_shard_record(
        args.output, protocol_sha=protocol_sha, shard=shard,
        seed=args.sampling_seed, pred=pred_path, real=real_path,
        pred_sha=pred_sha, real_sha=real_sha,
    )
    if resumed is not None:
        print(json.dumps({**resumed, "resumed": True}, sort_keys=True))
        return resumed

    parent_audit = audit_saved_parent_locked_h5ad(
        pred_path,
        max_singleton_fraction=PARENT_MAX_SINGLETON_FRACTION,
        max_endpoint_mean_drift=PARENT_MAX_ENDPOINT_DRIFT,
    )
    pred = ad.read_h5ad(pred_path, backed="r")
    real = ad.read_h5ad(real_path, backed="r")
    try:
        expected_cells = int(shard["expected_treated_cells"])
        expected_genes = int(protocol["expected"]["genes"])
        require(
            int(parent_audit["valid_cells"]) == expected_cells,
            "prediction shard Parent gate valid_cells differs from treated cells",
        )
        require(
            int(pred.uns[REFERENCE_KEY]["generated_cells"]) == expected_cells,
            "prediction shard Parent reference generated_cells differs from treated cells",
        )
        require(pred.shape == real.shape, f"shard pred/real shapes differ: {pred.shape}/{real.shape}")
        require(pred.shape == (expected_cells, expected_genes), f"shard shape changed: {pred.shape}")
        genes = list(map(str, pred.var_names))
        require(genes == list(map(str, real.var_names)), "shard gene order differs")
        require(len(set(genes)) == len(genes), "shard gene names are not unique")
        require(list(map(str, pred.obs_names)) == list(map(str, real.obs_names)), "shard obs order differs")
        for key in OBS_KEYS:
            require(np.array_equal(strings(pred, key), strings(real, key)), f"pred/real {key} order differs")
        physical_identity = assert_paired_identity_equal(
            real.obs,
            pred.obs,
            group_key="cytokine",
            reference="PBS",
            context=f"PBMC treated shard {int(args.shard_id)}",
            expected_role=TREATED_ROLE,
        )
        cytokines, cell_types, donors = (
            strings(pred, "cytokine"), strings(pred, "cell_type"), strings(pred, "donor")
        )
        observed_cytokines = set(cytokines.tolist())
        require(
            observed_cytokines == set(map(str, shard["cytokines"])),
            "shard actual cytokines differ from its frozen assignment",
        )
        require("PBS" not in observed_cytokines, "treated-only shard contains PBS")
        require(set(cell_types.tolist()).issubset(set(protocol["cell_types"])), "unexpected cell type")
        require(
            set(donors.tolist()) == set(protocol["heldout_donors"]),
            "shard does not contain exactly the four heldout donors",
        )
        pairs = set(zip(cytokines.tolist(), cell_types.tolist()))
        require(len(pairs) == int(shard["expected_nonempty_pairs"]), "shard pair count changed")
        require(GROUP_OBS_KEY in pred.obs and GROUP_OBS_KEY in real.obs, "shard lost Parent groups")
        pred_groups = np.asarray(pred.obs[GROUP_OBS_KEY], dtype=np.int64)
        real_groups = np.asarray(real.obs[GROUP_OBS_KEY], dtype=np.int64)
        require(np.array_equal(pred_groups, real_groups), "pred/real Parent group order differs")
        require(int(pred_groups.min()) >= 0, "treated shard has control group IDs")
        require(GATE_KEY in real.uns, "truth shard lost Parent gate metadata")
        real_gate = validate_gate_record(
            real.uns[GATE_KEY],
            max_singleton_fraction=PARENT_MAX_SINGLETON_FRACTION,
            max_endpoint_mean_drift=PARENT_MAX_ENDPOINT_DRIFT,
        )
        require(
            int(real_gate["valid_cells"]) == expected_cells,
            "truth shard Parent gate valid_cells differs from treated cells",
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "passed": True,
            "kind": "pbmc_u2_attempt2_treated_shard_audit",
            "shard_id": int(args.shard_id),
            "sampling_seed": int(args.sampling_seed),
            "cytokines": list(map(str, shard["cytokines"])),
            "cell_types_observed": sorted(set(cell_types.tolist())),
            "donors_observed": sorted(set(donors.tolist())),
            "treated_cells": expected_cells,
            "nonempty_pairs": len(pairs),
            "control_cells": 0,
            "genes": expected_genes,
            "gene_axis_sha256": axis_digest(genes),
            "obs_order_identical": True,
            "pred": str(pred_path), "pred_sha256": pred_sha,
            "real": str(real_path), "real_sha256": real_sha,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": protocol_sha,
            "parent_gate_passed": True,
            "parent_gate": _json_scalars(parent_audit),
            "physical_identity": physical_identity,
            "real_matrix_audit": _matrix_audit(real),
        }
    finally:
        pred.file.close()
        real.file.close()
    atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return record


def dense_mean(matrix: Any) -> np.ndarray:
    value = matrix.mean(axis=0)
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value).reshape(-1).astype(np.float64)


def matrix_rows(adata: ad.AnnData, rows: np.ndarray) -> sparse.csr_matrix:
    value = adata.X[np.asarray(rows, dtype=np.int64), :]
    if sparse.issparse(value):
        result = value.tocsr().astype(np.float32)
        values = result.data
    else:
        dense = np.asarray(value, dtype=np.float32)
        values = dense
        result = sparse.csr_matrix(dense)
    require(np.isfinite(values).all(), "staged PBMC expression contains non-finite values")
    require(not values.size or float(values.min()) >= 0.0, "staged PBMC expression is negative")
    return result


def _validate_shard_records(args: argparse.Namespace, protocol: dict) -> tuple[list[dict], str]:
    require(len(args.shard_record) == int(FORMAL_EXPECTED["shards"]), "merge needs seven records")
    require(len({path.resolve() for path in args.shard_record}) == len(args.shard_record), "duplicate shard record")
    protocol_sha = sha256(args.protocol)
    records = [load_json(path) for path in args.shard_record]
    require(
        {int(record.get("shard_id", -1)) for record in records}
        == set(range(int(FORMAL_EXPECTED["shards"]))),
        "merge requires exactly the seven shard IDs",
    )
    records.sort(key=lambda item: int(item["shard_id"]))
    pred_paths, real_paths, flattened = set(), set(), []
    gene_axis = None
    for record in records:
        spec = shard_spec(protocol, int(record["shard_id"]))
        require(record.get("schema_version") == SCHEMA_VERSION, "stale shard audit schema")
        require(record.get("passed") is True, "an input shard audit did not pass")
        require(record.get("kind") == "pbmc_u2_attempt2_treated_shard_audit", "wrong shard audit kind")
        require(int(record.get("sampling_seed", -1)) == int(args.sampling_seed), "mixed shard seeds")
        require(record.get("protocol_sha256") == protocol_sha, "mixed shard protocols")
        require(record.get("parent_gate_passed") is True, "a shard lacks Parent gate evidence")
        identity = record.get("physical_identity")
        require(isinstance(identity, dict), "a shard lacks physical identity evidence")
        require(
            identity.get("schema_version") == STAGE_IDENTITY_SCHEMA_VERSION
            and identity.get("passed") is True,
            "a shard has stale or failed physical identity evidence",
        )
        require(record.get("cytokines") == list(spec["cytokines"]), "shard assignment changed after audit")
        require(int(record.get("treated_cells", -1)) == int(spec["expected_treated_cells"]), "shard cell count changed")
        require(int(record.get("nonempty_pairs", -1)) == int(spec["expected_nonempty_pairs"]), "shard pair count changed")
        require(int(record.get("control_cells", -1)) == 0, "treated shard record contains controls")
        pred_path, real_path = Path(record["pred"]), Path(record["real"])
        require(pred_path.resolve() not in pred_paths, "prediction shard path reused")
        require(real_path.resolve() not in real_paths, "truth shard path reused")
        pred_paths.add(pred_path.resolve())
        real_paths.add(real_path.resolve())
        require(sha256(pred_path) == record["pred_sha256"], "prediction shard changed after audit")
        require(sha256(real_path) == record["real_sha256"], "truth shard changed after audit")
        gene_axis = record["gene_axis_sha256"] if gene_axis is None else gene_axis
        require(record["gene_axis_sha256"] == gene_axis, "shard gene axes differ")
        flattened.extend(map(str, record["cytokines"]))
    require(
        len(flattened) == len(set(flattened)) == int(protocol["expected"]["cytokines"]),
        "audited shards overlap or omit cytokines",
    )
    require(set(flattened) == set(protocol["official_test_cytokines"]), "audited shard union changed")
    require(
        sum(int(record["treated_cells"]) for record in records)
        == int(protocol["expected"]["treated_cells"]),
        "audited shards do not cover all treated cells",
    )
    return records, protocol_sha


def _publish_directory(work: Path, destination: Path) -> None:
    require(not destination.exists(), f"refusing to replace existing directory: {destination}")
    os.replace(work, destination)
    _fsync_directory(destination.parent)


def _audit_stage_physical_identity(
    pred_path: Path, real_path: Path, *, context: str
) -> dict[str, Any]:
    pred = ad.read_h5ad(pred_path, backed="r")
    real = ad.read_h5ad(real_path, backed="r")
    try:
        report = assert_paired_identity_equal(
            real.obs,
            pred.obs,
            group_key="cytokine",
            reference="PBS",
            context=context,
        )
        require(
            list(map(str, pred.obs_names)) == list(map(str, real.obs_names)),
            f"{context}: pred/real output obs order differs",
        )
        return report
    finally:
        pred.file.close()
        real.file.close()


def _resume_stage(
    destination: Path, *, cell_type: str, seed: int, input_fingerprint: str
) -> dict | None:
    if not destination.exists():
        return None
    manifest_path = destination / "stage_manifest.json"
    require(manifest_path.is_file(), f"partial PBMC stage directory exists: {destination}")
    record = load_json(manifest_path)
    require(record.get("schema_version") == SCHEMA_VERSION, "stale PBMC stage schema")
    require(record.get("passed") is True, "existing PBMC stage did not pass")
    require(record.get("cell_type") == cell_type, "PBMC stage cell type changed")
    require(int(record.get("sampling_seed", -1)) == int(seed), "PBMC stage seed changed")
    require(record.get("input_fingerprint") == input_fingerprint, "PBMC stage inputs changed")
    require(record.get("parent_gate_passed") is True, "PBMC stage Parent gate did not pass")
    validate_gate_record(
        record.get("parent_gate", {}),
        max_singleton_fraction=PARENT_MAX_SINGLETON_FRACTION,
        max_endpoint_mean_drift=PARENT_MAX_ENDPOINT_DRIFT,
    )
    require(
        int(record["parent_gate"]["valid_cells"]) == int(record.get("treated_cells", -1)),
        "PBMC stage Parent gate valid_cells differs from treated cells",
    )
    saved_audit = record.get("parent_stage_audit", {})
    require(
        bool(saved_audit.get("saved_h5ad_audit_passed", False)),
        "PBMC stage lacks its on-disk Parent audit",
    )
    for key, hash_key in (("pred", "pred_sha256"), ("real", "real_sha256")):
        path = Path(record[key])
        require(path.is_file(), f"missing resumed PBMC stage {key}: {path}")
        require(sha256(path) == record[hash_key], f"resumed PBMC stage {key} changed")
    expected_identity = record.get("physical_identity")
    require(
        isinstance(expected_identity, dict)
        and expected_identity.get("schema_version") == STAGE_IDENTITY_SCHEMA_VERSION
        and expected_identity.get("passed") is True,
        "PBMC stage lacks passed physical identity evidence",
    )
    observed_identity = _audit_stage_physical_identity(
        Path(record["pred"]), Path(record["real"]), context=f"resumed stage {cell_type}"
    )
    require(observed_identity == expected_identity, "resumed PBMC stage identity changed")
    return record


def _reference_rows(
    pred: ad.AnnData, rows: np.ndarray, offset: int, *,
    partition_policy: str = "strict_unsplit",
    selected_matrix: sparse.csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    old_groups = np.asarray(pred.obs.iloc[rows][GROUP_OBS_KEY], dtype=np.int64)
    unique_groups = np.unique(old_groups)
    require(unique_groups.size > 0 and int(unique_groups[0]) >= 0, "invalid treated Parent groups")
    all_groups = np.asarray(pred.obs[GROUP_OBS_KEY], dtype=np.int64)
    require(int(all_groups.min()) >= 0, "treated shard has negative Parent groups")
    selected_counts = np.bincount(old_groups, minlength=int(all_groups.max()) + 1)[unique_groups]
    source_counts = np.bincount(all_groups, minlength=int(all_groups.max()) + 1)[unique_groups]
    split_groups = not np.array_equal(selected_counts, source_counts)
    require(
        partition_policy in {"strict_unsplit", "donor_celltype_repartition"},
        "unknown Parent partition policy",
    )
    if partition_policy == "strict_unsplit":
        require(not split_groups, "a Parent projection group is split across cell types")
    reference = pred.uns[REFERENCE_KEY]
    require(int(reference.get("schema_version", -1)) == 1, "unsupported Parent reference schema")
    source_ids = np.asarray(reference["projection_group_ids"], dtype=np.int64)
    source_means = np.asarray(reference["effective_parent_mean"], dtype=np.float32)
    require(np.array_equal(source_ids, np.arange(len(source_ids))), "Parent reference IDs are not contiguous")
    require(source_means.ndim == 2 and source_means.shape[1] == pred.n_vars, "Parent reference shape changed")
    require(int(unique_groups[-1]) < len(source_ids), "Parent group exceeds reference table")
    require(np.isfinite(source_means[unique_groups]).all(), "Parent reference is non-finite")
    local_groups = np.searchsorted(unique_groups, old_groups).astype(np.int64)
    remapped = local_groups + int(offset)
    if partition_policy == "donor_celltype_repartition":
        require(selected_matrix is not None, "donor Parent repartition needs the selected matrix")
        require(selected_matrix.shape == (len(rows), pred.n_vars), "selected Parent matrix shape changed")
        repartitioned = []
        for local_group in range(len(unique_groups)):
            selected = local_groups == local_group
            require(selected.any(), "repartitioned Parent group is empty")
            group_sum = np.asarray(
                selected_matrix[selected].sum(axis=0, dtype=np.float64)
            ).reshape(-1)
            repartitioned.append(group_sum / int(selected.sum()))
        references = np.asarray(repartitioned, dtype=np.float32)
    else:
        references = source_means[unique_groups]
    singleton_cells = int(selected_counts[selected_counts == 1].sum())
    return remapped, references, singleton_cells


def _combined_parent_gate(
    records: list[dict], *, valid_cells: int, singleton_cells: int
) -> dict:
    """Build a stage-sized gate while retaining strict source-shard bounds."""

    gates = [record["parent_gate"] for record in records]
    require(len(gates) == int(FORMAL_EXPECTED["shards"]), "Parent gate needs seven shards")
    policy_fields = ("singleton_policy", "projection_policy", "parent_mean_policy")
    for field in policy_fields:
        require(len({str(gate[field]) for gate in gates}) == 1, f"mixed Parent {field}")
    source_valid_cells = sum(int(gate["valid_cells"]) for gate in gates)
    source_singleton_cells = sum(int(gate["singleton_cells"]) for gate in gates)
    valid_cells, singleton_cells = int(valid_cells), int(singleton_cells)
    require(valid_cells > 0 and 0 <= singleton_cells <= valid_cells, "invalid combined Parent counts")
    endpoint_drift = max(
        max(
            float(gate["endpoint_mean_drift_max_abs"]),
            float(gate.get("saved_h5ad_endpoint_mean_drift_max_abs", 0.0)),
        )
        for gate in gates
    )
    minimum = min(
        min(
            float(gate["minimum_expression"]),
            float(gate.get("saved_h5ad_minimum_expression", gate["minimum_expression"])),
        )
        for gate in gates
    )
    negative_values = sum(
        int(gate["negative_values_after_projection"])
        + int(gate.get("saved_h5ad_negative_values", 0))
        for gate in gates
    )
    combined = {
        "schema_version": 2,
        "singleton_policy": str(gates[0]["singleton_policy"]),
        "projection_policy": str(gates[0]["projection_policy"]),
        "parent_mean_policy": str(gates[0]["parent_mean_policy"]),
        "valid_cells": valid_cells,
        "singleton_cells": singleton_cells,
        "singleton_fraction": singleton_cells / valid_cells,
        "endpoint_mean_drift_max_abs": endpoint_drift,
        "minimum_expression": minimum,
        "negative_values_after_projection": negative_values,
        "saved_h5ad_audit_required": True,
        "passed": all(
            bool(gate["passed"]) and bool(gate.get("saved_h5ad_audit_passed", False))
            for gate in gates
        ),
        "source_shards": len(gates),
        "source_valid_cells": source_valid_cells,
        "source_singleton_cells": source_singleton_cells,
        "combination_policy": "stage_counts_max_source_drift_min_source_expression_sum_source_negatives",
    }
    validate_gate_record(
        combined,
        max_singleton_fraction=PARENT_MAX_SINGLETON_FRACTION,
        max_endpoint_mean_drift=PARENT_MAX_ENDPOINT_DRIFT,
    )
    return combined


def _donor_parent_partition_audit(preds: list[ad.AnnData]) -> dict:
    total_groups = 0
    split_groups = 0
    for pred in preds:
        frame = pred.obs[[GROUP_OBS_KEY, "donor", "cytokine", "cell_type"]].copy()
        grouped = frame.groupby(GROUP_OBS_KEY, observed=True)[
            ["donor", "cytokine", "cell_type"]
        ].nunique()
        require((grouped["donor"] == 1).all(), "donor Parent group spans donors")
        require((grouped["cytokine"] == 1).all(), "donor Parent group spans cytokines")
        total_groups += int(len(grouped))
        split_groups += int((grouped["cell_type"] > 1).sum())
    require(total_groups > 0 and split_groups > 0, "donor Parent partition audit found no split groups")
    return {
        "schema_version": 1,
        "passed": True,
        "policy": "donor_celltype_repartition",
        "source_projection_groups": total_groups,
        "source_groups_split_across_cell_types": split_groups,
        "source_groups_are_donor_pure": True,
        "source_groups_are_cytokine_pure": True,
        "prediction_values_unchanged": True,
        "stage_reference_policy": "selected_celltype_empirical_group_mean_v1",
    }


def _build_stage(
    *,
    args: argparse.Namespace,
    protocol: dict,
    records: list[dict],
    preds: list[ad.AnnData],
    reals: list[ad.AnnData],
    controls: ad.AnnData,
    cell_type: str,
    genes: list[str],
    controls_sha: str,
    input_fingerprint: str,
) -> dict:
    partition_policy = str(getattr(args, "parent_partition_policy", "strict_unsplit"))
    slug = safe_slug(cell_type)
    destination = args.output_dir / slug
    resumed = _resume_stage(
        destination, cell_type=cell_type, seed=args.sampling_seed,
        input_fingerprint=input_fingerprint,
    )
    if resumed is not None:
        return resumed

    pred_blocks, real_blocks, obs_blocks, reference_blocks = [], [], [], []
    group_offset = 0
    stage_singleton_cells = 0
    for record, pred, real in zip(records, preds, reals):
        rows = np.flatnonzero(strings(pred, "cell_type") == cell_type)
        if rows.size == 0:
            continue
        pred_block = matrix_rows(pred, rows)
        pred_blocks.append(pred_block)
        real_blocks.append(matrix_rows(real, rows))
        stage_obs_columns = [
            *OBS_KEYS, *PHYSICAL_IDENTITY_COLUMNS, SOURCE_FILE_COLUMN
        ]
        missing = [column for column in stage_obs_columns if column not in pred.obs]
        require(not missing, f"treated shard lost physical identity columns: {missing}")
        obs = pred.obs.iloc[rows][stage_obs_columns].copy()
        real_identity = real.obs.iloc[rows][stage_obs_columns]
        require(obs.equals(real_identity), "selected pred/real physical identity differs")
        obs["source_shard"] = int(record["shard_id"])
        remapped, references, singleton_cells = _reference_rows(
            pred, rows, group_offset, partition_policy=partition_policy,
            selected_matrix=pred_block,
        )
        obs[GROUP_OBS_KEY] = remapped
        group_offset += references.shape[0]
        stage_singleton_cells += singleton_cells
        reference_blocks.append(references)
        obs_blocks.append(obs)

    ctrl_rows = np.flatnonzero(strings(controls, "cell_type") == cell_type)
    expected_controls = int(protocol["control_cells_by_cell_type"][cell_type])
    require(len(ctrl_rows) == expected_controls, f"control count changed for {cell_type}")
    control_x = matrix_rows(controls, ctrl_rows)
    control_obs = controls.obs.iloc[ctrl_rows][
        [*OBS_KEYS, *PHYSICAL_IDENTITY_COLUMNS, SOURCE_FILE_COLUMN]
    ].copy()
    control_obs["source_shard"] = -1
    control_obs[GROUP_OBS_KEY] = -1
    require(set(control_obs["cytokine"].astype(str)) == {"PBS"}, "non-PBS canonical control")
    require(
        set(control_obs["donor"].astype(str)).issubset(set(protocol["heldout_donors"])),
        "control donor leaked",
    )

    treated = sum(block.shape[0] for block in pred_blocks)
    require(treated > 0 and reference_blocks, f"no treated PBMC cells for {cell_type}")
    pred_x = sparse.vstack([*pred_blocks, control_x], format="csr")
    real_x = sparse.vstack([*real_blocks, control_x], format="csr")
    obs = pd.concat([*obs_blocks, control_obs], ignore_index=True)
    obs.index = pd.Index([f"{slug}_{index}" for index in range(len(obs))], dtype=str)
    physical_identity = assert_paired_identity_equal(
        obs,
        obs,
        group_key="cytokine",
        reference="PBS",
        context=f"PBMC cell-type stage {cell_type}",
    )
    require(pred_x.shape == real_x.shape == (treated + expected_controls, len(genes)), "stage shape mismatch")
    require(
        np.array_equal(pred_x[-expected_controls:].toarray(), real_x[-expected_controls:].toarray()),
        "canonical PBS differs between pred and real",
    )
    cytokines = set(obs.loc[obs["cytokine"].astype(str) != "PBS", "cytokine"].astype(str))
    ordered_cytokines = [item for item in protocol["official_test_cytokines"] if item in cytokines]
    require(len(ordered_cytokines) == len(cytokines), "stage contains an unknown cytokine")
    var = pd.DataFrame(index=pd.Index(genes, dtype=str))
    pred_out = ad.AnnData(X=pred_x, obs=obs.copy(), var=var.copy())
    real_out = ad.AnnData(X=real_x, obs=obs.copy(), var=var.copy())
    pred_out.uns[REFERENCE_KEY] = {
        "schema_version": 1,
        "projection_group_ids": np.arange(group_offset, dtype=np.int64),
        "effective_parent_mean": np.concatenate(reference_blocks, axis=0),
        "generated_cells": treated,
        "control_group_id": -1,
    }
    pred_out.uns[GATE_KEY] = _combined_parent_gate(
        records, valid_cells=treated, singleton_cells=stage_singleton_cells
    )
    require(
        int(pred_out.uns[GATE_KEY]["valid_cells"]) == treated,
        "stage Parent gate valid_cells differs from treated cells",
    )
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "canonical_control_sha256": controls_sha,
        "input_fingerprint": input_fingerprint,
        "cell_type": cell_type,
        "parent_partition_policy": partition_policy,
        "parent_reference_values_recomputed_from_stage_prediction": (
            partition_policy == "donor_celltype_repartition"
        ),
        "source_prediction_values_unchanged": True,
    }
    pred_out.uns["pbmc_celltype_stage"] = lineage
    real_out.uns["pbmc_celltype_stage"] = lineage
    physical_provenance_json = provenance_uns_payload(
        physical_identity["real"]["source_bindings"]
    )
    pred_out.uns["pbmc_physical_row_provenance_json"] = physical_provenance_json
    real_out.uns["pbmc_physical_row_provenance_json"] = physical_provenance_json

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{slug}.stage.", dir=args.output_dir))
    temporary_pred, temporary_real = work / "pred.h5ad", work / "real.h5ad"
    pred_out.write_h5ad(temporary_pred, compression="gzip", compression_opts=1)
    real_out.write_h5ad(temporary_real, compression="gzip", compression_opts=1)
    on_disk_physical_identity = _audit_stage_physical_identity(
        temporary_pred,
        temporary_real,
        context=f"new stage {cell_type}",
    )
    require(on_disk_physical_identity == physical_identity, "on-disk stage identity changed")
    final_pred, final_real = destination / "pred.h5ad", destination / "real.h5ad"
    parent_stage_audit = _json_scalars(
        audit_saved_parent_locked_h5ad(
            temporary_pred,
            max_singleton_fraction=PARENT_MAX_SINGLETON_FRACTION,
            max_endpoint_mean_drift=PARENT_MAX_ENDPOINT_DRIFT,
        )
    )
    parent_stage_audit["saved_h5ad"] = str(final_pred.resolve())
    require(
        int(parent_stage_audit["valid_cells"]) == treated,
        "on-disk Parent audit valid_cells differs from treated cells",
    )
    stage_record = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "kind": "pbmc_u2_attempt2_celltype_stage",
        "sampling_seed": int(args.sampling_seed),
        "cell_type": cell_type, "slug": slug,
        "treated_cells": treated,
        "control_cells": expected_controls,
        "total_cells": treated + expected_controls,
        "nonempty_cytokines": len(ordered_cytokines),
        "cytokines": ordered_cytokines,
        "donors": list(protocol["heldout_donors"]),
        "genes": len(genes), "gene_axis_sha256": axis_digest(genes),
        "canonical_control_sha256": controls_sha,
        "canonical_controls_appended_once": True,
        "parent_gate_passed": True,
        "parent_gate": _json_scalars(pred_out.uns[GATE_KEY]),
        "parent_stage_audit": parent_stage_audit,
        "parent_partition_policy": partition_policy,
        "parent_reference_values_recomputed_from_stage_prediction": (
            partition_policy == "donor_celltype_repartition"
        ),
        "source_prediction_values_unchanged": True,
        "physical_identity": on_disk_physical_identity,
        "input_fingerprint": input_fingerprint,
        "pred": str(final_pred.resolve()), "pred_sha256": sha256(temporary_pred),
        "real": str(final_real.resolve()), "real_sha256": sha256(temporary_real),
    }
    atomic_json(work / "stage_manifest.json", stage_record)
    _publish_directory(work, destination)
    return stage_record


def merge_celltypes(args: argparse.Namespace) -> dict:
    partition_policy = str(getattr(args, "parent_partition_policy", "strict_unsplit"))
    require(
        partition_policy in {"strict_unsplit", "donor_celltype_repartition"},
        "unknown Parent partition policy",
    )
    protocol = load_protocol(args.protocol)
    records, protocol_sha = _validate_shard_records(args, protocol)
    controls_path = args.controls.resolve()
    require(controls_path.is_file(), f"missing canonical PBMC controls: {controls_path}")
    controls_sha = sha256(controls_path)
    input_fingerprint = canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_sha256": protocol_sha,
            "sampling_seed": int(args.sampling_seed),
            "controls_sha256": controls_sha,
            "parent_partition_policy": partition_policy,
            "shards": [
                {
                    "id": int(record["shard_id"]),
                    "pred_sha256": record["pred_sha256"],
                    "physical_identity": record["physical_identity"],
                    "real_sha256": record["real_sha256"],
                    "parent_gate": record["parent_gate"],
                }
                for record in records
            ],
        }
    )
    preds = [ad.read_h5ad(record["pred"], backed="r") for record in records]
    reals = [ad.read_h5ad(record["real"], backed="r") for record in records]
    controls_source = ad.read_h5ad(controls_path, backed="r")
    attach_control_source_identity(
        controls_source.obs,
        source_path=controls_path,
        dataset_name="pbmc_control",
        reference="PBS",
    )
    if partition_policy == "donor_celltype_repartition":
        parent_partition_audit = _donor_parent_partition_audit(preds)
    else:
        parent_partition_audit = {
            "schema_version": 1, "passed": True, "policy": "strict_unsplit",
            "prediction_values_unchanged": True,
            "stage_reference_policy": "source_effective_parent_mean_v1",
        }
    try:
        genes = list(map(str, preds[0].var_names))
        require(len(genes) == int(protocol["expected"]["genes"]), "PBMC gene count changed")
        require(len(set(genes)) == len(genes), "PBMC gene names are not unique")
        for pred, real in zip(preds, reals):
            require(list(map(str, pred.var_names)) == genes, "prediction shard gene order changed")
            require(list(map(str, real.var_names)) == genes, "truth shard gene order changed")
        control_gene_to_index = {
            str(gene): index for index, gene in enumerate(controls_source.var_names)
        }
        require(len(control_gene_to_index) == controls_source.n_vars, "control genes are not unique")
        missing = [gene for gene in genes if gene not in control_gene_to_index]
        require(not missing, f"canonical controls miss selected genes: {missing[:5]}")
        positions = np.asarray([control_gene_to_index[gene] for gene in genes], dtype=np.int64)
        controls = controls_source[:, positions]
        require(list(map(str, controls.var_names)) == genes, "canonical control gene order differs")
        require(controls.n_obs == int(protocol["expected"]["control_cells"]), "canonical PBS count changed")
        require(set(strings(controls, "cytokine")) == {"PBS"}, "control artifact is not PBS-only")
        require(set(strings(controls, "cell_type")) == set(protocol["cell_types"]), "control cell types changed")
        require(set(strings(controls, "donor")) == set(protocol["heldout_donors"]), "control donors changed")
        require(len(set(map(str, controls.obs_names))) == controls.n_obs, "control obs names are not unique")
        observed_control_counts = pd.Series(strings(controls, "cell_type")).value_counts().to_dict()
        require(
            {key: int(value) for key, value in observed_control_counts.items()}
            == {key: int(value) for key, value in protocol["control_cells_by_cell_type"].items()},
            "canonical control counts by cell type changed",
        )
        stages = [
            _build_stage(
                args=args, protocol=protocol, records=records, preds=preds,
                reals=reals, controls=controls, cell_type=cell_type,
                genes=genes, controls_sha=controls_sha,
                input_fingerprint=input_fingerprint,
            )
            for cell_type in protocol["cell_types"]
        ]
    finally:
        for item in [*preds, *reals, controls_source]:
            item.file.close()

    treated = sum(int(item["treated_cells"]) for item in stages)
    controls_count = sum(int(item["control_cells"]) for item in stages)
    pairs = sum(int(item["nonempty_cytokines"]) for item in stages)
    require(treated == int(protocol["expected"]["treated_cells"]), "staged treated coverage changed")
    require(controls_count == int(protocol["expected"]["control_cells"]), "staged control coverage changed")
    require(treated + controls_count == int(protocol["expected"]["total_cells"]), "staged total changed")
    require(pairs == int(protocol["expected"]["nonempty_cytokine_cell_type_pairs"]), "staged pair coverage changed")
    require(all(item.get("parent_gate_passed") is True for item in stages), "a staged Parent gate failed")
    require(
        all(
            item.get("physical_identity", {}).get("passed") is True
            for item in stages
        ),
        "a stage lacks physical identity evidence",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION, "passed": True,
        "kind": "pbmc_u2_attempt2_celltype_merge",
        "sampling_seed": int(args.sampling_seed),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": protocol_sha,
        "canonical_controls": str(controls_path),
        "canonical_controls_sha256": controls_sha,
        "input_fingerprint": input_fingerprint,
        "parent_partition_policy": partition_policy,
        "parent_partition_audit": parent_partition_audit,
        "source_prediction_values_unchanged": True,
        "treated_cells": treated, "control_cells": controls_count,
        "total_cells": treated + controls_count,
        "genes": int(protocol["expected"]["genes"]),
        "gene_axis_sha256": stages[0]["gene_axis_sha256"],
        "cell_types": len(stages),
        "nonempty_cytokine_cell_type_pairs": pairs,
        "canonical_controls_partitioned_without_duplication": True,
        "all_celltype_parent_gates_passed": True,
        "physical_identity": {
            "schema_version": STAGE_IDENTITY_SCHEMA_VERSION,
            "passed": True,
            "identity_semantics": PHYSICAL_IDENTITY_SEMANTICS,
            "identity_columns": list(PHYSICAL_IDENTITY_COLUMNS),
            "source_file_column": SOURCE_FILE_COLUMN,
            "canonical_roles": {
                "treated": TREATED_ROLE,
                "reference": REFERENCE_ROLE,
            },
            "physical_locator_unique_within_each_stage": True,
            "physical_locator_unique_across_stages": "deferred_to_pooled_builder",
        },
        "shard_records": [str(path.resolve()) for path in args.shard_record],
        "stages": stages,
    }
    atomic_json(args.output_dir / "merge_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def add_r2_and_counts(frame: pd.DataFrame, pred: ad.AnnData, real: ad.AnnData) -> pd.DataFrame:
    rows = []
    real_labels, pred_labels = strings(real, "cytokine"), strings(pred, "cytokine")
    for cytokine in sorted(set(real_labels.tolist()) - {"PBS"}):
        real_mask, pred_mask = real_labels == cytokine, pred_labels == cytokine
        require(real_mask.any() and pred_mask.any(), f"missing pseudobulk rows for {cytokine}")
        rows.append(
            {
                "perturbation": cytokine,
                "n_real": int(real_mask.sum()), "n_pred": int(pred_mask.sum()),
                "R2": float(r2_score(dense_mean(real.X[real_mask]), dense_mean(pred.X[pred_mask]))),
            }
        )
    return frame.merge(pd.DataFrame(rows), on="perturbation", how="outer", validate="one_to_one")


def _validate_stage_record(record: dict, *, seed: int, gene_count: int) -> None:
    require(record.get("schema_version") == SCHEMA_VERSION, "stale stage record schema")
    require(record.get("passed") is True, "PBMC stage did not pass")
    require(record.get("kind") == "pbmc_u2_attempt2_celltype_stage", "wrong stage kind")
    require(int(record.get("sampling_seed", -1)) == int(seed), "PBMC stage seed changed")
    require(int(record.get("genes", -1)) == int(gene_count), "PBMC stage gene count changed")
    for key, hash_key in (("pred", "pred_sha256"), ("real", "real_sha256")):
        path = Path(record[key])
        require(path.is_file(), f"missing PBMC stage file: {path}")
        require(sha256(path) == record[hash_key], f"PBMC stage {key} hash changed")
    expected_identity = record.get("physical_identity")
    require(
        isinstance(expected_identity, dict)
        and expected_identity.get("schema_version") == STAGE_IDENTITY_SCHEMA_VERSION
        and expected_identity.get("passed") is True,
        "PBMC stage lacks passed physical identity evidence",
    )
    observed_identity = _audit_stage_physical_identity(
        Path(record["pred"]), Path(record["real"]), context=str(record["cell_type"])
    )
    require(observed_identity == expected_identity, "PBMC stage physical identity changed")


def _resume_evaluation(
    destination: Path, *, cell_type: str, seed: int, stage_fingerprint: str,
    cell_eval_version: str, pdex_version: str,
) -> tuple[dict, pd.DataFrame] | None:
    if not destination.exists():
        return None
    manifest_path = destination / "evaluation_manifest.json"
    require(manifest_path.is_file(), f"partial PBMC evaluation exists: {destination}")
    record = load_json(manifest_path)
    require(record.get("schema_version") == SCHEMA_VERSION, "stale evaluation schema")
    require(record.get("passed") is True, "existing cell-type evaluation did not pass")
    require(record.get("cell_type") == cell_type, "evaluation cell type changed")
    require(int(record.get("sampling_seed", -1)) == int(seed), "evaluation seed changed")
    require(record.get("stage_fingerprint") == stage_fingerprint, "evaluation stage changed")
    require(record.get("cell_eval") == cell_eval_version, "Cell-Eval version changed")
    require(record.get("pdex") == pdex_version, "pdex version changed")
    guard = record.get("pdex_fdr_nan_guard")
    require(isinstance(guard, dict), "evaluation lacks pdex FDR NaN guard audit")
    require(guard.get("policy") == PDEX_FDR_NAN_GUARD_POLICY, "pdex FDR NaN guard changed")
    for key, hash_key in (
        ("per_cytokine", "per_cytokine_sha256"),
        ("pred_de", "pred_de_sha256"), ("real_de", "real_de_sha256"),
    ):
        path = Path(record[key])
        require(path.is_file(), f"missing resumed evaluation file: {path}")
        require(sha256(path) == record[hash_key], f"resumed evaluation {key} changed")
    frame = pd.read_csv(record["per_cytokine"])
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"resumed Cell-Eval metric {metric} is non-finite")
    undefined_audit = record.get("cell_eval_undefined_de_guard")
    if undefined_audit is None:
        # Compatibility is safe because the immutable pre-policy CSV has no
        # non-finite value, so the new conditional policy would not be invoked.
        undefined_audit = _empty_undefined_de_audit()
        undefined_audit["resume_compatibility"] = "pre_policy_output_all_metrics_finite"
        record = dict(record)
        record["cell_eval_undefined_de_guard"] = undefined_audit
    else:
        require(isinstance(undefined_audit, dict), "invalid undefined-DE audit")
        require(
            undefined_audit.get("policy") == CELL_EVAL_UNDEFINED_DE_POLICY,
            "undefined-DE policy changed",
        )
    return record, frame


def _evaluate_cell_type(
    *, args: argparse.Namespace, stage: dict, evaluator_factory: Callable[..., Any],
    cell_eval_version: str, pdex_version: str,
    pdex_fdr_guard: Callable[..., Any] | None,
) -> tuple[dict, pd.DataFrame]:
    cell_type, slug = str(stage["cell_type"]), str(stage["slug"])
    stage_fingerprint = canonical_digest(
        {
            "pred_sha256": stage["pred_sha256"], "real_sha256": stage["real_sha256"],
            "cytokines": stage["cytokines"], "sampling_seed": int(args.sampling_seed),
            "skip_metrics": SKIP_METRICS,
            "pdex_fdr_nan_guard_policy": PDEX_FDR_NAN_GUARD_POLICY,
        }
    )
    destination = args.output_dir / slug
    resumed = _resume_evaluation(
        destination, cell_type=cell_type, seed=args.sampling_seed,
        stage_fingerprint=stage_fingerprint, cell_eval_version=cell_eval_version,
        pdex_version=pdex_version,
    )
    if resumed is not None:
        return resumed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{slug}.eval.", dir=args.output_dir))
    celleval_dir = work / "celleval"
    guard_before = _pdex_guard_snapshot(pdex_fdr_guard)
    evaluator = evaluator_factory(
        adata_pred=stage["pred"], adata_real=stage["real"],
        control_pert="PBS", pert_col="cytokine", num_threads=int(args.num_threads),
        batch_size=64, outdir=str(celleval_dir),
    )
    results, _ = evaluator.compute(
        profile="full", skip_metrics=SKIP_METRICS, write_csv=False,
        break_on_error=True,
    )
    guard_after = _pdex_guard_snapshot(pdex_fdr_guard)
    guard_audit = _pdex_guard_delta(
        guard_before, guard_after, active=pdex_fdr_guard is not None
    )
    frame = results.to_pandas().rename(columns=RENAME)
    require("perturbation" in frame, "Cell-Eval result has no perturbation column")
    frame["perturbation"] = frame["perturbation"].astype(str)
    require(frame["perturbation"].is_unique, "Cell-Eval returned duplicate cytokines")
    frame = add_r2_and_counts(frame, evaluator.anndata_pair.pred, evaluator.anndata_pair.real)
    temporary_pred_de = celleval_dir / "pred_de.csv"
    temporary_real_de = celleval_dir / "real_de.csv"
    require(temporary_pred_de.is_file() and temporary_real_de.is_file(), "Cell-Eval omitted DE outputs")
    frame, undefined_de_audit = repair_undefined_de_metrics(
        frame, temporary_real_de, temporary_pred_de
    )
    expected_cytokines = list(map(str, stage["cytokines"]))
    require(set(frame["perturbation"].astype(str)) == set(expected_cytokines), "Cell-Eval cytokines changed")
    for metric in METRICS:
        require(metric in frame, f"Cell-Eval omitted metric {metric}")
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        require(np.isfinite(numeric.to_numpy()).all(), f"Cell-Eval metric {metric} is non-finite")
        frame[metric] = numeric.astype(float)
    require((frame["n_real"] > 0).all() and (frame["n_pred"] > 0).all(), "empty Cell-Eval group")
    require((frame["n_real"] == frame["n_pred"]).all(), "pred/real group sizes changed")
    es_values = frame["ES"].to_numpy(dtype=float)
    require(np.allclose(es_values, es_values[0], rtol=0.0, atol=1.0e-12), "ES is not one cell-type scalar")
    order = {name: index for index, name in enumerate(expected_cytokines)}
    frame.insert(0, "cell_type", cell_type)
    frame["_order"] = frame["perturbation"].map(order)
    frame = frame.sort_values("_order").drop(columns="_order")
    frame = frame[["cell_type", "perturbation", "n_real", "n_pred", *METRICS]]
    temporary_csv = work / "per_cytokine.csv"
    frame.to_csv(temporary_csv, index=False)
    final_csv = destination / "per_cytokine.csv"
    final_pred_de = destination / "celleval" / "pred_de.csv"
    final_real_de = destination / "celleval" / "real_de.csv"
    evaluation_record = {
        "schema_version": SCHEMA_VERSION, "passed": True,
        "kind": "pbmc_u2_attempt2_celltype_celleval",
        "sampling_seed": int(args.sampling_seed),
        "cell_type": cell_type, "slug": slug,
        "cell_eval": cell_eval_version, "pdex": pdex_version,
        "pdex_fdr_nan_guard": guard_audit,
        "cell_eval_undefined_de_guard": undefined_de_audit,
        "stage_fingerprint": stage_fingerprint,
        "nonempty_cytokines": len(frame), "es_celltype": float(es_values[0]),
        "per_cytokine": str(final_csv.resolve()),
        "per_cytokine_sha256": sha256(temporary_csv),
        "pred_de": str(final_pred_de.resolve()),
        "pred_de_sha256": sha256(temporary_pred_de),
        "real_de": str(final_real_de.resolve()),
        "real_de_sha256": sha256(temporary_real_de),
    }
    atomic_json(work / "evaluation_manifest.json", evaluation_record)
    del evaluator
    gc.collect()
    _publish_directory(work, destination)
    return evaluation_record, frame


def evaluate(
    args: argparse.Namespace, *, evaluator_factory: Callable[..., Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> dict:
    if package_versions is None:
        package_versions = {
            "cell-eval": importlib.metadata.version("cell-eval"),
            "pdex": importlib.metadata.version("pdex"),
        }
    cell_eval_version = str(package_versions["cell-eval"])
    pdex_version = str(package_versions["pdex"])
    require(cell_eval_version == "0.6.6", "PBMC evaluation requires Cell-Eval 0.6.6")
    require(pdex_version == "0.1.26", "PBMC evaluation requires pdex 0.1.26")
    pdex_fdr_guard = None
    if evaluator_factory is None:
        pdex_fdr_guard = install_pdex_fdr_nan_guard()
        from cell_eval import MetricsEvaluator
        evaluator_factory = MetricsEvaluator

    merge_path = args.merge_dir / "merge_manifest.json"
    merge = load_json(merge_path)
    require(merge.get("schema_version") == SCHEMA_VERSION, "stale PBMC merge schema")
    require(merge.get("passed") is True, "PBMC merge did not pass")
    require(merge.get("kind") == "pbmc_u2_attempt2_celltype_merge", "wrong merge kind")
    require(int(merge.get("sampling_seed", -1)) == int(args.sampling_seed), "merge seed changed")
    protocol_path = Path(merge["protocol"])
    protocol = load_protocol(protocol_path)
    require(sha256(protocol_path) == merge["protocol_sha256"], "protocol changed after merge")
    for key in ("treated_cells", "control_cells", "total_cells"):
        require(int(merge.get(key, -1)) == int(protocol["expected"][key]), f"merge {key} changed")
    stages = merge.get("stages")
    require(isinstance(stages, list) and len(stages) == int(protocol["expected"]["cell_types"]), "merge lacks 18 stages")
    require([stage["cell_type"] for stage in stages] == protocol["cell_types"], "stage order changed")
    for stage in stages:
        _validate_stage_record(stage, seed=args.sampling_seed, gene_count=int(protocol["expected"]["genes"]))

    evaluations = [
        _evaluate_cell_type(
            args=args, stage=stage, evaluator_factory=evaluator_factory,
            cell_eval_version=cell_eval_version, pdex_version=pdex_version,
            pdex_fdr_guard=pdex_fdr_guard,
        )[0]
        for stage in stages
    ]
    pairs = sum(int(record["nonempty_cytokines"]) for record in evaluations)
    require(pairs == int(protocol["expected"]["nonempty_cytokine_cell_type_pairs"]), "Cell-Eval pair coverage changed")
    manifest = {
        "schema_version": SCHEMA_VERSION, "passed": True,
        "kind": "pbmc_u2_attempt2_celltype_celleval_suite",
        "sampling_seed": int(args.sampling_seed),
        "cell_eval": cell_eval_version, "pdex": pdex_version,
        "pdex_fdr_nan_guard": {
            "policy": PDEX_FDR_NAN_GUARD_POLICY,
            "active": pdex_fdr_guard is not None,
            "fdr_calls": sum(
                int(record["pdex_fdr_nan_guard"]["fdr_calls"])
                for record in evaluations
            ),
            "calls_with_nan": sum(
                int(record["pdex_fdr_nan_guard"]["calls_with_nan"])
                for record in evaluations
            ),
            "nan_pvalues_replaced": sum(
                int(record["pdex_fdr_nan_guard"]["nan_pvalues_replaced"])
                for record in evaluations
            ),
        },
        "cell_eval_undefined_de_guard": {
            "policy": CELL_EVAL_UNDEFINED_DE_POLICY,
            "replacement_value": 0.0,
            "values_replaced": sum(
                int(record["cell_eval_undefined_de_guard"]["values_replaced"])
                for record in evaluations
            ),
            "cell_types_with_replacements": sum(
                int(record["cell_eval_undefined_de_guard"]["values_replaced"] > 0)
                for record in evaluations
            ),
            "metrics": {
                metric: sum(
                    int(record["cell_eval_undefined_de_guard"]["metrics"][metric]["values_replaced"])
                    for record in evaluations
                )
                for metric in CELL_EVAL_UNDEFINED_DE_METRICS
            },
        },
        "merge_manifest": str(merge_path.resolve()),
        "merge_manifest_sha256": sha256(merge_path),
        "protocol": str(protocol_path.resolve()), "protocol_sha256": sha256(protocol_path),
        "cell_types": len(evaluations),
        "nonempty_cytokine_cell_type_pairs": pairs,
        "aggregation_deferred_to_strict_summarizer": True,
        "evaluations": evaluations,
    }
    atomic_json(args.output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("audit-shard")
    shard.add_argument("--protocol", type=Path, required=True)
    shard.add_argument("--shard-id", type=int, choices=range(7), required=True)
    shard.add_argument("--sampling-seed", type=int, choices=(42, 20260811), required=True)
    shard.add_argument("--pred", type=Path, required=True)
    shard.add_argument("--real", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--protocol", type=Path, required=True)
    merge.add_argument("--shard-record", type=Path, action="append", required=True)
    merge.add_argument("--controls", type=Path, required=True)
    merge.add_argument("--sampling-seed", type=int, choices=(42, 20260811), required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument(
        "--parent-partition-policy",
        choices=("strict_unsplit", "donor_celltype_repartition"),
        default="strict_unsplit",
    )
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--merge-dir", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--sampling-seed", type=int, choices=(42, 20260811), required=True)
    evaluation.add_argument("--num-threads", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-shard":
        audit_shard(args)
    elif args.command == "merge":
        merge_celltypes(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

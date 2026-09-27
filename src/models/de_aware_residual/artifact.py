"""Audited, train-only DE-label artifact for P2 residual objectives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = "perturbdiff.de_aware_residual.v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_split_sha256(
    mask: np.ndarray,
    cellline_names: np.ndarray,
    perturbation_names: np.ndarray,
) -> str:
    """Hash the ordered audit mask, including both vocabularies."""

    payload = {
        "audit_condition_mask": np.asarray(mask, dtype=np.uint8).tolist(),
        "cellline_names": np.asarray(cellline_names).astype(str).tolist(),
        "perturbation_names": np.asarray(perturbation_names).astype(str).tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_support_sha256(
    mask: np.ndarray,
    cellline_names: np.ndarray,
    perturbation_names: np.ndarray,
) -> str:
    payload = {
        "parent_calibration_support_mask": np.asarray(
            mask, dtype=np.uint8
        ).tolist(),
        "cellline_names": np.asarray(cellline_names).astype(str).tolist(),
        "perturbation_names": np.asarray(perturbation_names).astype(str).tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required(archive, key: str) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"DE-aware artifact is missing {key!r}")
    value = np.asarray(archive[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in DE-aware artifact key {key!r}")
    return value


def _scalar_string(value: np.ndarray, name: str) -> str:
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    return str(flattened[0])


def _parent_effective_control_count(
    parent, expected_lines: int
) -> np.ndarray:
    """Resolve direct or explicitly pinned fallback Parent control counts."""

    direct = _required(parent, "control_count").astype(np.int64)
    if direct.shape != (int(expected_lines),):
        raise ValueError("Parent control_count must have shape [L]")
    has_source = "control_mean_source_count" in parent.files
    has_fallback = "control_mean_fallback_mask" in parent.files
    if has_source != has_fallback:
        raise ValueError("Parent control fallback provenance is incomplete")
    if not has_source:
        return direct
    source = _required(parent, "control_mean_source_count").astype(np.int64)
    fallback = _required(parent, "control_mean_fallback_mask").astype(bool)
    if source.shape != direct.shape or fallback.shape != direct.shape:
        raise ValueError("Parent control fallback arrays must have shape [L]")
    if np.any(source < direct):
        raise ValueError("Parent effective control count is below direct count")
    if np.any(~fallback & (source != direct)):
        raise ValueError("Parent non-fallback control count differs from direct count")
    if np.any(fallback & (direct != 0)):
        raise ValueError("Parent fallback mask must identify zero direct controls")
    return source


@dataclass(frozen=True)
class TrainDELookup:
    labels: torch.Tensor
    train_available: torch.Tensor
    audit_condition: torch.Tensor
    validation_condition: torch.Tensor
    test_condition: torch.Tensor
    target_count: torch.Tensor
    control_count: torch.Tensor


class TrainDELabelBank(nn.Module):
    """Load labels while proving alignment with the Parent artifact.

    Only boolean training labels and counts are loaded.  No validation/test
    treated expression or statistic is permitted in this artifact.  All
    buffers are non-persistent so checkpoints cannot redistribute supervision.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        parent_artifact_path: str | Path,
        *,
        expected_sha256: str,
        expected_parent_sha256: str | None,
        expected_gene_dim: int,
        expected_normalization_divisor: float,
        expected_parent_calibration: str = "no_intercept",
    ) -> None:
        super().__init__()
        artifact_path = Path(artifact_path).expanduser().resolve()
        parent_path = Path(parent_artifact_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        actual_sha = sha256_file(artifact_path)
        expected_sha = str(expected_sha256 or "").lower()
        if not expected_sha or expected_sha in {"none", "null"}:
            raise ValueError("enabled DE-aware training requires an artifact SHA256")
        if actual_sha != expected_sha:
            raise ValueError(
                f"DE-aware artifact SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        parent_sha = sha256_file(parent_path)
        configured_parent_sha = str(expected_parent_sha256 or "").lower()
        if configured_parent_sha not in {"", "none", "null"} and parent_sha != configured_parent_sha:
            raise ValueError(
                "Parent artifact SHA256 mismatch while loading DE-aware labels: "
                f"expected {configured_parent_sha}, got {parent_sha}"
            )

        with np.load(parent_path, allow_pickle=False) as parent, np.load(
            artifact_path, allow_pickle=False
        ) as loaded:
            for archive_name, archive in (("parent", parent), ("DE-aware", loaded)):
                for key in archive.files:
                    if archive[key].dtype == object:
                        raise TypeError(
                            f"unsafe object dtype in {archive_name} artifact key {key!r}"
                        )
            schema = _scalar_string(_required(loaded, "schema_version"), "schema_version")
            if schema != SCHEMA_VERSION:
                raise ValueError(f"unsupported DE-aware schema {schema!r}")
            recorded_parent_sha = _scalar_string(
                _required(loaded, "source_parent_artifact_sha256"),
                "source_parent_artifact_sha256",
            ).lower()
            if recorded_parent_sha != parent_sha:
                raise ValueError(
                    "DE-aware labels were built from a different Parent artifact: "
                    f"recorded {recorded_parent_sha}, runtime {parent_sha}"
                )

            genes = _required(loaded, "gene_names").astype(str)
            parent_genes = _required(parent, "gene_names").astype(str)
            lines = _required(loaded, "cellline_names").astype(str)
            parent_lines = _required(parent, "cellline_names").astype(str)
            perturbations = _required(loaded, "perturbation_names").astype(str)
            parent_perturbations = _required(parent, "perturbation_names").astype(str)
            if genes.shape != (int(expected_gene_dim),):
                raise ValueError("DE-aware gene dimension does not match the model")
            if not np.array_equal(genes, parent_genes):
                raise ValueError("DE-aware/Parent artifact gene order mismatch")
            if not np.array_equal(lines, parent_lines):
                raise ValueError("DE-aware/Parent cell-line order mismatch")
            if not np.array_equal(perturbations, parent_perturbations):
                raise ValueError("DE-aware/Parent perturbation order mismatch")
            divisor = float(_required(loaded, "normalization_divisor").reshape(-1)[0])
            parent_divisor = float(
                _required(parent, "normalization_divisor").reshape(-1)[0]
            )
            if not np.isclose(divisor, parent_divisor) or not np.isclose(
                divisor, float(expected_normalization_divisor)
            ):
                raise ValueError(
                    "DE-aware/Parent/runtime normalization mismatch: "
                    f"{divisor}, {parent_divisor}, {expected_normalization_divisor}"
                )

            train_mask = _required(loaded, "train_condition_mask").astype(bool)
            audit_mask = _required(loaded, "audit_condition_mask").astype(bool)
            validation_mask = _required(loaded, "validation_condition_mask").astype(bool)
            test_mask = _required(loaded, "test_condition_mask").astype(bool)
            parent_train = _required(parent, "train_condition_mask").astype(bool)
            parent_validation = _required(parent, "validation_condition_mask").astype(bool)
            parent_test = _required(parent, "test_condition_mask").astype(bool)
            expected_condition_shape = (len(lines), len(perturbations))
            for name, mask in (
                ("train", train_mask),
                ("audit", audit_mask),
                ("validation", validation_mask),
                ("test", test_mask),
                ("Parent train", parent_train),
                ("Parent validation", parent_validation),
                ("Parent test", parent_test),
            ):
                if mask.shape != expected_condition_shape:
                    raise ValueError(f"{name} condition mask has the wrong [L,P] shape")
            if not (
                np.array_equal(validation_mask, parent_validation)
                and np.array_equal(test_mask, parent_test)
            ):
                raise ValueError("DE-aware held-out masks differ from the Parent artifact")
            if np.any(audit_mask & ~parent_train):
                raise ValueError("audit conditions must be carved only from Parent train")
            if np.any(audit_mask & (validation_mask | test_mask)):
                raise ValueError("audit and validation/test condition masks overlap")
            expected_train_mask = parent_train & ~audit_mask
            if not np.array_equal(train_mask, expected_train_mask):
                raise ValueError(
                    "DE-aware train mask must equal Parent train minus audit conditions"
                )
            if np.any(train_mask & (audit_mask | validation_mask | test_mask)):
                raise ValueError("forbidden condition appears in DE-aware training mask")
            recorded_audit_sha = _scalar_string(
                _required(loaded, "audit_split_sha256"), "audit_split_sha256"
            ).lower()
            computed_audit_sha = _audit_split_sha256(
                audit_mask, lines, perturbations
            )
            if recorded_audit_sha != computed_audit_sha:
                raise ValueError(
                    "DE-aware audit split hash mismatch: "
                    f"recorded {recorded_audit_sha}, computed {computed_audit_sha}"
                )

            calibration_mode = _scalar_string(
                _required(loaded, "parent_calibration_mode"),
                "parent_calibration_mode",
            ).lower()
            runtime_calibration = str(expected_parent_calibration).lower()
            if runtime_calibration not in {"none", "no_intercept", "affine"}:
                raise ValueError(
                    "Parent calibration must be none, no_intercept, or affine"
                )
            if calibration_mode != runtime_calibration:
                raise ValueError(
                    "P2/Parent runtime calibration mode mismatch: "
                    f"artifact {calibration_mode!r}, runtime {runtime_calibration!r}"
                )
            recorded_support = _required(
                loaded, "parent_calibration_support_mask"
            ).astype(bool)
            if recorded_support.shape != expected_condition_shape:
                raise ValueError(
                    "parent_calibration_support_mask must have shape [L,P]"
                )
            if calibration_mode == "none":
                expected_support = np.zeros_like(parent_train, dtype=bool)
            else:
                expected_support = _required(
                    parent, "ridge_support_mask"
                ).astype(bool)
                if expected_support.shape != expected_condition_shape:
                    raise ValueError(
                        "Parent ridge_support_mask has the wrong [L,P] shape"
                    )
            if not np.array_equal(recorded_support, expected_support):
                raise ValueError(
                    "P2 calibration support provenance differs from Parent runtime"
                )
            if np.any(recorded_support & audit_mask):
                raise ValueError(
                    "Parent calibration leakage: ridge support contains audit "
                    "conditions; rebuild/derive Parent calibration without audit "
                    "or use parent_residual_calibration=none"
                )
            if np.any(recorded_support & (validation_mask | test_mask)):
                raise ValueError(
                    "Parent calibration support contains validation/test conditions"
                )
            recorded_support_sha = _scalar_string(
                _required(loaded, "parent_calibration_support_sha256"),
                "parent_calibration_support_sha256",
            ).lower()
            computed_support_sha = _calibration_support_sha256(
                recorded_support, lines, perturbations
            )
            if recorded_support_sha != computed_support_sha:
                raise ValueError(
                    "Parent calibration support hash mismatch: "
                    f"recorded {recorded_support_sha}, computed {computed_support_sha}"
                )
            labels = _required(loaded, "train_de_label").astype(bool)
            if labels.shape != expected_condition_shape + (len(genes),):
                raise ValueError("train_de_label must have shape [L,P,G]")
            available = _required(loaded, "label_available_mask").astype(bool)
            if available.shape != expected_condition_shape:
                raise ValueError("label_available_mask must have shape [L,P]")
            if not np.array_equal(available, train_mask):
                raise ValueError("every and only train condition must have DE labels")
            if labels[~train_mask].any():
                raise ValueError("held-out/non-train conditions contain DE labels")
            target_count = _required(loaded, "train_condition_count").astype(np.int64)
            parent_target_count = _required(parent, "train_condition_count").astype(np.int64)
            control_count = _required(loaded, "control_count").astype(np.int64)
            parent_control_count = _parent_effective_control_count(
                parent, len(lines)
            )
            if target_count.shape != expected_condition_shape:
                raise ValueError("train_condition_count must have shape [L,P]")
            if control_count.shape != (len(lines),):
                raise ValueError("control_count must have shape [L]")
            expected_target_count = np.where(train_mask, parent_target_count, 0)
            if not np.array_equal(target_count, expected_target_count):
                raise ValueError(
                    "DE-aware counts must equal Parent counts on effective train only"
                )
            if not np.array_equal(control_count, parent_control_count):
                raise ValueError(
                    "DE-aware/Parent effective control-source counts differ"
                )
            if np.any(target_count[~train_mask] != 0):
                raise ValueError("audit/held-out/non-train target counts must be zero")
            if np.any(target_count[train_mask] < 1):
                raise ValueError(
                    "effective-train DE labels require a positive treated count"
                )
            if np.any(control_count < 2):
                raise ValueError(
                    "DE-aware statistics require at least two controls per cell line"
                )

        self.artifact_path = str(artifact_path)
        self.artifact_sha256 = actual_sha
        self.parent_artifact_sha256 = parent_sha
        self.audit_split_sha256 = recorded_audit_sha
        self.parent_calibration_mode = calibration_mode
        self.parent_calibration_support_sha256 = recorded_support_sha
        self.register_buffer(
            "parent_calibration_support_mask",
            torch.from_numpy(recorded_support.copy()),
            persistent=False,
        )
        self.register_buffer("train_de_label", torch.from_numpy(labels.copy()), persistent=False)
        self.register_buffer("train_condition_mask", torch.from_numpy(train_mask.copy()), persistent=False)
        self.register_buffer("audit_condition_mask", torch.from_numpy(audit_mask.copy()), persistent=False)
        self.register_buffer(
            "validation_condition_mask", torch.from_numpy(validation_mask.copy()), persistent=False
        )
        self.register_buffer("test_condition_mask", torch.from_numpy(test_mask.copy()), persistent=False)
        self.register_buffer("train_condition_count", torch.from_numpy(target_count.copy()), persistent=False)
        self.register_buffer("control_count", torch.from_numpy(control_count.copy()), persistent=False)

    def lookup(
        self,
        line_ids: torch.Tensor,
        perturbation_ids: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> TrainDELookup:
        """Return labels only for effective-train rows; fail before label access."""

        if not torch.is_tensor(line_ids) or not torch.is_tensor(perturbation_ids):
            raise TypeError("artifact condition IDs must be tensors")
        if line_ids.dtype != torch.long or perturbation_ids.dtype != torch.long:
            raise TypeError("artifact condition IDs must be torch.long")
        if line_ids.ndim != 1 or perturbation_ids.shape != line_ids.shape:
            raise ValueError("artifact condition IDs must share shape [B]")
        if ((line_ids < 0) | (line_ids >= self.train_condition_mask.shape[0])).any():
            raise ValueError("artifact cell-line ID out of range")
        if ((perturbation_ids < 0) | (perturbation_ids >= self.train_condition_mask.shape[1])).any():
            raise ValueError("artifact perturbation ID out of range")
        if active_mask is None:
            active = torch.ones_like(line_ids, dtype=torch.bool)
        else:
            if not torch.is_tensor(active_mask) or active_mask.shape != line_ids.shape:
                raise ValueError("active_mask must be a tensor with shape [B]")
            active = active_mask.to(device=line_ids.device, dtype=torch.bool)

        train_available = self.train_condition_mask[line_ids, perturbation_ids]
        audit_condition = self.audit_condition_mask[line_ids, perturbation_ids]
        validation_condition = self.validation_condition_mask[line_ids, perturbation_ids]
        test_condition = self.test_condition_mask[line_ids, perturbation_ids]
        forbidden = active & ~train_available
        if forbidden.any():
            audit_rows = int((forbidden & audit_condition).sum().item())
            validation_rows = int((forbidden & validation_condition).sum().item())
            test_rows = int((forbidden & test_condition).sum().item())
            other_rows = int(forbidden.sum().item()) - audit_rows - validation_rows - test_rows
            raise RuntimeError(
                "DE-aware lookup rejected forbidden treated conditions before "
                "labels/statistics were accessed: "
                f"audit_rows={audit_rows}, validation_rows={validation_rows}, "
                f"test_rows={test_rows}, other_rows={other_rows}"
            )

        # Inactive/padded IDs can name any split, but must not index a treated
        # label or contribute a sample count. Replace them before label lookup.
        safe_line_ids = torch.where(active, line_ids, torch.zeros_like(line_ids))
        safe_perturbation_ids = torch.where(
            active, perturbation_ids, torch.zeros_like(perturbation_ids)
        )
        labels = self.train_de_label[safe_line_ids, safe_perturbation_ids]
        labels = labels & active.unsqueeze(-1)
        target_count = self.train_condition_count[
            safe_line_ids, safe_perturbation_ids
        ]
        control_count = self.control_count[safe_line_ids]
        target_count = torch.where(active, target_count, torch.zeros_like(target_count))
        control_count = torch.where(active, control_count, torch.zeros_like(control_count))
        return TrainDELookup(
            labels=labels,
            train_available=train_available,
            audit_condition=audit_condition,
            validation_condition=validation_condition,
            test_condition=test_condition,
            target_count=target_count,
            control_count=control_count,
        )


def _fixed_unicode(values: Iterable[str]) -> np.ndarray:
    values = [str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def build_train_de_label_artifact(
    *,
    parent_artifact_path: str | Path,
    de_csv_by_cell_line: Mapping[str, str | Path],
    output_path: str | Path,
    expected_parent_sha256: str | None = None,
    fdr_threshold: float = 0.05,
    audit_conditions: Iterable[tuple[str, str]] = (),
    parent_calibration: str = "no_intercept",
) -> dict[str, object]:
    """Build labels from per-line DE tables that contain train conditions only.

    Each CSV must contain ``target``, ``feature`` and ``fdr``.  A row whose
    target belongs to audit/validation/test (or any non-training condition) is
    a hard error. Audit conditions are carved out of the Parent train mask and
    complete gene coverage is required for every remaining training condition.
    """

    parent_path = Path(parent_artifact_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    parent_sha = sha256_file(parent_path)
    expected = str(expected_parent_sha256 or "").lower()
    if expected not in {"", "none", "null"} and parent_sha != expected:
        raise ValueError(
            f"Parent artifact SHA256 mismatch: expected {expected}, got {parent_sha}"
        )
    if not 0.0 < float(fdr_threshold) < 1.0:
        raise ValueError("fdr_threshold must lie in (0,1)")

    with np.load(parent_path, allow_pickle=False) as parent:
        for key in parent.files:
            if parent[key].dtype == object:
                raise TypeError(f"unsafe object dtype in Parent artifact key {key!r}")
        genes = np.asarray(parent["gene_names"]).astype(str)
        lines = np.asarray(parent["cellline_names"]).astype(str)
        perturbations = np.asarray(parent["perturbation_names"]).astype(str)
        divisor = np.asarray(parent["normalization_divisor"], dtype=np.float32)
        train_mask = np.asarray(parent["train_condition_mask"], dtype=bool)
        validation_mask = np.asarray(parent["validation_condition_mask"], dtype=bool)
        test_mask = np.asarray(parent["test_condition_mask"], dtype=bool)
        target_count = np.asarray(parent["train_condition_count"], dtype=np.int64)
        control_count = np.asarray(parent["control_count"], dtype=np.int64)
        parent_ridge_support = (
            np.asarray(parent["ridge_support_mask"], dtype=bool)
            if "ridge_support_mask" in parent.files
            else None
        )
    if np.any(train_mask & (validation_mask | test_mask)):
        raise ValueError("Parent artifact train/held-out masks overlap")
    if set(de_csv_by_cell_line) != set(lines.tolist()):
        missing = sorted(set(lines.tolist()) - set(de_csv_by_cell_line))
        extra = sorted(set(de_csv_by_cell_line) - set(lines.tolist()))
        raise ValueError(f"DE CSV cell-line coverage mismatch; missing={missing}, extra={extra}")

    line_index = {name: index for index, name in enumerate(lines)}
    pert_index = {name: index for index, name in enumerate(perturbations)}
    gene_index = {name: index for index, name in enumerate(genes)}
    audit_pairs = [(str(line), str(pert)) for line, pert in audit_conditions]
    if len(set(audit_pairs)) != len(audit_pairs):
        raise ValueError("duplicate audit condition")
    parent_train_mask = train_mask.copy()
    audit_mask = np.zeros_like(parent_train_mask, dtype=bool)
    for line_name, perturbation_name in audit_pairs:
        if line_name not in line_index:
            raise ValueError(f"unknown audit cell line {line_name!r}")
        if perturbation_name not in pert_index:
            raise ValueError(f"unknown audit perturbation {perturbation_name!r}")
        line_id = line_index[line_name]
        pert_id = pert_index[perturbation_name]
        if validation_mask[line_id, pert_id] or test_mask[line_id, pert_id]:
            raise ValueError(
                f"audit condition {(line_name, perturbation_name)!r} overlaps validation/test"
            )
        if not parent_train_mask[line_id, pert_id]:
            raise ValueError(
                f"audit condition {(line_name, perturbation_name)!r} is not Parent train"
            )
        audit_mask[line_id, pert_id] = True
    train_mask = parent_train_mask & ~audit_mask
    target_count = np.where(train_mask, target_count, 0)
    audit_sha = _audit_split_sha256(audit_mask, lines, perturbations)
    calibration_mode = str(parent_calibration).lower()
    if calibration_mode not in {"none", "no_intercept", "affine"}:
        raise ValueError(
            "parent_calibration must be none, no_intercept, or affine"
        )
    if calibration_mode == "none":
        calibration_support = np.zeros_like(parent_train_mask, dtype=bool)
    else:
        if parent_ridge_support is None:
            raise ValueError(
                "calibrated Parent artifact is missing ridge_support_mask provenance"
            )
        if parent_ridge_support.shape != parent_train_mask.shape:
            raise ValueError("Parent ridge_support_mask has the wrong [L,P] shape")
        calibration_support = parent_ridge_support.copy()
    calibration_overlap = calibration_support & audit_mask
    if calibration_overlap.any():
        examples = [
            (str(lines[line_id]), str(perturbations[perturbation_id]))
            for line_id, perturbation_id in np.argwhere(calibration_overlap)[:5]
        ]
        raise ValueError(
            "Parent calibration leakage: ridge_support_mask contains audit "
            f"conditions (count={int(calibration_overlap.sum())}, examples={examples}); "
            "rebuild/derive Parent calibration without audit or explicitly use "
            "parent_calibration='none'"
        )
    if np.any(calibration_support & (validation_mask | test_mask)):
        raise ValueError(
            "Parent calibration support contains validation/test conditions"
        )
    calibration_support_sha = _calibration_support_sha256(
        calibration_support, lines, perturbations
    )
    if np.any(target_count[train_mask] < 1):
        raise ValueError(
            "effective-train DE labels require a positive treated count"
        )
    if np.any(control_count < 2):
        raise ValueError(
            "DE-aware statistics require at least two controls per cell line"
        )
    labels = np.zeros(train_mask.shape + (len(genes),), dtype=bool)
    seen = np.zeros_like(labels, dtype=bool)
    csv_hashes: dict[str, str] = {}

    for line_name in lines:
        line_id = line_index[str(line_name)]
        csv_path = Path(de_csv_by_cell_line[str(line_name)]).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        csv_hashes[str(line_name)] = sha256_file(csv_path)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"target", "feature", "fdr"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain {sorted(required)}")
            for row_number, row in enumerate(reader, start=2):
                target = str(row["target"])
                feature = str(row["feature"])
                if target not in pert_index:
                    raise ValueError(f"{csv_path}:{row_number}: unknown target {target!r}")
                if feature not in gene_index:
                    raise ValueError(f"{csv_path}:{row_number}: unknown feature {feature!r}")
                pert_id = pert_index[target]
                if not train_mask[line_id, pert_id]:
                    heldout = "audit" if audit_mask[line_id, pert_id] else (
                        "validation" if validation_mask[line_id, pert_id] else (
                            "test" if test_mask[line_id, pert_id] else "non-train"
                        )
                    )
                    raise ValueError(
                        f"{csv_path}:{row_number}: {heldout} target {target!r} is forbidden"
                    )
                gene_id = gene_index[feature]
                if seen[line_id, pert_id, gene_id]:
                    raise ValueError(
                        f"{csv_path}:{row_number}: duplicate ({target!r}, {feature!r})"
                    )
                fdr = float(row["fdr"])
                if not np.isfinite(fdr) or not 0.0 <= fdr <= 1.0:
                    raise ValueError(f"{csv_path}:{row_number}: invalid FDR {fdr}")
                labels[line_id, pert_id, gene_id] = fdr < float(fdr_threshold)
                seen[line_id, pert_id, gene_id] = True

    expected_seen = np.broadcast_to(train_mask[..., None], seen.shape)
    if not np.array_equal(seen, expected_seen):
        missing = np.argwhere(expected_seen & ~seen)
        example = missing[0].tolist() if len(missing) else None
        raise ValueError(
            "DE tables do not cover every gene of every train condition; "
            f"first_missing={example}"
        )
    if labels[~train_mask].any():
        raise AssertionError("internal error: held-out labels were populated")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_parent_artifact": str(parent_path),
        "source_parent_artifact_sha256": parent_sha,
        "source_de_csv_sha256": csv_hashes,
        "fdr_threshold": float(fdr_threshold),
        "audit_split_sha256": audit_sha,
        "audit_conditions": sorted([list(pair) for pair in audit_pairs]),
        "audit_treated_statistics_present": False,
        "validation_test_treated_statistics_present": False,
        "parent_calibration_mode": calibration_mode,
        "parent_calibration_support_sha256": calibration_support_sha,
        "parent_calibration_support_overlaps_audit": False,
    }
    payload = {
        "schema_version": _fixed_unicode([SCHEMA_VERSION]),
        "source_parent_artifact_sha256": _fixed_unicode([parent_sha]),
        "audit_split_sha256": _fixed_unicode([audit_sha]),
        "parent_calibration_mode": _fixed_unicode([calibration_mode]),
        "parent_calibration_support_sha256": _fixed_unicode(
            [calibration_support_sha]
        ),
        "parent_calibration_support_mask": calibration_support,
        "metadata_json": _fixed_unicode([json.dumps(metadata, sort_keys=True)]),
        "gene_names": _fixed_unicode(genes),
        "cellline_names": _fixed_unicode(lines),
        "perturbation_names": _fixed_unicode(perturbations),
        "normalization_divisor": divisor,
        "train_condition_mask": train_mask,
        "audit_condition_mask": audit_mask,
        "validation_condition_mask": validation_mask,
        "test_condition_mask": test_mask,
        "label_available_mask": train_mask.copy(),
        "train_de_label": labels,
        "train_condition_count": target_count,
        "control_count": control_count,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "parent_sha256": parent_sha,
        "audit_split_sha256": audit_sha,
        "audit_conditions": int(audit_mask.sum()),
        "parent_calibration_mode": calibration_mode,
        "parent_calibration_support_sha256": calibration_support_sha,
        "train_conditions": int(train_mask.sum()),
        "positive_labels": int(labels.sum()),
        "total_labels": int(seen.sum()),
    }


__all__ = [
    "SCHEMA_VERSION",
    "TrainDELabelBank",
    "TrainDELookup",
    "build_train_de_label_artifact",
    "sha256_file",
]

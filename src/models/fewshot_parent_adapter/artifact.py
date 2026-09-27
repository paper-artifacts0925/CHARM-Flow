"""Safe response-basis artifact for the few-shot Parent adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .episodes import SUPPORT_SIZES, deterministic_support_order


SCHEMA_VERSION = "fewshot_parent_adapter_v1"
PARENT_SCHEMA = "parent_residual_replogle_v1"
DEFAULT_AUDIT_SALT = "fewshot_parent_adapter_audit_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"artifact is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in artifact key {key!r}")
    return value


def _fixed_unicode(values: Sequence[str] | str) -> np.ndarray:
    if isinstance(values, str):
        values = [values]
    maximum = max(1, *(len(str(value)) for value in values))
    return np.asarray([str(value) for value in values], dtype=f"<U{maximum}")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(temporary, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(f"unsafe object dtype in output key {key!r}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonicalize_component_signs(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    pivots = np.abs(result).argmax(axis=1)
    signs = np.sign(result[np.arange(len(result)), pivots])
    signs[signs == 0] = 1.0
    result *= signs[:, None]
    return result


def _frozen_audit_split(
    perturbation_names: np.ndarray,
    train_mask: np.ndarray,
    *,
    audit_size: int,
    audit_salt: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Select an expression-independent audit set by salted perturbation hash."""

    audit_size = int(audit_size)
    if audit_size < 1:
        raise ValueError("audit_size must be positive")
    # Every line must own a target value so the same frozen perturbation set can
    # evaluate all LOLO episodes. Expression values never enter selection.
    candidates = np.flatnonzero(np.asarray(train_mask, dtype=bool).all(axis=0))
    if len(candidates) <= audit_size:
        raise ValueError(
            f"only {len(candidates)} perturbations are train-observed in every "
            f"line; need more than audit_size={audit_size}"
        )
    scored = []
    for perturbation_id in candidates:
        name = str(perturbation_names[int(perturbation_id)])
        digest = hashlib.sha256(
            (str(audit_salt) + "\0" + name).encode("utf-8")
        ).hexdigest()
        scored.append((digest, name, int(perturbation_id)))
    selected = sorted(scored)[:audit_size]
    ids = np.asarray([item[2] for item in selected], dtype=np.int64)
    names = np.asarray([item[1] for item in selected], dtype=str)
    split_hash = hashlib.sha256(
        ("\n".join(names.tolist()) + "\n").encode("utf-8")
    ).hexdigest()
    return ids, names, split_hash


def _fit_lolo_basis(
    train_delta: np.ndarray,
    train_mask: np.ndarray,
    target_line: int,
    max_rank: int,
    seed: int,
    source_line_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_mask = train_mask.copy()
    source_mask[int(target_line)] = False
    if source_line_mask is not None:
        allowed = np.asarray(source_line_mask, dtype=bool)
        if allowed.shape != (train_mask.shape[0],):
            raise ValueError("source_line_mask must have shape [L]")
        source_mask &= allowed[:, None]
    values = train_delta[source_mask].astype(np.float64)
    if len(values) < int(max_rank):
        raise ValueError(
            f"target line {target_line} has only {len(values)} source conditions; "
            f"cannot fit rank {max_rank}"
        )
    _, singular_values, right = randomized_svd(
        values,
        n_components=int(max_rank),
        n_iter=7,
        random_state=int(seed) + int(target_line),
    )
    basis = _canonicalize_component_signs(right)
    return basis.astype(np.float32), singular_values.astype(np.float32)


def build_fewshot_parent_artifact(
    *,
    parent_artifact_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    max_rank: int = 64,
    max_donors: int = 3,
    seed: int = 42,
    audit_size: int = 64,
    audit_salt: str = DEFAULT_AUDIT_SALT,
    heldout_deployment_line_id: int | None = None,
    minimum_support_condition_cells: int = 1,
) -> dict:
    """Build LOLO response bases without reading held-out target expression."""

    parent_path = Path(parent_artifact_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    metadata_output = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else output.with_suffix(".metadata.json")
    )
    if int(max_rank) not in (32, 64):
        raise ValueError("max_rank must be 32 or 64")
    if int(max_donors) != 3:
        raise ValueError("the first artifact schema fixes max_donors=3")
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent_sha = sha256_file(parent_path)
    with np.load(parent_path, allow_pickle=False) as loaded:
        for key in loaded.files:
            if loaded[key].dtype == object:
                raise TypeError(f"unsafe object dtype in Parent key {key!r}")
        schema = str(_required(loaded, "schema_version").reshape(-1)[0])
        if schema != PARENT_SCHEMA:
            raise ValueError(f"unsupported Parent artifact schema {schema!r}")
        genes = _required(loaded, "gene_names").astype(str)
        lines = _required(loaded, "cellline_names").astype(str)
        perturbations = _required(loaded, "perturbation_names").astype(str)
        divisor = float(_required(loaded, "normalization_divisor"))
        train_delta = _required(loaded, "train_condition_delta").astype(np.float32)
        train_mask = _required(loaded, "train_condition_mask").astype(bool)
        train_count = _required(loaded, "train_condition_count").astype(np.int64)
        validation_mask = _required(loaded, "validation_condition_mask").astype(bool)
        test_mask = _required(loaded, "test_condition_mask").astype(bool)
        prior_mask = _required(loaded, "cross_line_equal_line_mask").astype(bool)
    expected = (len(lines), len(perturbations), len(genes))
    if train_delta.shape != expected or train_mask.shape != expected[:2]:
        raise ValueError("Parent train arrays do not match mappings")
    if train_count.shape != train_mask.shape or (train_count < 0).any():
        raise ValueError("Parent train condition counts do not match mappings")
    if not np.array_equal(train_count > 0, train_mask):
        raise ValueError("Parent train condition count/mask are inconsistent")
    minimum_support_condition_cells = int(minimum_support_condition_cells)
    if minimum_support_condition_cells < 1:
        raise ValueError("minimum_support_condition_cells must be positive")
    if validation_mask.shape != train_mask.shape or test_mask.shape != train_mask.shape:
        raise ValueError("Parent held-out masks do not match train mask")
    if np.any(train_mask & (validation_mask | test_mask)):
        raise ValueError("held-out target condition appears in Parent training mask")
    heldout = validation_mask | test_mask
    if np.any(train_delta[heldout] != 0):
        raise ValueError("held-out target expression is nonzero in Parent train delta")
    if not np.isfinite(train_delta).all():
        raise ValueError("Parent train deltas must be finite")

    line_count, perturbation_count, gene_dim = train_delta.shape
    deployment_line_id = (
        -1
        if heldout_deployment_line_id is None
        else int(heldout_deployment_line_id)
    )
    if deployment_line_id < -1 or deployment_line_id >= line_count:
        raise ValueError("heldout_deployment_line_id is out of range")
    audit_ids, audit_names, audit_split_sha256 = _frozen_audit_split(
        perturbations,
        train_mask,
        audit_size=int(audit_size),
        audit_salt=str(audit_salt),
    )
    audit_perturbation_mask = np.zeros(perturbation_count, dtype=bool)
    audit_perturbation_mask[audit_ids] = True
    audit_condition_mask = (
        train_mask & audit_perturbation_mask[None, :]
    )
    adapter_train_mask = (
        train_mask & ~audit_perturbation_mask[None, :]
    )
    bases = np.empty((line_count, int(max_rank), gene_dim), dtype=np.float32)
    singular_values = np.empty((line_count, int(max_rank)), dtype=np.float32)
    basis_source_line_mask = np.ones((line_count, line_count), dtype=bool)
    np.fill_diagonal(basis_source_line_mask, False)
    if deployment_line_id >= 0:
        # The deployment line is never a meta-training source. Its responses
        # beyond N support remain evaluation-only under every LOLO source-line
        # episode, not merely under its own target episode.
        basis_source_line_mask[:, deployment_line_id] = False
    donor_line_ids = np.full((line_count, int(max_donors)), -1, dtype=np.int16)
    donor_line_mask = np.zeros_like(donor_line_ids, dtype=bool)
    episodic_eligible = np.zeros_like(train_mask)
    support_priority = np.full_like(train_mask, -1, dtype=np.int32)
    for target_line in range(line_count):
        basis, values = _fit_lolo_basis(
            train_delta,
            adapter_train_mask,
            target_line,
            int(max_rank),
            int(seed),
            source_line_mask=basis_source_line_mask[target_line],
        )
        bases[target_line] = basis
        singular_values[target_line] = values
        donors = np.asarray(
            [
                line
                for line in range(line_count)
                if line != target_line
                and (
                    deployment_line_id < 0
                    or target_line == deployment_line_id
                    or line != deployment_line_id
                )
            ],
            dtype=np.int16,
        )[: int(max_donors)]
        donor_line_ids[target_line, : len(donors)] = donors
        donor_line_mask[target_line, : len(donors)] = True
        donor_available = adapter_train_mask[donors.astype(np.int64)].any(axis=0)
        eligible = (
            adapter_train_mask[target_line]
            & donor_available
            & prior_mask[target_line]
            & (
                train_count[target_line]
                >= minimum_support_condition_cells
            )
        )
        episodic_eligible[target_line] = eligible
        order = deterministic_support_order(
            eligible, target_line=target_line, seed=int(seed)
        )
        support_priority[target_line, order] = np.arange(len(order), dtype=np.int32)
        if len(order) <= max(SUPPORT_SIZES):
            raise ValueError(
                f"target line {lines[target_line]!r} has {len(order)} eligible "
                "conditions; the N=64 protocol also needs at least one query"
            )

    arrays = {
        "schema_version": _fixed_unicode(SCHEMA_VERSION),
        "parent_artifact_sha256": _fixed_unicode(parent_sha),
        "normalization_divisor": np.asarray(divisor, dtype=np.float32),
        "gene_names": _fixed_unicode(genes.tolist()),
        "cellline_names": _fixed_unicode(lines.tolist()),
        "perturbation_names": _fixed_unicode(perturbations.tolist()),
        "max_rank": np.asarray(int(max_rank), dtype=np.int16),
        "support_sizes": np.asarray(SUPPORT_SIZES, dtype=np.int16),
        "minimum_support_condition_cells": np.asarray(
            minimum_support_condition_cells, dtype=np.int32
        ),
        "audit_salt": _fixed_unicode(str(audit_salt)),
        "audit_split_sha256": _fixed_unicode(audit_split_sha256),
        "audit_perturbation_ids": audit_ids.astype(np.int32),
        "audit_perturbation_names": _fixed_unicode(audit_names.tolist()),
        "audit_perturbation_mask": audit_perturbation_mask,
        "audit_condition_mask": audit_condition_mask,
        "heldout_deployment_line_id": np.asarray(
            deployment_line_id, dtype=np.int16
        ),
        "heldout_deployment_line_name": _fixed_unicode(
            "" if deployment_line_id < 0 else str(lines[deployment_line_id])
        ),
        "response_basis_by_target_line": bases,
        "response_singular_values_by_target_line": singular_values,
        "basis_source_line_mask": basis_source_line_mask,
        "donor_line_ids": donor_line_ids,
        "donor_line_mask": donor_line_mask,
        "episodic_eligible_mask": episodic_eligible,
        "support_priority": support_priority,
        "train_condition_mask_audit": train_mask,
        "adapter_train_condition_mask_audit": adapter_train_mask,
        "validation_condition_mask_audit": validation_mask,
        "test_condition_mask_audit": test_mask,
    }
    _atomic_npz(output, arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": (
            "strict_nshot"
            if deployment_line_id >= 0
            else "generic_lolo_preflight"
        ),
        "full_support_parent_calibration_permitted": False,
        "runtime_parent_floor": (
            "raw leakage-safe equal-line prior with no-intercept calibration "
            "refit from exactly the selected N target-line supports"
        ),
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": parent_sha,
        "max_rank": int(max_rank),
        "max_donors": int(max_donors),
        "seed": int(seed),
        "support_sizes": list(SUPPORT_SIZES),
        "minimum_support_condition_cells": minimum_support_condition_cells,
        "audit_split": {
            "size": int(len(audit_ids)),
            "salt": str(audit_salt),
            "sha256": audit_split_sha256,
            "perturbation_names": audit_names.tolist(),
            "selection": (
                "lowest salted SHA256 among perturbations train-observed in every line"
            ),
        },
        "heldout_deployment_line_id": (
            None if deployment_line_id < 0 else deployment_line_id
        ),
        "heldout_deployment_line_name": (
            None
            if deployment_line_id < 0
            else str(lines[deployment_line_id])
        ),
        "basis_scope": (
            "for target line l, SVD uses only adapter-train deltas from lines "
            "!= l and never uses the held-out deployment line"
        ),
        "episode_scope": (
            "support/query are disjoint target-line train perturbations with at least "
            "one source-line observation"
        ),
        "leakage_audit": {
            "heldout_target_conditions_in_train": int(
                np.sum(train_mask & heldout)
            ),
            "heldout_target_delta_nonzero": int(np.count_nonzero(train_delta[heldout])),
            "target_line_used_in_own_basis": False,
            "deployment_line_used_in_any_basis_or_source_donor": False,
            "audit_used_in_basis_ridge_or_support": False,
            "audit_split_sha256": audit_split_sha256,
            "pickle_required_to_load": False,
        },
        "dimensions": {
            "cell_lines": int(line_count),
            "perturbations": int(perturbation_count),
            "genes": int(gene_dim),
        },
    }
    _atomic_json(metadata_output, metadata)
    return metadata


class FewShotParentArtifact:
    """Strict pickle-free loader with Parent provenance verification."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_parent_sha256: str | None = None,
        expected_normalization_divisor: float = 10.0,
    ):
        artifact = Path(path).expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual_sha = sha256_file(artifact)
        if expected_sha256 not in (None, "", "null") and actual_sha != str(expected_sha256).lower():
            raise ValueError(
                f"few-shot artifact SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
            )
        with np.load(artifact, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(f"unsafe object dtype in adapter key {key!r}")
            schema = str(_required(loaded, "schema_version").reshape(-1)[0])
            if schema != SCHEMA_VERSION:
                raise ValueError(f"unsupported few-shot schema {schema!r}")
            parent_sha = str(
                _required(loaded, "parent_artifact_sha256").reshape(-1)[0]
            )
            if expected_parent_sha256 not in (None, "", "null") and parent_sha != str(expected_parent_sha256).lower():
                raise ValueError("few-shot artifact references a different Parent artifact")
            divisor = float(_required(loaded, "normalization_divisor"))
            if not np.isclose(divisor, float(expected_normalization_divisor)):
                raise ValueError("few-shot artifact/runtime normalization mismatch")
            self.gene_names = _required(loaded, "gene_names").astype(str)
            self.cellline_names = _required(loaded, "cellline_names").astype(str)
            self.perturbation_names = _required(loaded, "perturbation_names").astype(str)
            self.max_rank = int(_required(loaded, "max_rank"))
            self.support_sizes = tuple(
                int(value) for value in _required(loaded, "support_sizes").tolist()
            )
            self.minimum_support_condition_cells = int(
                np.asarray(
                    loaded["minimum_support_condition_cells"]
                    if "minimum_support_condition_cells" in loaded.files
                    else 1
                )
            )
            self.audit_salt = str(
                _required(loaded, "audit_salt").reshape(-1)[0]
            )
            self.audit_split_sha256 = str(
                _required(loaded, "audit_split_sha256").reshape(-1)[0]
            )
            self.audit_perturbation_ids = _required(
                loaded, "audit_perturbation_ids"
            ).astype(np.int64)
            self.audit_perturbation_names = _required(
                loaded, "audit_perturbation_names"
            ).astype(str)
            self.audit_perturbation_mask = _required(
                loaded, "audit_perturbation_mask"
            ).astype(bool)
            self.audit_condition_mask = _required(
                loaded, "audit_condition_mask"
            ).astype(bool)
            self.heldout_deployment_line_id = int(
                _required(loaded, "heldout_deployment_line_id")
            )
            self.heldout_deployment_line_name = str(
                _required(
                    loaded, "heldout_deployment_line_name"
                ).reshape(-1)[0]
            )
            self.response_basis = _required(
                loaded, "response_basis_by_target_line"
            ).astype(np.float32)
            self.response_singular_values = _required(
                loaded, "response_singular_values_by_target_line"
            ).astype(np.float32)
            self.basis_source_line_mask = _required(
                loaded, "basis_source_line_mask"
            ).astype(bool)
            self.donor_line_ids = _required(loaded, "donor_line_ids").astype(np.int64)
            self.donor_line_mask = _required(loaded, "donor_line_mask").astype(bool)
            self.episodic_eligible_mask = _required(
                loaded, "episodic_eligible_mask"
            ).astype(bool)
            self.support_priority = _required(loaded, "support_priority").astype(np.int64)
            train = _required(loaded, "train_condition_mask_audit").astype(bool)
            self.adapter_train_condition_mask = _required(
                loaded, "adapter_train_condition_mask_audit"
            ).astype(bool)
            validation = _required(
                loaded, "validation_condition_mask_audit"
            ).astype(bool)
            test = _required(loaded, "test_condition_mask_audit").astype(bool)
            self.train_condition_mask = train
            self.validation_condition_mask = validation
            self.test_condition_mask = test
        lines = len(self.cellline_names)
        perts = len(self.perturbation_names)
        genes = len(self.gene_names)
        if self.response_basis.shape != (lines, self.max_rank, genes):
            raise ValueError("few-shot response basis has the wrong [L,R,G] shape")
        if self.response_singular_values.shape != (lines, self.max_rank):
            raise ValueError("few-shot singular values have the wrong [L,R] shape")
        if self.basis_source_line_mask.shape != (lines, lines):
            raise ValueError("few-shot basis-source mask has the wrong [L,L] shape")
        if self.donor_line_ids.shape != self.donor_line_mask.shape or self.donor_line_ids.shape != (lines, 3):
            raise ValueError("few-shot donors must have shape [L,3]")
        if self.episodic_eligible_mask.shape != (lines, perts):
            raise ValueError("few-shot eligible mask has the wrong [L,P] shape")
        if self.support_priority.shape != (lines, perts):
            raise ValueError("few-shot support priority has the wrong [L,P] shape")
        if self.audit_perturbation_mask.shape != (perts,):
            raise ValueError("few-shot audit mask has the wrong [P] shape")
        if self.audit_condition_mask.shape != (lines, perts):
            raise ValueError("few-shot audit condition mask has the wrong [L,P] shape")
        if not -1 <= self.heldout_deployment_line_id < lines:
            raise ValueError("few-shot held-out deployment line ID is invalid")
        expected_deployment_name = (
            ""
            if self.heldout_deployment_line_id < 0
            else str(self.cellline_names[self.heldout_deployment_line_id])
        )
        if self.heldout_deployment_line_name != expected_deployment_name:
            raise ValueError("few-shot deployment line ID/name are inconsistent")
        if train.shape != (lines, perts):
            raise ValueError("few-shot train audit mask has the wrong [L,P] shape")
        if validation.shape != (lines, perts) or test.shape != (lines, perts):
            raise ValueError("few-shot held-out audit masks have the wrong [L,P] shape")
        if self.adapter_train_condition_mask.shape != (lines, perts):
            raise ValueError("few-shot adapter-train mask has the wrong [L,P] shape")
        if np.any(train & (validation | test)):
            raise ValueError("few-shot audit masks contain held-out train overlap")
        if np.any(validation & test):
            raise ValueError("few-shot validation/test masks overlap")
        if len(self.audit_perturbation_ids) < 1:
            raise ValueError("few-shot frozen audit split cannot be empty")
        if len(np.unique(self.audit_perturbation_ids)) != len(
            self.audit_perturbation_ids
        ):
            raise ValueError("few-shot audit perturbation IDs must be unique")
        if (
            self.audit_perturbation_ids.min() < 0
            or self.audit_perturbation_ids.max() >= perts
        ):
            raise ValueError("few-shot audit perturbation ID is out of range")
        expected_audit_perturbation_mask = np.zeros(perts, dtype=bool)
        expected_audit_perturbation_mask[self.audit_perturbation_ids] = True
        if not np.array_equal(
            self.audit_perturbation_mask, expected_audit_perturbation_mask
        ):
            raise ValueError("few-shot audit IDs/mask are inconsistent")
        expected_audit_condition_mask = (
            train & self.audit_perturbation_mask[None, :]
        )
        if not np.array_equal(
            self.audit_condition_mask, expected_audit_condition_mask
        ):
            raise ValueError("few-shot audit condition mask is inconsistent")
        expected_adapter_train = (
            train & ~self.audit_perturbation_mask[None, :]
        )
        if not np.array_equal(
            self.adapter_train_condition_mask, expected_adapter_train
        ):
            raise ValueError("few-shot adapter-train mask is inconsistent")
        if np.any(
            self.adapter_train_condition_mask & self.audit_condition_mask
        ):
            raise ValueError("hash-frozen audit conditions entered adapter training")
        if np.any(self.episodic_eligible_mask & self.audit_condition_mask):
            raise ValueError("hash-frozen audit conditions entered episode support/query")
        if np.any(self.support_priority[self.audit_condition_mask] >= 0):
            raise ValueError("hash-frozen audit perturbation entered support order")
        expected_names = self.perturbation_names[self.audit_perturbation_ids]
        if not np.array_equal(expected_names, self.audit_perturbation_names):
            raise ValueError("few-shot audit IDs/names are inconsistent")
        expected_audit_hash = hashlib.sha256(
            ("\n".join(self.audit_perturbation_names.tolist()) + "\n").encode(
                "utf-8"
            )
        ).hexdigest()
        if expected_audit_hash != self.audit_split_sha256:
            raise ValueError("few-shot audit split hash mismatch")
        if self.support_sizes != SUPPORT_SIZES:
            raise ValueError("few-shot artifact has an unsupported support-size protocol")
        if self.minimum_support_condition_cells < 1:
            raise ValueError("few-shot minimum support condition cells must be positive")
        if not np.isfinite(self.response_basis).all():
            raise ValueError("few-shot response basis must be finite")
        for line in range(lines):
            gram = self.response_basis[line] @ self.response_basis[line].T
            if not np.allclose(gram, np.eye(self.max_rank), atol=2e-4, rtol=2e-4):
                raise ValueError(f"few-shot basis for line {line} is not orthonormal")
            if self.basis_source_line_mask[line, line]:
                raise ValueError("target line appears in its own response basis scope")
            if (
                self.heldout_deployment_line_id >= 0
                and self.basis_source_line_mask[
                    line, self.heldout_deployment_line_id
                ]
            ):
                raise ValueError(
                    "deployment line appears in a response-basis scope"
                )
            active_donors = self.donor_line_ids[line, self.donor_line_mask[line]]
            if len(active_donors) and (
                active_donors.min() < 0 or active_donors.max() >= lines
            ):
                raise ValueError("few-shot donor ID is out of range")
            if line in active_donors.tolist() or len(np.unique(active_donors)) != len(active_donors):
                raise ValueError("few-shot donor IDs are invalid")
            if (
                self.heldout_deployment_line_id >= 0
                and line != self.heldout_deployment_line_id
                and self.heldout_deployment_line_id in active_donors.tolist()
            ):
                raise ValueError("deployment line appears as a meta-training donor")
            priority = self.support_priority[line]
            active_support = np.flatnonzero(priority >= 0)
            if not np.array_equal(
                np.sort(priority[active_support]),
                np.arange(len(active_support), dtype=np.int64),
            ):
                raise ValueError("few-shot support priorities are not contiguous")
            if not np.array_equal(
                priority >= 0, self.episodic_eligible_mask[line]
            ):
                raise ValueError("few-shot support order/eligibility are inconsistent")
        self.path = str(artifact)
        self.sha256 = actual_sha
        self.parent_sha256 = parent_sha
        self.normalization_divisor = divisor

    def support_order(self, target_line: int) -> np.ndarray:
        priority = self.support_priority[int(target_line)]
        active = np.flatnonzero(priority >= 0)
        return active[np.argsort(priority[active], kind="stable")]


__all__ = [
    "DEFAULT_AUDIT_SALT",
    "FewShotParentArtifact",
    "SCHEMA_VERSION",
    "build_fewshot_parent_artifact",
    "sha256_file",
]

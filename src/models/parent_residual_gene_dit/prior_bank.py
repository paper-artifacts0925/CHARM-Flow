"""Safe, name-aligned priors for Parent-Residual Gene-DiT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


_SCHEMA = "parent_residual_replogle_v1"
_PRIOR_KEYS = {
    "equal_line": (
        "cross_line_delta_prior_equal_line",
        "cross_line_equal_line_mask",
    ),
    "count_weighted": (
        "cross_line_delta_prior_count_weighted",
        "cross_line_count_weighted_mask",
    ),
}
_CALIBRATIONS = frozenset({"none", "no_intercept", "affine"})


@dataclass(frozen=True)
class ParentPriorLookup:
    """Condition-level tensors aligned to one runtime batch."""

    control_mean: torch.Tensor
    prior_delta: torch.Tensor
    prior_available: torch.Tensor
    target_delta: torch.Tensor
    target_available: torch.Tensor
    target_count: torch.Tensor
    artifact_cellline_ids: torch.Tensor
    artifact_perturbation_ids: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _runtime_mapping(config, name: str) -> dict[str, int]:
    raw = getattr(config, name, None)
    if raw is None:
        raise ValueError(f"covariate config must provide {name}")
    mapping = {str(key): int(value) for key, value in dict(raw).items()}
    if not mapping:
        raise ValueError(f"covariate config {name} cannot be empty")
    values = sorted(mapping.values())
    if values != list(range(len(values))):
        raise ValueError(f"covariate config {name} IDs must be contiguous from zero")
    return mapping


def _name_alignment(
    runtime: Mapping[str, int],
    artifact_names: np.ndarray,
    kind: str,
) -> torch.Tensor:
    names = [str(value) for value in artifact_names.tolist()]
    if len(names) != len(set(names)):
        raise ValueError(f"artifact {kind} names are not unique")
    artifact = {name: index for index, name in enumerate(names)}
    missing = sorted(set(runtime).difference(artifact))
    extra = sorted(set(artifact).difference(runtime))
    if missing or extra:
        raise ValueError(
            f"runtime/artifact {kind} names differ; missing={missing[:8]}, "
            f"extra={extra[:8]}"
        )
    aligned = torch.empty(len(runtime), dtype=torch.long)
    for name, runtime_id in runtime.items():
        aligned[runtime_id] = artifact[name]
    _validate_bijective_alignment(aligned, len(names), kind)
    return aligned


def _suffix_name_alignment(
    runtime: Mapping[str, int],
    artifact_names: np.ndarray,
    kind: str,
) -> torch.Tensor:
    """Align dataset-prefixed runtime names to canonical donor names."""

    names = [str(value) for value in artifact_names.tolist()]
    if len(names) != len(set(names)):
        raise ValueError(f"artifact {kind} names are not unique")
    aligned = torch.empty(len(runtime), dtype=torch.long)
    for runtime_name, runtime_id in runtime.items():
        matches = [
            index
            for index, artifact_name in enumerate(names)
            if runtime_name == artifact_name
            or runtime_name.endswith("_" + artifact_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"runtime {kind} {runtime_name!r} does not match exactly one "
                "artifact name"
            )
        aligned[runtime_id] = matches[0]
    _validate_bijective_alignment(aligned, len(names), kind)
    return aligned

def _composite_name_alignment(
    runtime_donors: Mapping[str, int],
    runtime_celltypes: Mapping[str, int],
    artifact_names: np.ndarray,
) -> torch.Tensor:
    """Align flattened runtime donor/cell-type pairs to composite artifact rows."""

    names = [str(value) for value in artifact_names.tolist()]
    if len(names) != len(set(names)):
        raise ValueError("artifact donor/cell-type names are not unique")
    parsed = []
    for name in names:
        if "::" not in name:
            raise ValueError(f"invalid donor/cell-type artifact name {name!r}")
        parsed.append(tuple(name.split("::", 1)))
    artifact = {name: index for index, name in enumerate(names)}
    artifact_donors = sorted({donor for donor, _ in parsed})
    artifact_celltypes = np.asarray(
        sorted({celltype for _, celltype in parsed}), dtype=str
    )
    donor_alignment = {}
    for runtime_name, runtime_id in runtime_donors.items():
        matches = [
            donor
            for donor in artifact_donors
            if runtime_name == donor or runtime_name.endswith("_" + donor)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"runtime donor {runtime_name!r} does not match exactly one "
                "artifact donor"
            )
        donor_alignment[int(runtime_id)] = matches[0]
    celltype_alignment = _name_alignment(
        runtime_celltypes, artifact_celltypes, "cell-type"
    )
    aligned = torch.full(
        (len(runtime_donors), len(runtime_celltypes)),
        -1,
        dtype=torch.long,
    )
    for runtime_donor_id, donor in donor_alignment.items():
        for runtime_celltype_id in range(len(runtime_celltypes)):
            celltype = str(
                artifact_celltypes[int(celltype_alignment[runtime_celltype_id])]
            )
            name = f"{donor}::{celltype}"
            if name in artifact:
                aligned[runtime_donor_id, runtime_celltype_id] = artifact[name]
    valid = aligned[aligned >= 0]
    expected = torch.arange(len(names), dtype=torch.long)
    if not torch.equal(torch.unique(valid, sorted=True), expected):
        raise ValueError("runtime donor/cell-type pairs do not cover artifact rows")
    return aligned.reshape(-1)



def _validate_bijective_alignment(
    alignment: torch.Tensor,
    artifact_size: int,
    kind: str,
) -> None:
    if alignment.dtype != torch.long or alignment.shape != (int(artifact_size),):
        raise ValueError(
            f"{kind} runtime-to-artifact alignment must be long "
            f"[{int(artifact_size)}]"
        )
    expected = torch.arange(
        int(artifact_size), device=alignment.device, dtype=torch.long
    )
    if not torch.equal(torch.sort(alignment).values, expected):
        unique = int(torch.unique(alignment).numel())
        minimum = int(alignment.min()) if alignment.numel() else None
        maximum = int(alignment.max()) if alignment.numel() else None
        raise ValueError(
            f"{kind} runtime-to-artifact alignment must be a bijection over "
            f"[0,{int(artifact_size)}); unique={unique}, min={minimum}, "
            f"max={maximum}"
        )


def _required_array(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"Parent-Residual artifact is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in artifact key {key!r}")
    return value


class ParentResidualPriorBank(nn.Module):
    """Inference-safe cross-line prior plus non-persistent train supervision.

    Inference tensors are persistent buffers, so a checkpoint records the
    exact prior used by the run.  Target-line training pseudobulks are loaded
    as non-persistent buffers and never enter a checkpoint.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        covariate_config,
        *,
        gene_dim: int,
        expected_sha256: str | None = None,
        prior_mode: str = "equal_line",
        calibration: str = "no_intercept",
        parent_grouping: str = "celltype",
        expected_normalization_divisor: float = 10.0,
    ):
        super().__init__()
        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256(path)
        if expected_sha256 not in (None, "", "null"):
            if actual_sha256 != str(expected_sha256).lower():
                raise ValueError(
                    "Parent-Residual artifact SHA256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        prior_mode = str(prior_mode).lower()
        if prior_mode not in _PRIOR_KEYS:
            raise ValueError(f"prior_mode must be one of {sorted(_PRIOR_KEYS)}")
        calibration = str(calibration).lower()
        if calibration not in _CALIBRATIONS:
            raise ValueError(
                f"calibration must be one of {sorted(_CALIBRATIONS)}"
            )
        parent_grouping = str(parent_grouping).lower()
        if parent_grouping not in {"celltype", "donor", "donor_celltype"}:
            raise ValueError(
                "parent_grouping must be celltype, donor, or donor_celltype"
            )

        with np.load(path, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(f"unsafe object dtype in artifact key {key!r}")
            schema = str(_required_array(loaded, "schema_version").reshape(-1)[0])
            if schema != _SCHEMA:
                raise ValueError(f"unsupported Parent-Residual schema {schema!r}")
            divisor = float(_required_array(loaded, "normalization_divisor"))
            if not np.isclose(divisor, float(expected_normalization_divisor)):
                raise ValueError(
                    "artifact/runtime normalization mismatch: "
                    f"{divisor} vs {expected_normalization_divisor}"
                )
            artifact_parent_grouping = "celltype"
            if "parent_context_kind" in loaded.files:
                artifact_parent_grouping = str(
                    _required_array(loaded, "parent_context_kind").reshape(-1)[0]
                ).lower()
            if artifact_parent_grouping != parent_grouping:
                raise ValueError(
                    "Parent artifact grouping mismatch: "
                    f"artifact={artifact_parent_grouping}, runtime={parent_grouping}"
                )

            runtime_perts = _runtime_mapping(covariate_config, "pert_dict")
            artifact_lines = _required_array(loaded, "cellline_names")
            runtime_batch_count = 0
            runtime_composite_celltype_count = 0
            if parent_grouping == "donor_celltype":
                runtime_donors = _runtime_mapping(covariate_config, "batch_dict")
                runtime_celltypes = _runtime_mapping(
                    covariate_config, "cell_type_dict"
                )
                line_alignment = _composite_name_alignment(
                    runtime_donors, runtime_celltypes, artifact_lines
                )
                runtime_batch_count = len(runtime_donors)
                runtime_composite_celltype_count = len(runtime_celltypes)
            elif parent_grouping == "donor":
                runtime_lines = _runtime_mapping(covariate_config, "batch_dict")
                line_alignment = _suffix_name_alignment(
                    runtime_lines, artifact_lines, "donor"
                )
            else:
                runtime_lines = _runtime_mapping(
                    covariate_config, "cell_type_dict"
                )
                line_alignment = _name_alignment(
                    runtime_lines, artifact_lines, "cell-line"
                )
            artifact_perturbations = _required_array(
                loaded, "perturbation_names"
            )
            pert_alignment = _name_alignment(
                runtime_perts,
                artifact_perturbations,
                "perturbation",
            )
            # Replogle perturbations and measured genes share canonical names.
            # Only an exact, unique equality is considered reliable enough for
            # target-gene exclusion in Child PDS. Other datasets/artifacts
            # safely retain -1 and therefore use every expression dimension.
            target_gene_index = np.full(
                artifact_perturbations.shape[0], -1, dtype=np.int64
            )
            if "gene_names" in loaded.files:
                artifact_genes = _required_array(loaded, "gene_names")
                if artifact_genes.shape != (int(gene_dim),):
                    raise ValueError(
                        "artifact gene_names must have shape [gene_dim]"
                    )
                gene_names = [str(value) for value in artifact_genes.tolist()]
                if len(gene_names) != len(set(gene_names)):
                    raise ValueError("artifact gene_names are not unique")
                gene_to_index = {
                    name: index for index, name in enumerate(gene_names)
                }
                target_gene_index = np.asarray(
                    [
                        gene_to_index.get(str(name), -1)
                        for name in artifact_perturbations.tolist()
                    ],
                    dtype=np.int64,
                )

            module_ids = _required_array(loaded, "module_ids").astype(np.int64)
            if module_ids.shape != (int(gene_dim),):
                raise ValueError(
                    f"artifact module_ids must have shape [{int(gene_dim)}]"
                )
            if module_ids.min() < 0:
                raise ValueError("artifact module_ids cannot be negative")
            module_count = int(module_ids.max()) + 1
            if np.any(np.bincount(module_ids, minlength=module_count) == 0):
                raise ValueError("artifact module_ids contains an empty module")

            control_mean = _required_array(loaded, "control_mean").astype(
                np.float32
            )
            prior_key, mask_key = _PRIOR_KEYS[prior_mode]
            prior = _required_array(loaded, prior_key).astype(np.float32)
            prior_mask = _required_array(loaded, mask_key).astype(bool)
            target_delta = _required_array(
                loaded, "train_condition_delta"
            ).astype(np.float32)
            target_mask = _required_array(
                loaded, "train_condition_mask"
            ).astype(bool)
            target_count = _required_array(
                loaded, "train_condition_count"
            ).astype(np.int64)
            validation_mask = _required_array(
                loaded, "validation_condition_mask"
            ).astype(bool)
            test_mask = _required_array(
                loaded, "test_condition_mask"
            ).astype(bool)
            if np.any(target_mask & (validation_mask | test_mask)):
                raise ValueError("held-out target condition appears in train supervision")
            expected_shape = (
                control_mean.shape[0],
                len(runtime_perts),
                int(gene_dim),
            )
            if prior.shape != expected_shape or target_delta.shape != expected_shape:
                raise ValueError(
                    "artifact prior/target shape does not match [L,P,G]="
                    f"{expected_shape}"
                )
            if prior_mask.shape != expected_shape[:2]:
                raise ValueError("artifact prior mask has the wrong [L,P] shape")
            if target_mask.shape != expected_shape[:2] or target_count.shape != expected_shape[:2]:
                raise ValueError("artifact train supervision has the wrong [L,P] shape")
            if control_mean.shape[1] != int(gene_dim):
                raise ValueError("artifact control mean has the wrong gene dimension")

            if calibration == "none":
                scale = np.ones_like(control_mean, dtype=np.float32)
                intercept = np.zeros_like(control_mean, dtype=np.float32)
            elif calibration == "no_intercept":
                scale = _required_array(
                    loaded, "ridge_scale_no_intercept"
                ).astype(np.float32)
                intercept = np.zeros_like(scale, dtype=np.float32)
            else:
                scale = _required_array(loaded, "ridge_affine_scale").astype(
                    np.float32
                )
                intercept = _required_array(
                    loaded, "ridge_affine_intercept"
                ).astype(np.float32)

        finite_arrays = {
            "control_mean": control_mean,
            "prior": prior,
            "target_delta": target_delta,
            "calibration_scale": scale,
            "calibration_intercept": intercept,
        }
        for name, value in finite_arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"artifact {name} must be finite")

        self.artifact_path = str(path)
        self.artifact_sha256 = actual_sha256
        self.prior_mode = prior_mode
        self.calibration = calibration
        self.parent_grouping = parent_grouping
        self.gene_dim = int(gene_dim)
        self.num_modules = module_count
        self._runtime_cellline_count = int(line_alignment.numel())
        self._runtime_perturbation_count = int(pert_alignment.numel())
        self.register_buffer("module_ids", torch.from_numpy(module_ids), persistent=True)
        self._artifact_parent_count = int(control_mean.shape[0])
        self._runtime_batch_count = int(runtime_batch_count)
        self._runtime_composite_celltype_count = int(runtime_composite_celltype_count)
        self.register_buffer(
            "runtime_cellline_to_artifact", line_alignment, persistent=True
        )
        self.register_buffer(
            "runtime_perturbation_to_artifact", pert_alignment, persistent=True
        )
        self.register_buffer(
            "control_mean", torch.from_numpy(control_mean.copy()), persistent=True
        )
        self.register_buffer(
            "cross_line_delta", torch.from_numpy(prior.copy()), persistent=True
        )
        self.register_buffer(
            "cross_line_mask", torch.from_numpy(prior_mask.copy()), persistent=True
        )
        self.register_buffer(
            "calibration_scale", torch.from_numpy(scale.copy()), persistent=True
        )
        self.register_buffer(
            "calibration_intercept",
            torch.from_numpy(intercept.copy()),
            persistent=True,
        )
        self.register_buffer(
            "train_target_delta",
            torch.from_numpy(target_delta.copy()),
            persistent=False,
        )
        self.register_buffer(
            "train_target_mask",
            torch.from_numpy(target_mask.copy()),
            persistent=False,
        )
        self.register_buffer(
            "train_target_count",
            torch.from_numpy(target_count.copy()),
            persistent=False,
        )
        self.register_buffer(
            "target_gene_index",
            torch.from_numpy(target_gene_index.copy()),
            persistent=False,
        )
        self._validate_runtime_alignments()

    def _validate_runtime_alignments(self) -> None:
        if self.parent_grouping == "donor_celltype":
            alignment = self.runtime_cellline_to_artifact
            expected_shape = (
                self._runtime_batch_count
                * self._runtime_composite_celltype_count
            )
            if alignment.dtype != torch.long or alignment.shape != (expected_shape,):
                raise ValueError("donor/cell-type alignment has the wrong shape")
            valid = alignment[alignment >= 0]
            expected = torch.arange(
                self._artifact_parent_count,
                device=alignment.device,
                dtype=torch.long,
            )
            if not torch.equal(torch.unique(valid, sorted=True), expected):
                raise ValueError("donor/cell-type alignment does not cover artifact rows")
        else:
            _validate_bijective_alignment(
                self.runtime_cellline_to_artifact,
                self._runtime_cellline_count,
                "cell-line",
            )
        _validate_bijective_alignment(
            self.runtime_perturbation_to_artifact,
            self._runtime_perturbation_count,
            "perturbation",
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        try:
            self._validate_runtime_alignments()
        except (TypeError, ValueError) as exc:
            error_msgs.append(f"{prefix}runtime alignment validation failed: {exc}")

    @staticmethod
    def _condition_ids(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.dtype != torch.long:
            raise TypeError(f"{name} must be a torch.long tensor")
        if value.ndim == 1:
            return value
        if value.ndim != 2 or value.shape[1] < 1:
            raise ValueError(f"{name} must have shape [B] or [B,S]")
        if not torch.equal(value, value[:, :1].expand_as(value)):
            raise ValueError(f"all cells in a row must share one {name}")
        return value[:, 0]

    def runtime_parent_ids(self, cov_celltype, cov_batch=None):
        if self.parent_grouping == "donor_celltype":
            if cov_batch is None:
                raise ValueError(
                    "donor_celltype Parent grouping requires cov_batch"
                )
            celltype = self._condition_ids(cov_celltype, "cov_celltype")
            batch = self._condition_ids(cov_batch, "cov_batch")
            if celltype.shape != batch.shape:
                raise ValueError("cov_celltype and cov_batch must share batch shape")
            if (
                (celltype < 0)
                | (celltype >= self._runtime_composite_celltype_count)
            ).any():
                raise ValueError("cov_celltype contains an out-of-range runtime ID")
            if ((batch < 0) | (batch >= self._runtime_batch_count)).any():
                raise ValueError("cov_batch contains an out-of-range runtime ID")
            return batch * self._runtime_composite_celltype_count + celltype
        if self.parent_grouping == "donor":
            if cov_batch is None:
                raise ValueError("donor Parent grouping requires cov_batch")
            value, name = cov_batch, "cov_batch"
        else:
            value, name = cov_celltype, "cov_celltype"
        runtime = self._condition_ids(value, name)
        if (
            (runtime < 0)
            | (runtime >= self.runtime_cellline_to_artifact.numel())
        ).any():
            raise ValueError(f"{name} contains an out-of-range runtime ID")
        return runtime

    def lookup(
        self,
        cov_celltype: torch.Tensor,
        cov_perturbation: torch.Tensor,
        cov_batch: torch.Tensor | None = None,
    ) -> ParentPriorLookup:
        runtime_line = self.runtime_parent_ids(cov_celltype, cov_batch)
        runtime_pert = self._condition_ids(cov_perturbation, "cov_pert")
        if runtime_line.shape != runtime_pert.shape:
            raise ValueError("Parent condition and cov_pert must share batch shape")
        if ((runtime_pert < 0) | (runtime_pert >= self.runtime_perturbation_to_artifact.numel())).any():
            raise ValueError("cov_pert contains an out-of-range runtime ID")
        line = self.runtime_cellline_to_artifact.index_select(0, runtime_line)
        perturbation = self.runtime_perturbation_to_artifact.index_select(
            0, runtime_pert
        )
        if (line < 0).any():
            raise ValueError("runtime condition has no Parent artifact row")
        control = self.control_mean[line]
        available = self.cross_line_mask[line, perturbation]
        raw_prior = self.cross_line_delta[line, perturbation]
        calibrated = (
            raw_prior * self.calibration_scale[line]
            + self.calibration_intercept[line]
        )
        calibrated = torch.where(
            available.unsqueeze(-1), calibrated, torch.zeros_like(calibrated)
        )
        target_available = self.train_target_mask[line, perturbation]
        target = self.train_target_delta[line, perturbation]
        target = torch.where(
            target_available.unsqueeze(-1), target, torch.zeros_like(target)
        )
        return ParentPriorLookup(
            control_mean=control,
            prior_delta=calibrated,
            prior_available=available,
            target_delta=target,
            target_available=target_available,
            target_count=self.train_target_count[line, perturbation],
            artifact_cellline_ids=line,
            artifact_perturbation_ids=perturbation,
        )


__all__ = ["ParentPriorLookup", "ParentResidualPriorBank"]

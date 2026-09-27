"""Leakage-safe real-control residual reservoir."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.parent_residual_gene_dit.source import ParentResidualSource


REAL_CONTROL_RESERVOIR_SCHEMA = "real_control_child_reservoir_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _required_array(loaded, key: str) -> np.ndarray:
    if key not in loaded.files:
        raise ValueError(f"real-control reservoir is missing {key!r}")
    value = np.asarray(loaded[key])
    if value.dtype == object:
        raise TypeError(f"unsafe object dtype in reservoir key {key!r}")
    return value


def _runtime_cellline_mapping(covariate_config, artifact_names) -> torch.Tensor:
    raw = getattr(covariate_config, "cell_type_dict", None)
    if raw is None:
        raise ValueError("covariate config must provide cell_type_dict")
    runtime = {str(name): int(index) for name, index in dict(raw).items()}
    if sorted(runtime.values()) != list(range(len(runtime))):
        raise ValueError("cell_type_dict IDs must be contiguous from zero")
    names = [str(value) for value in artifact_names.tolist()]
    if len(names) != len(set(names)):
        raise ValueError("reservoir cell-line names are not unique")
    artifact = {name: index for index, name in enumerate(names)}
    missing = sorted(set(runtime).difference(artifact))
    extra = sorted(set(artifact).difference(runtime))
    if missing or extra:
        raise ValueError(
            "runtime/reservoir cell-line names differ; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    aligned = torch.empty(len(runtime), dtype=torch.long)
    for name, runtime_index in runtime.items():
        aligned[runtime_index] = artifact[name]
    _validate_bijective_alignment(aligned, len(names))
    return aligned


def _runtime_group_mapping(
    covariate_config,
    artifact_names,
    grouping,
    artifact_group_sizes=None,
):
    """Map runtime categorical IDs onto a control artifact grouping."""

    mode = str(grouping)
    if mode == "celltype":
        return _runtime_cellline_mapping(covariate_config, artifact_names)
    names = [str(value) for value in artifact_names.tolist()]
    artifact = {name: index for index, name in enumerate(names)}
    if len(artifact) != len(names):
        raise ValueError("reservoir control-group names are not unique")
    group_sizes = np.ones(len(names), dtype=np.int64)
    if artifact_group_sizes is not None:
        group_sizes = np.asarray(artifact_group_sizes, dtype=np.int64)
        if group_sizes.shape != (len(names),) or (group_sizes <= 0).any():
            raise ValueError("artifact_group_sizes must be positive with shape [L]")
    raw_batches = getattr(covariate_config, "batch_dict", None)
    raw_celltypes = getattr(covariate_config, "cell_type_dict", None)
    if raw_batches is None or raw_celltypes is None:
        raise ValueError("donor grouping requires batch_dict and cell_type_dict")
    batches = {str(name): int(index) for name, index in dict(raw_batches).items()}
    celltypes = {str(name): int(index) for name, index in dict(raw_celltypes).items()}
    if sorted(batches.values()) != list(range(len(batches))):
        raise ValueError("batch_dict IDs must be contiguous from zero")
    if sorted(celltypes.values()) != list(range(len(celltypes))):
        raise ValueError("cell_type_dict IDs must be contiguous from zero")

    def donor_name(runtime_name):
        candidates = [
            name for name in names
            if runtime_name == name
            or runtime_name.endswith("_" + name.split("::", 1)[0])
        ]
        donors = sorted({name.split("::", 1)[0] for name in candidates})
        if len(donors) != 1:
            raise ValueError(
                f"cannot align runtime batch {runtime_name!r} to one donor"
            )
        return donors[0]

    if mode == "donor":
        aligned = torch.empty(len(batches), dtype=torch.long)
        for runtime_name, runtime_index in batches.items():
            donor = donor_name(runtime_name)
            if donor not in artifact:
                raise ValueError(f"donor {donor!r} is missing from reservoir")
            aligned[runtime_index] = artifact[donor]
    elif mode == "donor_celltype":
        aligned = torch.full(
            (len(batches), len(celltypes)), -1, dtype=torch.long
        )
        for runtime_name, runtime_index in batches.items():
            donor = donor_name(runtime_name)
            for celltype, celltype_index in celltypes.items():
                group = f"{donor}::{celltype}"
                if group not in artifact:
                    suffix = f"::{celltype}"
                    candidates = [
                        name for name in names if name.endswith(suffix)
                    ]
                    if candidates:
                        group = sorted(
                            candidates,
                            key=lambda name: (
                                -int(group_sizes[artifact[name]]),
                                name,
                            ),
                        )[0]
                if group in artifact:
                    aligned[runtime_index, celltype_index] = artifact[group]
    else:
        raise ValueError(f"unsupported control grouping {mode!r}")
    valid = aligned[aligned >= 0]
    if not torch.equal(
        torch.unique(valid, sorted=True), torch.arange(len(names), dtype=torch.long)
    ):
        raise ValueError("runtime control groups do not cover the reservoir exactly")
    return aligned


def _validate_bijective_alignment(
    alignment: torch.Tensor,
    artifact_size: int,
) -> None:
    if alignment.dtype != torch.long or alignment.shape != (int(artifact_size),):
        raise ValueError(
            "real-control runtime-to-artifact cell-line alignment must be "
            f"long [{int(artifact_size)}]"
        )
    expected = torch.arange(
        int(artifact_size), device=alignment.device, dtype=torch.long
    )
    if not torch.equal(torch.sort(alignment).values, expected):
        unique = int(torch.unique(alignment).numel())
        minimum = int(alignment.min()) if alignment.numel() else None
        maximum = int(alignment.max()) if alignment.numel() else None
        raise ValueError(
            "real-control runtime-to-artifact cell-line alignment must be a "
            f"bijection over [0,{int(artifact_size)}); unique={unique}, "
            f"min={minimum}, max={maximum}"
        )


def _condition_ids(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.long:
        raise TypeError(f"{name} must be a torch.long tensor")
    if value.ndim == 1:
        return value
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape [B] or [B,S]")
    if not torch.equal(value, value[:, :1].expand_as(value)):
        raise ValueError(f"all response cells in a row must share one {name}")
    return value[:, 0]


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class RealControlResidualSample:
    """One independently sampled real control residual per response cell."""

    residual: torch.Tensor
    member_indices: torch.Tensor
    member_counts: torch.Tensor
    artifact_cellline_ids: torch.Tensor
    child_indices: torch.Tensor

@dataclass(frozen=True)
class RealControlGlobalSample:
    """Global control-marginal sample with no public Child identity."""

    residual: torch.Tensor
    member_indices: torch.Tensor
    member_counts: torch.Tensor
    artifact_cellline_ids: torch.Tensor



class RealControlResidualReservoir(nn.Module):
    """Load and sample a pickle-free ``[line, Child, member, gene]`` bank.

    Large expression buffers are deliberately non-persistent: the artifact
    hash and path are the source of truth, while checkpoints remain compact.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        covariate_config,
        *,
        gene_dim: int,
        child_capacity: int,
        sampling_mode: str = "child",
        expected_sha256: str | None = None,
        expected_normalization_divisor: float = 10.0,
        validate_runtime_prototypes: bool = True,
        control_grouping: str = "celltype",
        recenter_to_runtime_control_mean: bool = False,
        prototype_atol: float = 2e-5,
        prototype_rtol: float = 2e-4,
    ):
        super().__init__()
        if not isinstance(sampling_mode, str):
            raise TypeError("sampling_mode must be a string")
        sampling_mode = sampling_mode.strip().lower()
        if sampling_mode not in {"child", "global_marginal"}:
            raise ValueError(
                "sampling_mode must be 'child' or 'global_marginal'"
            )
        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256(path)
        if expected_sha256 not in (None, "", "null"):
            if actual_sha256 != str(expected_sha256).lower():
                raise ValueError(
                    "real-control reservoir SHA256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

        with np.load(path, allow_pickle=False) as loaded:
            for key in loaded.files:
                if loaded[key].dtype == object:
                    raise TypeError(
                        f"unsafe object dtype in reservoir key {key!r}"
                    )
            schema = str(_required_array(loaded, "schema_version").reshape(-1)[0])
            if schema != REAL_CONTROL_RESERVOIR_SCHEMA:
                raise ValueError(f"unsupported real-control schema {schema!r}")
            control_only = bool(
                _required_array(loaded, "control_only").reshape(-1)[0]
            )
            treated_used = bool(
                _required_array(loaded, "treated_expression_used").reshape(-1)[0]
            )
            if not control_only or treated_used:
                raise ValueError(
                    "reservoir leakage audit failed: only control expression is allowed"
                )
            divisor = float(
                _required_array(loaded, "normalization_divisor").reshape(-1)[0]
            )
            if not np.isclose(divisor, float(expected_normalization_divisor)):
                raise ValueError(
                    "reservoir/runtime normalization mismatch: "
                    f"{divisor} vs {expected_normalization_divisor}"
                )
            cellline_names = _required_array(loaded, "cellline_names")
            residuals = _required_array(loaded, "member_residuals").astype(
                np.float32
            )
            member_mask = _required_array(loaded, "member_mask").astype(bool)
            child_mask = _required_array(loaded, "child_mask").astype(bool)
            prototypes = _required_array(loaded, "child_prototypes").astype(
                np.float32
            )
            control_mean = _required_array(loaded, "control_mean").astype(
                np.float32
            )
            full_child_counts = (
                _required_array(loaded, "full_child_counts")
                if "full_child_counts" in loaded.files
                else None
            )

        if residuals.ndim != 4:
            raise ValueError("member_residuals must have shape [L,K,R,G]")
        lines, children, members, genes = residuals.shape
        if lines < 1 or children < 1 or members < 1 or genes < 1:
            raise ValueError("reservoir L,K,R,G dimensions must be non-empty")
        if genes != int(gene_dim):
            raise ValueError(
                f"reservoir gene dimension {genes} does not match {int(gene_dim)}"
            )
        if children != int(child_capacity):
            raise ValueError(
                f"reservoir Child capacity {children} does not match "
                f"configured {int(child_capacity)}"
            )
        if cellline_names.shape != (lines,):
            raise ValueError("cellline_names must have shape [L]")
        if member_mask.shape != (lines, children, members):
            raise ValueError("member_mask must have shape [L,K,R]")
        if child_mask.shape != (lines, children):
            raise ValueError("child_mask must have shape [L,K]")
        if prototypes.shape != (lines, children, genes):
            raise ValueError("child_prototypes must have shape [L,K,G]")
        if control_mean.shape != (lines, genes):
            raise ValueError("control_mean must have shape [L,G]")
        counts = member_mask.sum(axis=-1, dtype=np.int64)
        if np.any(child_mask & (counts == 0)):
            raise ValueError("every active Child needs at least one real member")
        if np.any((~child_mask) & (counts != 0)):
            raise ValueError("inactive Children cannot contain real members")
        if np.any(member_mask & ~child_mask[..., None]):
            raise ValueError("member_mask activates an inactive Child")
        if not np.isfinite(residuals[member_mask]).all():
            raise ValueError("valid member residuals must be finite")
        if not np.isfinite(prototypes[child_mask]).all():
            raise ValueError("active Child prototypes must be finite")
        if not np.isfinite(control_mean).all():
            raise ValueError("control_mean must be finite")
        if full_child_counts is None:
            if sampling_mode == "global_marginal":
                raise ValueError(
                    "global_marginal sampling requires full_child_counts"
                )
            # Legacy artifacts remain valid in the default per-Child mode.
            full_counts = counts.copy()
        else:
            if full_child_counts.dtype.kind not in {"i", "u"}:
                raise TypeError("full_child_counts must have an integer dtype")
            if full_child_counts.shape != (lines, children):
                raise ValueError("full_child_counts must have shape [L,K]")
            if (
                full_child_counts.dtype.kind == "u"
                and full_child_counts.size
                and int(full_child_counts.max()) > np.iinfo(np.int64).max
            ):
                raise ValueError("full_child_counts exceeds int64 range")
            full_counts = full_child_counts.astype(np.int64, copy=False)
            if np.any(full_counts < 0):
                raise ValueError("full_child_counts cannot be negative")
            if np.any(child_mask & (full_counts <= 0)):
                raise ValueError(
                    "every active Child needs positive full_child_counts"
                )
            if np.any((~child_mask) & (full_counts != 0)):
                raise ValueError(
                    "inactive Children must have zero full_child_counts"
                )
            if np.any(full_counts < counts):
                raise ValueError(
                    "full_child_counts cannot be smaller than stored members"
                )
            maximum = np.iinfo(np.int64).max
            if any(sum(map(int, row)) > maximum for row in full_counts):
                raise ValueError("per-line full_child_counts sum exceeds int64")
        residuals = np.where(member_mask[..., None], residuals, 0.0)
        prototypes = np.where(child_mask[..., None], prototypes, 0.0)

        self.artifact_path = str(path)
        self.artifact_sha256 = actual_sha256
        self.gene_dim = int(gene_dim)
        self.child_capacity = int(child_capacity)
        self.members_per_child = int(members)
        self.sampling_mode = sampling_mode
        self.control_grouping = str(control_grouping)
        if self.control_grouping not in {"celltype", "donor", "donor_celltype"}:
            raise ValueError("invalid real-control grouping")
        self.recenter_to_runtime_control_mean = bool(
            recenter_to_runtime_control_mean
        )
        self.validate_runtime_prototypes = bool(validate_runtime_prototypes)
        self.prototype_atol = _finite_nonnegative("prototype_atol", prototype_atol)
        self.prototype_rtol = _finite_nonnegative("prototype_rtol", prototype_rtol)
        alignment = _runtime_group_mapping(
            covariate_config,
            cellline_names,
            self.control_grouping,
            artifact_group_sizes=full_counts.sum(axis=1, dtype=np.int64),
        )
        self._artifact_group_count = int(len(cellline_names))
        self.register_buffer(
            "runtime_cellline_to_artifact", alignment, persistent=True
        )
        self.register_buffer(
            "member_residuals",
            torch.from_numpy(np.ascontiguousarray(residuals)),
            persistent=False,
        )
        self.register_buffer(
            "member_mask",
            torch.from_numpy(np.ascontiguousarray(member_mask)),
            persistent=False,
        )
        self.register_buffer(
            "member_counts",
            torch.from_numpy(np.ascontiguousarray(counts)),
            persistent=False,
        )
        self.register_buffer(
            "full_child_counts",
            torch.from_numpy(np.ascontiguousarray(full_counts)),
            persistent=False,
        )
        self.register_buffer(
            "child_mask",
            torch.from_numpy(np.ascontiguousarray(child_mask)),
            persistent=False,
        )
        self.register_buffer(
            "child_prototypes",
            torch.from_numpy(np.ascontiguousarray(prototypes)),
            persistent=False,
        )
        self.register_buffer(
            "control_mean",
            torch.from_numpy(np.ascontiguousarray(control_mean)),
            persistent=False,
        )
        self._validate_runtime_alignment()

    def _validate_runtime_alignment(self) -> None:
        alignment = self.runtime_cellline_to_artifact
        if self.control_grouping == "celltype":
            _validate_bijective_alignment(
                alignment,
                self._artifact_group_count,
            )
            return
        if alignment.dtype != torch.long or alignment.ndim not in {1, 2}:
            raise ValueError("runtime control-group alignment must be long [N] or [N,M]")
        valid = alignment[alignment >= 0]
        expected = torch.arange(
            self._artifact_group_count, device=alignment.device, dtype=torch.long
        )
        if not torch.equal(torch.unique(valid, sorted=True), expected):
            raise ValueError("runtime alignment does not cover every artifact group")

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
            self._validate_runtime_alignment()
        except (TypeError, ValueError) as exc:
            error_msgs.append(f"{prefix}runtime alignment validation failed: {exc}")

    def artifact_cellline_ids(
        self,
        cov_celltype: torch.Tensor,
        cov_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        celltype = _condition_ids(cov_celltype, "cov_celltype")
        if self.control_grouping == "celltype":
            if ((celltype < 0) | (celltype >= self.runtime_cellline_to_artifact.shape[0])).any():
                raise ValueError("cov_celltype contains an out-of-range runtime ID")
            result = self.runtime_cellline_to_artifact.index_select(0, celltype)
        else:
            if cov_batch is None:
                raise ValueError("donor-grouped reservoir requires cov_batch")
            batch = _condition_ids(cov_batch, "cov_batch")
            if ((batch < 0) | (batch >= self.runtime_cellline_to_artifact.shape[0])).any():
                raise ValueError("cov_batch contains an out-of-range runtime ID")
            if self.control_grouping == "donor":
                result = self.runtime_cellline_to_artifact.index_select(0, batch)
            else:
                if ((celltype < 0) | (celltype >= self.runtime_cellline_to_artifact.shape[1])).any():
                    raise ValueError("cov_celltype contains an out-of-range runtime ID")
                result = self.runtime_cellline_to_artifact[batch, celltype]
        if (result < 0).any():
            raise ValueError("runtime condition has no supported control group")
        return result

    def _validate_prototypes(
        self,
        *,
        artifact_lines: torch.Tensor,
        runtime_prototypes: torch.Tensor,
        bank_inverse: torch.Tensor,
    ) -> None:
        if not self.validate_runtime_prototypes:
            return
        if runtime_prototypes.ndim != 3:
            raise ValueError("runtime_prototypes must have shape [U,K,G]")
        if runtime_prototypes.shape[1:] != (
            self.child_capacity,
            self.gene_dim,
        ):
            raise ValueError("runtime_prototypes have incompatible [K,G]")
        if bank_inverse.dtype != torch.long or bank_inverse.ndim != 1:
            raise TypeError("bank_inverse must be a one-dimensional long tensor")
        if bank_inverse.shape != artifact_lines.shape:
            raise ValueError("bank_inverse must have shape [B]")
        if ((bank_inverse < 0) | (bank_inverse >= runtime_prototypes.shape[0])).any():
            raise ValueError("bank_inverse contains an out-of-range bank row")
        for bank_row in torch.unique(bank_inverse, sorted=True):
            rows = bank_inverse == bank_row
            line_values = torch.unique(artifact_lines[rows])
            if line_values.numel() != 1:
                raise ValueError(
                    "one runtime control bank cannot mix multiple cell lines"
                )
            line = line_values[0]
            expected_mask = self.child_mask[line]
            expected = self.child_prototypes[line, expected_mask]
            observed = runtime_prototypes[bank_row, expected_mask].float()
            if not torch.allclose(
                observed,
                expected.float(),
                atol=self.prototype_atol,
                rtol=self.prototype_rtol,
            ):
                maximum = float((observed - expected.float()).abs().max().cpu())
                raise ValueError(
                    "runtime Child order/prototypes do not match the reservoir; "
                    f"maximum absolute difference={maximum:.7g}"
                )

    def sample(
        self,
        *,
        cov_celltype: torch.Tensor,
        cov_batch: Optional[torch.Tensor] = None,
        selected_indices: torch.Tensor,
        stochastic: bool = True,
        generator: Optional[torch.Generator] = None,
        uniforms: Optional[torch.Tensor] = None,
        runtime_child_mask: Optional[torch.Tensor] = None,
        runtime_control_mean: Optional[torch.Tensor] = None,
        runtime_prototypes: Optional[torch.Tensor] = None,
        bank_inverse: Optional[torch.Tensor] = None,
    ) -> RealControlResidualSample:
        """Sample real controls with replacement using the configured mode.

        "child" samples inside each supplied Child exactly as before.
        "global_marginal" ignores the supplied Child values and uses each
        uniform once: first to select a Child by full-population mass, then to
        select uniformly among that Child's stored reservoir members.
        """

        if not torch.is_tensor(selected_indices) or selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must be a torch.long tensor")
        if selected_indices.ndim != 2 or selected_indices.shape[1] < 1:
            raise ValueError("selected_indices must have shape [B,S], S >= 1")
        artifact_lines = self.artifact_cellline_ids(cov_celltype, cov_batch)
        if artifact_lines.shape != (selected_indices.shape[0],):
            raise ValueError("cov_celltype B must match selected_indices")
        if selected_indices.device != self.member_residuals.device:
            raise ValueError("selected_indices and reservoir must share one device")
        if artifact_lines.device != selected_indices.device:
            artifact_lines = artifact_lines.to(selected_indices.device)
        lines = artifact_lines[:, None].expand_as(selected_indices)
        if self.sampling_mode == "child":
            if (
                (selected_indices < 0)
                | (selected_indices >= self.child_capacity)
            ).any():
                raise ValueError("selected_indices contains an out-of-range Child")
            active = self.child_mask[lines, selected_indices]
            if not active.all():
                raise ValueError(
                    "selected_indices selects an inactive reservoir Child"
                )
        if runtime_child_mask is not None:
            if (
                runtime_child_mask.dtype != torch.bool
                or runtime_child_mask.shape
                != (selected_indices.shape[0], self.child_capacity)
            ):
                raise ValueError("runtime_child_mask must be bool with shape [B,K]")
            expected_child_mask = self.child_mask.index_select(0, artifact_lines)
            if not torch.equal(runtime_child_mask, expected_child_mask):
                raise ValueError(
                    "runtime active Child mask does not match the reservoir"
                )
        runtime_recenter_shift = None
        if runtime_control_mean is not None:
            if (
                not torch.is_tensor(runtime_control_mean)
                or not runtime_control_mean.is_floating_point()
                or runtime_control_mean.shape
                != (selected_indices.shape[0], self.gene_dim)
            ):
                raise ValueError(
                    "runtime_control_mean must be floating point with shape [B,G]"
                )
            expected_control_mean = self.control_mean.index_select(
                0, artifact_lines
            ).to(runtime_control_mean)
            if self.recenter_to_runtime_control_mean:
                runtime_recenter_shift = (
                    expected_control_mean - runtime_control_mean
                )
            elif not torch.allclose(
                runtime_control_mean.float(),
                expected_control_mean.float(),
                atol=self.prototype_atol,
                rtol=self.prototype_rtol,
            ):
                raise ValueError(
                    "runtime Parent and reservoir control means do not match"
                )
        if self.sampling_mode == "global_marginal":
            if runtime_prototypes is not None or bank_inverse is not None:
                raise ValueError(
                    "runtime_prototypes and bank_inverse must be omitted in "
                    "global_marginal mode"
                )
        else:
            if (runtime_prototypes is None) != (bank_inverse is None):
                raise ValueError(
                    "runtime_prototypes and bank_inverse must be supplied together"
                )
            if runtime_prototypes is not None:
                self._validate_prototypes(
                    artifact_lines=artifact_lines,
                    runtime_prototypes=runtime_prototypes,
                    bank_inverse=bank_inverse,
                )

        if uniforms is not None:
            if generator is not None:
                raise ValueError("pass either uniforms or generator, not both")
            if not torch.is_tensor(uniforms) or not uniforms.is_floating_point():
                raise TypeError("uniforms must be a floating-point tensor")
            if uniforms.shape != selected_indices.shape:
                raise ValueError("uniforms must have shape [B,S]")
            if uniforms.device != selected_indices.device:
                raise ValueError("uniforms must share the selected-index device")
            if not torch.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms >= 1).any():
                raise ValueError("uniforms must be finite values in [0,1)")
            random_values = uniforms.float()
        elif bool(stochastic):
            if generator is not None:
                if not isinstance(generator, torch.Generator):
                    raise TypeError("generator must be a torch.Generator")
                generator_device = torch.device(generator.device)
                if generator_device.type != selected_indices.device.type:
                    raise ValueError("generator and reservoir must share device type")
            random_values = torch.rand(
                selected_indices.shape,
                dtype=torch.float32,
                device=selected_indices.device,
                generator=generator,
            )
        else:
            random_values = torch.zeros(
                selected_indices.shape,
                dtype=torch.float32,
                device=selected_indices.device,
            )
        sampled_children = selected_indices
        within_child_uniforms = random_values
        if self.sampling_mode == "global_marginal":
            line_full_counts = self.full_child_counts.index_select(
                0, artifact_lines
            )
            totals = line_full_counts.sum(dim=-1)
            if (totals <= 0).any():
                raise RuntimeError(
                    "global_marginal control distribution has zero total mass"
                )
            cumulative = line_full_counts.cumsum(dim=-1)
            scaled = random_values.double() * totals[:, None].double()
            mass_slots = torch.floor(scaled).to(torch.long)
            mass_slots = torch.minimum(mass_slots, totals[:, None] - 1)
            sampled_children = torch.searchsorted(
                cumulative.contiguous(),
                mass_slots.contiguous(),
                right=True,
            )
            sampled_full_counts = line_full_counts.gather(
                1, sampled_children
            )
            lower_mass = cumulative.gather(1, sampled_children) - (
                sampled_full_counts
            )
            within_child_uniforms = (
                scaled - lower_mass.double()
            ) / sampled_full_counts.double()
            within_child_uniforms = within_child_uniforms.clamp(
                min=0.0,
                max=torch.nextafter(
                    torch.ones((), dtype=torch.float64, device=scaled.device),
                    torch.zeros((), dtype=torch.float64, device=scaled.device),
                ),
            )

        counts = self.member_counts[lines, sampled_children]
        if (counts <= 0).any():
            raise RuntimeError("sampled active Child has no real control members")
        member_indices = torch.floor(
            within_child_uniforms * counts.to(within_child_uniforms.dtype)
        ).to(torch.long)
        member_indices = torch.minimum(member_indices, counts - 1)
        residual = self.member_residuals[
            lines, sampled_children, member_indices
        ]
        if runtime_recenter_shift is not None:
            residual = residual + runtime_recenter_shift[:, None, :]
        if residual.shape != (*selected_indices.shape, self.gene_dim):
            raise RuntimeError("real-control gather returned an invalid shape")
        if not torch.isfinite(residual).all():
            raise RuntimeError("sampled real-control residual contains non-finite values")
        return RealControlResidualSample(
            residual=residual,
            member_indices=member_indices,
            member_counts=counts,
            artifact_cellline_ids=artifact_lines,
            child_indices=sampled_children,
        )
    def sample_global(
        self,
        *,
        cov_celltype: torch.Tensor,
        num_samples: int,
        cov_batch: Optional[torch.Tensor] = None,
        stochastic: bool = True,
        generator: Optional[torch.Generator] = None,
        uniforms: Optional[torch.Tensor] = None,
        runtime_control_mean: Optional[torch.Tensor] = None,
    ) -> RealControlGlobalSample:
        """Sample the full control marginal without accepting a Child ID."""

        if self.sampling_mode != "global_marginal":
            raise RuntimeError(
                "sample_global requires sampling_mode='global_marginal'"
            )
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        batch_size = int(
            _condition_ids(cov_celltype, "cov_celltype").shape[0]
        )
        opaque = torch.zeros(
            batch_size,
            int(num_samples),
            dtype=torch.long,
            device=self.member_residuals.device,
        )
        sampled = self.sample(
            cov_celltype=cov_celltype,
            cov_batch=cov_batch,
            selected_indices=opaque,
            stochastic=stochastic,
            generator=generator,
            uniforms=uniforms,
            runtime_child_mask=None,
            runtime_control_mean=runtime_control_mean,
            runtime_prototypes=None,
            bank_inverse=None,
        )
        return RealControlGlobalSample(
            residual=sampled.residual,
            member_indices=sampled.member_indices,
            member_counts=sampled.member_counts,
            artifact_cellline_ids=sampled.artifact_cellline_ids,
        )




def build_real_control_parent_source(
    *,
    parent_mean: torch.Tensor,
    sample: RealControlResidualSample,
    prior_probs: torch.Tensor,
    child_mask: torch.Tensor,
    scale: float = 1.0,
) -> ParentResidualSource:
    """Combine a detached Parent mean with a sampled real control residual."""

    scale = _finite_nonnegative("scale", scale)
    if not torch.is_tensor(parent_mean) or not parent_mean.is_floating_point():
        raise TypeError("parent_mean must be a floating-point tensor")
    if parent_mean.ndim != 2:
        raise ValueError("parent_mean must have shape [B,G]")
    residual = sample.residual
    if residual.ndim != 3 or residual.shape[0] != parent_mean.shape[0]:
        raise ValueError("sample residual must have shape [B,S,G]")
    if residual.shape[2] != parent_mean.shape[1]:
        raise ValueError("sample residual and parent_mean must share G")
    if residual.device != parent_mean.device or residual.dtype != parent_mean.dtype:
        residual = residual.to(parent_mean)
    if prior_probs.ndim != 2 or prior_probs.shape[0] != parent_mean.shape[0]:
        raise ValueError("prior_probs must have shape [B,K]")
    if child_mask.shape != prior_probs.shape or child_mask.dtype != torch.bool:
        raise ValueError("child_mask must be bool with shape [B,K]")
    if not torch.isfinite(parent_mean).all() or not torch.isfinite(residual).all():
        raise ValueError("parent_mean and real residual must be finite")
    scaled_residual = residual * scale
    source = parent_mean.detach()[:, None, :] + scaled_residual
    # The residual is already a realized member, so there is no extra
    # diagonal-Gaussian term. Parent-lock performs exact condition centering.
    return ParentResidualSource(
        source=source,
        mean=source,
        center=parent_mean.new_zeros(parent_mean.shape),
        residual=scaled_residual,
        std=torch.zeros_like(source),
        prior_probs=prior_probs,
        mask=child_mask,
        selected_indices=sample.child_indices,
    )


__all__ = [
    "REAL_CONTROL_RESERVOIR_SCHEMA",
    "RealControlResidualReservoir",
    "RealControlResidualSample",
    "build_real_control_parent_source",
]

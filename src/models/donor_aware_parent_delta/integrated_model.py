"""Opt-in integration with the existing Parent-Residual Gene-DiT wrapper."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from src.models.parent_residual_gene_dit.wrapper import (
    ParentMeanPrediction,
    ParentResidualGeneDiTModel,
)

from .config import DonorAwareParentDeltaConfig
from .head import DonorAwareParentDeltaHead


@dataclass(frozen=True)
class DonorAwareParentMeanPrediction(ParentMeanPrediction):
    donor_aware_correction_delta: torch.Tensor | None = None
    donor_aware_raw_adjustment: torch.Tensor | None = None
    donor_aware_reliability: torch.Tensor | None = None
    donor_aware_control_state_rms: torch.Tensor | None = None
    donor_aware_proposal_rms: torch.Tensor | None = None
    donor_aware_realized_rms: torch.Tensor | None = None
    donor_aware_response_strength_gate_probability: torch.Tensor | None = None
    donor_aware_response_strength_gate_logit: torch.Tensor | None = None


def matched_control_group_summary(
    control: torch.Tensor,
    cov_celltype: torch.Tensor,
    cov_batch: torch.Tensor,
    *,
    num_celltypes: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean matched PBS controls by donor x cell type, returned per row."""

    if not torch.is_tensor(control) or not control.is_floating_point():
        raise TypeError("control must be a floating-point tensor")
    if control.ndim != 3 or min(control.shape) < 1:
        raise ValueError("control must have non-empty shape [B,S,G]")
    if not torch.isfinite(control).all():
        raise ValueError("control must be finite")
    batch_size, set_size, genes = control.shape
    if isinstance(num_celltypes, bool) or not isinstance(num_celltypes, int):
        raise TypeError("num_celltypes must be an integer")
    if num_celltypes < 1:
        raise ValueError("num_celltypes must be positive")

    def expand_ids(name: str, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value) or value.dtype != torch.long:
            raise TypeError(f"{name} must be a torch.long tensor")
        if value.device != control.device:
            raise ValueError(f"{name} must share the control device")
        if tuple(value.shape) == (batch_size,):
            return value[:, None].expand(-1, set_size)
        if tuple(value.shape) != (batch_size, set_size):
            raise ValueError(f"{name} must have shape [B] or [B,S]")
        return value

    celltype = expand_ids("cov_celltype", cov_celltype)
    donor = expand_ids("cov_batch", cov_batch)
    if (celltype < 0).any() or (celltype >= num_celltypes).any():
        raise ValueError("cov_celltype contains an out-of-range ID")
    if (donor < 0).any():
        raise ValueError("cov_batch contains a negative ID")
    if valid_mask is None:
        valid = torch.ones(
            batch_size, set_size, device=control.device, dtype=torch.bool
        )
    else:
        if not torch.is_tensor(valid_mask) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if valid_mask.device != control.device:
            raise ValueError("valid_mask must share the control device")
        if tuple(valid_mask.shape) == (batch_size,):
            valid = valid_mask[:, None].expand(-1, set_size)
        elif tuple(valid_mask.shape) == (batch_size, set_size):
            valid = valid_mask
        else:
            raise ValueError("valid_mask must have shape [B] or [B,S]")
    if not valid.any(dim=1).all():
        raise ValueError("every row needs at least one valid matched control")

    group = donor * int(num_celltypes) + celltype
    valid_groups = group[valid]
    unique, inverse = torch.unique(valid_groups, sorted=True, return_inverse=True)
    values = control[valid].to(torch.float32)
    sums = torch.zeros(
        unique.numel(), genes, device=control.device, dtype=torch.float32
    ).index_add(0, inverse, values)
    counts = torch.zeros(
        unique.numel(), device=control.device, dtype=torch.float32
    ).index_add(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    means = sums / counts[:, None].clamp_min(1.0)

    row_group = group[:, 0]
    if not torch.equal(group, row_group[:, None].expand_as(group)):
        raise ValueError("all cells in a row must share donor and cell type")
    positions = torch.searchsorted(unique, row_group)
    if (positions >= unique.numel()).any() or not torch.equal(
        unique.index_select(0, positions), row_group
    ):
        raise RuntimeError("failed to recover a matched-control group summary")
    return (
        means.index_select(0, positions).to(control).detach(),
        counts.index_select(0, positions).detach(),
    )


class DonorAwareParentResidualGeneDiTModel(ParentResidualGeneDiTModel):
    """Parent wrapper with one bounded PBS-state x perturbation interaction."""

    def __init__(self, model_cfg, covariate_config):
        super().__init__(model_cfg, covariate_config)
        if self.prior_bank.parent_grouping != "donor_celltype":
            raise ValueError(
                "donor-aware Parent delta requires "
                "parent_residual_parent_grouping=donor_celltype"
            )
        incompatible = {
            "parent_residual_soft_gated_regression_enabled": bool(
                self.soft_gated_regression_enabled
            ),
            "parent_residual_child_guidance_enabled": bool(
                self.child_guidance_enabled
            ),
            "parent_residual_child_marginal_enabled": bool(
                self.child_marginal_enabled
            ),
            "parent_residual_child_consensus_bridge_enabled": bool(
                self.child_consensus_bridge_enabled
            ),
            "parent_residual_fewshot_adapter_enabled": (
                self.fewshot_parent_adapter is not None
            ),
        }
        active = [name for name, enabled in incompatible.items() if enabled]
        if active:
            raise ValueError(
                "the first donor-aware Parent experiment requires these "
                f"switches=false: {', '.join(active)}"
            )
        context_dim = int(
            getattr(model_cfg, "parent_residual_context_hidden_dim", 128)
        )
        gene_identity_dim = int(
            self.gene_dit.main_tokenizer.gene_identity.shape[1]
        )
        cfg = DonorAwareParentDeltaConfig(
            gene_dim=self.gene_dim,
            perturbation_dim=context_dim,
            hidden_dim=int(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_hidden_dim",
                    128,
                )
            ),
            gene_identity_dim=gene_identity_dim,
            gene_rank=int(
                getattr(model_cfg, "parent_residual_donor_aware_gene_rank", 64)
            ),
            minimum_control_cells=int(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_minimum_control_cells",
                    20,
                )
            ),
            reliability_tau=float(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_reliability_tau",
                    64.0,
                )
            ),
            max_delta_rms_ratio=float(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_max_delta_rms_ratio",
                    0.75,
                )
            ),
            correction_scale=float(self.correction_max_scale),
            norm_eps=float(
                getattr(model_cfg, "parent_residual_norm_eps", 1.0e-6)
            ),
            detach_control_summary=True,
            detach_perturbation_context=True,
            response_strength_gate_enabled=bool(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_response_strength_gate_enabled",
                    False,
                )
            ),
            response_strength_gate_hidden_dim=int(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_response_strength_gate_hidden_dim",
                    16,
                )
            ),
            response_strength_gate_initial_probability=float(
                getattr(
                    model_cfg,
                    "parent_residual_donor_aware_response_strength_gate_initial_probability",
                    0.999,
                )
            ),
        ).validate()
        with torch.random.fork_rng(devices=[]):
            self.donor_aware_parent_delta_head = DonorAwareParentDeltaHead(cfg)
        self.donor_aware_parent_delta_enabled = True
        self._donor_aware_num_celltypes = int(
            self.prior_bank._runtime_composite_celltype_count
        )
        self._register_celltype_control_reference()

    def _register_celltype_control_reference(self) -> None:
        path = Path(self.prior_bank.artifact_path)
        with np.load(path, allow_pickle=False) as archive:
            if "control_count" not in archive.files:
                raise ValueError(
                    "donor-aware Parent artifact is missing control_count"
                )
            counts = np.asarray(archive["control_count"])
        if not np.issubdtype(counts.dtype, np.integer):
            raise TypeError("Parent artifact control_count must be integer")
        if counts.shape != (self.prior_bank._artifact_parent_count,):
            raise ValueError("Parent artifact control_count has the wrong shape")
        if (counts < 0).any():
            raise ValueError("Parent artifact control_count cannot be negative")

        alignment = self.prior_bank.runtime_cellline_to_artifact
        if (alignment < 0).any():
            raise ValueError(
                "donor-aware Parent requires every runtime donor/cell type pair"
            )
        runtime_counts = torch.from_numpy(counts.astype(np.float32)).index_select(
            0, alignment.cpu()
        )
        donors = int(self.prior_bank._runtime_batch_count)
        celltypes = int(self.prior_bank._runtime_composite_celltype_count)
        runtime_controls = self.prior_bank.control_mean.detach().cpu().index_select(
            0, alignment.cpu()
        )
        runtime_counts = runtime_counts.reshape(donors, celltypes)
        runtime_controls = runtime_controls.reshape(
            donors, celltypes, self.gene_dim
        )
        denominators = runtime_counts.sum(dim=0)
        if (denominators <= 0).any():
            raise ValueError("every cell type needs at least one PBS control")
        reference = (
            runtime_controls * runtime_counts[:, :, None]
        ).sum(dim=0) / denominators[:, None]
        self.register_buffer(
            "donor_aware_celltype_control_reference",
            reference.contiguous(),
            persistent=True,
        )

    def attach_matched_control_summary(
        self,
        context,
        *,
        control: torch.Tensor,
        cov_celltype: torch.Tensor,
        cov_batch: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        mean, count = matched_control_group_summary(
            control,
            cov_celltype,
            cov_batch,
            num_celltypes=self._donor_aware_num_celltypes,
            valid_mask=valid_mask,
        )
        context.donor_aware_matched_control_mean = mean
        context.donor_aware_matched_control_count = count

    def predict_parent_mean(
        self, context, cov_celltype, cov_pert, cov_batch=None
    ) -> DonorAwareParentMeanPrediction:
        if cov_batch is None:
            raise ValueError("donor-aware Parent prediction requires cov_batch")
        if not hasattr(context, "donor_aware_matched_control_mean") or not hasattr(
            context, "donor_aware_matched_control_count"
        ):
            raise ValueError(
                "donor-aware Parent prediction requires an explicitly attached "
                "matched PBS control summary"
            )
        base = super().predict_parent_mean(
            context, cov_celltype, cov_pert, cov_batch=cov_batch
        )
        celltype = self.prior_bank._condition_ids(
            cov_celltype, "cov_celltype"
        )
        reference = self.donor_aware_celltype_control_reference.index_select(
            0, celltype.to(
                device=self.donor_aware_celltype_control_reference.device,
                dtype=torch.long,
            )
        ).to(base.parent_mean)
        control_mean = base.lookup.control_mean.to(base.parent_mean)
        base_delta = base.parent_mean - control_mean
        output = self.donor_aware_parent_delta_head(
            base_delta=base_delta,
            base_raw_correction=base.raw_correction,
            base_bounded_correction=base.correction,
            matched_control_mean=context.donor_aware_matched_control_mean.to(
                base.parent_mean
            ),
            reference_control_mean=reference,
            perturbation_context=context.perturbation_query,
            gene_identity=self.gene_dit.main_tokenizer.gene_identity,
            control_count=context.donor_aware_matched_control_count,
        )
        payload = {
            field.name: getattr(base, field.name)
            for field in fields(ParentMeanPrediction)
        }
        payload.update(
            raw_correction=output.combined_raw_correction,
            correction=output.total_correction,
            parent_mean=control_mean + output.predicted_delta,
            effective_delta=output.predicted_delta,
            donor_aware_correction_delta=output.correction_delta,
            donor_aware_raw_adjustment=output.raw_adjustment,
            donor_aware_reliability=output.reliability,
            donor_aware_control_state_rms=output.donor_state_rms,
            donor_aware_proposal_rms=output.proposal_rms,
            donor_aware_realized_rms=output.realized_rms,
            donor_aware_response_strength_gate_probability=(
                output.response_strength_gate_probability
            ),
            donor_aware_response_strength_gate_logit=(
                output.response_strength_gate_logit
            ),
        )
        return DonorAwareParentMeanPrediction(**payload)

    def initialize_weights(self):
        super().initialize_weights()
        self.donor_aware_parent_delta_head.reset_output_projection()


__all__ = [
    "DonorAwareParentMeanPrediction",
    "DonorAwareParentResidualGeneDiTModel",
    "matched_control_group_summary",
]

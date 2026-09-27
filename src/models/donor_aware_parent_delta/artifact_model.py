"""Artifact-stable donor-aware integration for Parent-Residual Gene-DiT."""

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
class ArtifactDonorAwareParentMeanPrediction(ParentMeanPrediction):
    """Parent prediction plus diagnostics for the isolated donor arm."""

    donor_aware_correction_delta: torch.Tensor | None = None
    donor_aware_raw_adjustment: torch.Tensor | None = None
    donor_aware_reliability: torch.Tensor | None = None
    donor_aware_control_state_rms: torch.Tensor | None = None
    donor_aware_proposal_rms: torch.Tensor | None = None
    donor_aware_realized_rms: torch.Tensor | None = None
    donor_aware_response_strength_gate_probability: torch.Tensor | None = None
    donor_aware_response_strength_gate_logit: torch.Tensor | None = None


def _load_control_counts(
    path: Path, expected_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load the true source support, including audited fallback controls."""

    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if archive[key].dtype == object:
                raise TypeError(f"unsafe object dtype in Parent artifact key {key!r}")
        if "control_count" not in archive.files:
            raise ValueError("donor-aware Parent artifact is missing control_count")
        raw = np.asarray(archive["control_count"])
        if "control_mean_source_count" in archive.files:
            effective = np.asarray(archive["control_mean_source_count"])
        else:
            effective = raw
        fallback = (
            np.asarray(archive["control_mean_fallback_mask"], dtype=bool)
            if "control_mean_fallback_mask" in archive.files
            else np.zeros(raw.shape, dtype=bool)
        )
    for name, value in (("control_count", raw), ("effective count", effective)):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"Parent artifact {name} must be integer")
        if value.shape != (int(expected_rows),):
            raise ValueError(f"Parent artifact {name} has the wrong shape")
        if np.issubdtype(value.dtype, np.signedinteger) and (value < 0).any():
            raise ValueError(f"Parent artifact {name} cannot be negative")
    if fallback.shape != raw.shape:
        raise ValueError("control_mean_fallback_mask has the wrong shape")
    if (fallback & (effective <= 0)).any():
        raise ValueError("fallback control means need positive source support")
    if ((~fallback) & (effective != raw)).any():
        raise ValueError(
            "control_mean_source_count may differ from control_count only on "
            "explicit fallback rows"
        )
    return (
        raw.astype(np.float32, copy=True),
        effective.astype(np.float32, copy=True),
    )


class ArtifactDonorAwareParentResidualGeneDiTModel(
    ParentResidualGeneDiTModel
):
    """Add one low-rank PBS donor-state x perturbation Parent correction."""

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
        config = DonorAwareParentDeltaConfig(
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
            self.donor_aware_parent_delta_head = DonorAwareParentDeltaHead(
                config
            )
        self.donor_aware_parent_delta_enabled = True
        self._register_artifact_control_state()

    def _register_artifact_control_state(self) -> None:
        bank = self.prior_bank
        direct_count, effective_count = _load_control_counts(
            Path(bank.artifact_path), bank._artifact_parent_count
        )
        alignment = bank.runtime_cellline_to_artifact.detach().cpu()
        if (alignment < 0).any():
            raise ValueError(
                "donor-aware Parent requires every runtime donor/cell type pair"
            )
        donors = int(bank._runtime_batch_count)
        celltypes = int(bank._runtime_composite_celltype_count)
        expected = donors * celltypes
        if alignment.shape != (expected,):
            raise ValueError("runtime donor/cell-type alignment has wrong shape")

        artifact_count = torch.from_numpy(effective_count)
        direct_artifact_count = torch.from_numpy(direct_count)
        runtime_count = direct_artifact_count.index_select(0, alignment)
        runtime_control = bank.control_mean.detach().cpu().index_select(
            0, alignment
        )
        runtime_count_grid = runtime_count.reshape(donors, celltypes)
        runtime_control_grid = runtime_control.reshape(
            donors, celltypes, self.gene_dim
        )
        celltype_count = runtime_count_grid.sum(dim=0)
        if (celltype_count <= 0).any():
            raise ValueError("every cell type needs positive PBS source support")
        reference = (
            runtime_control_grid * runtime_count_grid[:, :, None]
        ).sum(dim=0) / celltype_count[:, None]

        self.register_buffer(
            "donor_aware_artifact_control_count",
            artifact_count.contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "donor_aware_celltype_control_reference",
            reference.contiguous(),
            persistent=True,
        )

    def predict_parent_mean(
        self, context, cov_celltype, cov_pert, cov_batch=None
    ) -> ArtifactDonorAwareParentMeanPrediction:
        if cov_batch is None:
            raise ValueError("donor-aware Parent prediction requires cov_batch")
        base = super().predict_parent_mean(
            context, cov_celltype, cov_pert, cov_batch=cov_batch
        )
        celltype = self.prior_bank._condition_ids(
            cov_celltype, "cov_celltype"
        )
        reference = self.donor_aware_celltype_control_reference.index_select(
            0,
            celltype.to(
                device=self.donor_aware_celltype_control_reference.device,
                dtype=torch.long,
            ),
        ).to(base.parent_mean)
        control_mean = base.lookup.control_mean.to(base.parent_mean)
        support = self.donor_aware_artifact_control_count.index_select(
            0,
            base.lookup.artifact_cellline_ids.to(
                device=self.donor_aware_artifact_control_count.device,
                dtype=torch.long,
            ),
        )
        output = self.donor_aware_parent_delta_head(
            base_delta=base.parent_mean - control_mean,
            base_raw_correction=base.raw_correction,
            base_bounded_correction=base.correction,
            matched_control_mean=control_mean,
            reference_control_mean=reference,
            perturbation_context=context.perturbation_query,
            gene_identity=self.gene_dit.main_tokenizer.gene_identity,
            control_count=support,
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
        return ArtifactDonorAwareParentMeanPrediction(**payload)

    def initialize_weights(self):
        super().initialize_weights()
        self.donor_aware_parent_delta_head.reset_output_projection()


__all__ = [
    "ArtifactDonorAwareParentMeanPrediction",
    "ArtifactDonorAwareParentResidualGeneDiTModel",
]

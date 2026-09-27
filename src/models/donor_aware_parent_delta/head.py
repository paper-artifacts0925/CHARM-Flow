"""Low-rank continuous-donor interaction for a bounded Parent correction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.models.parent_residual_gene_dit.child_guidance import rms_trust_clip
from src.models.parent_residual_gene_dit.tied_gene_correction import (
    TiedGeneParentCorrection,
)

from .config import DonorAwareParentDeltaConfig


@dataclass(frozen=True)
class DonorAwareParentDeltaOutput:
    """Auditable output of the donor-aware bounded correction composition."""

    predicted_delta: torch.Tensor
    total_correction: torch.Tensor
    correction_delta: torch.Tensor
    raw_adjustment: torch.Tensor
    combined_raw_correction: torch.Tensor
    reliability: torch.Tensor
    donor_state_rms: torch.Tensor
    proposal_rms: torch.Tensor
    realized_rms: torch.Tensor
    response_strength_gate_probability: torch.Tensor | None
    response_strength_gate_logit: torch.Tensor | None


def _matrix(
    name: str,
    value: torch.Tensor,
    *,
    rows: int,
    columns: int,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tuple(value.shape) != (rows, columns):
        raise ValueError(f"{name} must have shape [{rows},{columns}]")
    if value.device != device:
        raise ValueError(f"{name} must share the base_delta device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


class _ResponseStrengthGate(nn.Module):
    """Condition-level target-free scalar gate for donor-aware strength."""

    input_dim = 4

    def __init__(self, *, hidden_dim: int, initial_probability: float) -> None:
        super().__init__()
        self.initial_probability = float(initial_probability)
        self.input_projection = nn.Linear(self.input_dim, int(hidden_dim))
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(int(hidden_dim), 1)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        initial_logit = math.log(
            self.initial_probability / (1.0 - self.initial_probability)
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, initial_logit)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("response-strength features must have shape [B,4]")
        if not torch.isfinite(features).all():
            raise ValueError("response-strength features must be finite")
        hidden = self.activation(self.input_projection(features))
        return self.output_projection(hidden).to(dtype=torch.float32)


class DonorAwareParentDeltaHead(nn.Module):
    """Predict only the donor-by-perturbation departure from an existing Parent.

    The donor signal is a continuous PBS-control offset, never a donor ID.  A
    Hadamard interaction makes the new arm vanish when either donor state or
    perturbation context carries no signal.  The tied gene decoder reuses the
    Gene-DiT response-gene identities and is zero-output initialized.

    The final correction is a convex interpolation between the historical
    bounded correction and a new bounded candidate.  It therefore retains the
    existing ``+-correction_scale`` invariant required by HS3 signature losses.
    """

    def __init__(self, config: DonorAwareParentDeltaConfig) -> None:
        super().__init__()
        self.config = config.validate()
        cfg = self.config
        self.donor_encoder = nn.Sequential(
            nn.LayerNorm(
                cfg.gene_dim,
                eps=cfg.norm_eps,
                elementwise_affine=False,
            ),
            nn.Linear(cfg.gene_dim, cfg.hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(
                cfg.hidden_dim,
                eps=cfg.norm_eps,
                elementwise_affine=False,
            ),
        )
        self.perturbation_encoder = nn.Sequential(
            nn.LayerNorm(
                cfg.perturbation_dim,
                eps=cfg.norm_eps,
                elementwise_affine=False,
            ),
            nn.Linear(cfg.perturbation_dim, cfg.hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(
                cfg.hidden_dim,
                eps=cfg.norm_eps,
                elementwise_affine=False,
            ),
        )
        self.interaction_norm = nn.LayerNorm(
            cfg.hidden_dim,
            eps=cfg.norm_eps,
            elementwise_affine=False,
        )
        self.gene_decoder = TiedGeneParentCorrection(
            hidden_dim=cfg.hidden_dim,
            gene_identity_dim=cfg.gene_identity_dim,
            rank=cfg.gene_rank,
            detach_gene_identity=True,
            norm_eps=cfg.norm_eps,
        )
        self.response_strength_gate = (
            _ResponseStrengthGate(
                hidden_dim=cfg.response_strength_gate_hidden_dim,
                initial_probability=cfg.response_strength_gate_initial_probability,
            )
            if cfg.response_strength_gate_enabled
            else None
        )

    def reset_output_projection(self) -> None:
        """Restore exact historical Parent values and first-order gradients."""

        self.gene_decoder.reset_output_projection()
        if self.response_strength_gate is not None:
            self.response_strength_gate.reset_output_projection()

    def _reliability(
        self,
        control_count: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(control_count):
            raise TypeError("control_count must be a torch.Tensor")
        if tuple(control_count.shape) == (batch_size,):
            count = control_count[:, None]
        elif tuple(control_count.shape) == (batch_size, 1):
            count = control_count
        else:
            raise ValueError("control_count must have shape [B] or [B,1]")
        if count.device != reference.device:
            raise ValueError("control_count must share the base_delta device")
        count = count.detach().to(dtype=torch.float32)
        if not torch.isfinite(count).all() or (count < 0).any():
            raise ValueError("control_count must be finite and non-negative")
        reliability = count / (count + float(self.config.reliability_tau))
        supported = count >= int(self.config.minimum_control_cells)
        return torch.where(supported, reliability, torch.zeros_like(reliability)).to(
            reference
        )

    def forward(
        self,
        *,
        base_delta: torch.Tensor,
        base_raw_correction: torch.Tensor,
        base_bounded_correction: torch.Tensor,
        matched_control_mean: torch.Tensor,
        reference_control_mean: torch.Tensor,
        perturbation_context: torch.Tensor,
        gene_identity: torch.Tensor,
        control_count: torch.Tensor,
    ) -> DonorAwareParentDeltaOutput:
        cfg = self.config
        if not torch.is_tensor(base_delta) or not base_delta.is_floating_point():
            raise TypeError("base_delta must be a floating-point tensor")
        if base_delta.ndim != 2 or base_delta.shape[1] != cfg.gene_dim:
            raise ValueError(f"base_delta must have shape [B,{cfg.gene_dim}]")
        if not torch.isfinite(base_delta).all():
            raise ValueError("base_delta must be finite")
        batch_size = int(base_delta.shape[0])
        device = base_delta.device
        raw = _matrix(
            "base_raw_correction",
            base_raw_correction,
            rows=batch_size,
            columns=cfg.gene_dim,
            device=device,
        )
        bounded = _matrix(
            "base_bounded_correction",
            base_bounded_correction,
            rows=batch_size,
            columns=cfg.gene_dim,
            device=device,
        )
        matched = _matrix(
            "matched_control_mean",
            matched_control_mean,
            rows=batch_size,
            columns=cfg.gene_dim,
            device=device,
        )
        reference = _matrix(
            "reference_control_mean",
            reference_control_mean,
            rows=batch_size,
            columns=cfg.gene_dim,
            device=device,
        )
        perturbation = _matrix(
            "perturbation_context",
            perturbation_context,
            rows=batch_size,
            columns=cfg.perturbation_dim,
            device=device,
        )
        identity = _matrix(
            "gene_identity",
            gene_identity,
            rows=cfg.gene_dim,
            columns=cfg.gene_identity_dim,
            device=device,
        )
        reliability = self._reliability(
            control_count,
            batch_size=batch_size,
            reference=base_delta,
        )

        donor_state = matched - reference
        donor_input = donor_state.detach() if cfg.detach_control_summary else donor_state
        perturbation_input = (
            perturbation.detach()
            if cfg.detach_perturbation_context
            else perturbation
        )
        donor_hidden = self.donor_encoder(donor_input)
        perturbation_hidden = self.perturbation_encoder(perturbation_input)
        interaction = self.interaction_norm(donor_hidden * perturbation_hidden)
        raw_adjustment = self.gene_decoder(interaction, identity).to(raw)

        scale = float(cfg.correction_scale)
        combined_raw = raw + raw_adjustment
        if not torch.isfinite(combined_raw).all():
            raise RuntimeError("donor-aware raw correction must remain finite")
        candidate = scale * torch.tanh(combined_raw)

        # Under BF16, a configured decimal bound such as 0.05 is represented
        # by the nearby value 0.050048828125.  The historical Parent computes
        # its endpoint in ``raw.dtype`` and may then store it in FP32.  Compare
        # both endpoints against that dtype-realized limit, rather than against
        # the Python decimal plus an unrelated FP32 epsilon.
        endpoint_dtype = torch.promote_types(bounded.dtype, candidate.dtype)
        realized_limit = torch.as_tensor(
            scale,
            dtype=raw.dtype,
            device=device,
        ).abs().to(dtype=endpoint_dtype)
        bounded_endpoint = bounded.detach().to(dtype=endpoint_dtype)
        candidate_endpoint = candidate.detach().to(dtype=endpoint_dtype)
        if bool((bounded_endpoint.abs() > realized_limit).any()):
            raise RuntimeError(
                "historical Parent correction endpoint exceeds its "
                "dtype-realized bound"
            )
        if bool((candidate_endpoint.abs() > realized_limit).any()):
            raise RuntimeError(
                "donor-aware candidate endpoint exceeds its "
                "dtype-realized bound"
            )

        proposal = candidate - bounded
        trusted = rms_trust_clip(
            proposal,
            base_delta,
            float(cfg.max_delta_rms_ratio),
        )
        response_strength_gate_probability = None
        response_strength_gate_logit = None
        if self.response_strength_gate is None:
            # Keep the historical arithmetic expression untouched when the
            # optional gate is disabled.  This is the old checkpoint path.
            correction_delta = reliability * trusted
        else:
            accumulation_dtype = (
                torch.float32
                if base_delta.dtype in (torch.float16, torch.bfloat16)
                else base_delta.dtype
            )

            def detached_rms(value: torch.Tensor) -> torch.Tensor:
                return (
                    value.detach()
                    .to(dtype=accumulation_dtype)
                    .square()
                    .mean(dim=-1, keepdim=True)
                    .sqrt()
                )

            base_delta_rms = detached_rms(base_delta)
            donor_state_rms = detached_rms(donor_state)
            proposal_rms = detached_rms(proposal)
            proposal_base_rms_ratio = proposal_rms / base_delta_rms.clamp_min(
                float(cfg.norm_eps)
            )
            gate_features = torch.cat(
                (
                    base_delta_rms,
                    donor_state_rms,
                    proposal_base_rms_ratio,
                    reliability.detach().to(dtype=accumulation_dtype),
                ),
                dim=-1,
            )
            gate_parameter = self.response_strength_gate.input_projection.weight
            response_strength_gate_logit = self.response_strength_gate(
                gate_features.to(
                    device=gate_parameter.device,
                    dtype=gate_parameter.dtype,
                )
            )
            response_strength_gate_probability = torch.sigmoid(
                response_strength_gate_logit
            )
            correction_delta = (
                reliability
                * response_strength_gate_probability.to(trusted)
                * trusted
            )
        total_correction = bounded + correction_delta

        # Reliability and the trust scale both lie in [0,1], so the result is
        # a convex interpolation of the two checked endpoints.  Repair only a
        # strict floating-point escape from that interval.  Detached masks keep
        # equality on the original path, preserving the zero-initialized
        # head's first-order gradient and bitwise historical Parent values.
        bounded_hull = bounded.to(dtype=total_correction.dtype)
        candidate_hull = candidate.to(dtype=total_correction.dtype)
        lower = torch.minimum(bounded_hull, candidate_hull)
        upper = torch.maximum(bounded_hull, candidate_hull)
        below = total_correction.detach() < lower.detach()
        above = total_correction.detach() > upper.detach()
        escaped = below | above
        total_correction = torch.where(
            below,
            lower,
            torch.where(above, upper, total_correction),
        )
        if bool(escaped.any()):
            repaired_delta = total_correction - bounded_hull
            correction_delta = torch.where(
                escaped,
                repaired_delta,
                correction_delta,
            )
        predicted_delta = base_delta + correction_delta

        accumulation_dtype = (
            torch.float32
            if base_delta.dtype in (torch.float16, torch.bfloat16)
            else base_delta.dtype
        )

        def rms(value: torch.Tensor) -> torch.Tensor:
            return value.detach().to(accumulation_dtype).square().mean(
                dim=-1, keepdim=True
            ).sqrt()

        return DonorAwareParentDeltaOutput(
            predicted_delta=predicted_delta,
            total_correction=total_correction,
            correction_delta=correction_delta,
            raw_adjustment=raw_adjustment,
            combined_raw_correction=combined_raw,
            reliability=reliability,
            donor_state_rms=rms(donor_state),
            proposal_rms=rms(proposal),
            realized_rms=rms(correction_delta),
            response_strength_gate_probability=(
                response_strength_gate_probability
            ),
            response_strength_gate_logit=response_strength_gate_logit,
        )


__all__ = ["DonorAwareParentDeltaHead", "DonorAwareParentDeltaOutput"]

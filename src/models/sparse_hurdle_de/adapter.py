"""Opt-in adapter over frozen Parent means and condition features."""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional

import torch
import torch.nn as nn

from .config import SparseHurdleDEConfig
from .head import ConditionSparseHurdleHead
from .output import (
    SparseHurdleDEAdapterOutput,
    SparseHurdleGeneProgramOutput,
)


def _finite_unit_alpha(alpha: float | torch.Tensor) -> float:
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha_mu tensor must be scalar")
        alpha = float(alpha.detach().cpu())
    elif isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha_mu must be a real scalar")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha_mu must lie in [0, 1]")
    return value


def _finite_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_input(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.device != device:
        raise ValueError(f"{name} must share the condition device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


class SparseHurdleDEAdapter(nn.Module):
    """Learn sparse corrections without changing the frozen base by default.

    ``predict_gene_program`` is the G10 single-evaluation boundary.  It creates
    one support probability and one signed slab from the ungated legacy Parent
    delta.  Passing that program back to ``forward`` makes DE-BCE, count,
    zero-rate and terminal-H fields reuse the exact same tensors used by Parent
    regression.  Calling ``forward`` without a program preserves the historical
    standalone adapter semantics.
    """

    def __init__(self, config: Optional[SparseHurdleDEConfig] = None):
        super().__init__()
        self.config = (
            SparseHurdleDEConfig() if config is None else config
        ).validate()
        self.head = ConditionSparseHurdleHead(self.config)

    def _validate_condition_and_base(
        self,
        condition: torch.Tensor,
        base_parent_delta: torch.Tensor,
        control_mean: torch.Tensor,
    ) -> tuple[int, tuple[int, int]]:
        cfg = self.config
        if not torch.is_tensor(condition) or not condition.is_floating_point():
            raise TypeError("condition must be a floating-point tensor")
        if condition.ndim != 2 or condition.shape[1] != cfg.condition_dim:
            raise ValueError(
                f"condition must have shape [B,{cfg.condition_dim}]"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition must be finite")
        batch_size = condition.shape[0]
        expected = (batch_size, cfg.gene_dim)
        for name, value in (
            ("base_parent_delta", base_parent_delta),
            ("control_mean", control_mean),
        ):
            _validate_input(name, value, expected, device=condition.device)
        return batch_size, expected

    def _detach_choice(self, detach_base_features: Optional[bool]) -> bool:
        if detach_base_features is None:
            return self.config.detach_base_features
        if isinstance(detach_base_features, bool):
            return detach_base_features
        raise TypeError("detach_base_features must be boolean")

    def predict_gene_program(
        self,
        condition: torch.Tensor,
        base_parent_delta: torch.Tensor,
        control_mean: torch.Tensor,
        *,
        response_gene_identity: Optional[torch.Tensor] = None,
        detach_base_features: Optional[bool] = None,
        support_logit_offset: float = 0.0,
        support_temperature: float = 1.0,
        slab_scale: float = 1.0,
    ) -> SparseHurdleGeneProgramOutput:
        """Evaluate the shared condition-to-response-gene program exactly once."""

        self._validate_condition_and_base(
            condition, base_parent_delta, control_mean
        )
        detach = self._detach_choice(detach_base_features)
        logit_offset = _finite_scalar(
            "support_logit_offset", support_logit_offset
        )
        temperature = _finite_positive(
            "support_temperature", support_temperature
        )
        scale = _finite_positive("slab_scale", slab_scale)

        condition_input = condition.detach() if detach else condition
        delta_input = (
            base_parent_delta.detach() if detach else base_parent_delta
        )
        control_input = control_mean.detach() if detach else control_mean
        work_dtype = (
            torch.float64
            if any(
                value.dtype == torch.float64
                for value in (delta_input, control_input)
            )
            else torch.float32
        )
        delta = delta_input.to(work_dtype)
        control = control_input.to(work_dtype)
        head_output = self.head(
            condition_input,
            response_gene_identity=response_gene_identity,
        )
        treated = torch.clamp_min(control + delta, 0.0)
        control_nonnegative = torch.clamp_min(control, 0.0)
        base_effect_size = (
            torch.log1p(treated) - torch.log1p(control_nonnegative)
        ).abs()
        base_support_logits = (
            base_effect_size - float(self.config.support_center)
        ) / float(self.config.support_temperature)
        shared_support_logits = (
            base_support_logits
            + head_output.ranking_residual_logits.to(work_dtype)
            + head_output.condition_budget_shift.to(work_dtype)
            + logit_offset
        ) / temperature
        shared_support_probability = torch.sigmoid(shared_support_logits)
        # C1 detaches only pi. Its signed magnitude must still train the
        # Parent correction, while DE/support losses consume the detached
        # ``base_parent_delta`` above and cannot leak through this slab.
        signed_slab = scale * torch.tanh(
            base_parent_delta.to(work_dtype) / scale
        )
        for name, value in (
            ("base_effect_size", base_effect_size),
            ("base_support_logits", base_support_logits),
            ("shared_support_logits", shared_support_logits),
            ("shared_support_probability", shared_support_probability),
            ("signed_slab", signed_slab),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"gene program produced non-finite {name}"
                )
        return SparseHurdleGeneProgramOutput(
            head=head_output,
            base_parent_delta=delta,
            base_effect_size=base_effect_size,
            base_support_logits=base_support_logits,
            shared_support_logits=shared_support_logits,
            shared_support_probability=shared_support_probability,
            signed_slab=signed_slab,
        )

    def forward(
        self,
        condition: torch.Tensor,
        base_parent_delta: torch.Tensor,
        control_mean: torch.Tensor,
        control_zero_rate: torch.Tensor,
        *,
        alpha_mu: float | torch.Tensor = 0.0,
        detach_base_features: Optional[bool] = None,
        response_gene_identity: Optional[torch.Tensor] = None,
        program: Optional[SparseHurdleGeneProgramOutput] = None,
    ) -> SparseHurdleDEAdapterOutput:
        cfg = self.config
        _, expected = self._validate_condition_and_base(
            condition, base_parent_delta, control_mean
        )
        _validate_input(
            "control_zero_rate",
            control_zero_rate,
            expected,
            device=condition.device,
        )
        if ((control_zero_rate < 0) | (control_zero_rate > 1)).any():
            raise ValueError("control_zero_rate must lie in [0, 1]")
        detach = self._detach_choice(detach_base_features)
        alpha_value = _finite_unit_alpha(alpha_mu)

        condition_input = condition.detach() if detach else condition
        delta_input = (
            base_parent_delta.detach() if detach else base_parent_delta
        )
        control_input = control_mean.detach() if detach else control_mean
        zero_input = (
            control_zero_rate.detach() if detach else control_zero_rate
        )
        work_dtype = (
            torch.float64
            if any(
                value.dtype == torch.float64
                for value in (delta_input, control_input, zero_input)
            )
            else torch.float32
        )
        control = control_input.to(work_dtype)
        q_control = zero_input.to(work_dtype)
        calibration_offset = math.log(
            float(cfg.ranking_false_positive_weight)
        )

        if program is None:
            delta = delta_input.to(work_dtype)
            head_output = self.head(
                condition_input,
                response_gene_identity=response_gene_identity,
            )
            treated = torch.clamp_min(control + delta, 0.0)
            control_nonnegative = torch.clamp_min(control, 0.0)
            base_effect_size = (
                torch.log1p(treated) - torch.log1p(control_nonnegative)
            ).abs()
            base_support_logits = (
                base_effect_size - float(cfg.support_center)
            ) / float(cfg.support_temperature)
            ranking_support_logits = (
                base_support_logits
                + head_output.ranking_residual_logits.to(work_dtype)
            )
            ranking_probability = torch.sigmoid(ranking_support_logits)
            calibrated_support_logits = (
                ranking_support_logits.detach()
                + calibration_offset
                + head_output.condition_budget_shift.to(work_dtype)
            )
            calibrated_support_probability = torch.sigmoid(
                calibrated_support_logits
            )
        else:
            if not isinstance(program, SparseHurdleGeneProgramOutput):
                raise TypeError(
                    "program must be a SparseHurdleGeneProgramOutput"
                )
            delta = program.base_parent_delta
            expected_delta = delta_input.to(delta)
            if (
                tuple(delta.shape) != expected
                or delta.device != condition.device
                or not torch.equal(delta, expected_delta)
            ):
                raise ValueError(
                    "cached gene program does not match base_parent_delta"
                )
            for name, value in (
                ("base_effect_size", program.base_effect_size),
                ("base_support_logits", program.base_support_logits),
                ("shared_support_logits", program.shared_support_logits),
                (
                    "shared_support_probability",
                    program.shared_support_probability,
                ),
                ("signed_slab", program.signed_slab),
            ):
                if (
                    tuple(value.shape) != expected
                    or value.device != condition.device
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        f"cached gene program {name} must be finite [B,G]"
                    )
            head_output = program.head
            base_effect_size = program.base_effect_size
            base_support_logits = program.base_support_logits
            # Object identity here is deliberate: BCE, count, zero/H and
            # Parent regression all consume this one cached score/probability.
            ranking_support_logits = program.shared_support_logits
            ranking_probability = program.shared_support_probability
            calibrated_support_logits = program.shared_support_logits
            calibrated_support_probability = (
                program.shared_support_probability
            )

        eps = float(cfg.probability_eps)
        safe_q_control = q_control.clamp(min=eps, max=1.0 - eps)
        control_zero_logits = torch.log(safe_q_control) - torch.log1p(
            -safe_q_control
        )
        bounded_zero_shift = float(cfg.zero_rate_max_shift) * torch.tanh(
            head_output.zero_rate_shift.to(work_dtype)
        )
        predicted_zero_logits = (
            control_zero_logits
            + calibrated_support_probability * bounded_zero_shift
        )
        predicted_zero_rate = torch.sigmoid(predicted_zero_logits)

        if alpha_value == 0.0:
            adjusted_delta = delta
            mean_shrinkage = torch.zeros_like(delta)
        else:
            scale = float(cfg.mean_shrink_scale)
            magnitude = delta.abs()
            shrink_magnitude = (
                alpha_value
                * (1.0 - calibrated_support_probability)
                * scale
                * torch.tanh(magnitude / scale)
            )
            remaining_magnitude = (
                magnitude - shrink_magnitude
            ).clamp_min(0.0)
            adjusted_delta = torch.sign(delta) * remaining_magnitude
            mean_shrinkage = delta - adjusted_delta

        for name, value in (
            ("base_effect_size", base_effect_size),
            ("ranking_support_logits", ranking_support_logits),
            ("calibrated_support_logits", calibrated_support_logits),
            ("predicted_zero_rate", predicted_zero_rate),
            ("adjusted_parent_delta", adjusted_delta),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"adapter produced non-finite {name}")
        return SparseHurdleDEAdapterOutput(
            head=head_output,
            base_effect_size=base_effect_size,
            base_support_logits=base_support_logits,
            ranking_support_logits=ranking_support_logits,
            ranking_probability=ranking_probability,
            calibrated_support_logits=calibrated_support_logits,
            calibrated_support_probability=calibrated_support_probability,
            ranking_logit_calibration_offset=delta.new_tensor(
                0.0 if program is not None else calibration_offset
            ),
            control_zero_logits=control_zero_logits,
            predicted_zero_logits=predicted_zero_logits,
            predicted_zero_rate=predicted_zero_rate,
            base_parent_delta=delta,
            adjusted_parent_delta=adjusted_delta,
            mean_shrinkage=mean_shrinkage,
            alpha_mu=delta.new_tensor(alpha_value),
            uses_unified_gene_program=program is not None,
        )


__all__ = ["SparseHurdleDEAdapter"]

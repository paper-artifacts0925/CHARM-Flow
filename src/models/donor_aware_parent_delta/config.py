"""Configuration for the opt-in donor-aware Parent delta head."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _open_unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return parsed


@dataclass(frozen=True)
class DonorAwareParentDeltaConfig:
    """Dimensions and safety limits for a continuous donor-response head.

    ``perturbation_dim`` is the existing target-free perturbation-query width;
    no categorical donor ID is accepted by this component.
    """

    gene_dim: int = 2000
    perturbation_dim: int = 256
    hidden_dim: int = 128
    gene_identity_dim: int = 384
    gene_rank: int = 64
    minimum_control_cells: int = 20
    reliability_tau: float = 64.0
    max_delta_rms_ratio: float = 0.75
    correction_scale: float = 0.05
    norm_eps: float = 1.0e-6
    detach_control_summary: bool = True
    detach_perturbation_context: bool = True
    response_strength_gate_enabled: bool = False
    response_strength_gate_hidden_dim: int = 16
    response_strength_gate_initial_probability: float = 0.999

    def validate(self) -> "DonorAwareParentDeltaConfig":
        for name in (
            "gene_dim",
            "perturbation_dim",
            "hidden_dim",
            "gene_identity_dim",
            "gene_rank",
            "minimum_control_cells",
            "response_strength_gate_hidden_dim",
        ):
            _positive_integer(name, getattr(self, name))
        for name in (
            "reliability_tau",
            "max_delta_rms_ratio",
            "correction_scale",
            "norm_eps",
        ):
            _finite_positive(name, getattr(self, name))
        _open_unit_interval(
            "response_strength_gate_initial_probability",
            self.response_strength_gate_initial_probability,
        )
        for name in (
            "detach_control_summary",
            "detach_perturbation_context",
            "response_strength_gate_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        return self


__all__ = ["DonorAwareParentDeltaConfig"]

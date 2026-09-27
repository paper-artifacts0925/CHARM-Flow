"""Configuration objects for posterior-routed set transport training."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PosteriorPriorMixConfig:
    """Anneal target-aware source assignment toward the inference prior.

    ``posterior_fraction`` is held at ``start_fraction`` through
    ``transition_start_step`` and reaches ``end_fraction`` at
    ``transition_end_step``.  A value of one makes every FM source use the
    Hungarian posterior; zero makes every source use the target-free Router.
    """

    start_fraction: float = 1.0
    end_fraction: float = 0.0
    transition_start_step: int = 500
    transition_end_step: int = 5000
    curve: str = "cosine"

    def validate(self) -> "PosteriorPriorMixConfig":
        for name, value in (
            ("start_fraction", self.start_fraction),
            ("end_fraction", self.end_fraction),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if int(self.transition_start_step) < 0:
            raise ValueError("transition_start_step must be non-negative")
        if int(self.transition_end_step) <= int(self.transition_start_step):
            raise ValueError(
                "transition_end_step must be greater than transition_start_step"
            )
        if self.curve not in {"linear", "cosine"}:
            raise ValueError("curve must be 'linear' or 'cosine'")
        return self


@dataclass(frozen=True)
class ResidualSetLossConfig:
    """Weights and numerical controls for an unordered residual-set loss."""

    energy_weight: float = 1.0
    log_variance_weight: float = 0.25
    correlation_weight: float = 0.05
    mean_weight: float = 0.0
    min_cells_per_group: int = 8
    max_cells_per_group: int = 256
    normalize_projection_distance: bool = True
    eps: float = 1e-6

    def validate(self) -> "ResidualSetLossConfig":
        for name, value in (
            ("energy_weight", self.energy_weight),
            ("log_variance_weight", self.log_variance_weight),
            ("correlation_weight", self.correlation_weight),
            ("mean_weight", self.mean_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.min_cells_per_group) < 2:
            raise ValueError("min_cells_per_group must be at least two")
        if int(self.max_cells_per_group) < int(self.min_cells_per_group):
            raise ValueError(
                "max_cells_per_group must be >= min_cells_per_group"
            )
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be finite and positive")
        return self


__all__ = ["PosteriorPriorMixConfig", "ResidualSetLossConfig"]

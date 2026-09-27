"""Configuration for train-only DE-aware residual objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class DEAwareResidualConfig:
    """Weights and numerical controls for the P2 objective family.

    Component losses are normalized to approximately order one before the
    small outer weights below are applied.  The entire auxiliary is then
    linearly warmed up and, optionally, capped using its exact gradient with
    respect to the locked-flow raw velocity tensor.
    """

    signed_wilcoxon_weight: float = 2.0e-4
    zero_rate_weight: float = 2.0e-4
    log_variance_weight: float = 1.0e-4
    quantile_weight: float = 1.0e-4
    rank_weight: float = 2.0e-5
    call_weight: float = 1.0e-4
    warmup_steps: int = 500
    min_cells_per_group: int = 8
    gene_chunk_size: int = 256
    wilcoxon_temperature_scale: float = 0.2
    wilcoxon_temperature_min: float = 3.0e-3
    wilcoxon_temperature_max: float = 3.0e-2
    zero_threshold: float = 0.0
    zero_temperature: float = 3.0e-3
    variance_eps: float = 1.0e-6
    quantile_scale_floor: float = 3.0e-3
    quantile_scale_ceiling: float = 1.0e-1
    quantile_levels: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)
    rank_temperature: float = 0.5
    rank_max_positive: int = 256
    rank_max_negative: int = 256
    call_z_threshold: float = 4.0
    call_temperature: float = 0.5
    call_false_positive_weight: float = 2.0
    gradient_cap_ratio: float = 0.3
    gradient_cap_eps: float = 1.0e-12

    def validate(self) -> "DEAwareResidualConfig":
        for name in (
            "signed_wilcoxon_weight",
            "zero_rate_weight",
            "log_variance_weight",
            "quantile_weight",
            "rank_weight",
            "call_weight",
            "gradient_cap_ratio",
        ):
            _nonnegative(name, getattr(self, name))
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(self.min_cells_per_group) < 2:
            raise ValueError("min_cells_per_group must be at least two")
        if int(self.gene_chunk_size) < 1:
            raise ValueError("gene_chunk_size must be positive")
        if int(self.rank_max_positive) < 1 or int(self.rank_max_negative) < 1:
            raise ValueError("rank hard-example caps must be positive")
        for name in (
            "wilcoxon_temperature_scale",
            "wilcoxon_temperature_min",
            "wilcoxon_temperature_max",
            "zero_temperature",
            "variance_eps",
            "quantile_scale_floor",
            "quantile_scale_ceiling",
            "rank_temperature",
            "call_temperature",
            "call_false_positive_weight",
            "gradient_cap_eps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.wilcoxon_temperature_max < self.wilcoxon_temperature_min:
            raise ValueError("wilcoxon temperature maximum must be >= minimum")
        if self.quantile_scale_ceiling < self.quantile_scale_floor:
            raise ValueError("quantile scale ceiling must be >= floor")
        if not math.isfinite(float(self.zero_threshold)):
            raise ValueError("zero_threshold must be finite")
        if not math.isfinite(float(self.call_z_threshold)):
            raise ValueError("call_z_threshold must be finite")
        levels = tuple(float(value) for value in self.quantile_levels)
        if not levels or any(not 0.0 <= value <= 1.0 for value in levels):
            raise ValueError("quantile_levels must be a non-empty subset of [0,1]")
        if tuple(sorted(set(levels))) != levels:
            raise ValueError("quantile_levels must be strictly increasing")
        return self


__all__ = ["DEAwareResidualConfig"]

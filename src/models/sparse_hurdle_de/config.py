"""Configuration for the standalone sparse hurdle/DE core."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SparseHurdleDEConfig:
    """Static dimensions and conservative defaults for the opt-in core.

    The module is deliberately independent from Parent-Residual Gene-DiT.  In
    particular, ``projection_alpha=0`` means callers can construct and train a
    head without changing the historical fixed-Parent projection.
    """

    condition_dim: int = 512
    gene_dim: int = 2000
    hidden_dim: int = 256
    dropout: float = 0.0
    norm_eps: float = 1.0e-6
    zero_rate_temperature: float = 1.0
    projection_alpha: float = 0.0
    support_center: float = 0.065
    support_temperature: float = 0.015
    zero_rate_max_shift: float = 4.0
    mean_shrink_scale: float = 1.0
    probability_eps: float = 1.0e-6
    detach_base_features: bool = True
    ranking_false_positive_weight: float = 4.0
    gene_relation_enabled: bool = False
    gene_relation_rank: int = 64
    gene_relation_identity_dim: int = 0

    def validate(self) -> "SparseHurdleDEConfig":
        for name, value in (
            ("condition_dim", self.condition_dim),
            ("gene_dim", self.gene_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(
            self.dropout
        ) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(float(self.norm_eps)) or float(self.norm_eps) <= 0.0:
            raise ValueError("norm_eps must be finite and positive")
        if (
            not math.isfinite(float(self.zero_rate_temperature))
            or float(self.zero_rate_temperature) <= 0.0
        ):
            raise ValueError("zero_rate_temperature must be finite and positive")
        if (
            not math.isfinite(float(self.projection_alpha))
            or not 0.0 <= float(self.projection_alpha) <= 1.0
        ):
            raise ValueError("projection_alpha must lie in [0, 1]")
        if not math.isfinite(float(self.support_center)) or float(
            self.support_center
        ) < 0.0:
            raise ValueError("support_center must be finite and nonnegative")
        if (
            not math.isfinite(float(self.support_temperature))
            or float(self.support_temperature) <= 0.0
        ):
            raise ValueError("support_temperature must be finite and positive")
        if (
            not math.isfinite(float(self.zero_rate_max_shift))
            or float(self.zero_rate_max_shift) < 0.0
        ):
            raise ValueError("zero_rate_max_shift must be finite and nonnegative")
        if (
            not math.isfinite(float(self.mean_shrink_scale))
            or float(self.mean_shrink_scale) <= 0.0
        ):
            raise ValueError("mean_shrink_scale must be finite and positive")
        if (
            not math.isfinite(float(self.probability_eps))
            or not 0.0 < float(self.probability_eps) < 0.5
        ):
            raise ValueError("probability_eps must lie in (0, 0.5)")
        if not isinstance(self.detach_base_features, bool):
            raise TypeError("detach_base_features must be boolean")
        if (
            not math.isfinite(float(self.ranking_false_positive_weight))
            or float(self.ranking_false_positive_weight) <= 0.0
        ):
            raise ValueError(
                "ranking_false_positive_weight must be finite and positive"
            )
        if not isinstance(self.gene_relation_enabled, bool):
            raise TypeError("gene_relation_enabled must be boolean")
        if isinstance(self.gene_relation_rank, bool) or int(
            self.gene_relation_rank
        ) < 1:
            raise ValueError("gene_relation_rank must be a positive integer")
        if isinstance(self.gene_relation_identity_dim, bool) or int(
            self.gene_relation_identity_dim
        ) < 0:
            raise ValueError(
                "gene_relation_identity_dim must be a non-negative integer"
            )
        if self.gene_relation_enabled and int(
            self.gene_relation_identity_dim
        ) < 1:
            raise ValueError(
                "gene_relation_identity_dim must be positive when the "
                "gene relation is enabled"
            )
        return self


__all__ = ["SparseHurdleDEConfig"]

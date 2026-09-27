"""Configuration for the leakage-safe few-shot Parent adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FewShotParentAdapterConfig:
    """Static analytic adapter settings.

    The production ranks and support sizes are deliberately constrained to the
    two/four values evaluated by the protocol.  Unit tests may opt into a
    smaller rank through ``allow_test_rank`` without changing production
    defaults.
    """

    rank: int = 32
    support_size: int = 32
    support_sizes: tuple[int, ...] = (8, 16, 32, 64)
    ridge_lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    parent_calibration_lambdas: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    )
    parent_calibration_default_lambda: float = 1e-2
    donor_ridge: float = 1e-2
    cv_folds: int = 4
    minimum_relative_cv_gain: float = 0.0
    cv_tolerance: float = 1e-12
    max_donors: int = 3
    seed: int = 42
    gate_maximum: float = 1.0
    heldout_deployment_line_id: int | None = None
    allow_test_rank: bool = False

    def validate(self) -> "FewShotParentAdapterConfig":
        rank = int(self.rank)
        if rank < 1:
            raise ValueError("rank must be positive")
        if not self.allow_test_rank and rank not in (32, 64):
            raise ValueError("production few-shot rank must be 32 or 64")
        sizes = tuple(int(value) for value in self.support_sizes)
        if sizes != (8, 16, 32, 64):
            raise ValueError("support_sizes must be exactly (8, 16, 32, 64)")
        if int(self.support_size) not in sizes:
            raise ValueError("support_size must be one of 8, 16, 32, or 64")
        lambdas = tuple(float(value) for value in self.ridge_lambdas)
        if not lambdas or any(not math.isfinite(value) or value < 0 for value in lambdas):
            raise ValueError("ridge_lambdas must be finite and non-negative")
        if len(set(lambdas)) != len(lambdas):
            raise ValueError("ridge_lambdas must be unique")
        calibration_lambdas = tuple(
            float(value) for value in self.parent_calibration_lambdas
        )
        if not calibration_lambdas or any(
            not math.isfinite(value) or value < 0
            for value in calibration_lambdas
        ):
            raise ValueError(
                "parent_calibration_lambdas must be finite and non-negative"
            )
        if len(set(calibration_lambdas)) != len(calibration_lambdas):
            raise ValueError("parent_calibration_lambdas must be unique")
        for name, value in (
            ("donor_ridge", self.donor_ridge),
            ("minimum_relative_cv_gain", self.minimum_relative_cv_gain),
            ("cv_tolerance", self.cv_tolerance),
            ("gate_maximum", self.gate_maximum),
            (
                "parent_calibration_default_lambda",
                self.parent_calibration_default_lambda,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be at least two")
        if int(self.max_donors) != 3:
            raise ValueError("the first protocol fixes max_donors=3")
        if (
            self.heldout_deployment_line_id is not None
            and int(self.heldout_deployment_line_id) < 0
        ):
            raise ValueError("heldout_deployment_line_id must be non-negative")
        return self


__all__ = ["FewShotParentAdapterConfig"]

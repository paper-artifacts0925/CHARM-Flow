"""Matched source routing and unordered residual-set objectives."""

from .diagnostics import (
    ConditionRoutingDiagnostics,
    condition_level_posterior_prior_kl,
    condition_level_posterior_prior_diagnostics,
)
from .config import PosteriorPriorMixConfig, ResidualSetLossConfig
from .losses import ResidualSetLoss, grouped_residual_set_loss
from .routing import (
    MixedChildSelection,
    posterior_fraction,
    posterior_prior_kl,
    select_mixed_child_indices,
)
from .sampling import sample_grouped_child_indices

__all__ = [
    "ConditionRoutingDiagnostics",
    "MixedChildSelection",
    "PosteriorPriorMixConfig",
    "condition_level_posterior_prior_kl",
    "ResidualSetLoss",
    "ResidualSetLossConfig",
    "condition_level_posterior_prior_diagnostics",
    "sample_grouped_child_indices",
    "grouped_residual_set_loss",
    "posterior_fraction",
    "posterior_prior_kl",
    "select_mixed_child_indices",
]

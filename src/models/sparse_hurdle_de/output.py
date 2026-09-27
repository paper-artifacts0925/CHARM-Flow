"""Typed outputs for the standalone sparse hurdle/DE core."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseHurdleHeadOutput:
    """Function-neutral residual predictions from one condition embedding.

    ``de_support_logits`` is the ranking exit. False-positive-aware DE
    supervision is intentionally allowed to make it non-calibrated. The
    separate scalar ``condition_budget_shift`` calibrates that ranking score
    for hurdle and mean paths without changing its gene ordering.
    """

    de_support_logits: torch.Tensor
    zero_rate_logits: torch.Tensor
    de_support_probability: torch.Tensor
    zero_rate_probability: torch.Tensor
    condition_budget_shift: torch.Tensor

    @property
    def support_residual_logits(self) -> torch.Tensor:
        """Legacy alias for the ranking residual logits."""

        return self.de_support_logits

    @property
    def ranking_residual_logits(self) -> torch.Tensor:
        return self.de_support_logits

    @property
    def ranking_residual_probability(self) -> torch.Tensor:
        return self.de_support_probability

    @property
    def zero_rate_shift(self) -> torch.Tensor:
        """Alias exposing the adapter semantics of ``zero_rate_logits``."""

        return self.zero_rate_logits


@dataclass(frozen=True)
class SparseHurdleGeneProgramOutput:
    """One cached condition-to-gene program shared by Parent, S and H.

    The support probability is computed exactly once from the ungated legacy
    Parent delta.  ``signed_slab`` is the bounded, signed magnitude proposed by
    that same program.  Callers must reuse these tensor objects rather than
    evaluating the head again on an already gated Parent.
    """

    head: SparseHurdleHeadOutput
    base_parent_delta: torch.Tensor
    base_effect_size: torch.Tensor
    base_support_logits: torch.Tensor
    shared_support_logits: torch.Tensor
    shared_support_probability: torch.Tensor
    signed_slab: torch.Tensor

    @property
    def pi(self) -> torch.Tensor:
        return self.shared_support_probability


@dataclass(frozen=True)
class SparseHurdleDEAdapterOutput:
    """Condition-level DE and zero-rate estimates from a frozen base model.

    Gene-shaped tensors use ``[B,G]`` and ``condition_budget_shift`` uses
    ``[B,1]``.  The adapter never invokes the hard-zero projector; its runtime
    hurdle continuation therefore remains an external, independently gated
    choice.
    """

    head: SparseHurdleHeadOutput
    base_effect_size: torch.Tensor
    base_support_logits: torch.Tensor
    ranking_support_logits: torch.Tensor
    ranking_probability: torch.Tensor
    calibrated_support_logits: torch.Tensor
    calibrated_support_probability: torch.Tensor
    ranking_logit_calibration_offset: torch.Tensor
    control_zero_logits: torch.Tensor
    predicted_zero_logits: torch.Tensor
    predicted_zero_rate: torch.Tensor
    base_parent_delta: torch.Tensor
    adjusted_parent_delta: torch.Tensor
    mean_shrinkage: torch.Tensor
    alpha_mu: torch.Tensor
    uses_unified_gene_program: bool = False

    @property
    def pi(self) -> torch.Tensor:
        """Calibrated support gate used by hurdle and Parent-shrink paths."""

        return self.calibrated_support_probability

    @property
    def combined_support_logits(self) -> torch.Tensor:
        """Legacy alias for the DE ranking logits."""

        return self.ranking_support_logits

    @property
    def de_probability(self) -> torch.Tensor:
        """Legacy DE-facing alias for the ranking probability."""

        return self.ranking_probability

    @property
    def de_support_probability(self) -> torch.Tensor:
        return self.ranking_probability

    @property
    def hurdle_gate_probability(self) -> torch.Tensor:
        return self.calibrated_support_probability

    @property
    def rank_scores(self) -> torch.Tensor:
        """Scores for DE ranking and any rank-preserving projector."""

        return self.ranking_probability

    @property
    def support_residual_logits(self) -> torch.Tensor:
        return self.head.support_residual_logits

    @property
    def zero_rate_shift(self) -> torch.Tensor:
        return self.head.zero_rate_shift

    @property
    def condition_budget_shift(self) -> torch.Tensor:
        return self.head.condition_budget_shift

    @property
    def virtual_delta(self) -> torch.Tensor:
        return self.adjusted_parent_delta


@dataclass(frozen=True)
class SparseHurdleProjection:
    """A nonnegative, fixed-Parent expression with explicit hard zeros.

    Expression-like tensors retain ``[B,S,G]``.  Requested and realized zero
    fractions use compact condition groups and have shape ``[Q,G]``.
    """

    value: torch.Tensor
    residual: torch.Tensor
    effective_parent: torch.Tensor
    group_ids: torch.Tensor
    counts: torch.Tensor
    valid_mask: torch.Tensor
    exact_mask: torch.Tensor
    singleton_mask: torch.Tensor
    raw_group_parent_mean: torch.Tensor
    effective_group_parent_mean: torch.Tensor
    group_mean_drift_max_abs: torch.Tensor
    raw_parent_adjustment_max_abs: torch.Tensor
    negative_values_before: torch.Tensor
    enforced_zero_mask: torch.Tensor
    requested_zero_fraction: torch.Tensor
    realized_zero_fraction: torch.Tensor
    alpha: torch.Tensor
    empty_support_fallbacks: torch.Tensor
    tie_expansion_count: torch.Tensor
    used_hurdle: bool


__all__ = [
    "SparseHurdleDEAdapterOutput",
    "SparseHurdleGeneProgramOutput",
    "SparseHurdleHeadOutput",
    "SparseHurdleProjection",
]

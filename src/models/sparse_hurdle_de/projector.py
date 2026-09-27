"""Rank-based hard-zero projection with an exactly fixed Parent mean."""

from __future__ import annotations

from dataclasses import replace

import math
from numbers import Real
from typing import Optional

import torch

from ..parent_locked_residual.centering import (
    NonnegativeFixedMeanProjection,
    project_group_nonnegative_fixed_mean,
)
from .output import SparseHurdleProjection


def _finite_alpha(alpha: float | torch.Tensor) -> float:
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be scalar")
        alpha = float(alpha.detach().cpu())
    elif isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real scalar")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return value


def _finite_temperature(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("zero_rate_temperature must be finite and positive")
    return value


def _expand_cell_gene(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the expression device")
    batch_size, set_size, genes = reference.shape
    if tuple(value.shape) == (batch_size, genes):
        value = value[:, None, :].expand(-1, set_size, -1)
    elif tuple(value.shape) == (batch_size, 1, genes):
        value = value.expand(-1, set_size, -1)
    elif tuple(value.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} must have shape [B,G], [B,1,G], or [B,S,G]"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _group_zero_fraction(
    value: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    genes = value.shape[-1]
    valid_positions = torch.nonzero(
        valid_mask.reshape(-1), as_tuple=False
    ).flatten()
    inverse = group_ids.reshape(-1).index_select(0, valid_positions)
    valid_value = value.reshape(-1, genes).index_select(0, valid_positions)
    zero = (valid_value == 0).to(torch.float32)
    totals = torch.zeros(
        counts.numel(), genes, device=value.device, dtype=torch.float32
    ).index_add(0, inverse, zero)
    return totals / counts.to(torch.float32)[:, None]


def _masked_simplex_project_columns(
    values: torch.Tensor,
    target_sum: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Project columns while forcing entries outside ``keep_mask`` to zero."""

    if values.ndim != 2 or target_sum.ndim != 1:
        raise ValueError("masked simplex inputs must have shape [N,G] and [G]")
    if keep_mask.shape != values.shape or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be boolean with shape [N,G]")
    if values.shape[1] != target_sum.shape[0] or values.shape[0] < 1:
        raise ValueError("masked simplex inputs have incompatible shapes")
    if not torch.isfinite(values).all() or not torch.isfinite(target_sum).all():
        raise ValueError("masked simplex inputs must be finite")
    if (target_sum < 0).any():
        raise ValueError("masked simplex target sums must be nonnegative")
    positive_target = target_sum > 0
    if (positive_target & ~keep_mask.any(dim=0)).any():
        raise ValueError("positive target columns require at least one kept cell")

    negative_infinity = torch.full_like(values, -torch.inf)
    masked_values = torch.where(keep_mask, values, negative_infinity)
    ordered = torch.sort(masked_values, dim=0, descending=True).values
    ordered_valid = torch.isfinite(ordered)
    safe_ordered = torch.where(ordered_valid, ordered, 0.0)
    ranks = ordered_valid.to(values.dtype).cumsum(dim=0).clamp_min(1.0)
    cumulative = safe_ordered.cumsum(dim=0) - target_sum[None, :]
    theta_candidates = cumulative / ranks
    active = ordered_valid & (ordered - theta_candidates > 0)
    rho = active.sum(dim=0).sub(1).clamp_min(0)
    theta = theta_candidates.gather(0, rho[None, :]).squeeze(0)
    projected = torch.where(
        keep_mask,
        torch.clamp_min(values - theta[None, :], 0.0),
        0.0,
    )
    projected = torch.where(positive_target[None, :], projected, 0.0)

    # Repair floating-point accumulation error on a retained maximum.  A
    # second repair handles the rare ulp introduced by the first indexed add.
    for _ in range(2):
        correction = target_sum - projected.sum(dim=0)
        largest = projected.argmax(dim=0, keepdim=True)
        repair = torch.zeros_like(projected).scatter(
            0, largest, correction[None, :]
        )
        projected = torch.where(
            keep_mask,
            torch.clamp_min(projected + repair, 0.0),
            0.0,
        )
        projected = torch.where(positive_target[None, :], projected, 0.0)
    return projected


def _terminal_fixed_support_additive_repair(
    values: torch.Tensor,
    target_sum: torch.Tensor,
    keep_mask: torch.Tensor,
    repair_rank: torch.Tensor,
) -> torch.Tensor:
    """Apply terminal H affine projection without closing the support.

    Once terminal H has selected an exact zero mask, the L2 projection on its
    fixed survivor support is an equal additive shift. Unlike a second closed
    simplex projection, it cannot silently threshold a small survivor to zero.
    A stable ordinal chooses the final floating-point repair coordinate.
    """

    if values.ndim != 2 or target_sum.ndim != 1:
        raise ValueError(
            "terminal fixed-support inputs must have shape [N,G] and [G]"
        )
    if keep_mask.shape != values.shape or keep_mask.dtype != torch.bool:
        raise ValueError("terminal keep_mask must be boolean with shape [N,G]")
    if repair_rank.shape != values.shape:
        raise ValueError("terminal repair_rank must have shape [N,G]")
    if values.shape[1] != target_sum.shape[0] or values.shape[0] < 1:
        raise ValueError("terminal fixed-support inputs have incompatible shapes")
    if (
        not torch.isfinite(values).all()
        or not torch.isfinite(target_sum).all()
        or not torch.isfinite(repair_rank).all()
    ):
        raise ValueError("terminal fixed-support inputs must be finite")
    if (values < 0).any() or (target_sum < 0).any():
        raise RuntimeError(
            "terminal fixed-support repair requires a legacy-feasible "
            "nonnegative endpoint"
        )

    positive_target = target_sum > 0
    survivor_count = keep_mask.sum(dim=0)
    missing_support = positive_target & (survivor_count == 0)
    if missing_support.any():
        raise RuntimeError(
            "terminal fixed-support repair has positive Parent mass but no "
            f"survivor columns={int(missing_support.sum().item())}"
        )
    unexpected_support = ~positive_target & (survivor_count != 0)
    if unexpected_support.any():
        raise RuntimeError(
            "terminal fixed-support repair retained cells for a zero-mass "
            f"Parent column count={int(unexpected_support.sum().item())}"
        )

    base = torch.where(keep_mask, values, 0.0)
    correction = target_sum - base.sum(dim=0)
    shift = correction / survivor_count.clamp_min(1).to(values.dtype)
    projected = torch.where(keep_mask, base + shift[None, :], 0.0)
    projected = torch.where(positive_target[None, :], projected, 0.0)

    lost_support = keep_mask & positive_target[None, :] & (projected <= 0)
    if lost_support.any():
        # The strict-positive fixed support is an open set. When its affine L2
        # solution reaches the boundary, retain the selected survivor at the
        # smallest representable positive value and reconcile that ULP-scale
        # mass on a deterministic maximum below. This preserves the discrete
        # support; if the maximum cannot pay for it, the repair fails closed.
        minimum_positive = torch.nextafter(
            torch.zeros((), device=values.device, dtype=values.dtype),
            torch.full((), torch.inf, device=values.device, dtype=values.dtype),
        )
        projected = torch.where(lost_support, minimum_positive, projected)

    negative_infinity = torch.full_like(projected, -torch.inf)
    kept_values = torch.where(keep_mask, projected, negative_infinity)
    maxima = kept_values.max(dim=0).values
    maximum_mask = keep_mask & (projected == maxima[None, :])
    ranked_maxima = torch.where(maximum_mask, repair_rank, negative_infinity)
    repair_rows = ranked_maxima.argmax(dim=0)
    columns = torch.arange(values.shape[1], device=values.device)

    for _ in range(4):
        residual = target_sum - projected.sum(dim=0)
        current = projected[repair_rows, columns]
        repaired = current + residual
        stalled = positive_target & (residual != 0) & (repaired == current)
        direction = torch.where(
            residual >= 0,
            torch.full_like(current, torch.inf),
            torch.full_like(current, -torch.inf),
        )
        repaired = torch.where(
            stalled,
            torch.nextafter(current, direction),
            repaired,
        )
        invalid_repair = positive_target & (
            ~torch.isfinite(repaired) | (repaired <= 0)
        )
        if invalid_repair.any():
            raise RuntimeError(
                "terminal fixed-support ULP repair cannot preserve positive "
                f"support columns={int(invalid_repair.sum().item())}"
            )
        projected = projected.index_put((repair_rows, columns), repaired)
        projected = torch.where(positive_target[None, :], projected, 0.0)

    final_lost_support = keep_mask & positive_target[None, :] & (projected <= 0)
    if final_lost_support.any() or not torch.isfinite(projected).all():
        raise RuntimeError("terminal fixed-support repair returned invalid support")
    return projected


def repair_terminal_hard_zero_fixed_support(
    expression: torch.Tensor,
    projection: SparseHurdleProjection,
    *,
    repair_rank: torch.Tensor,
) -> SparseHurdleProjection:
    """Repair terminal-H mass on its already selected survivor support only.

    This inference-only operation intentionally starts from the incoming
    legacy-feasible endpoint. The generic projector remains a closed-simplex
    projection for observed-target training and all non-terminal callers.
    """

    if not torch.is_tensor(expression) or expression.ndim != 3:
        raise ValueError("expression must have shape [B,S,G]")
    if not projection.used_hurdle:
        raise ValueError("terminal fixed-support repair requires an H projection")
    if tuple(projection.value.shape) != tuple(expression.shape):
        raise ValueError("terminal projection and expression shapes differ")
    if (
        projection.enforced_zero_mask.dtype != torch.bool
        or tuple(projection.enforced_zero_mask.shape) != tuple(expression.shape)
    ):
        raise ValueError("terminal projection has an invalid enforced-zero mask")
    expanded_rank = _expand_cell_gene(repair_rank, expression, "repair_rank")

    low_precision = expression.dtype in (torch.float16, torch.bfloat16)
    work_dtype = torch.float32 if low_precision else expression.dtype
    output_dtype = projection.value.dtype
    batch_size, set_size, genes = expression.shape
    flat_valid = projection.valid_mask.reshape(-1)
    valid_positions = torch.nonzero(flat_valid, as_tuple=False).flatten()
    inverse = projection.group_ids.reshape(-1).index_select(0, valid_positions)
    valid_expression = expression.reshape(-1, genes).index_select(
        0, valid_positions
    ).to(work_dtype)
    valid_rank = expanded_rank.reshape(-1, genes).index_select(
        0, valid_positions
    ).detach().to(work_dtype)
    valid_keep = ~projection.enforced_zero_mask.reshape(-1, genes).index_select(
        0, valid_positions
    )
    group_count = int(projection.counts.numel())
    if inverse.numel() == 0 or int(inverse.max().item()) >= group_count:
        raise ValueError("terminal projection has invalid compact group IDs")

    effective_mean = projection.effective_group_parent_mean.to(work_dtype)
    target_sum = effective_mean * projection.counts.to(work_dtype)[:, None]
    repaired_valid = torch.zeros_like(valid_expression)
    for group_index in range(group_count):
        positions = torch.nonzero(
            inverse == group_index, as_tuple=False
        ).flatten()
        if positions.numel() != int(projection.counts[group_index].item()):
            raise ValueError("terminal projection group count is inconsistent")
        repaired_group = _terminal_fixed_support_additive_repair(
            valid_expression.index_select(0, positions),
            target_sum[group_index],
            valid_keep.index_select(0, positions),
            valid_rank.index_select(0, positions),
        )
        repaired_valid = repaired_valid.index_copy(
            0, positions, repaired_group
        )

    repaired_flat = torch.zeros(
        batch_size * set_size,
        genes,
        device=expression.device,
        dtype=work_dtype,
    ).index_copy(0, valid_positions, repaired_valid)
    effective_valid = effective_mean.index_select(0, inverse)
    residual_valid = repaired_valid - effective_valid
    residual_flat = torch.zeros_like(repaired_flat).index_copy(
        0, valid_positions, residual_valid
    )
    effective_flat = torch.zeros_like(repaired_flat).index_copy(
        0, valid_positions, effective_valid
    )
    repaired_sums = torch.zeros(
        group_count, genes, device=expression.device, dtype=work_dtype
    ).index_add(0, inverse, repaired_valid)
    repaired_means = (
        repaired_sums / projection.counts.to(work_dtype)[:, None]
    )
    drift = (repaired_means - effective_mean).abs().max()
    value = repaired_flat.reshape(expression.shape).to(output_dtype)
    realized_fraction = _group_zero_fraction(
        value,
        projection.group_ids,
        projection.valid_mask,
        projection.counts,
    ).to(output_dtype)
    return replace(
        projection,
        value=value,
        residual=residual_flat.reshape(expression.shape).to(output_dtype),
        effective_parent=(
            effective_flat.reshape(expression.shape).to(output_dtype)
        ),
        group_mean_drift_max_abs=drift.to(output_dtype),
        negative_values_before=(valid_expression < 0).sum(),
        realized_zero_fraction=realized_fraction,
    )


def _legacy_output(legacy, alpha: float) -> SparseHurdleProjection:
    realized = _group_zero_fraction(
        legacy.value,
        legacy.group_ids,
        legacy.valid_mask,
        legacy.counts,
    )
    return SparseHurdleProjection(
        value=legacy.value,
        residual=legacy.residual,
        effective_parent=legacy.effective_parent,
        group_ids=legacy.group_ids,
        counts=legacy.counts,
        valid_mask=legacy.valid_mask,
        exact_mask=legacy.exact_mask,
        singleton_mask=legacy.singleton_mask,
        raw_group_parent_mean=legacy.raw_group_parent_mean,
        effective_group_parent_mean=legacy.effective_group_parent_mean,
        group_mean_drift_max_abs=legacy.group_mean_drift_max_abs,
        raw_parent_adjustment_max_abs=legacy.raw_parent_adjustment_max_abs,
        negative_values_before=legacy.negative_values_before,
        enforced_zero_mask=torch.zeros_like(legacy.value, dtype=torch.bool),
        requested_zero_fraction=realized.new_zeros(realized.shape),
        realized_zero_fraction=realized,
        alpha=legacy.value.new_tensor(alpha),
        empty_support_fallbacks=torch.zeros(
            (), device=legacy.value.device, dtype=torch.long
        ),
        tie_expansion_count=torch.zeros(
            (), device=legacy.value.device, dtype=torch.long
        ),
        used_hurdle=False,
    )


def project_ranked_hard_zero_fixed_parent_mean(
    expression: torch.Tensor,
    parent_mean: torch.Tensor,
    *,
    zero_rate_logits: Optional[torch.Tensor] = None,
    alpha: float | torch.Tensor = 0.0,
    zero_rate_temperature: float = 1.0,
    rank_scores: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    reference_projection: Optional[NonnegativeFixedMeanProjection] = None,
) -> SparseHurdleProjection:
    """Create exact zeros without changing the effective Parent group mean.

    ``zero_rate_logits`` is condition-level (``[B,G]``) but may also be
    explicitly repeated over a set axis.  For each compact condition group and
    gene, its mean logit determines a requested zero count.  Cells are ranked
    by ``rank_scores`` (the raw expression by default), the lowest cells are
    forced to zero, and the retained values are projected onto a simplex whose
    sum is ``group_size * clamp(group_mean(parent), 0)``.

    ``alpha=0`` is an explicit early return through the historical projector;
    zero-rate inputs are neither required nor inspected on that path.
    """

    alpha_value = _finite_alpha(alpha)
    if not torch.is_tensor(expression) or expression.ndim != 3:
        raise ValueError("expression must have shape [B,S,G]")
    legacy = project_group_nonnegative_fixed_mean(
        expression,
        parent_mean,
        group_ids=group_ids,
        valid_mask=valid_mask,
        reference_projection=reference_projection,
    )
    if alpha_value == 0.0:
        return _legacy_output(legacy, alpha_value)
    if zero_rate_logits is None:
        raise ValueError("zero_rate_logits is required when alpha is nonzero")
    temperature = _finite_temperature(zero_rate_temperature)
    expanded_logits = _expand_cell_gene(
        zero_rate_logits, expression, "zero_rate_logits"
    )
    if rank_scores is None:
        expanded_rank = expression
    else:
        expanded_rank = _expand_cell_gene(
            rank_scores, expression, "rank_scores"
        )

    low_precision = expression.dtype in (torch.float16, torch.bfloat16)
    work_dtype = torch.float32 if low_precision else expression.dtype
    output_dtype = work_dtype
    batch_size, set_size, genes = expression.shape
    flat_valid = legacy.valid_mask.reshape(-1)
    valid_positions = torch.nonzero(flat_valid, as_tuple=False).flatten()
    inverse = legacy.group_ids.reshape(-1).index_select(0, valid_positions)
    valid_expression = expression.reshape(-1, genes).index_select(
        0, valid_positions
    ).to(work_dtype)
    valid_rank = expanded_rank.reshape(-1, genes).index_select(
        0, valid_positions
    ).detach().to(work_dtype)
    valid_logits = expanded_logits.reshape(-1, genes).index_select(
        0, valid_positions
    ).float()
    group_count = int(legacy.counts.numel())
    counts_float = legacy.counts.to(torch.float32)[:, None]
    grouped_logits = torch.zeros(
        group_count, genes, device=expression.device, dtype=torch.float32
    ).index_add(0, inverse, valid_logits)
    grouped_logits = grouped_logits / counts_float
    zero_probability = torch.sigmoid(grouped_logits / temperature)

    effective_mean = legacy.effective_group_parent_mean.to(work_dtype)
    target_sum = effective_mean * legacy.counts.to(work_dtype)[:, None]
    current_zero_count = torch.zeros(
        group_count, genes, device=expression.device, dtype=torch.long
    ).index_add(0, inverse, (valid_expression == 0).to(torch.long))
    target_zero_count = torch.floor(
        zero_probability * counts_float + 0.5
    ).to(torch.long)
    # Alpha is a monotone continuation from the expression's current point
    # mass, not a replacement zero rate. Existing exact zeros therefore
    # cannot disappear merely because the head requests a smaller fraction.
    requested_additional = (
        target_zero_count - current_zero_count
    ).clamp_min(0)
    activated_additional = torch.floor(
        float(alpha_value) * requested_additional.to(torch.float32) + 0.5
    ).to(torch.long)
    desired_zero_count = current_zero_count + activated_additional
    positive_target = target_sum > 0
    maximum_for_positive = legacy.counts[:, None].sub(1).clamp_min(0)
    maximum_zero_count = torch.where(
        positive_target,
        maximum_for_positive.expand(-1, genes),
        legacy.counts[:, None].expand(-1, genes),
    )
    desired_zero_count = torch.minimum(
        desired_zero_count, maximum_zero_count
    ).clamp_min(0)
    requested_fraction = desired_zero_count.to(torch.float32) / counts_float

    projected_valid = torch.zeros_like(valid_expression)
    enforced_valid = torch.zeros_like(valid_expression, dtype=torch.bool)
    empty_support_fallbacks = 0
    tie_expansion_count = 0
    for group_index in range(group_count):
        positions = torch.nonzero(
            inverse == group_index, as_tuple=False
        ).flatten()
        group_value = valid_expression.index_select(0, positions)
        group_rank = valid_rank.index_select(0, positions)
        requested_count = desired_zero_count[group_index]
        original_zero = group_value == 0
        original_count = original_zero.sum(dim=0)
        additional_count = (requested_count - original_count).clamp_min(0)
        # Rank only cells that are not already exact zeros. This both keeps the
        # original hurdle mass and makes alpha control newly introduced zeros.
        candidate_rank = torch.where(
            original_zero,
            torch.full_like(group_rank, torch.inf),
            group_rank,
        )
        sorted_rank = torch.sort(candidate_rank, dim=0).values
        threshold_index = additional_count.sub(1).clamp(
            min=0, max=group_rank.shape[0] - 1
        )
        threshold = sorted_rank.gather(
            0, threshold_index[None, :]
        ).squeeze(0)
        newly_zero = (
            ~original_zero
            & (candidate_rank <= threshold[None, :])
            & (additional_count[None, :] > 0)
        )
        zero_mask = original_zero | newly_zero
        zero_target = ~positive_target[group_index]
        zero_mask[:, zero_target] = True

        # Inclusive thresholds make tied ranks permutation equivariant but can
        # consume the whole support.  Retain every maximum in that rare case;
        # if all scores tie, the safe result is the legacy all-active support.
        no_support = zero_mask.all(dim=0) & positive_target[group_index]
        if no_support.any():
            maxima = group_rank.max(dim=0).values
            keep_maxima = group_rank == maxima[None, :]
            zero_mask[:, no_support] = ~keep_maxima[:, no_support]
            empty_support_fallbacks += int(no_support.sum().item())

        actual_count = zero_mask.sum(dim=0)
        tie_expansion_count += int(
            (
                (actual_count > requested_count)
                & positive_target[group_index]
            ).sum().item()
        )
        keep_mask = ~zero_mask
        keep_mask[:, zero_target] = False
        group_projected = _masked_simplex_project_columns(
            group_value,
            target_sum[group_index],
            keep_mask,
        )
        projected_valid = projected_valid.index_copy(
            0, positions, group_projected
        )
        enforced_valid = enforced_valid.index_copy(
            0, positions, zero_mask
        )

    projected_flat = torch.zeros(
        batch_size * set_size,
        genes,
        device=expression.device,
        dtype=work_dtype,
    ).index_copy(0, valid_positions, projected_valid)
    enforced_flat = torch.zeros(
        batch_size * set_size,
        genes,
        device=expression.device,
        dtype=torch.bool,
    ).index_copy(0, valid_positions, enforced_valid)
    effective_valid = effective_mean.index_select(0, inverse)
    residual_valid = projected_valid - effective_valid
    residual_flat = torch.zeros_like(projected_flat).index_copy(
        0, valid_positions, residual_valid
    )
    effective_flat = torch.zeros_like(projected_flat).index_copy(
        0, valid_positions, effective_valid
    )

    projected_sums = torch.zeros(
        group_count, genes, device=expression.device, dtype=work_dtype
    ).index_add(0, inverse, projected_valid)
    projected_means = projected_sums / legacy.counts.to(work_dtype)[:, None]
    drift = (projected_means - effective_mean).abs().max()
    value = projected_flat.reshape(expression.shape).to(output_dtype)
    residual = residual_flat.reshape(expression.shape).to(output_dtype)
    effective_parent = effective_flat.reshape(expression.shape).to(output_dtype)
    realized_fraction = _group_zero_fraction(
        value,
        legacy.group_ids,
        legacy.valid_mask,
        legacy.counts,
    )
    return SparseHurdleProjection(
        value=value,
        residual=residual,
        effective_parent=effective_parent,
        group_ids=legacy.group_ids,
        counts=legacy.counts,
        valid_mask=legacy.valid_mask,
        exact_mask=legacy.exact_mask,
        singleton_mask=legacy.singleton_mask,
        raw_group_parent_mean=legacy.raw_group_parent_mean.to(output_dtype),
        effective_group_parent_mean=effective_mean.to(output_dtype),
        group_mean_drift_max_abs=drift.to(output_dtype),
        raw_parent_adjustment_max_abs=(
            effective_mean
            - legacy.raw_group_parent_mean.to(work_dtype)
        ).abs().max().to(output_dtype),
        negative_values_before=(valid_expression < 0).sum(),
        enforced_zero_mask=enforced_flat.reshape(expression.shape),
        requested_zero_fraction=requested_fraction.to(output_dtype),
        realized_zero_fraction=realized_fraction.to(output_dtype),
        alpha=value.new_tensor(alpha_value),
        empty_support_fallbacks=torch.tensor(
            empty_support_fallbacks,
            device=expression.device,
            dtype=torch.long,
        ),
        tie_expansion_count=torch.tensor(
            tie_expansion_count,
            device=expression.device,
            dtype=torch.long,
        ),
        used_hurdle=True,
    )


__all__ = [
    "project_ranked_hard_zero_fixed_parent_mean",
    "repair_terminal_hard_zero_fixed_support",
]

"""Detached, grouped semi-balanced optimal transport for Child routing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SemiBalancedOTResult:
    """Grouped transport plan and stable aggregate diagnostics."""

    child_distribution: torch.Tensor
    coupling: torch.Tensor
    column_marginal: torch.Tensor
    target_occupancy: torch.Tensor
    column_l1: torch.Tensor
    column_kl: torch.Tensor
    effective_children: torch.Tensor
    cost_scale: torch.Tensor


def _validate_inputs(
    cost,
    occupancy,
    group_ids,
    child_mask,
    row_mask,
    epsilon,
    rho,
    iterations,
):
    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError("semi-balanced OT cost must have shape [N,K] and be non-empty")
    if occupancy.shape != cost.shape:
        raise ValueError("semi-balanced OT occupancy must have the same [N,K] shape")
    if group_ids is None:
        raise ValueError("semi-balanced OT requires dataset/cell-line/perturbation group_ids")
    if group_ids.ndim not in (1, 2) or group_ids.shape[0] != cost.shape[0]:
        raise ValueError("semi-balanced OT group_ids must have shape [N] or [N,D]")
    if child_mask is not None and child_mask.shape != cost.shape:
        raise ValueError("semi-balanced OT child_mask must have shape [N,K]")
    if row_mask is not None and row_mask.shape not in {
        (cost.shape[0],),
        (cost.shape[0], 1),
    }:
        raise ValueError("semi-balanced OT row_mask must have shape [N] or [N,1]")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("semi-balanced OT epsilon must be finite and positive")
    if not math.isfinite(float(rho)) or float(rho) < 0.0:
        raise ValueError("semi-balanced OT rho must be finite and non-negative")
    if int(iterations) < 1:
        raise ValueError("semi-balanced OT iterations must be positive")


def _group_cost_scale(group_cost, support):
    """Return a robust, positive per-group scale without coupling groups."""

    finite = group_cost[support]
    if finite.numel() == 0:
        return group_cost.new_tensor(1.0)
    shifted = finite - finite.min()
    positive = shifted[shifted > 0.0]
    if positive.numel() == 0:
        return group_cost.new_tensor(1.0)
    scale = positive.median()
    if not torch.isfinite(scale) or scale <= 1.0e-8:
        scale = positive.mean()
    if not torch.isfinite(scale) or scale <= 1.0e-8:
        scale = group_cost.new_tensor(1.0)
    return scale.clamp_min(1.0e-8)


def _target_occupancy(group_occupancy, support):
    """Aggregate row occupancies after respecting row-specific Child masks."""

    weights = group_occupancy.clamp_min(0.0).masked_fill(~support, 0.0)
    row_total = weights.sum(dim=-1, keepdim=True)
    supported_count = support.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = support.to(weights.dtype) / supported_count
    weights = torch.where(row_total > 0.0, weights / row_total.clamp_min(1.0e-30), uniform)
    target = weights.mean(dim=0)
    available = support.any(dim=0)
    target = target.masked_fill(~available, 0.0)
    total = target.sum()
    if not torch.isfinite(total) or total <= 0.0:
        target = available.to(weights.dtype)
        total = target.sum()
    return target / total.clamp_min(1.0)


def _solve_one_group(
    cost,
    occupancy,
    support,
    *,
    epsilon,
    rho,
    iterations,
    normalize_cost_by_median,
):
    rows = cost.shape[0]
    if (~support.any(dim=-1)).any():
        raise ValueError("every valid OT row needs at least one active finite Child")

    scale = (
        _group_cost_scale(cost, support)
        if normalize_cost_by_median
        else cost.new_tensor(1.0)
    )
    normalized_cost = cost / scale
    # Per-row shifts leave the exact-row solution invariant and substantially
    # improve the range of the log kernel for extreme biological costs.
    row_min = normalized_cost.masked_fill(~support, torch.inf).min(dim=-1).values
    log_kernel = -(normalized_cost - row_min[:, None]) / float(epsilon)
    log_kernel = log_kernel.masked_fill(~support, -torch.inf)

    target = _target_occupancy(occupancy, support)
    available = support.any(dim=0)
    log_target = target.clamp_min(1.0e-30).log().masked_fill(~available, -torch.inf)
    log_row_mass = cost.new_full((rows,), -math.log(float(rows)))
    log_v = cost.new_zeros(cost.shape[1]).masked_fill(~available, -torch.inf)
    exponent = float(rho) / (float(rho) + float(epsilon))

    if exponent > 0.0:
        for _ in range(int(iterations)):
            log_u = log_row_mass - torch.logsumexp(
                log_kernel + log_v[None, :], dim=-1
            )
            log_column_without_v = torch.logsumexp(
                log_kernel + log_u[:, None], dim=0
            )
            log_v = exponent * (log_target - log_column_without_v)
            log_v = log_v.masked_fill(~available, -torch.inf)

    log_u = log_row_mass - torch.logsumexp(
        log_kernel + log_v[None, :], dim=-1
    )
    log_coupling = log_kernel + log_u[:, None] + log_v[None, :]
    # Re-impose the exact row marginal after the finite Sinkhorn iteration
    # budget.  This keeps the downstream posterior normalized even at FP32.
    log_coupling = (
        log_coupling
        - torch.logsumexp(log_coupling, dim=-1, keepdim=True)
        + log_row_mass[:, None]
    )
    coupling = log_coupling.exp().masked_fill(~support, 0.0)
    distribution = coupling * float(rows)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(
        1.0e-30
    )
    distribution = distribution.masked_fill(~support, 0.0)
    coupling = distribution / float(rows)
    return distribution, coupling, target, scale


@torch.no_grad()
def solve_grouped_semi_balanced_ot(
    cost: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    child_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    epsilon: float = 0.1,
    rho: float = 1.0,
    iterations: int = 50,
    normalize_cost_by_median: bool = True,
) -> SemiBalancedOTResult:
    """Solve independent semi-balanced OT problems for condition groups.

    ``rho=0`` is exactly the masked row-wise soft assignment
    ``softmax(-cost / epsilon)``.  Positive ``rho`` softly attracts the Child
    column marginal to the unequal control occupancy while every valid row
    retains mass ``1 / group_size``.
    """

    _validate_inputs(
        cost,
        occupancy,
        group_ids,
        child_mask,
        row_mask,
        epsilon,
        rho,
        iterations,
    )
    device = cost.device
    cost32 = cost.detach().to(device=device, dtype=torch.float32)
    occupancy32 = occupancy.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(occupancy32).all() or (occupancy32 < 0.0).any():
        raise ValueError("semi-balanced OT occupancy must be finite and non-negative")
    mask = (
        torch.ones_like(cost32, dtype=torch.bool)
        if child_mask is None
        else child_mask.detach().to(device=device, dtype=torch.bool)
    )
    mask = mask & torch.isfinite(cost32)
    valid_rows = (
        torch.ones(cost32.shape[0], device=device, dtype=torch.bool)
        if row_mask is None
        else row_mask.detach().to(device=device, dtype=torch.bool).reshape(-1)
    )
    ids = group_ids.detach().to(device=device, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids[:, None]

    distribution = torch.zeros_like(cost32)
    coupling = torch.zeros_like(cost32)
    column_marginal = torch.zeros_like(cost32)
    target_occupancy = torch.zeros_like(cost32)
    row_column_l1 = torch.zeros(cost32.shape[0], device=device)
    row_column_kl = torch.zeros(cost32.shape[0], device=device)
    row_effective = torch.zeros(cost32.shape[0], device=device)
    row_scale = torch.zeros(cost32.shape[0], device=device)

    _, inverse = torch.unique(ids, dim=0, return_inverse=True)
    solved_groups = 0
    for group_index in range(int(inverse.max().item()) + 1):
        rows = (inverse == group_index) & valid_rows
        if not rows.any():
            continue
        group_rows = torch.nonzero(rows, as_tuple=False).flatten()
        group_distribution, group_coupling, target, scale = _solve_one_group(
            cost32.index_select(0, group_rows),
            occupancy32.index_select(0, group_rows),
            mask.index_select(0, group_rows),
            epsilon=float(epsilon),
            rho=float(rho),
            iterations=int(iterations),
            normalize_cost_by_median=bool(normalize_cost_by_median),
        )
        marginal = group_coupling.sum(dim=0)
        safe_marginal = marginal.clamp_min(1.0e-30)
        safe_target = target.clamp_min(1.0e-30)
        column_l1 = (marginal - target).abs().sum()
        column_kl = (
            marginal * (safe_marginal.log() - safe_target.log())
        ).sum()
        effective = (-(marginal * safe_marginal.log()).sum()).exp()

        group_size = group_rows.numel()
        distribution.index_copy_(0, group_rows, group_distribution)
        coupling.index_copy_(0, group_rows, group_coupling)
        column_marginal.index_copy_(
            0, group_rows, marginal.unsqueeze(0).expand(group_size, -1)
        )
        target_occupancy.index_copy_(
            0, group_rows, target.unsqueeze(0).expand(group_size, -1)
        )
        row_column_l1.index_copy_(
            0, group_rows, column_l1.expand(group_size)
        )
        row_column_kl.index_copy_(
            0, group_rows, column_kl.expand(group_size)
        )
        row_effective.index_copy_(
            0, group_rows, effective.expand(group_size)
        )
        row_scale.index_copy_(0, group_rows, scale.expand(group_size))
        solved_groups += 1

    if solved_groups == 0:
        zero = cost32.new_zeros(())
        mean_l1 = mean_kl = mean_effective = mean_scale = zero
    else:
        weights = valid_rows.to(cost32.dtype)
        denominator = weights.sum().clamp_min(1.0)
        mean_l1 = (row_column_l1 * weights).sum() / denominator
        mean_kl = (row_column_kl * weights).sum() / denominator
        mean_effective = (row_effective * weights).sum() / denominator
        mean_scale = (row_scale * weights).sum() / denominator

    return SemiBalancedOTResult(
        child_distribution=distribution.detach(),
        coupling=coupling.detach(),
        column_marginal=column_marginal.detach(),
        target_occupancy=target_occupancy.detach(),
        column_l1=mean_l1.detach(),
        column_kl=mean_kl.detach(),
        effective_children=mean_effective.detach(),
        cost_scale=mean_scale.detach(),
    )


__all__ = ["SemiBalancedOTResult", "solve_grouped_semi_balanced_ot"]

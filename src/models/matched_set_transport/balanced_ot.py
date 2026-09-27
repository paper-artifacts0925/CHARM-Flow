"""Detached grouped OT with exact row and Child occupancy marginals."""

from __future__ import annotations

import math

import torch

from .semi_balanced_ot import (
    SemiBalancedOTResult,
    _group_cost_scale,
    _target_occupancy,
    _validate_inputs,
)


def _solve_one_balanced_group(
    cost,
    occupancy,
    support,
    *,
    epsilon,
    iterations,
    normalize_cost_by_median,
    column_tolerance,
):
    rows = cost.shape[0]
    target = _target_occupancy(occupancy, support)
    positive_target = target > 0.0
    support = support & positive_target[None, :]
    if (~support.any(dim=-1)).any():
        raise ValueError(
            "balanced OT row has no support in the positive occupancy marginal"
        )

    scale = (
        _group_cost_scale(cost, support)
        if normalize_cost_by_median
        else cost.new_tensor(1.0)
    )
    normalized_cost = cost / scale
    row_min = normalized_cost.masked_fill(~support, torch.inf).min(dim=-1).values
    log_kernel = -(normalized_cost - row_min[:, None]) / float(epsilon)
    log_kernel = log_kernel.masked_fill(~support, -torch.inf)

    log_a = cost.new_full((rows,), -math.log(float(rows)))
    log_b = target.clamp_min(1.0e-30).log().masked_fill(
        ~positive_target, -torch.inf
    )
    log_v = cost.new_zeros(cost.shape[1]).masked_fill(
        ~positive_target, -torch.inf
    )
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(
            log_kernel + log_v[None, :], dim=-1
        )
        log_v = log_b - torch.logsumexp(
            log_kernel + log_u[:, None], dim=0
        )
        log_v = log_v.masked_fill(~positive_target, -torch.inf)

    # Finish with the exact-row projection.  Converged Sinkhorn scaling also
    # retains the requested column marginal; the explicit tolerance below
    # prevents a finite-iteration proxy from being mislabeled as balanced.
    log_u = log_a - torch.logsumexp(
        log_kernel + log_v[None, :], dim=-1
    )
    log_coupling = log_kernel + log_u[:, None] + log_v[None, :]
    coupling = log_coupling.exp().masked_fill(~support, 0.0)
    row_residual = (
        coupling.sum(dim=-1) - coupling.new_full((rows,), 1.0 / rows)
    ).abs().max()
    column_residual = (coupling.sum(dim=0) - target).abs().sum()
    if not torch.isfinite(row_residual) or not torch.isfinite(column_residual):
        raise FloatingPointError("balanced OT produced non-finite marginals")
    if float(row_residual) > 2.0e-5:
        raise RuntimeError(
            "balanced OT row marginal failed: "
            f"residual={float(row_residual):.6g}"
        )
    if float(column_residual) > float(column_tolerance):
        raise RuntimeError(
            "balanced OT column marginal failed: "
            f"residual={float(column_residual):.6g} "
            f"tolerance={float(column_tolerance):.6g}; increase iterations"
        )
    distribution = coupling * float(rows)
    return distribution, coupling, target, scale


@torch.no_grad()
def solve_grouped_balanced_ot(
    cost: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    child_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    epsilon: float = 0.1,
    iterations: int = 300,
    normalize_cost_by_median: bool = True,
    column_tolerance: float = 1.0e-3,
) -> SemiBalancedOTResult:
    """Solve exact-marginal entropic OT independently per condition group."""

    _validate_inputs(
        cost,
        occupancy,
        group_ids,
        child_mask,
        row_mask,
        epsilon,
        0.0,
        iterations,
    )
    if not math.isfinite(float(column_tolerance)) or float(column_tolerance) <= 0.0:
        raise ValueError("balanced OT column_tolerance must be finite and positive")

    device = cost.device
    cost32 = cost.detach().to(device=device, dtype=torch.float32)
    occupancy32 = occupancy.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(occupancy32).all() or (occupancy32 < 0.0).any():
        raise ValueError("balanced OT occupancy must be finite and non-negative")
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
    for group_index in range(int(inverse.max().item()) + 1):
        rows = (inverse == group_index) & valid_rows
        if not rows.any():
            continue
        group_rows = torch.nonzero(rows, as_tuple=False).flatten()
        group_distribution, group_coupling, target, scale = (
            _solve_one_balanced_group(
                cost32.index_select(0, group_rows),
                occupancy32.index_select(0, group_rows),
                mask.index_select(0, group_rows),
                epsilon=float(epsilon),
                iterations=int(iterations),
                normalize_cost_by_median=bool(normalize_cost_by_median),
                column_tolerance=float(column_tolerance),
            )
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

    weights = valid_rows.to(cost32.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return SemiBalancedOTResult(
        child_distribution=distribution.detach(),
        coupling=coupling.detach(),
        column_marginal=column_marginal.detach(),
        target_occupancy=target_occupancy.detach(),
        column_l1=((row_column_l1 * weights).sum() / denominator).detach(),
        column_kl=((row_column_kl * weights).sum() / denominator).detach(),
        effective_children=((row_effective * weights).sum() / denominator).detach(),
        cost_scale=((row_scale * weights).sum() / denominator).detach(),
    )


__all__ = ["solve_grouped_balanced_ot"]

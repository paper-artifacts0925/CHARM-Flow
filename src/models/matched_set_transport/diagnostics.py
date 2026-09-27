"""Condition-level posterior/prior diagnostics for Child routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from .routing import _normalized_probabilities, _validate_probability_inputs


@dataclass(frozen=True)
class ConditionRoutingDiagnostics:
    """Unweighted condition-level summaries of q-bar versus Router prior."""

    kl_mean: torch.Tensor
    js_mean: torch.Tensor
    js_max: torch.Tensor
    qbar_entropy_mean: torch.Tensor
    prior_entropy_mean: torch.Tensor
    valid_groups: torch.Tensor
    mean_group_size: torch.Tensor


def _validate_group_ids(group_ids: torch.Tensor, rows: int, device) -> torch.Tensor:
    if not torch.is_tensor(group_ids):
        raise TypeError("group_ids must be a torch tensor")
    if group_ids.ndim not in (1, 2) or group_ids.shape[0] != rows:
        raise ValueError("group_ids must have shape [B] or [B,D]")
    if group_ids.is_floating_point() or group_ids.dtype == torch.bool:
        raise TypeError("group_ids must contain integer identifiers")
    return group_ids.to(device=device, dtype=torch.long)


def condition_level_posterior_prior_kl(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    child_mask: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable, condition-balanced KL(stopgrad(q-bar) || p-bar)."""

    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    child_mask = _validate_probability_inputs(posterior, prior, child_mask)
    posterior = posterior.detach().float()
    prior = prior.float()
    child_mask = child_mask.to(device=posterior.device)
    q, q_valid = _normalized_probabilities(posterior, child_mask)
    p, p_valid = _normalized_probabilities(prior, child_mask)
    if not p_valid.all():
        raise ValueError("every prior row must have positive active mass")

    valid = q_valid
    if row_mask is not None:
        if not torch.is_tensor(row_mask) or row_mask.shape != (posterior.shape[0],):
            raise ValueError("row_mask must have shape [B]")
        valid = valid & row_mask.to(device=posterior.device, dtype=torch.bool)

    identifiers = _validate_group_ids(
        group_ids, posterior.shape[0], posterior.device
    )
    if identifiers.ndim == 1:
        _, inverse = torch.unique(identifiers, sorted=True, return_inverse=True)
    else:
        _, inverse = torch.unique(
            identifiers, dim=0, sorted=True, return_inverse=True
        )

    values = []
    for group_index in range(int(inverse.max().item()) + 1):
        members = (inverse == group_index) & valid
        if not members.any():
            continue
        qbar = q[members].mean(dim=0)
        pbar = p[members].mean(dim=0)
        qbar = qbar / qbar.sum().clamp_min(float(eps))
        pbar = pbar / pbar.sum().clamp_min(float(eps))
        values.append(
            (
                qbar
                * (
                    qbar.clamp_min(float(eps)).log()
                    - pbar.clamp_min(float(eps)).log()
                )
            ).sum()
        )
    if not values:
        return prior.sum() * 0.0
    return torch.stack(values).mean()


@torch.no_grad()
def condition_level_posterior_prior_diagnostics(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    child_mask: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> ConditionRoutingDiagnostics:
    """Aggregate cell posteriors within each condition before KL/JS.

    Every valid condition contributes once, irrespective of its number of
    response cells.  This exposes whether a target-free Router learns the
    condition's aggregate Hungarian occupancy without forcing individual
    cells to have identical assignments.
    """

    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    child_mask = _validate_probability_inputs(posterior, prior, child_mask)
    posterior = posterior.detach().float()
    prior = prior.detach().float()
    child_mask = child_mask.to(device=posterior.device)
    q, q_valid = _normalized_probabilities(posterior, child_mask)
    p, p_valid = _normalized_probabilities(prior, child_mask)
    if not p_valid.all():
        raise ValueError("every prior row must have positive active mass")

    valid = q_valid
    if row_mask is not None:
        if not torch.is_tensor(row_mask) or row_mask.shape != (posterior.shape[0],):
            raise ValueError("row_mask must have shape [B]")
        valid = valid & row_mask.to(device=posterior.device, dtype=torch.bool)

    identifiers = _validate_group_ids(
        group_ids, posterior.shape[0], posterior.device
    )
    if identifiers.ndim == 1:
        _, inverse = torch.unique(identifiers, sorted=True, return_inverse=True)
    else:
        _, inverse = torch.unique(
            identifiers, dim=0, sorted=True, return_inverse=True
        )

    kl_values = []
    js_values = []
    q_entropy_values = []
    p_entropy_values = []
    group_sizes = []
    for group_index in range(int(inverse.max().item()) + 1):
        members = (inverse == group_index) & valid
        count = int(members.sum().item())
        if count < 1:
            continue
        qbar = q[members].mean(dim=0)
        pbar = p[members].mean(dim=0)
        qbar = qbar / qbar.sum().clamp_min(float(eps))
        pbar = pbar / pbar.sum().clamp_min(float(eps))
        midpoint = 0.5 * (qbar + pbar)

        q_log = qbar.clamp_min(float(eps)).log()
        p_log = pbar.clamp_min(float(eps)).log()
        midpoint_log = midpoint.clamp_min(float(eps)).log()
        kl_values.append((qbar * (q_log - p_log)).sum())
        js_values.append(
            0.5 * (qbar * (q_log - midpoint_log)).sum()
            + 0.5 * (pbar * (p_log - midpoint_log)).sum()
        )
        q_entropy_values.append(-(qbar * q_log).sum())
        p_entropy_values.append(-(pbar * p_log).sum())
        group_sizes.append(qbar.new_tensor(float(count)))

    zero = posterior.new_zeros(())
    if not kl_values:
        return ConditionRoutingDiagnostics(
            kl_mean=zero,
            js_mean=zero,
            js_max=zero,
            qbar_entropy_mean=zero,
            prior_entropy_mean=zero,
            valid_groups=zero,
            mean_group_size=zero,
        )
    kl = torch.stack(kl_values)
    js = torch.stack(js_values)
    return ConditionRoutingDiagnostics(
        kl_mean=kl.mean(),
        js_mean=js.mean(),
        js_max=js.max(),
        qbar_entropy_mean=torch.stack(q_entropy_values).mean(),
        prior_entropy_mean=torch.stack(p_entropy_values).mean(),
        valid_groups=zero.new_tensor(float(len(kl_values))),
        mean_group_size=torch.stack(group_sizes).mean(),
    )


__all__ = [
    "ConditionRoutingDiagnostics",
    "condition_level_posterior_prior_diagnostics",
    "condition_level_posterior_prior_kl",
]

"""Low-variance inference-time Child sampling within biological conditions."""

from __future__ import annotations

from typing import Optional

import torch


_VALID_MODES = frozenset({"legacy", "quota", "stratified"})


def _group_inverse(
    group_ids: Optional[torch.Tensor],
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    if group_ids is None:
        return torch.arange(rows, device=device, dtype=torch.long)
    if not torch.is_tensor(group_ids):
        raise TypeError("group_ids must be a torch tensor")
    if group_ids.ndim not in (1, 2) or group_ids.shape[0] != rows:
        raise ValueError("group_ids must have shape [B] or [B,D]")
    if group_ids.is_floating_point() or group_ids.dtype == torch.bool:
        raise TypeError("group_ids must contain integer identifiers")
    group_ids = group_ids.to(device=device, dtype=torch.long)
    if group_ids.ndim == 1:
        _, inverse = torch.unique(
            group_ids, sorted=True, return_inverse=True
        )
        return inverse
    _, inverse = torch.unique(
        group_ids, dim=0, sorted=True, return_inverse=True
    )
    return inverse


def _quota_assignments(
    probability: torch.Tensor,
    total: int,
    *,
    stochastic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    expected = probability * int(total)
    counts = expected.floor().to(torch.long)
    remainder = int(total) - int(counts.sum().item())
    if remainder:
        fractions = expected - counts.to(expected)
        order = torch.argsort(fractions, descending=True, stable=True)
        counts[order[:remainder]] += 1
    assignments = torch.arange(
        probability.shape[0], device=probability.device
    ).repeat_interleave(counts)
    if stochastic and assignments.numel() > 1:
        order = torch.randperm(
            assignments.numel(),
            device=assignments.device,
            generator=generator,
        )
        assignments = assignments[order]
    return assignments


def _stratified_assignments(
    probability: torch.Tensor,
    total: int,
    *,
    stochastic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    offsets = (
        torch.rand(
            int(total),
            device=probability.device,
            generator=generator,
            dtype=probability.dtype,
        )
        if stochastic
        else probability.new_full((int(total),), 0.5)
    )
    positions = (
        torch.arange(int(total), device=probability.device, dtype=probability.dtype)
        + offsets
    ) / float(total)
    assignments = torch.searchsorted(
        probability.cumsum(dim=0), positions, right=False
    ).clamp_max(probability.shape[0] - 1)
    if stochastic and assignments.numel() > 1:
        order = torch.randperm(
            assignments.numel(),
            device=assignments.device,
            generator=generator,
        )
        assignments = assignments[order]
    return assignments


@torch.no_grad()
def sample_grouped_child_indices(
    probability: torch.Tensor,
    *,
    num_samples: int,
    mode: str = "legacy",
    group_ids: Optional[torch.Tensor] = None,
    stochastic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample [B,S] Child indices, optionally sharing quota by condition.

    legacy is exactly the historical row-wise multinomial/argmax path.
    quota uses largest-remainder counts from the mean prior of a condition.
    stratified draws one point per equal-mass stratum. The latter two
    reduce multinomial occupancy noise but still return one Child per cell.
    """

    if not torch.is_tensor(probability) or not probability.is_floating_point():
        raise TypeError("probability must be a floating point tensor")
    if probability.ndim != 2 or min(probability.shape) < 1:
        raise ValueError("probability must have non-empty shape [B,K]")
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("probability must be finite and non-negative")
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    mode = str(mode).lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            "mode must be one of " + ", ".join(sorted(_VALID_MODES))
        )
    mass = probability.sum(dim=-1, keepdim=True)
    if not (mass > 0).all():
        raise ValueError("every probability row must have positive mass")
    probability = probability / mass

    if mode == "legacy":
        if stochastic:
            return torch.multinomial(
                probability.float(),
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )
        return probability.argmax(dim=-1, keepdim=True).expand(-1, num_samples)

    inverse = _group_inverse(
        group_ids,
        rows=probability.shape[0],
        device=probability.device,
    )
    result = torch.empty(
        probability.shape[0],
        num_samples,
        dtype=torch.long,
        device=probability.device,
    )
    for group_index in range(int(inverse.max().item()) + 1):
        rows = torch.nonzero(inverse == group_index, as_tuple=False)[:, 0]
        group_probability = probability[rows].mean(dim=0)
        group_probability = group_probability / group_probability.sum()
        total = int(rows.numel()) * num_samples
        if mode == "quota":
            assignments = _quota_assignments(
                group_probability,
                total,
                stochastic=stochastic,
                generator=generator,
            )
        else:
            assignments = _stratified_assignments(
                group_probability,
                total,
                stochastic=stochastic,
                generator=generator,
            )
        result[rows] = assignments.reshape(rows.numel(), num_samples)
    return result


__all__ = ["sample_grouped_child_indices"]

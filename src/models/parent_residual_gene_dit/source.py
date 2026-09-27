"""Target-free prior source construction for Parent-Residual Gene-DiT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

import torch


@dataclass(frozen=True)
class ParentResidualSource:
    """A prior-centred source batch.

    Shapes are ``source/mean/residual/std: [B,S,G]``, ``center: [B,G]``,
    ``prior_probs/mask: [B,K]``, and ``selected_indices: [B,S]``.
    ``mean`` is the deterministic part of ``source`` and ``std`` already
    includes ``beta_std``.
    """

    source: torch.Tensor
    mean: torch.Tensor
    center: torch.Tensor
    residual: torch.Tensor
    std: torch.Tensor
    prior_probs: torch.Tensor
    mask: torch.Tensor
    selected_indices: torch.Tensor


def _floating_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value


def _integer_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    return value


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_inputs(
    parent_mean: torch.Tensor,
    prototypes: torch.Tensor,
    stds: torch.Tensor,
    prior_probs: torch.Tensor,
    bank_inverse: torch.Tensor,
    selected_indices: torch.Tensor,
    child_mask: Optional[torch.Tensor],
):
    parent_mean = _floating_tensor("parent_mean", parent_mean)
    prototypes = _floating_tensor("prototypes", prototypes)
    stds = _floating_tensor("stds", stds)
    prior_probs = _floating_tensor("prior_probs", prior_probs)
    bank_inverse = _integer_tensor("bank_inverse", bank_inverse)
    selected_indices = _integer_tensor("selected_indices", selected_indices)

    if parent_mean.ndim != 2:
        raise ValueError("parent_mean must have shape [B,G]")
    if prototypes.ndim != 3:
        raise ValueError("prototypes must have shape [U,K,G]")
    if stds.shape != prototypes.shape:
        raise ValueError("stds must have the same [U,K,G] shape as prototypes")

    batch_size, genes = parent_mean.shape
    unique_banks, children, prototype_genes = prototypes.shape
    if genes != prototype_genes:
        raise ValueError("parent_mean and prototypes must share gene dimension G")
    if prior_probs.shape != (batch_size, children):
        raise ValueError("prior_probs must have shape [B,K]")
    if bank_inverse.shape != (batch_size,):
        raise ValueError("bank_inverse must have shape [B]")
    if selected_indices.ndim != 2 or selected_indices.shape[0] != batch_size:
        raise ValueError("selected_indices must have shape [B,S]")
    if selected_indices.shape[1] < 1:
        raise ValueError("selected_indices must contain at least one source cell")

    floating = (parent_mean, prototypes, stds, prior_probs)
    if any(value.device != parent_mean.device for value in floating[1:]):
        raise ValueError("all floating-point inputs must share one device")
    if any(value.dtype != parent_mean.dtype for value in floating[1:]):
        raise TypeError("all floating-point inputs must share one dtype")
    if bank_inverse.device != parent_mean.device:
        raise ValueError("bank_inverse must share the input device")
    if selected_indices.device != parent_mean.device:
        raise ValueError("selected_indices must share the input device")

    for name, value in (
        ("parent_mean", parent_mean),
        ("prototypes", prototypes),
        ("stds", stds),
        ("prior_probs", prior_probs),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if (stds < 0).any():
        raise ValueError("stds cannot be negative")
    if (prior_probs < 0).any():
        raise ValueError("prior_probs cannot be negative")
    if unique_banks < 1 or children < 1 or genes < 1:
        raise ValueError("B, U, K, and G dimensions must all be non-empty")
    if ((bank_inverse < 0) | (bank_inverse >= unique_banks)).any():
        raise ValueError("bank_inverse contains an out-of-range bank row")
    if ((selected_indices < 0) | (selected_indices >= children)).any():
        raise ValueError("selected_indices contains an out-of-range Child")

    if child_mask is None:
        child_mask = torch.ones(
            batch_size,
            children,
            dtype=torch.bool,
            device=parent_mean.device,
        )
    else:
        if not torch.is_tensor(child_mask):
            raise TypeError("child_mask must be a torch.Tensor")
        if child_mask.dtype != torch.bool:
            raise TypeError("child_mask must have dtype torch.bool")
        if child_mask.shape != (batch_size, children):
            raise ValueError("child_mask must have shape [B,K]")
        if child_mask.device != parent_mean.device:
            raise ValueError("child_mask must share the input device")
    if not child_mask.any(dim=1).all():
        raise ValueError("every row of child_mask needs at least one active Child")
    if (prior_probs.masked_select(~child_mask) != 0).any():
        raise ValueError("prior_probs assigns mass to a padded Child")
    if not child_mask.gather(1, selected_indices).all():
        raise ValueError("selected_indices selects a padded Child")

    masked_probs = prior_probs.masked_fill(~child_mask, 0.0)
    probability_mass = masked_probs.sum(dim=1, keepdim=True)
    if (probability_mass <= 0).any():
        raise ValueError("each row of prior_probs must have positive active mass")
    normalized_probs = masked_probs / probability_mass
    return (
        parent_mean,
        prototypes,
        stds,
        normalized_probs,
        bank_inverse,
        selected_indices,
        child_mask,
    )


def _banked_center(
    prototypes: torch.Tensor,
    prior_probs: torch.Tensor,
    bank_inverse: torch.Tensor,
) -> torch.Tensor:
    """Compute [B,G] means without materializing a potentially huge [B,K,G]."""

    row_chunks = []
    center_chunks = []
    for bank_row in torch.unique(bank_inverse, sorted=True):
        rows = torch.nonzero(bank_inverse == bank_row, as_tuple=False).flatten()
        row_chunks.append(rows)
        center_chunks.append(prior_probs.index_select(0, rows) @ prototypes[bank_row])
    grouped_rows = torch.cat(row_chunks, dim=0)
    grouped_centers = torch.cat(center_chunks, dim=0)
    return grouped_centers.index_select(0, torch.argsort(grouped_rows))


def _validate_epsilon(
    epsilon: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    epsilon = _floating_tensor("epsilon", epsilon)
    if epsilon.shape != expected.shape:
        raise ValueError("epsilon must have shape [B,S,G]")
    if epsilon.device != expected.device:
        raise ValueError("epsilon must share the source device")
    if epsilon.dtype != expected.dtype:
        raise TypeError("epsilon must share the source dtype")
    if not torch.isfinite(epsilon).all():
        raise ValueError("epsilon must be finite")
    return epsilon


def _sample_epsilon(
    expected: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        generator_device = torch.device(generator.device)
        if generator_device.type != expected.device.type:
            raise ValueError("generator and source must use the same device type")
        if (
            generator_device.type == "cuda"
            and generator_device.index is not None
            and expected.device.index is not None
            and generator_device.index != expected.device.index
        ):
            raise ValueError("generator and source must use the same CUDA device")
    return torch.randn(
        expected.shape,
        dtype=expected.dtype,
        device=expected.device,
        generator=generator,
    )


def build_parent_residual_source(
    *,
    parent_mean: torch.Tensor,
    prototypes: torch.Tensor,
    stds: torch.Tensor,
    prior_probs: torch.Tensor,
    bank_inverse: torch.Tensor,
    selected_indices: torch.Tensor,
    child_mask: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    beta_std: float = 1.0,
    epsilon: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> ParentResidualSource:
    """Build the common training/inference source from a target-free prior.

    For row ``b`` and selected Child ``k`` this computes

    ``center_b = sum_j p_bj * prototype_bank(b),j``

    ``residual_bk = prototype_bank(b),k - center_b``

    ``source_bsk = stopgrad(parent_mean_b) + beta * residual_bk
                   + beta_std * std_bk * epsilon_bsk``.

    Thus the complete Child residual bank has probability-weighted mean zero.
    Probabilities are normalized after masking.  Passing a fixed ``epsilon``
    or a newly seeded ``generator`` makes construction exactly reproducible.
    The Parent term is always detached; residual/router paths remain
    differentiable.
    """

    beta = _finite_scalar("beta", beta)
    beta_std = _finite_scalar("beta_std", beta_std)
    (
        parent_mean,
        prototypes,
        stds,
        prior_probs,
        bank_inverse,
        selected_indices,
        child_mask,
    ) = _validate_inputs(
        parent_mean,
        prototypes,
        stds,
        prior_probs,
        bank_inverse,
        selected_indices,
        child_mask,
    )

    center = _banked_center(prototypes, prior_probs, bank_inverse)
    bank_rows = bank_inverse[:, None].expand_as(selected_indices)
    selected_prototypes = prototypes[bank_rows, selected_indices]
    selected_stds = stds[bank_rows, selected_indices]
    residual = selected_prototypes - center[:, None, :]
    parent_base = parent_mean.detach()[:, None, :]
    mean = parent_base + beta * residual
    scaled_std = beta_std * selected_stds

    if epsilon is None:
        if beta_std == 0.0:
            epsilon = torch.zeros_like(mean)
        else:
            epsilon = _sample_epsilon(mean, generator)
    else:
        if generator is not None:
            raise ValueError("pass either epsilon or generator, not both")
        epsilon = _validate_epsilon(epsilon, mean)
    source = mean + scaled_std * epsilon
    if not torch.isfinite(source).all():
        raise ValueError("constructed source contains non-finite values")

    return ParentResidualSource(
        source=source,
        mean=mean,
        center=center,
        residual=residual,
        std=scaled_std,
        prior_probs=prior_probs,
        mask=child_mask,
        selected_indices=selected_indices,
    )


# Explicit semantic alias for integration code that calls this a prior source.
build_parent_residual_prior_source = build_parent_residual_source


__all__ = [
    "ParentResidualSource",
    "build_parent_residual_prior_source",
    "build_parent_residual_source",
]

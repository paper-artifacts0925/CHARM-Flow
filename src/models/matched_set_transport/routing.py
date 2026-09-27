"""Posterior-to-prior source routing for matched flow training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from .config import PosteriorPriorMixConfig


@dataclass(frozen=True)
class MixedChildSelection:
    """A discrete Child assignment used to construct the actual FM source.

    All index/mask tensors have shape ``[B,S]``.  ``posterior_mask`` is true
    exactly where the selected index came from target-aware matching rather
    than the inference-time Router.  ``mixture_distribution`` is diagnostic;
    the hard ``indices`` tensor is what must be passed to the source builder.
    """

    indices: torch.Tensor
    posterior_indices: torch.Tensor
    prior_indices: torch.Tensor
    posterior_mask: torch.Tensor
    posterior_eligible: torch.Tensor
    posterior_fraction: torch.Tensor
    mixture_distribution: torch.Tensor
    posterior_distribution: torch.Tensor
    prior_distribution: torch.Tensor

    @property
    def realized_posterior_fraction(self) -> torch.Tensor:
        if self.posterior_mask.numel() == 0:
            return self.posterior_fraction.new_zeros(())
        return self.posterior_mask.float().mean()


def posterior_fraction(
    step: int,
    config: PosteriorPriorMixConfig,
) -> float:
    """Return the target-aware source fraction at one optimizer step."""

    config.validate()
    step = int(step)
    if step < 0:
        raise ValueError("step must be non-negative")
    if step <= int(config.transition_start_step):
        return float(config.start_fraction)
    if step >= int(config.transition_end_step):
        return float(config.end_fraction)
    progress = (
        (step - int(config.transition_start_step))
        / (int(config.transition_end_step) - int(config.transition_start_step))
    )
    if config.curve == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(config.start_fraction) + progress * (
        float(config.end_fraction) - float(config.start_fraction)
    )


def _validate_probability_inputs(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    child_mask: Optional[torch.Tensor],
):
    if not torch.is_tensor(posterior) or not torch.is_tensor(prior):
        raise TypeError("posterior and prior must be torch tensors")
    if not posterior.is_floating_point() or not prior.is_floating_point():
        raise TypeError("posterior and prior must be floating point")
    if posterior.ndim != 2 or posterior.shape != prior.shape:
        raise ValueError("posterior and prior must share shape [B,K]")
    if posterior.device != prior.device:
        raise ValueError("posterior and prior must share a device")
    if not torch.isfinite(posterior).all() or not torch.isfinite(prior).all():
        raise ValueError("posterior and prior must be finite")
    if (posterior < 0).any() or (prior < 0).any():
        raise ValueError("posterior and prior cannot contain negative mass")
    if posterior.shape[0] < 1 or posterior.shape[1] < 1:
        raise ValueError("posterior and prior cannot be empty")
    if child_mask is None:
        child_mask = torch.ones_like(posterior, dtype=torch.bool)
    else:
        if not torch.is_tensor(child_mask) or child_mask.dtype != torch.bool:
            raise TypeError("child_mask must be a boolean tensor")
        if child_mask.shape != posterior.shape:
            raise ValueError("child_mask must have shape [B,K]")
        if child_mask.device != posterior.device:
            raise ValueError("child_mask must share the probability device")
    if not child_mask.any(dim=-1).all():
        raise ValueError("every row needs at least one active Child")
    if (prior.masked_select(~child_mask) != 0).any():
        raise ValueError("prior assigns mass to a masked Child")
    if (posterior.masked_select(~child_mask) != 0).any():
        raise ValueError("posterior assigns mass to a masked Child")
    return child_mask


def _normalized_probabilities(
    probability: torch.Tensor,
    child_mask: torch.Tensor,
):
    masked = probability.masked_fill(~child_mask, 0.0)
    mass = masked.sum(dim=-1, keepdim=True)
    normalized = masked / mass.clamp_min(torch.finfo(masked.dtype).tiny)
    return normalized, mass[:, 0] > 0


def _categorical_indices(
    probability: torch.Tensor,
    num_samples: int,
    stochastic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if stochastic:
        return torch.multinomial(
            probability.float(),
            num_samples=int(num_samples),
            replacement=True,
            generator=generator,
        )
    return probability.argmax(dim=-1, keepdim=True).expand(-1, int(num_samples))


@torch.no_grad()
def select_mixed_child_indices(
    *,
    posterior: torch.Tensor,
    prior: torch.Tensor,
    step: int,
    config: PosteriorPriorMixConfig,
    num_samples: int = 1,
    child_mask: Optional[torch.Tensor] = None,
    stochastic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> MixedChildSelection:
    """Select the Child indices that will form the FM source.

    Matching and Router distributions are sampled separately.  An independent
    Bernoulli gate for each response cell chooses between them, so assignment
    of one cell never consumes or suppresses a token for another cell.  Rows
    without posterior mass safely fall back to the prior.
    """

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    child_mask = _validate_probability_inputs(posterior, prior, child_mask)
    posterior = posterior.detach()
    prior = prior.detach()
    normalized_prior, prior_valid = _normalized_probabilities(prior, child_mask)
    if not prior_valid.all():
        raise ValueError("every prior row must have positive active mass")
    normalized_posterior, posterior_valid = _normalized_probabilities(
        posterior, child_mask
    )
    # torch.multinomial rejects zero-mass rows even when their draw will later
    # be discarded.  Replacing only those rows with the prior preserves the
    # explicit fallback semantics.
    safe_posterior = torch.where(
        posterior_valid[:, None], normalized_posterior, normalized_prior
    )
    posterior_indices = _categorical_indices(
        safe_posterior, num_samples, bool(stochastic), generator
    )
    prior_indices = _categorical_indices(
        normalized_prior, num_samples, bool(stochastic), generator
    )

    fraction = posterior_fraction(step, config)
    fraction_tensor = prior.new_tensor(fraction, dtype=torch.float32)
    if fraction <= 0.0:
        use_posterior = torch.zeros(
            posterior.shape[0], num_samples, dtype=torch.bool, device=prior.device
        )
    elif fraction >= 1.0:
        use_posterior = torch.ones(
            posterior.shape[0], num_samples, dtype=torch.bool, device=prior.device
        )
    else:
        use_posterior = torch.rand(
            posterior.shape[0],
            num_samples,
            device=prior.device,
            generator=generator,
        ) < fraction
    use_posterior = use_posterior & posterior_valid[:, None]
    indices = torch.where(use_posterior, posterior_indices, prior_indices)

    row_fraction = posterior_valid.to(normalized_prior.dtype)[:, None] * fraction
    mixture = (
        row_fraction * normalized_posterior
        + (1.0 - row_fraction) * normalized_prior
    )
    return MixedChildSelection(
        indices=indices,
        posterior_indices=posterior_indices,
        prior_indices=prior_indices,
        posterior_mask=use_posterior,
        posterior_eligible=posterior_valid,
        posterior_fraction=fraction_tensor,
        mixture_distribution=mixture,
        posterior_distribution=normalized_posterior,
        prior_distribution=normalized_prior,
    )


def posterior_prior_kl(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    *,
    child_mask: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distil a detached matching posterior into the target-free Router."""

    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    child_mask = _validate_probability_inputs(posterior, prior, child_mask)
    posterior_normalized, posterior_valid = _normalized_probabilities(
        posterior.detach(), child_mask
    )
    prior_normalized, prior_valid = _normalized_probabilities(prior, child_mask)
    if not prior_valid.all():
        raise ValueError("every prior row must have positive active mass")
    valid = posterior_valid
    if row_mask is not None:
        if not torch.is_tensor(row_mask):
            raise TypeError("row_mask must be a torch tensor")
        if row_mask.shape != (posterior.shape[0],):
            raise ValueError("row_mask must have shape [B]")
        valid = valid & row_mask.to(device=posterior.device, dtype=torch.bool)
    log_ratio = (
        posterior_normalized.clamp_min(float(eps)).log()
        - prior_normalized.clamp_min(float(eps)).log()
    )
    per_row = (posterior_normalized * log_ratio).sum(dim=-1)
    weights = valid.to(per_row.dtype)
    return (per_row * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = [
    "MixedChildSelection",
    "posterior_fraction",
    "posterior_prior_kl",
    "select_mixed_child_indices",
]

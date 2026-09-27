"""Differentiable consistency across raw-batch views of one perturbation."""

from __future__ import annotations

import torch


def episode_distribution_consistency(distributions, episode_ids, eps=1e-8):
    """Return Jensen-Shannon consistency for views in the same episode.

    ``distributions`` contains the differentiable Child posterior of every
    set-view. Hard endpoint coupling remains Hungarian; this soft companion is
    used only to let A+B, A+C, B+C observations train a shared routing rule.
    """
    if distributions.ndim != 2:
        raise ValueError("distributions must have shape [num_views,K]")
    episode_ids = episode_ids.to(
        device=distributions.device, dtype=torch.long
    ).reshape(-1)
    if episode_ids.shape[0] != distributions.shape[0]:
        raise ValueError("episode_ids must contain one ID per set-view")

    terms = []
    for episode_id in torch.unique(episode_ids):
        members = distributions[episode_ids == episode_id]
        if members.shape[0] < 2:
            continue
        members = members.clamp_min(float(eps))
        members = members / members.sum(dim=-1, keepdim=True)
        center = members.mean(dim=0).clamp_min(float(eps))
        terms.append(
            (members * (members.log() - center.log())).sum(dim=-1).mean()
        )
    if not terms:
        return distributions.new_zeros(())
    return torch.stack(terms).mean()


__all__ = ["episode_distribution_consistency"]

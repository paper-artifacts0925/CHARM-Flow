"""State-preserving A/B/C consistency objectives for adaptive anchors."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.models.adaptive_state_anchor import safe_root_mean_square


def state_conditioned_pair_consistency(
    anchor_delta: torch.Tensor,
    responsibilities: torch.Tensor,
    valid_cell_mask: torch.Tensor,
    source_mask: torch.Tensor,
    num_views: int = 3,
    min_occupancy: float = 0.05,
    episode_ids: torch.Tensor | None = None,
    view_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare delta direction/magnitude across fixed or ragged batch views.

    ``responsibilities`` are treated as an EM assignment, not a differentiable
    shortcut.  Each view first forms one response signature per state; pairwise
    terms then use the smaller occupancy in the two views as confidence.  This
    keeps state identity instead of collapsing all anchors into a single
    population mean and avoids forcing genuinely different state proportions
    to match.

    With ``episode_ids`` and ``view_ids``, each episode may contain two to five
    views. All within-episode view pairs are used, pairs are averaged inside
    each episode, and episodes are then averaged equally. Omitting both IDs
    preserves the legacy fixed-``num_views`` reduction exactly.

    Returns the scalar loss and the mean number of confidently shared states
    per view pair.
    """
    if anchor_delta.ndim != 4:
        raise ValueError("anchor_delta must have shape [B,K,S,G]")
    if responsibilities.ndim != 3:
        raise ValueError("responsibilities must have shape [B,S,K]")
    batch_size, num_states, set_size, _ = anchor_delta.shape
    if responsibilities.shape != (batch_size, set_size, num_states):
        raise ValueError("responsibilities do not match anchor_delta")
    if valid_cell_mask.shape != (batch_size, set_size):
        raise ValueError("valid_cell_mask must have shape [B,S]")
    if source_mask.shape != (batch_size, num_states):
        raise ValueError("source_mask must have shape [B,K]")
    if (episode_ids is None) != (view_ids is None):
        raise ValueError("episode_ids and view_ids must be provided together")
    explicit_views = episode_ids is not None
    if not explicit_views and (
        int(num_views) < 2 or batch_size % int(num_views) != 0
    ):
        raise ValueError("batch size must contain complete multi-view episodes")

    q_bks = responsibilities.detach().float().permute(0, 2, 1)
    valid = valid_cell_mask.to(device=anchor_delta.device, dtype=torch.float32)
    state_cell_weight = q_bks * valid.unsqueeze(1)
    occupancy_count = state_cell_weight.sum(dim=-1)
    valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    occupancy = occupancy_count / valid_count
    state_signature = (
        state_cell_weight.unsqueeze(-1) * anchor_delta.float()
    ).sum(dim=2) / occupancy_count.clamp_min(1e-8).unsqueeze(-1)

    if explicit_views:
        episode_ids = episode_ids.detach().to(
            device=anchor_delta.device, dtype=torch.long
        ).reshape(-1)
        view_ids = view_ids.detach().to(
            device=anchor_delta.device, dtype=torch.long
        ).reshape(-1)
        if episode_ids.numel() != batch_size or view_ids.numel() != batch_size:
            raise ValueError("episode_ids and view_ids must each contain B values")

        episode_losses = []
        episode_active_states = []
        for episode_id in torch.unique(episode_ids, sorted=True):
            episode_indices = torch.nonzero(
                episode_ids == episode_id, as_tuple=False
            ).flatten()
            episode_view_ids = view_ids.index_select(0, episode_indices)
            unique_views, view_counts = torch.unique(
                episode_view_ids, sorted=True, return_counts=True
            )
            episode_num_views = int(unique_views.numel())
            if not view_counts.eq(1).all():
                raise ValueError(
                    "each (episode_id, view_id) pair must identify exactly one view"
                )
            if not 2 <= episode_num_views <= 5:
                raise ValueError("every episode must contain between 2 and 5 views")
            view_order = torch.argsort(episode_view_ids, stable=True)
            episode_indices = episode_indices.index_select(0, view_order)
            episode_signatures = state_signature.index_select(0, episode_indices)
            episode_occupancies = occupancy.index_select(0, episode_indices)
            episode_masks = source_mask.bool().index_select(0, episode_indices)

            view_pair_losses = []
            view_pair_active_states = []
            for left in range(episode_num_views):
                for right in range(left + 1, episode_num_views):
                    left_signature = episode_signatures[left]
                    right_signature = episode_signatures[right]
                    left_centered = left_signature - left_signature.mean(
                        dim=-1, keepdim=True
                    )
                    right_centered = right_signature - right_signature.mean(
                        dim=-1, keepdim=True
                    )
                    left_magnitude = safe_root_mean_square(left_centered, dim=-1)
                    right_magnitude = safe_root_mean_square(right_centered, dim=-1)
                    direction_defined = (left_magnitude > 1e-6) & (
                        right_magnitude > 1e-6
                    )
                    direction = 1.0 - F.cosine_similarity(
                        left_centered, right_centered, dim=-1, eps=1e-6
                    )
                    direction = torch.where(
                        direction_defined, direction, torch.zeros_like(direction)
                    )
                    magnitude = F.smooth_l1_loss(
                        torch.log1p(left_magnitude),
                        torch.log1p(right_magnitude),
                        reduction="none",
                    )
                    confidence = torch.minimum(
                        episode_occupancies[left], episode_occupancies[right]
                    )
                    shared = (
                        episode_masks[left]
                        & episode_masks[right]
                        & (confidence >= float(min_occupancy))
                    )
                    confidence = torch.where(
                        shared, confidence, torch.zeros_like(confidence)
                    )
                    view_pair_losses.append(
                        ((direction + magnitude) * confidence).sum()
                        / confidence.sum().clamp_min(1e-8)
                    )
                    view_pair_active_states.append(shared.float().sum())

            episode_losses.append(torch.stack(view_pair_losses).mean())
            episode_active_states.append(
                torch.stack(view_pair_active_states).mean()
            )

        return (
            torch.stack(episode_losses).mean(),
            torch.stack(episode_active_states).mean(),
        )

    episodes = batch_size // int(num_views)
    signatures = state_signature.reshape(
        episodes, int(num_views), num_states, -1
    )
    occupancies = occupancy.reshape(episodes, int(num_views), num_states)
    masks = source_mask.bool().reshape(episodes, int(num_views), num_states)

    pair_losses = []
    pair_active_states = []
    for left in range(int(num_views)):
        for right in range(left + 1, int(num_views)):
            left_signature = signatures[:, left]
            right_signature = signatures[:, right]
            left_centered = left_signature - left_signature.mean(
                dim=-1, keepdim=True
            )
            right_centered = right_signature - right_signature.mean(
                dim=-1, keepdim=True
            )
            left_magnitude = safe_root_mean_square(left_centered, dim=-1)
            right_magnitude = safe_root_mean_square(right_centered, dim=-1)
            direction_defined = (left_magnitude > 1e-6) & (
                right_magnitude > 1e-6
            )
            direction = 1.0 - F.cosine_similarity(
                left_centered,
                right_centered,
                dim=-1,
                eps=1e-6,
            )
            direction = torch.where(
                direction_defined, direction, torch.zeros_like(direction)
            )
            magnitude = F.smooth_l1_loss(
                torch.log1p(left_magnitude),
                torch.log1p(right_magnitude),
                reduction="none",
            )

            confidence = torch.minimum(
                occupancies[:, left], occupancies[:, right]
            )
            shared = (
                masks[:, left]
                & masks[:, right]
                & (confidence >= float(min_occupancy))
            )
            confidence = torch.where(
                shared, confidence, torch.zeros_like(confidence)
            )
            pair_losses.append(
                ((direction + magnitude) * confidence).sum()
                / confidence.sum().clamp_min(1e-8)
            )
            pair_active_states.append(shared.float().sum(dim=-1).mean())

    return (
        torch.stack(pair_losses).mean(),
        torch.stack(pair_active_states).mean(),
    )


__all__ = ["state_conditioned_pair_consistency"]

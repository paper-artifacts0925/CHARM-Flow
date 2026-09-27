"""Pure matched hierarchical reconstruction objective."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MatchedHierarchicalReconstructionResult:
    loss: Tensor
    charbonnier: Tensor
    rmse: Tensor
    cosine: Tensor
    reconstruction: Tensor
    matched_child_mean: Tensor
    centered_child: Tensor
    condition_child_mean: Tensor
    unique_condition_ids: Tensor
    valid_cell_count: Tensor
    positive_match_count: Tensor
    condition_count: Tensor
    positive_matches_per_cell_mean: Tensor
    positive_matches_per_cell_min: Tensor
    positive_matches_per_cell_max: Tensor
    cosine_weight_mean: Tensor
    centered_group_mean_max_abs: Tensor


def matched_hierarchical_reconstruction(
    parent_endpoint: Tensor,
    matched_child_endpoint: Tensor,
    target: Tensor,
    condition_group_ids: Tensor,
    positive_mask: Tensor,
    row_valid: Tensor,
    *,
    charbonnier_weight: float = 0.4,
    rmse_weight: float = 0.35,
    cosine_weight: float = 0.25,
    charbonnier_eps: float = 1e-3,
    rmse_eps: float = 1e-8,
    cosine_eps: float = 1e-8,
    cosine_norm_scale: float = 1.0,
) -> MatchedHierarchicalReconstructionResult:
    """Reconstruct each valid target once from Parent plus centered Child."""
    if parent_endpoint.ndim != 2:
        raise ValueError("parent_endpoint must have shape [B, G]")
    b, g = parent_endpoint.shape
    if matched_child_endpoint.ndim != 3 or matched_child_endpoint.shape[:1] != (b,) or matched_child_endpoint.shape[2] != g:
        raise ValueError("matched_child_endpoint must have shape [B, R, G]")
    r = matched_child_endpoint.shape[1]
    if target.shape != (b, 1, g):
        raise ValueError("target must have shape [B, 1, G]")
    if condition_group_ids.shape != (b,):
        raise ValueError("condition_group_ids must have shape [B]")
    if positive_mask.shape != (b, r):
        raise ValueError("positive_mask must have shape [B, R]")
    if row_valid.shape != (b,):
        raise ValueError("row_valid must have shape [B]")
    if positive_mask.dtype != torch.bool or row_valid.dtype != torch.bool:
        raise TypeError("positive_mask and row_valid must be boolean")
    if condition_group_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("condition_group_ids must be integer")
    tensors = (matched_child_endpoint, target, condition_group_ids, positive_mask, row_valid)
    if any(x.device != parent_endpoint.device for x in tensors):
        raise ValueError("all inputs must be on the same device")
    if not (charbonnier_eps > 0 and rmse_eps > 0 and cosine_eps > 0 and cosine_norm_scale > 0):
        raise ValueError("epsilon and cosine_norm_scale values must be positive")
    if min(charbonnier_weight, rmse_weight, cosine_weight) < 0:
        raise ValueError("loss weights must be nonnegative")

    valid_positions = row_valid.nonzero(as_tuple=False).squeeze(1)
    if valid_positions.numel() == 0:
        raise ValueError("at least one row must be valid")
    pair_mask = row_valid[:, None] & positive_mask
    row_counts = pair_mask.sum(dim=1)
    if torch.any(row_counts[row_valid] == 0):
        raise ValueError("every valid row must have at least one positive match")
    if torch.any(condition_group_ids[row_valid] < 0):
        raise ValueError("valid condition group ids must be nonnegative")

    safe_child = torch.where(pair_mask[..., None], matched_child_endpoint, torch.zeros_like(matched_child_endpoint))
    child_mean = safe_child.sum(dim=1) / row_counts.clamp_min(1).to(parent_endpoint.dtype)[:, None]
    valid_child = child_mean.index_select(0, valid_positions)
    valid_groups = condition_group_ids.index_select(0, valid_positions)
    unique_groups, inverse = torch.unique(valid_groups, sorted=True, return_inverse=True)
    group_sum = valid_child.new_zeros((unique_groups.numel(), g)).index_add(0, inverse, valid_child)
    group_count = torch.bincount(inverse, minlength=unique_groups.numel())
    group_mean = group_sum / group_count.to(valid_child.dtype)[:, None]
    centered_valid = valid_child - group_mean.index_select(0, inverse)
    centered_child = torch.zeros_like(parent_endpoint).index_copy(0, valid_positions, centered_valid)
    reconstruction = parent_endpoint + centered_child

    valid_reconstruction = reconstruction.index_select(0, valid_positions)
    valid_target = target[:, 0, :].index_select(0, valid_positions).detach()
    if not torch.isfinite(valid_reconstruction).all() or not torch.isfinite(valid_target).all():
        raise ValueError("effective predictions and targets must be finite")
    error = (valid_reconstruction - valid_target).float()
    charbonnier_per_cell = torch.sqrt(error.square() + charbonnier_eps**2).mean(dim=-1) - charbonnier_eps
    rmse_per_cell = torch.sqrt(error.square().mean(dim=-1) + rmse_eps**2) - rmse_eps
    prediction32 = valid_reconstruction.float()
    target32 = valid_target.float()
    prediction_norm = torch.linalg.vector_norm(prediction32, dim=-1)
    target_norm = torch.linalg.vector_norm(target32, dim=-1)
    similarity = (prediction32 * target32).sum(dim=-1) / (
        prediction_norm.clamp_min(cosine_eps) * target_norm.clamp_min(cosine_eps)
    )
    similarity = similarity.clamp(-1.0, 1.0)
    norm_weight = target_norm / (target_norm + cosine_norm_scale)
    charbonnier = charbonnier_per_cell.mean()
    rmse = rmse_per_cell.mean()
    cosine = (norm_weight * (1.0 - similarity)).mean()
    loss = charbonnier_weight * charbonnier + rmse_weight * rmse + cosine_weight * cosine

    centered_sum = centered_valid.new_zeros(group_mean.shape).index_add(0, inverse, centered_valid)
    centered_group_mean = centered_sum / group_count.to(centered_valid.dtype)[:, None]
    valid_counts = row_counts[row_valid].float()
    return MatchedHierarchicalReconstructionResult(
        loss=loss,
        charbonnier=charbonnier,
        rmse=rmse,
        cosine=cosine,
        reconstruction=reconstruction,
        matched_child_mean=child_mean,
        centered_child=centered_child,
        condition_child_mean=group_mean,
        unique_condition_ids=unique_groups,
        valid_cell_count=row_valid.sum(),
        positive_match_count=pair_mask.sum(),
        condition_count=unique_groups.new_tensor(unique_groups.numel()),
        positive_matches_per_cell_mean=valid_counts.mean(),
        positive_matches_per_cell_min=valid_counts.min(),
        positive_matches_per_cell_max=valid_counts.max(),
        cosine_weight_mean=norm_weight.mean(),
        centered_group_mean_max_abs=centered_group_mean.abs().max(),
    )

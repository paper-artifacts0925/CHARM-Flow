"""Low-dimensional, permutation-invariant supervision for generated cell sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .config import ResidualSetLossConfig


@dataclass(frozen=True)
class ResidualSetLoss:
    """Scalar objectives and collapse diagnostics aggregated over conditions."""

    total: torch.Tensor
    energy: torch.Tensor
    log_variance: torch.Tensor
    correlation: torch.Tensor
    mean: torch.Tensor
    variance_ratio: torch.Tensor
    valid_groups: torch.Tensor
    mean_group_size: torch.Tensor


def _flatten_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
):
    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch tensors")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share shape [...,G]")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if not torch.is_tensor(group_ids):
        raise TypeError("group_ids must be a torch tensor")
    if group_ids.device != prediction.device:
        raise ValueError("group_ids must share the prediction device")

    leading = prediction.shape[:-1]
    if tuple(group_ids.shape) == tuple(leading):
        flat_groups = group_ids.reshape(-1, 1)
    elif group_ids.ndim >= 2 and tuple(group_ids.shape[:-1]) == tuple(leading):
        flat_groups = group_ids.reshape(-1, group_ids.shape[-1])
    else:
        raise ValueError(
            "group_ids must have shape [...] or [...,C] matching prediction"
        )
    if flat_groups.is_floating_point() and not torch.isfinite(flat_groups).all():
        raise ValueError("group_ids must be finite")

    if valid_mask is None:
        flat_valid = torch.ones(
            prediction.numel() // prediction.shape[-1],
            dtype=torch.bool,
            device=prediction.device,
        )
    else:
        if not torch.is_tensor(valid_mask):
            raise TypeError("valid_mask must be a torch tensor")
        if tuple(valid_mask.shape) != tuple(leading):
            raise ValueError("valid_mask must match prediction leading dimensions")
        if valid_mask.device != prediction.device:
            raise ValueError("valid_mask must share the prediction device")
        flat_valid = valid_mask.reshape(-1).to(dtype=torch.bool)
    if not flat_valid.any():
        return (
            prediction.reshape(-1, prediction.shape[-1])[:0],
            target.reshape(-1, target.shape[-1])[:0],
            flat_groups[:0],
        )
    return (
        prediction.reshape(-1, prediction.shape[-1])[flat_valid],
        target.reshape(-1, target.shape[-1])[flat_valid],
        flat_groups[flat_valid],
    )


def _project(
    value: torch.Tensor,
    projection: Optional[torch.Tensor],
    whitening_scale: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    if projection is None:
        projected = value
    else:
        if not torch.is_tensor(projection) or not projection.is_floating_point():
            raise TypeError("projection must be a floating-point tensor")
        if projection.ndim != 2 or projection.shape[0] != value.shape[-1]:
            raise ValueError("projection must have shape [G,D]")
        if not torch.isfinite(projection).all():
            raise ValueError("projection must be finite")
        projected = value @ projection.detach().to(value)
    if whitening_scale is not None:
        if not torch.is_tensor(whitening_scale) or not whitening_scale.is_floating_point():
            raise TypeError("whitening_scale must be a floating-point tensor")
        if whitening_scale.shape != (projected.shape[-1],):
            raise ValueError("whitening_scale must have shape [D]")
        if not torch.isfinite(whitening_scale).all() or (whitening_scale <= 0).any():
            raise ValueError("whitening_scale must be finite and positive")
        projected = projected / whitening_scale.detach().to(projected).clamp_min(eps)
    return projected


def _quantile_subsample(value: torch.Tensor, max_cells: int) -> torch.Tensor:
    """Order-invariant deterministic cap for the quadratic energy term."""

    if value.shape[0] <= int(max_cells):
        return value
    # Sorting a detached scalar key makes selection permutation invariant in
    # the usual non-tied case while preserving gradients through chosen rows.
    key = value.detach().float().square().mean(dim=-1)
    if value.shape[-1]:
        key = key + 1e-7 * value.detach().float()[:, 0]
    order = torch.argsort(key)
    positions = torch.linspace(
        0,
        value.shape[0] - 1,
        int(max_cells),
        device=value.device,
    ).round().long()
    return value.index_select(0, order.index_select(0, positions))


def _energy_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalize_dimension: bool,
) -> torch.Tensor:
    cross = torch.cdist(prediction.float(), target.float(), p=2).mean()
    pred_self = torch.cdist(prediction.float(), prediction.float(), p=2).mean()
    target_self = torch.cdist(target.float(), target.float(), p=2).mean()
    result = 2.0 * cross - pred_self - target_self
    if normalize_dimension:
        result = result / math.sqrt(float(prediction.shape[-1]))
    # The empirical energy distance is non-negative; clamp only absorbs tiny
    # floating-point violations of that property.
    return result.clamp_min(0.0).to(prediction)


def _correlation_matrix(value: torch.Tensor, eps: float) -> torch.Tensor:
    covariance = value.transpose(0, 1) @ value / float(value.shape[0])
    std = covariance.diagonal().clamp_min(0.0).add(float(eps)).sqrt()
    return covariance / (std[:, None] * std[None, :]).clamp_min(float(eps))


def grouped_residual_set_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    projection: Optional[torch.Tensor] = None,
    whitening_scale: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    config: Optional[ResidualSetLossConfig] = None,
) -> ResidualSetLoss:
    """Compare unordered, condition-homogeneous endpoint residual sets.

    Each condition is centered before the Energy, variance and correlation
    terms are computed.  Consequently these terms cannot move the Parent
    pseudobulk mean.  ``mean_weight`` is available as an explicit, separately
    reported endpoint-mean objective and defaults to zero.

    ``group_ids`` should include every identity needed to prevent accidental
    mixing, normally ``(dataset, cell_line, perturbation)``.  Inputs may be
    ``[N,G]`` or ``[B,S,G]``; no response-to-response pairing is used.
    """

    config = ResidualSetLossConfig() if config is None else config
    config.validate()
    prediction, target, group_ids = _flatten_inputs(
        prediction, target, group_ids, valid_mask
    )
    zero = prediction.sum() * 0.0
    if prediction.shape[0] == 0:
        return ResidualSetLoss(
            total=zero,
            energy=zero,
            log_variance=zero,
            correlation=zero,
            mean=zero,
            variance_ratio=zero.detach(),
            valid_groups=zero.detach(),
            mean_group_size=zero.detach(),
        )

    _, inverse = torch.unique(group_ids, dim=0, return_inverse=True)
    energy_terms = []
    variance_terms = []
    correlation_terms = []
    mean_terms = []
    variance_ratios = []
    group_sizes = []
    for group_index in range(int(inverse.max().item()) + 1):
        rows = torch.nonzero(inverse == group_index, as_tuple=False).flatten()
        if rows.numel() < int(config.min_cells_per_group):
            continue
        pred_group = prediction.index_select(0, rows)
        target_group = target.index_select(0, rows)
        pred_mean = pred_group.mean(dim=0, keepdim=True)
        target_mean = target_group.mean(dim=0, keepdim=True)
        pred_residual = pred_group - pred_mean
        target_residual = target_group - target_mean
        pred_projected = _project(
            pred_residual, projection, whitening_scale, config.eps
        )
        target_projected = _project(
            target_residual, projection, whitening_scale, config.eps
        )
        pred_energy = _quantile_subsample(
            pred_projected, config.max_cells_per_group
        )
        target_energy = _quantile_subsample(
            target_projected, config.max_cells_per_group
        )
        energy_terms.append(
            _energy_distance(
                pred_energy,
                target_energy,
                config.normalize_projection_distance,
            )
        )

        pred_variance = pred_projected.float().square().mean(dim=0)
        target_variance = target_projected.float().square().mean(dim=0)
        variance_terms.append(
            F.smooth_l1_loss(
                (pred_variance + config.eps).log(),
                (target_variance + config.eps).log(),
            ).to(prediction)
        )
        correlation_terms.append(
            F.smooth_l1_loss(
                _correlation_matrix(pred_projected.float(), config.eps),
                _correlation_matrix(target_projected.float(), config.eps),
            ).to(prediction)
        )
        projected_mean_delta = _project(
            pred_mean - target_mean,
            projection,
            whitening_scale,
            config.eps,
        )
        mean_terms.append(projected_mean_delta.float().square().mean().to(prediction))
        variance_ratios.append(
            (
                pred_variance.mean()
                / target_variance.mean().clamp_min(config.eps)
            ).detach().to(prediction)
        )
        group_sizes.append(prediction.new_tensor(float(rows.numel())))

    if not energy_terms:
        return ResidualSetLoss(
            total=zero,
            energy=zero,
            log_variance=zero,
            correlation=zero,
            mean=zero,
            variance_ratio=zero.detach(),
            valid_groups=zero.detach(),
            mean_group_size=zero.detach(),
        )

    energy = torch.stack(energy_terms).mean()
    log_variance = torch.stack(variance_terms).mean()
    correlation = torch.stack(correlation_terms).mean()
    mean = torch.stack(mean_terms).mean()
    total = (
        float(config.energy_weight) * energy
        + float(config.log_variance_weight) * log_variance
        + float(config.correlation_weight) * correlation
        + float(config.mean_weight) * mean
    )
    return ResidualSetLoss(
        total=total,
        energy=energy,
        log_variance=log_variance,
        correlation=correlation,
        mean=mean,
        variance_ratio=torch.stack(variance_ratios).mean(),
        valid_groups=prediction.new_tensor(float(len(energy_terms))),
        mean_group_size=torch.stack(group_sizes).mean(),
    )


__all__ = ["ResidualSetLoss", "grouped_residual_set_loss"]

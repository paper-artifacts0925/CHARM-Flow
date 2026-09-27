"""Shared training/inference coordinates for Parent-locked residual flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .centering import (
    NonnegativeFixedMeanProjection,
    project_group_nonnegative_fixed_mean,
    project_group_zero_mean,
)


@dataclass(frozen=True)
class ParentLockedResidualPair:
    """Residual FM endpoints with one protected effective Parent mean."""

    parent_mean: torch.Tensor
    source_residual: torch.Tensor
    target_residual: torch.Tensor
    source_projection: NonnegativeFixedMeanProjection
    target_projection: NonnegativeFixedMeanProjection


@dataclass(frozen=True)
class ParentLockedComposition:
    """A nonnegative expression and its effective-Parent residual."""

    expression: torch.Tensor
    residual: torch.Tensor
    parent_mean: torch.Tensor
    projection: NonnegativeFixedMeanProjection


@dataclass(frozen=True)
class MeanCarryingTarget:
    """Condition-level endpoint mean used by the Child mean-carrying flow.

    ``supervision_mask`` is true only for valid cells whose condition group has
    at least one observed target mean.  In particular, a supervised singleton
    is a valid mean-training example even though it cannot supervise centred
    within-condition heterogeneity.
    """

    target_mean: torch.Tensor
    mean_shift: torch.Tensor
    supervision_mask: torch.Tensor


def _validate_expression(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,S,G]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _expand_mean(
    name: str,
    mean: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(mean) or not mean.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if mean.device != reference.device:
        raise ValueError(f"{name} must share the expression device")
    if mean.dtype != reference.dtype:
        mean = mean.to(dtype=reference.dtype)
    batch_size, set_size, genes = reference.shape
    if tuple(mean.shape) == (batch_size, genes):
        mean = mean[:, None, :].expand(-1, set_size, -1)
    elif tuple(mean.shape) == (batch_size, 1, genes):
        mean = mean.expand(-1, set_size, -1)
    elif tuple(mean.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} must have shape [B,G], [B,1,G], or [B,S,G]"
        )
    if not torch.isfinite(mean).all():
        raise ValueError(f"{name} must be finite")
    return mean


def _expand_cell_mask(
    name: str,
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size, set_size = reference.shape[:2]
    if mask is None:
        return torch.ones(
            batch_size,
            set_size,
            dtype=torch.bool,
            device=reference.device,
        )
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean tensor")
    if mask.device != reference.device:
        raise ValueError(f"{name} must share the expression device")
    if mask.ndim == 1 and mask.shape[0] == batch_size:
        return mask[:, None].expand(-1, set_size)
    if tuple(mask.shape) != (batch_size, set_size):
        raise ValueError(f"{name} must have shape [B] or [B,S]")
    return mask


def prepare_mean_carrying_target(
    *,
    target_condition_mean: torch.Tensor,
    effective_parent: torch.Tensor,
    reference: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    supervision_mask: Optional[torch.Tensor] = None,
) -> MeanCarryingTarget:
    """Broadcast one supervised Child endpoint mean per condition group.

    Target means are group-averaged over supervised valid cells.  This makes
    the common mode invariant to the ``[B,S,G]`` versus grouped ``[B,1,G]``
    layout.  Groups without mean supervision fall back to their effective
    Parent and receive no common-mode loss.  All returned mean targets are
    detached: the Child flow learns *towards* these coordinates but cannot use
    them to backpropagate into the Parent/lookup path.
    """

    _validate_expression("reference", reference)
    expanded_target = _expand_mean(
        "target_condition_mean", target_condition_mean, reference
    ).detach()
    expanded_parent = _expand_mean(
        "effective_parent", effective_parent, reference
    ).detach()
    topology = project_group_zero_mean(
        torch.zeros_like(reference),
        group_ids=group_ids,
        valid_mask=valid_mask,
        singleton_policy="zero",
    )
    requested = _expand_cell_mask(
        "supervision_mask", supervision_mask, reference
    )
    supervised = topology.valid_mask & requested

    flat_supervised = supervised.reshape(-1)
    valid_positions = torch.nonzero(
        topology.valid_mask.reshape(-1), as_tuple=False
    ).flatten()
    compact = topology.group_ids.reshape(-1).index_select(0, valid_positions)
    supervised_valid = flat_supervised.index_select(0, valid_positions)

    feature_size = reference.shape[-1]
    target_valid = expanded_target.reshape(-1, feature_size).index_select(
        0, valid_positions
    )
    accumulation_dtype = (
        torch.float32
        if target_valid.dtype in (torch.float16, torch.bfloat16)
        else target_valid.dtype
    )
    weighted_target = target_valid.to(accumulation_dtype) * supervised_valid[
        :, None
    ].to(accumulation_dtype)
    group_sums = torch.zeros(
        topology.counts.numel(),
        feature_size,
        device=reference.device,
        dtype=accumulation_dtype,
    ).index_add(0, compact, weighted_target)
    group_supervised_counts = torch.zeros(
        topology.counts.numel(),
        device=reference.device,
        dtype=accumulation_dtype,
    ).index_add(0, compact, supervised_valid.to(accumulation_dtype))
    group_has_target = group_supervised_counts > 0
    group_targets = group_sums / group_supervised_counts.clamp_min(1)[:, None]

    parent_valid = expanded_parent.reshape(-1, feature_size).index_select(
        0, valid_positions
    )
    selected_target = torch.where(
        group_has_target.index_select(0, compact)[:, None],
        group_targets.index_select(0, compact).to(parent_valid),
        parent_valid,
    )
    target_flat = torch.zeros_like(expanded_parent.reshape(-1, feature_size))
    target_flat = target_flat.index_copy(0, valid_positions, selected_target)
    target_mean = target_flat.reshape_as(expanded_parent)
    group_is_supervised = group_has_target.index_select(0, compact)
    output_mask_flat = torch.zeros_like(topology.valid_mask.reshape(-1))
    output_mask_flat = output_mask_flat.index_copy(
        0, valid_positions, group_is_supervised
    )
    output_mask = output_mask_flat.reshape_as(topology.valid_mask)
    return MeanCarryingTarget(
        target_mean=target_mean,
        mean_shift=target_mean - expanded_parent,
        supervision_mask=output_mask,
    )


def prepare_source_residual(
    source_expression: torch.Tensor,
    parent_mean: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    reference_projection: Optional[NonnegativeFixedMeanProjection] = None,
) -> NonnegativeFixedMeanProjection:
    """Project a source endpoint and express it around the effective Parent.

    Training and inference use this exact operation. A singleton is the
    effective Parent itself; it never retains an unsupported free residual.
    """

    _validate_expression("source_expression", source_expression)
    expanded_parent = _expand_mean("parent_mean", parent_mean, source_expression)
    return project_group_nonnegative_fixed_mean(
        source_expression,
        expanded_parent.detach(),
        group_ids=group_ids,
        valid_mask=valid_mask,
        reference_projection=reference_projection,
    )


def prepare_training_residual_pair(
    *,
    target_expression: torch.Tensor,
    source_expression: torch.Tensor,
    parent_mean: torch.Tensor,
    target_condition_mean: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    target_projection_override: Optional[object] = None,
) -> ParentLockedResidualPair:
    """Prepare source and target on the same nonnegative fixed-mean support.

    ``target_condition_mean`` remains accepted for checkpoint/API compatibility
    and is validated when provided, but the old translate-then-centre endpoint
    is intentionally replaced. Both observed targets and prior sources are now
    Euclidean-projected to the feasible Parent simplex before FM supervision,
    preventing a train/inference support-domain mismatch. The Parent is detached
    and remains learnable only through its explicit mean objective.
    """

    _validate_expression("target_expression", target_expression)
    _validate_expression("source_expression", source_expression)
    if source_expression.shape != target_expression.shape:
        raise ValueError("source_expression and target_expression must share [B,S,G]")
    if target_condition_mean is not None:
        _expand_mean("target_condition_mean", target_condition_mean, target_expression)

    source_projection = prepare_source_residual(
        source_expression,
        parent_mean,
        group_ids=group_ids,
        valid_mask=valid_mask,
    )
    expanded_parent = _expand_mean(
        "parent_mean", parent_mean, target_expression
    ).detach()
    if target_projection_override is None:
        target_projection = project_group_nonnegative_fixed_mean(
            target_expression,
            expanded_parent,
            group_ids=group_ids,
            valid_mask=valid_mask,
        )
    else:
        target_projection = target_projection_override
        required = (
            "value", "residual", "effective_parent", "group_ids",
            "counts", "valid_mask",
        )
        if any(not hasattr(target_projection, name) for name in required):
            raise TypeError("target_projection_override lacks projection fields")
        if not torch.equal(target_projection.value, target_expression):
            raise ValueError("target projection/value mismatch")
        for name in ("group_ids", "counts", "valid_mask", "effective_parent"):
            if not torch.equal(
                getattr(target_projection, name), getattr(source_projection, name)
            ):
                raise ValueError(
                    f"target projection {name} differs from source support"
                )
        if not torch.equal(
            target_projection.residual,
            target_projection.value - target_projection.effective_parent,
        ):
            raise ValueError("target projection residual is inconsistent")
    if not torch.equal(
        source_projection.effective_parent,
        target_projection.effective_parent,
    ):
        raise AssertionError("source and target effective Parent supports differ")

    return ParentLockedResidualPair(
        parent_mean=source_projection.effective_parent[:, 0, :].detach(),
        source_residual=source_projection.residual,
        target_residual=target_projection.residual,
        source_projection=source_projection,
        target_projection=target_projection,
    )


def compose_parent_residual(
    *,
    parent_mean: torch.Tensor,
    residual: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    enforce_zero_mean: bool = True,
    detach_parent_mean: bool = True,
) -> ParentLockedComposition:
    """Compose through the Parent skip and enforce the feasible simplex.

    ``enforce_zero_mean`` is retained for API compatibility. The fixed-mean
    simplex is now the sole endpoint support and therefore always enforces the
    corresponding zero-mean residual around ``clamp(group_mean(Parent), 0)``.
    """

    _validate_expression("residual", residual)
    del enforce_zero_mean
    expanded_parent = _expand_mean("parent_mean", parent_mean, residual)
    if detach_parent_mean:
        expanded_parent = expanded_parent.detach()
    projection = project_group_nonnegative_fixed_mean(
        expanded_parent + residual,
        expanded_parent,
        group_ids=group_ids,
        valid_mask=valid_mask,
    )
    return ParentLockedComposition(
        expression=projection.value,
        residual=projection.residual,
        parent_mean=projection.effective_parent[:, 0, :],
        projection=projection,
    )


__all__ = [
    "MeanCarryingTarget",
    "ParentLockedComposition",
    "ParentLockedResidualPair",
    "compose_parent_residual",
    "prepare_mean_carrying_target",
    "prepare_source_residual",
    "prepare_training_residual_pair",
]

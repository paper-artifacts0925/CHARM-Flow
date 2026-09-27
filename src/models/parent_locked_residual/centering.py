"""Condition-set centering primitives for Parent-locked residual flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GroupProjection:
    """Result of subtracting condition-set means.

    ``value`` and all masks have the original ``[B,S,...]`` layout.
    ``group_ids`` contains compact non-negative identifiers for valid cells and
    ``-1`` for padding.  ``counts`` is indexed by those compact identifiers.
    ``exact_mask`` marks cells in groups with at least two observations, for
    which the returned group mean is exactly zero up to floating-point error.
    """

    value: torch.Tensor
    group_ids: torch.Tensor
    counts: torch.Tensor
    valid_mask: torch.Tensor
    exact_mask: torch.Tensor
    singleton_mask: torch.Tensor


@dataclass(frozen=True)
class NonnegativeFixedMeanProjection:
    """Euclidean projection onto a nonnegative condition-mean simplex.

    ``raw_group_parent_mean`` is the mean requested by the Parent model.
    Negative expression means are infeasible, so ``effective_group_parent_mean``
    is its elementwise clamp at zero. ``residual`` is measured from that
    effective Parent. Low-precision CPU inputs are promoted to float32 on
    output for reliable sort/cumsum support and fixed-sum accuracy.
    """

    value: torch.Tensor
    residual: torch.Tensor
    effective_parent: torch.Tensor
    group_ids: torch.Tensor
    counts: torch.Tensor
    valid_mask: torch.Tensor
    exact_mask: torch.Tensor
    singleton_mask: torch.Tensor
    raw_group_parent_mean: torch.Tensor
    effective_group_parent_mean: torch.Tensor
    group_mean_drift_max_abs: torch.Tensor
    raw_parent_adjustment_max_abs: torch.Tensor
    negative_values_before: torch.Tensor


def condition_group_ids(*components: torch.Tensor) -> torch.Tensor:
    """Build collision-free group IDs from integer condition components.

    Components may be ``[B]`` or ``[B,S]`` and are broadcast over ``S``.  This
    avoids fragile arithmetic packing such as ``cell_id * N + perturbation_id``.
    In practice callers should include dataset, cell-line and perturbation IDs,
    while deliberately excluding raw batch ID when batches are views of the
    same biological condition.
    """

    if not components:
        raise ValueError("at least one condition component is required")
    batch_size = None
    set_size = 1
    device = None
    saw_set_axis = False
    for component in components:
        if not torch.is_tensor(component):
            raise TypeError("condition components must be torch tensors")
        if component.dtype == torch.bool or component.is_floating_point():
            raise TypeError("condition components must have an integer dtype")
        if component.ndim not in (1, 2):
            raise ValueError("condition components must have shape [B] or [B,S]")
        if batch_size is None:
            batch_size = int(component.shape[0])
            device = component.device
        elif component.shape[0] != batch_size:
            raise ValueError("condition components must share batch size B")
        if component.device != device:
            raise ValueError("condition components must share one device")
        if component.ndim == 2:
            saw_set_axis = True
            if component.shape[1] < 1:
                raise ValueError("condition component set axes cannot be empty")
            if set_size not in (1, int(component.shape[1])):
                raise ValueError("condition components must share set size S")
            set_size = int(component.shape[1])

    expanded = []
    for component in components:
        value = component.to(dtype=torch.long)
        if value.ndim == 1:
            value = value[:, None].expand(-1, set_size)
        elif value.shape[1] == 1 and set_size > 1:
            value = value.expand(-1, set_size)
        elif value.shape[1] != set_size:
            raise ValueError("condition components must share set size S")
        expanded.append(value.reshape(-1))
    keys = torch.stack(expanded, dim=1)
    _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    result = inverse.reshape(batch_size, set_size)
    return result if saw_set_axis else result[:, 0]


def _validate_value(value: torch.Tensor) -> tuple[int, int]:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError("value must be a floating-point tensor")
    if value.ndim < 3:
        raise ValueError("value must have shape [B,S,...]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("B and S dimensions must be non-empty")
    if not torch.isfinite(value).all():
        raise ValueError("value must be finite")
    return int(value.shape[0]), int(value.shape[1])


def _expand_valid_mask(
    valid_mask: Optional[torch.Tensor],
    batch_size: int,
    set_size: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(batch_size, set_size, dtype=torch.bool, device=device)
    if not torch.is_tensor(valid_mask) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a boolean tensor")
    if valid_mask.device != device:
        raise ValueError("valid_mask must share the value device")
    if valid_mask.ndim == 1 and valid_mask.shape[0] == batch_size:
        valid_mask = valid_mask[:, None].expand(-1, set_size)
    elif tuple(valid_mask.shape) != (batch_size, set_size):
        raise ValueError("valid_mask must have shape [B] or [B,S]")
    if not valid_mask.any():
        raise ValueError("valid_mask must contain at least one valid cell")
    return valid_mask


def _expand_group_ids(
    group_ids: Optional[torch.Tensor],
    batch_size: int,
    set_size: int,
    device: torch.device,
) -> torch.Tensor:
    if group_ids is None:
        return torch.arange(batch_size, device=device, dtype=torch.long)[:, None].expand(
            -1, set_size
        )
    if not torch.is_tensor(group_ids):
        raise TypeError("group_ids must be a torch tensor")
    if group_ids.dtype == torch.bool or group_ids.is_floating_point():
        raise TypeError("group_ids must have an integer dtype")
    if group_ids.device != device:
        raise ValueError("group_ids must share the value device")
    if group_ids.ndim == 1 and group_ids.shape[0] == batch_size:
        return group_ids.to(torch.long)[:, None].expand(-1, set_size)
    if tuple(group_ids.shape) != (batch_size, set_size):
        raise ValueError("group_ids must have shape [B] or [B,S]")
    return group_ids.to(torch.long)


def project_group_zero_mean(
    value: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    singleton_policy: str = "keep",
) -> GroupProjection:
    """Subtract the mean of every condition group.

    Args:
        value: Floating tensor shaped ``[B,S,...]``.
        group_ids: Optional identifiers shaped ``[B]`` or ``[B,S]``.  With no
            identifiers, every batch row is one set (the natural ``S > 1``
            inference layout).
        valid_mask: Padding mask shaped ``[B]`` or ``[B,S]``.
        singleton_policy: ``"keep"`` preserves a singleton residual,
            ``"zero"`` removes it, and ``"error"`` rejects it.

    The operation is differentiable. Invalid cells are returned as zeros and
    do not contribute to group statistics.
    """

    batch_size, set_size = _validate_value(value)
    if singleton_policy not in {"keep", "zero", "error"}:
        raise ValueError("singleton_policy must be 'keep', 'zero', or 'error'")
    mask = _expand_valid_mask(valid_mask, batch_size, set_size, value.device)
    identifiers = _expand_group_ids(
        group_ids, batch_size, set_size, value.device
    )

    flat_mask = mask.reshape(-1)
    flat_identifiers = identifiers.reshape(-1)
    valid_positions = torch.nonzero(flat_mask, as_tuple=False).flatten()
    _, inverse = torch.unique(
        flat_identifiers.index_select(0, valid_positions),
        sorted=True,
        return_inverse=True,
    )
    counts = torch.bincount(inverse, minlength=int(inverse.max().item()) + 1)
    if singleton_policy == "error" and (counts < 2).any():
        raise ValueError(
            "condition centering found a singleton group; provide an explicit "
            "condition mean or use a condition-grouped batch"
        )

    feature_shape = value.shape[2:]
    feature_size = 1
    for dimension in feature_shape:
        feature_size *= int(dimension)
    flat_value = value.reshape(batch_size * set_size, feature_size)
    valid_value = flat_value.index_select(0, valid_positions)
    accumulation_dtype = (
        torch.float32
        if valid_value.dtype in (torch.float16, torch.bfloat16)
        else valid_value.dtype
    )
    accumulated = torch.zeros(
        counts.shape[0],
        feature_size,
        device=value.device,
        dtype=accumulation_dtype,
    ).index_add(0, inverse, valid_value.to(accumulation_dtype))
    means = accumulated / counts.to(accumulation_dtype)[:, None]
    centred = valid_value.to(accumulation_dtype) - means.index_select(0, inverse)
    singleton_for_valid = counts.index_select(0, inverse) == 1
    if singleton_policy == "keep":
        centred = torch.where(
            singleton_for_valid[:, None],
            valid_value.to(accumulation_dtype),
            centred,
        )
    centred = centred.to(value.dtype)
    projected_flat = torch.zeros_like(flat_value).index_copy(
        0, valid_positions, centred
    )

    compact_flat = torch.full(
        (batch_size * set_size,), -1, dtype=torch.long, device=value.device
    ).index_copy(0, valid_positions, inverse)
    compact = compact_flat.reshape(batch_size, set_size)
    cell_counts = torch.zeros_like(compact)
    cell_counts.reshape(-1).index_copy_(
        0, valid_positions, counts.index_select(0, inverse)
    )
    singleton_mask = mask & (cell_counts == 1)
    exact_mask = mask & (cell_counts > 1)
    return GroupProjection(
        value=projected_flat.reshape(value.shape),
        group_ids=compact,
        counts=counts,
        valid_mask=mask,
        exact_mask=exact_mask,
        singleton_mask=singleton_mask,
    )



def _simplex_project_columns(
    values: torch.Tensor,
    target_sum: torch.Tensor,
) -> torch.Tensor:
    """Project ``[N,F]`` columns onto ``x>=0, sum(x)=target_sum``."""

    if values.ndim != 2 or target_sum.ndim != 1:
        raise ValueError("simplex inputs must have shape [N,F] and [F]")
    if values.shape[1] != target_sum.shape[0] or values.shape[0] < 1:
        raise ValueError("simplex inputs have incompatible shapes")
    if (target_sum < 0).any() or not torch.isfinite(target_sum).all():
        raise ValueError("simplex target sums must be finite and nonnegative")

    ordered = torch.sort(values, dim=0, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0) - target_sum[None, :]
    ranks = torch.arange(
        1,
        values.shape[0] + 1,
        device=values.device,
        dtype=values.dtype,
    )[:, None]
    active = ordered - cumulative / ranks > 0
    rho = active.sum(dim=0).sub(1).clamp_min(0)
    theta = cumulative.gather(0, rho[None, :]).squeeze(0) / (
        rho.to(values.dtype) + 1
    )
    projected = torch.clamp_min(values - theta[None, :], 0.0)

    # Repair the last few floating-point ulps without changing nonnegativity.
    correction = target_sum - projected.sum(dim=0)
    largest = projected.argmax(dim=0, keepdim=True)
    repair = torch.zeros_like(projected).scatter(0, largest, correction[None, :])
    return torch.clamp_min(projected + repair, 0.0)


def project_group_nonnegative_fixed_mean(
    expression: torch.Tensor,
    parent_mean: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    reference_projection: Optional[NonnegativeFixedMeanProjection] = None,
) -> NonnegativeFixedMeanProjection:
    """Project every condition/gene onto a nonnegative fixed-sum simplex.

    For group ``k`` and gene ``g`` the feasible target is
    ``sum_i x[i,g] = n_k * clamp(mean_i(parent[i,g]), min=0)``.
    The operation is permutation-equivariant and piecewise differentiable with
    respect to ``expression``. Invalid cells are returned as zeros. A singleton
    is exactly the effective Parent in both training and inference.
    """

    batch_size, set_size = _validate_value(expression)
    if not torch.is_tensor(parent_mean) or not parent_mean.is_floating_point():
        raise TypeError("parent_mean must be a floating-point tensor")
    if parent_mean.device != expression.device:
        raise ValueError("parent_mean must share the expression device")
    if parent_mean.ndim == 2 and tuple(parent_mean.shape) == (
        batch_size,
        expression.shape[-1],
    ):
        parent_mean = parent_mean[:, None, :].expand(-1, set_size, -1)
    elif parent_mean.ndim == 3 and tuple(parent_mean.shape) == (
        batch_size,
        1,
        expression.shape[-1],
    ):
        parent_mean = parent_mean.expand(-1, set_size, -1)
    elif tuple(parent_mean.shape) != tuple(expression.shape):
        raise ValueError("parent_mean must have shape [B,G], [B,1,G], or [B,S,G]")
    if not torch.isfinite(parent_mean).all():
        raise ValueError("parent_mean must be finite")

    topology = project_group_zero_mean(
        expression,
        group_ids=group_ids,
        valid_mask=valid_mask,
        singleton_policy="keep",
    )
    low_precision = expression.dtype in (torch.float16, torch.bfloat16)
    work_dtype = torch.float32 if low_precision else expression.dtype
    output_dtype = (
        torch.float32
        if expression.device.type == "cpu" and low_precision
        else expression.dtype
    )
    feature_shape = expression.shape[2:]
    feature_size = 1
    for dimension in feature_shape:
        feature_size *= int(dimension)
    flat_expression = expression.reshape(-1, feature_size)
    flat_parent = parent_mean.reshape(-1, feature_size)
    flat_mask = topology.valid_mask.reshape(-1)
    valid_positions = torch.nonzero(flat_mask, as_tuple=False).flatten()
    inverse = topology.group_ids.reshape(-1).index_select(0, valid_positions)
    valid_expression = flat_expression.index_select(0, valid_positions).to(work_dtype)
    valid_parent = flat_parent.index_select(0, valid_positions).to(work_dtype)

    counts = topology.counts.to(work_dtype)[:, None]
    if reference_projection is None:
        parent_sums = torch.zeros(
            topology.counts.numel(),
            feature_size,
            device=expression.device,
            dtype=work_dtype,
        ).index_add(0, inverse, valid_parent)
        raw_group_parent_mean = parent_sums / counts
        effective_group_parent_mean = torch.clamp_min(raw_group_parent_mean, 0.0)
    else:
        reference = reference_projection
        if not isinstance(reference, NonnegativeFixedMeanProjection):
            raise TypeError(
                "reference_projection must be a NonnegativeFixedMeanProjection"
            )
        if tuple(reference.value.shape) != tuple(expression.shape):
            raise ValueError("reference projection and expression shapes differ")
        if reference.value.device != expression.device:
            raise ValueError("reference projection and expression devices differ")
        if reference.value.dtype != output_dtype:
            raise ValueError("reference projection and output dtypes differ")
        if not torch.equal(reference.group_ids, topology.group_ids):
            raise ValueError("reference projection group IDs differ")
        if not torch.equal(reference.counts, topology.counts):
            raise ValueError("reference projection group counts differ")
        if not torch.equal(reference.valid_mask, topology.valid_mask):
            raise ValueError("reference projection valid masks differ")
        expected_group_shape = (topology.counts.numel(), feature_size)
        if tuple(reference.raw_group_parent_mean.shape) != expected_group_shape:
            raise ValueError("reference raw Parent mean has an invalid shape")
        if tuple(reference.effective_group_parent_mean.shape) != expected_group_shape:
            raise ValueError("reference effective Parent mean has an invalid shape")
        for name, value in (
            ("raw", reference.raw_group_parent_mean),
            ("effective", reference.effective_group_parent_mean),
        ):
            if value.device != expression.device:
                raise ValueError(f"reference {name} Parent mean is on the wrong device")
            if value.dtype != output_dtype:
                raise ValueError(f"reference {name} Parent mean has the wrong dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"reference {name} Parent mean is not finite")
        raw_group_parent_mean = reference.raw_group_parent_mean.to(work_dtype)
        effective_group_parent_mean = (
            reference.effective_group_parent_mean.to(work_dtype)
        )
        if (effective_group_parent_mean < 0).any():
            raise ValueError("reference effective Parent mean is negative")
        parent_sums = raw_group_parent_mean * counts

    projected_valid = torch.zeros_like(valid_expression)
    for group_index in range(topology.counts.numel()):
        positions = torch.nonzero(inverse == group_index, as_tuple=False).flatten()
        group_value = valid_expression.index_select(0, positions)
        target_sum = effective_group_parent_mean[group_index] * positions.numel()
        group_projected = _simplex_project_columns(group_value, target_sum)
        projected_valid = projected_valid.index_copy(0, positions, group_projected)
    effective_valid = effective_group_parent_mean.index_select(0, inverse)
    residual_valid = projected_valid - effective_valid

    projected_flat = torch.zeros(
        flat_expression.shape,
        device=expression.device,
        dtype=work_dtype,
    ).index_copy(0, valid_positions, projected_valid)
    residual_flat = torch.zeros_like(projected_flat).index_copy(
        0, valid_positions, residual_valid
    )
    effective_flat = torch.zeros_like(projected_flat).index_copy(
        0, valid_positions, effective_valid
    )

    projected_sums = torch.zeros_like(parent_sums).index_add(
        0, inverse, projected_valid
    )
    projected_means = projected_sums / counts
    mean_drift = (projected_means - effective_group_parent_mean).abs().max()
    adjustment = (
        effective_group_parent_mean - raw_group_parent_mean
    ).abs().max()
    negative_before = (valid_expression < 0).sum()
    return NonnegativeFixedMeanProjection(
        value=projected_flat.reshape(expression.shape).to(output_dtype),
        residual=residual_flat.reshape(expression.shape).to(output_dtype),
        effective_parent=effective_flat.reshape(expression.shape).to(output_dtype),
        group_ids=topology.group_ids,
        counts=topology.counts,
        valid_mask=topology.valid_mask,
        exact_mask=topology.exact_mask,
        singleton_mask=topology.singleton_mask,
        raw_group_parent_mean=raw_group_parent_mean.to(output_dtype),
        effective_group_parent_mean=effective_group_parent_mean.to(output_dtype),
        group_mean_drift_max_abs=mean_drift.to(output_dtype),
        raw_parent_adjustment_max_abs=adjustment.to(output_dtype),
        negative_values_before=negative_before,
    )

__all__ = [
    "GroupProjection",
    "NonnegativeFixedMeanProjection",
    "condition_group_ids",
    "project_group_nonnegative_fixed_mean",
    "project_group_zero_mean",
]

"""Opt-in integration of matched Child reconstruction and Parent contrast."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .centered_parent_contrast import centered_parent_contrast_loss
from .matched_hierarchical_reconstruction import (
    matched_hierarchical_reconstruction,
)


_PREFIX = "parent_residual_joint_parent_child"
_CONFLICT_FLAGS = (
    "parent_residual_child_sparse_parent_enabled",
    "match_fm_reliable_gene_energy",
)


@dataclass(frozen=True)
class _MacroMHR:
    loss: torch.Tensor
    centered_group_mean_max_abs: torch.Tensor
    positive_match_count: torch.Tensor
    charbonnier: torch.Tensor
    rmse: torch.Tensor
    cosine: torch.Tensor


def _macro_mhr(
    *,
    parent: torch.Tensor,
    child: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    active: torch.Tensor,
    objective_kwargs: dict[str, float],
) -> _MacroMHR:
    """Give every biological condition equal objective weight."""
    losses = []
    center_errors = []
    positive_counts = []
    charbonnier = []
    rmse = []
    cosine = []
    for group in torch.unique(group_ids[active], sorted=True):
        positions = (active & (group_ids == group)).nonzero(
            as_tuple=False
        ).squeeze(1)
        size = int(positions.numel())
        output = matched_hierarchical_reconstruction(
            parent.index_select(0, positions),
            child.index_select(0, positions),
            target.index_select(0, positions),
            group_ids.index_select(0, positions),
            positive_mask.index_select(0, positions),
            torch.ones(size, dtype=torch.bool, device=parent.device),
            **objective_kwargs,
        )
        losses.append(output.loss)
        center_errors.append(output.centered_group_mean_max_abs)
        positive_counts.append(output.positive_match_count)
        charbonnier.append(output.charbonnier)
        rmse.append(output.rmse)
        cosine.append(output.cosine)
    return _MacroMHR(
        loss=torch.stack(losses).mean(),
        centered_group_mean_max_abs=torch.stack(center_errors).max(),
        positive_match_count=torch.stack(positive_counts).sum(),
        charbonnier=torch.stack(charbonnier).mean(),
        rmse=torch.stack(rmse).mean(),
        cosine=torch.stack(cosine).mean(),
    )


def _config(lightning: Any, name: str, default: Any) -> Any:
    if hasattr(lightning, name):
        return getattr(lightning, name)
    model_cfg = getattr(lightning, "model_cfg", None)
    return getattr(model_cfg, name, default)


def _finite_number(lightning: Any, suffix: str, default: float) -> float:
    name = f"{_PREFIX}_{suffix}"
    value = float(_config(lightning, name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(lightning: Any, suffix: str, default: float) -> float:
    value = _finite_number(lightning, suffix, default)
    if value < 0.0:
        raise ValueError(f"{_PREFIX}_{suffix} must be nonnegative")
    return value


def _positive_number(lightning: Any, suffix: str, default: float) -> float:
    value = _finite_number(lightning, suffix, default)
    if value <= 0.0:
        raise ValueError(f"{_PREFIX}_{suffix} must be positive")
    return value


def _positive_int(lightning: Any, suffix: str, default: int) -> int:
    name = f"{_PREFIX}_{suffix}"
    raw = _config(lightning, name, default)
    if isinstance(raw, bool) or int(raw) != raw or int(raw) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw)


def _require_detached(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _gradient(loss: torch.Tensor, endpoint: torch.Tensor, name: str) -> torch.Tensor:
    if loss.ndim != 0 or not loss.requires_grad:
        raise ValueError(f"{name} loss must be a differentiable scalar")
    if not endpoint.requires_grad:
        raise ValueError(f"{name} endpoint must require gradients")
    gradient = torch.autograd.grad(
        loss, endpoint, retain_graph=True, create_graph=False, allow_unused=True
    )[0]
    if gradient is None:
        raise ValueError(f"{name} gradient is disconnected")
    gradient = gradient.detach().float()
    _require_finite(f"{name} gradient", gradient)
    if gradient.square().sum() <= 0:
        raise ValueError(f"{name} gradient must be nonzero")
    return gradient


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.reshape(-1)).clamp_min(1.0e-12)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.dot(left.reshape(-1), right.reshape(-1)) / (
        _norm(left) * _norm(right)
    )


def _scale(
    *, target_ratio: float, reference: torch.Tensor, auxiliary: torch.Tensor,
    lower: float, upper: float,
) -> torch.Tensor:
    raw = reference.new_tensor(target_ratio) * _norm(reference) / _norm(auxiliary)
    result = raw.clamp(min=lower, max=upper).detach()
    _require_finite("gradient scale", result)
    if result <= 0:
        raise ValueError("gradient scale must be positive")
    return result


def _macro_mean(
    value: torch.Tensor, group_ids: torch.Tensor, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = active.nonzero(as_tuple=False).squeeze(1)
    valid_groups = group_ids.index_select(0, positions)
    unique, inverse = torch.unique(valid_groups, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique.numel())
    sums = value.new_zeros((unique.numel(), value.shape[-1])).index_add(
        0, inverse, value.index_select(0, positions)
    )
    means = sums / counts.to(value.dtype)[:, None]
    expanded = value.new_zeros(value.shape).index_copy(
        0, positions, means.index_select(0, inverse)
    )
    return means, expanded, positions, inverse


def _constant_macro(
    name: str,
    value: torch.Tensor,
    positions: torch.Tensor,
    inverse: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    selected = value.index_select(0, positions)
    representatives = []
    for group in range(group_count):
        members = selected[inverse == group]
        first = members[0]
        if members.is_floating_point():
            equal = torch.equal(members, first.expand_as(members))
        else:
            equal = torch.equal(members, first.expand_as(members))
        if not equal:
            raise ValueError(f"{name} must be constant inside each condition")
        representatives.append(first)
    return torch.stack(representatives)


def _update_guard(
    lightning: Any, *, actual_ratio: torch.Tensor, base_cosine: torch.Tensor,
    warm: float, patience: int, minimum_ratio: float, minimum_cosine: float,
) -> None:
    if warm < 1.0:
        return
    step = int(getattr(lightning, "global_step", 0))
    last = getattr(lightning, f"_{_PREFIX}_guard_step", None)
    if last == step:
        return
    setattr(lightning, f"_{_PREFIX}_guard_step", step)
    low_name = f"_{_PREFIX}_low_ratio_steps"
    conflict_name = f"_{_PREFIX}_conflict_steps"
    low = int(getattr(lightning, low_name, 0))
    conflict = int(getattr(lightning, conflict_name, 0))
    low = low + 1 if float(actual_ratio) < minimum_ratio else 0
    conflict = conflict + 1 if float(base_cosine) < minimum_cosine else 0
    setattr(lightning, low_name, low)
    setattr(lightning, conflict_name, conflict)
    if low >= patience:
        raise RuntimeError("joint Parent auxiliary gradient remained below budget")
    if conflict >= patience:
        raise RuntimeError("joint Parent auxiliary gradient remained antagonistic")


def augment_result_with_joint_parent_child_objective(
    lightning: Any,
    result: Any,
    prepared: Any,
    match: Any,
    batch: Any,
    supervision_mask: Any,
    parent_grouping: Any,
    parent_reference_loss: Any,
) -> None:
    """Add the joint objective, or return without observing any other input."""

    if not bool(_config(lightning, f"{_PREFIX}_enabled", False)):
        return

    if not bool(getattr(lightning, "training", False)):
        raise RuntimeError("joint Parent/Child objective is training-only")
    if not bool(_config(lightning, "parent_residual_matched_set_enabled", False)):
        raise ValueError("joint objective requires matched-set transport")
    if not bool(_config(lightning, "parent_residual_locked_flow_enabled", False)):
        raise ValueError("joint objective requires locked flow")
    conflicts = [name for name in _CONFLICT_FLAGS if bool(_config(lightning, name, False))]
    if conflicts:
        raise ValueError("joint objective conflicts with " + ", ".join(conflicts))
    if float(
        _config(lightning, "parent_residual_parent_listwise_pds_weight", 0.0)
    ) != 0.0:
        raise ValueError(
            "joint objective requires parent_residual_parent_listwise_pds_weight=0"
        )
    for suffix in ("delta", "direction", "magnitude"):
        name = f"parent_residual_teacher_{suffix}_weight"
        if float(_config(lightning, name, 0.0)) != 0.0:
            raise ValueError(f"joint objective requires {name}=0")

    target = batch["pert_emb"].detach()
    _require_detached("target", target)
    if target.ndim != 3 or target.shape[1] != 1:
        raise ValueError("joint objective requires one target cell per row")
    parent = prepared.parent.parent_mean
    control = prepared.parent.lookup.control_mean.to(parent).detach()
    target_available = prepared.parent.lookup.target_available.to(parent.device)
    _require_detached("control mean", control)
    _require_detached("target availability", target_available)
    if parent.shape != target[:, 0].shape or control.shape != parent.shape:
        raise ValueError("Parent, control, and target shapes must agree")
    row_valid = supervision_mask[:, 0].to(device=parent.device, dtype=torch.bool)
    active = row_valid & target_available.to(dtype=torch.bool)
    if not active.any():
        raise ValueError("joint objective has no active training rows")

    from src.models.parent_locked_residual.centering import (
        condition_group_ids as encode_condition_group_ids,
    )
    from src.models.parent_locked_residual.matched_integration import (
        condition_group_ids as tuple_condition_group_ids,
    )

    tuple_ids = tuple_condition_group_ids(
        batch, set_size=1, parent_grouping=parent_grouping
    )
    if tuple_ids.ndim != 3 or tuple_ids.shape[1] != 1:
        raise ValueError("condition tuple IDs must have shape [B,1,C]")
    group_ids = encode_condition_group_ids(*tuple_ids.unbind(dim=-1))[:, 0]
    parent_macro, parent_expanded, positions, inverse = _macro_mean(
        parent, group_ids, active
    )
    group_count = int(parent_macro.shape[0])
    counts = torch.bincount(inverse, minlength=group_count)
    minimum_group_size = _positive_int(lightning, "min_group_size", 32)
    if torch.any(counts < minimum_group_size):
        raise ValueError("every active condition must meet the minimum group size")

    lookup = prepared.parent.lookup
    line_ids = _constant_macro(
        "artifact cell-line IDs", lookup.artifact_cellline_ids.to(parent.device),
        positions, inverse, group_count,
    ).to(torch.long)
    perturbation_ids = _constant_macro(
        "artifact perturbation IDs", lookup.artifact_perturbation_ids.to(parent.device),
        positions, inverse, group_count,
    ).to(torch.long)
    _constant_macro("control mean", control, positions, inverse, group_count)
    target_delta = lookup.target_delta.to(parent).detach()
    _require_detached("lookup target delta", target_delta)
    _constant_macro("lookup target delta", target_delta, positions, inverse, group_count)

    matched_child = match.positive_controls.detach() + match.positive_delta
    positive_mask = match.positive_mask.to(device=parent.device, dtype=torch.bool)
    if matched_child.shape[:1] != parent.shape[:1] or matched_child.shape[2] != parent.shape[1]:
        raise ValueError("matched Child endpoint must have shape [B,R,G]")
    if positive_mask.shape != matched_child.shape[:2]:
        raise ValueError("positive mask must have shape [B,R]")
    _require_detached("positive controls", match.positive_controls.detach())
    _require_finite("matched Child endpoint", matched_child)

    mhr_kwargs = {
        "charbonnier_weight": _nonnegative(lightning, "mhr_charbonnier_weight", 0.45),
        "rmse_weight": _nonnegative(lightning, "mhr_rmse_weight", 0.45),
        "cosine_weight": _nonnegative(lightning, "mhr_cosine_weight", 0.10),
        "charbonnier_eps": _positive_number(lightning, "mhr_charbonnier_eps", 1.0e-3),
        "rmse_eps": _positive_number(lightning, "mhr_rmse_eps", 1.0e-8),
        "cosine_eps": _positive_number(lightning, "mhr_cosine_eps", 1.0e-8),
        "cosine_norm_scale": _positive_number(lightning, "mhr_cosine_norm_scale", 1.0),
    }
    mhr_parent = _macro_mhr(
        parent=parent_expanded,
        child=matched_child.detach(),
        target=target,
        group_ids=group_ids,
        positive_mask=positive_mask,
        active=active,
        objective_kwargs=mhr_kwargs,
    )
    mhr_child = _macro_mhr(
        parent=parent_expanded.detach(),
        child=matched_child,
        target=target,
        group_ids=group_ids,
        positive_mask=positive_mask,
        active=active,
        objective_kwargs=mhr_kwargs,
    )
    tolerance = _positive_number(lightning, "centering_tolerance", 1.0e-6)
    if tolerance > 1.0e-6:
        raise ValueError("centering tolerance cannot exceed 1e-6")
    for name, value in (
        ("Parent-branch Child centering", mhr_parent.centered_group_mean_max_abs),
        ("Child-branch Child centering", mhr_child.centered_group_mean_max_abs),
    ):
        if float(value.detach()) >= tolerance:
            raise ValueError(f"{name} exceeded tolerance")

    prior_bank = lightning.model.prior_bank
    bank = prior_bank.train_target_delta.detach().to(parent)
    bank_mask = prior_bank.train_target_mask.detach().to(parent.device)
    _require_detached("training signature bank", bank)
    _require_finite("training signature bank", bank)
    _require_detached("training signature mask", bank_mask)
    if bank_mask.dtype != torch.bool:
        raise TypeError("training signature mask must be boolean")
    if not torch.all(bank_mask[line_ids, perturbation_ids]):
        raise ValueError("every macro query must have a positive training signature")

    parent_delta_macro, _, _, _ = _macro_mean(parent - control, group_ids, active)
    exclude_target = bool(_config(
        lightning, f"{_PREFIX}_exclude_target_gene", False
    ))
    target_gene_index = None
    if exclude_target:
        target_gene_index = prior_bank.target_gene_index.detach().to(parent.device)
        _require_detached("target gene index", target_gene_index)
    cpc = centered_parent_contrast_loss(
        parent_prediction=parent_delta_macro,
        cell_line_ids=line_ids,
        perturbation_ids=perturbation_ids,
        train_signature_bank=bank,
        train_signature_mask=bank_mask,
        target_gene_index=target_gene_index,
        l1_weight=_nonnegative(lightning, "cpc_l1_weight", 1.0),
        l2_weight=_nonnegative(lightning, "cpc_l2_weight", 1.0),
        cosine_weight=_nonnegative(lightning, "cpc_cosine_weight", 0.5),
        positive_huber_weight=_nonnegative(lightning, "cpc_positive_huber_weight", 1.0),
        centered_l1_norm_weight=_nonnegative(lightning, "cpc_l1_norm_weight", 1.0),
        centered_l2_norm_weight=_nonnegative(lightning, "cpc_l2_norm_weight", 1.0),
        l1_temperature=_positive_number(lightning, "cpc_l1_temperature", 0.01),
        l2_temperature=_positive_number(lightning, "cpc_l2_temperature", 0.02),
        cosine_temperature=_positive_number(lightning, "cpc_cosine_temperature", 0.05),
        huber_delta=_positive_number(lightning, "cpc_huber_delta", 0.1),
        norm_target_scale=_positive_number(lightning, "cpc_norm_target_scale", 0.05),
        norm_huber_beta=_positive_number(lightning, "cpc_norm_huber_beta", 0.1),
        chunk_size=_positive_int(lightning, "cpc_chunk_size", 256),
        eps=_positive_number(lightning, "cpc_eps", 1.0e-8),
    )

    base_gradient = _gradient(parent_reference_loss, parent, "Parent reference")
    mhr_parent_gradient = _gradient(mhr_parent.loss, parent, "MHR Parent")
    cpc_parent_gradient = _gradient(cpc.loss, parent, "CPC Parent")
    child_reference_gradient = _gradient(match.delta_loss, match.positive_delta, "Child reference")
    child_aux_gradient = _gradient(mhr_child.loss, match.positive_delta, "MHR Child")

    warmup_steps = _positive_int(lightning, "warmup_steps", 500)
    warm = min((int(getattr(lightning, "global_step", 0)) + 1) / warmup_steps, 1.0)
    lower = _nonnegative(lightning, "scale_min", 1.0e-6)
    upper = _nonnegative(lightning, "scale_max", 1.0e6)
    if lower <= 0.0 or upper < lower:
        raise ValueError("gradient scale bounds must satisfy 0 < min <= max")
    mhr_parent_scale = _scale(
        target_ratio=warm * _positive_number(lightning, "mhr_parent_gradient_ratio", 0.10),
        reference=base_gradient, auxiliary=mhr_parent_gradient,
        lower=lower, upper=upper,
    )
    cpc_parent_scale = _scale(
        target_ratio=warm * _positive_number(lightning, "cpc_parent_gradient_ratio", 0.15),
        reference=base_gradient, auxiliary=cpc_parent_gradient,
        lower=lower, upper=upper,
    )
    combined_parent_gradient = (
        mhr_parent_scale * mhr_parent_gradient + cpc_parent_scale * cpc_parent_gradient
    )
    cap = _positive_number(lightning, "parent_gradient_ratio_cap", 0.30)
    combined_ratio = _norm(combined_parent_gradient) / _norm(base_gradient)
    if float(combined_ratio) > cap:
        cap_factor = combined_ratio.new_tensor(cap) / combined_ratio
        mhr_parent_scale = (mhr_parent_scale * cap_factor).detach()
        cpc_parent_scale = (cpc_parent_scale * cap_factor).detach()
        combined_parent_gradient = (
            mhr_parent_scale * mhr_parent_gradient + cpc_parent_scale * cpc_parent_gradient
        )
    actual_parent_ratio = _norm(combined_parent_gradient) / _norm(base_gradient)
    parent_base_cosine = _cosine(combined_parent_gradient, base_gradient)

    mhr_child_scale = _scale(
        target_ratio=warm * _positive_number(lightning, "mhr_child_gradient_ratio", 0.20),
        reference=child_reference_gradient, auxiliary=child_aux_gradient,
        lower=lower, upper=upper,
    )
    actual_child_ratio = (
        mhr_child_scale * _norm(child_aux_gradient) / _norm(child_reference_gradient)
    )
    weighted_mhr_parent = mhr_parent_scale * mhr_parent.loss
    weighted_cpc_parent = cpc_parent_scale * cpc.loss
    weighted_mhr_child = mhr_child_scale * mhr_child.loss
    auxiliary = weighted_mhr_parent + weighted_cpc_parent + weighted_mhr_child
    _require_finite("joint auxiliary loss", auxiliary)
    result["loss"] = result["loss"] + auxiliary

    patience = _positive_int(lightning, "guard_patience", 50)
    minimum_ratio = _nonnegative(lightning, "minimum_actual_parent_ratio", 0.05)
    minimum_cosine = _finite_number(lightning, "minimum_parent_base_cosine", -0.25)
    _update_guard(
        lightning,
        actual_ratio=actual_parent_ratio,
        base_cosine=parent_base_cosine,
        warm=warm,
        patience=patience,
        minimum_ratio=minimum_ratio,
        minimum_cosine=minimum_cosine,
    )

    scalar = parent.new_tensor
    result.update({
        f"{_PREFIX}_loss": auxiliary.detach(),
        f"{_PREFIX}_mhr_parent_raw": mhr_parent.loss.detach(),
        f"{_PREFIX}_mhr_child_raw": mhr_child.loss.detach(),
        f"{_PREFIX}_mhr_parent_charbonnier": mhr_parent.charbonnier.detach(),
        f"{_PREFIX}_mhr_parent_rmse": mhr_parent.rmse.detach(),
        f"{_PREFIX}_mhr_parent_cosine": mhr_parent.cosine.detach(),
        f"{_PREFIX}_mhr_child_charbonnier": mhr_child.charbonnier.detach(),
        f"{_PREFIX}_mhr_child_rmse": mhr_child.rmse.detach(),
        f"{_PREFIX}_mhr_child_cosine": mhr_child.cosine.detach(),
        f"{_PREFIX}_cpc_raw": cpc.loss.detach(),
        f"{_PREFIX}_mhr_parent_scale": mhr_parent_scale,
        f"{_PREFIX}_cpc_parent_scale": cpc_parent_scale,
        f"{_PREFIX}_mhr_child_scale": mhr_child_scale,
        f"{_PREFIX}_actual_parent_gradient_ratio": actual_parent_ratio.detach(),
        f"{_PREFIX}_actual_child_gradient_ratio": actual_child_ratio.detach(),
        f"{_PREFIX}_parent_base_cosine": parent_base_cosine.detach(),
        f"{_PREFIX}_mhr_parent_base_cosine": _cosine(mhr_parent_gradient, base_gradient).detach(),
        f"{_PREFIX}_cpc_parent_base_cosine": _cosine(cpc_parent_gradient, base_gradient).detach(),
        f"{_PREFIX}_mhr_cpc_gradient_cosine": _cosine(mhr_parent_gradient, cpc_parent_gradient).detach(),
        f"{_PREFIX}_parent_aux_combined_gradient_norm": _norm(combined_parent_gradient),
        f"{_PREFIX}_active_row_count": active.sum().detach(),
        f"{_PREFIX}_condition_count": scalar(float(group_count)),
        f"{_PREFIX}_minimum_group_count": counts.min().detach(),
        f"{_PREFIX}_coverage": active.float().mean().detach(),
        f"{_PREFIX}_positive_match_count": mhr_parent.positive_match_count.detach(),
        f"{_PREFIX}_centering_max_abs": torch.maximum(
            mhr_parent.centered_group_mean_max_abs,
            mhr_child.centered_group_mean_max_abs,
        ).detach(),
        f"{_PREFIX}_cpc_query_count": cpc.query_count.detach(),
        f"{_PREFIX}_cpc_listwise_l1": cpc.listwise_l1.detach(),
        f"{_PREFIX}_cpc_listwise_l2": cpc.listwise_l2.detach(),
        f"{_PREFIX}_cpc_listwise_cosine": cpc.listwise_cosine.detach(),
        f"{_PREFIX}_cpc_positive_huber": cpc.positive_huber.detach(),
        f"{_PREFIX}_cpc_l1_norm_calibration": cpc.centered_l1_norm_calibration.detach(),
        f"{_PREFIX}_cpc_l2_norm_calibration": cpc.centered_l2_norm_calibration.detach(),
        f"{_PREFIX}_cpc_candidate_count_mean": cpc.candidate_count_mean.detach(),
        f"{_PREFIX}_cpc_candidate_count_min": cpc.candidate_count_min.detach(),
        f"{_PREFIX}_cpc_loo_centroid_l1_norm": cpc.loo_centroid_l1_norm.detach(),
        f"{_PREFIX}_cpc_loo_centroid_l2_norm": cpc.loo_centroid_l2_norm.detach(),
        f"{_PREFIX}_cpc_norm_gate_mean": cpc.norm_calibration_gate_mean.detach(),
        f"{_PREFIX}_cpc_norm_gate_min": cpc.norm_calibration_gate_min.detach(),
        f"{_PREFIX}_cpc_norm_gate_max": cpc.norm_calibration_gate_max.detach(),
        f"{_PREFIX}_warmup_fraction": scalar(warm),
    })


__all__ = ["augment_result_with_joint_parent_child_objective"]

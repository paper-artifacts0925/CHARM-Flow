"""Opt-in end-to-end Child-support wiring for Parent-locked G7."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.parent_locked_residual.centering import (
    NonnegativeFixedMeanProjection,
)
from src.models.de_aware_residual.losses import (
    cap_auxiliary_gradient,
    warmup_fraction,
)
from src.models.train_hurdle_bank import TrainHurdleBank

from .output import SparseHurdleDEAdapterOutput, SparseHurdleProjection
from .projector import (
    project_ranked_hard_zero_fixed_parent_mean,
    repair_terminal_hard_zero_fixed_support,
)


@dataclass(frozen=True)
class EndToEndHurdleState:
    target_expression: torch.Tensor
    adapter_output: Optional[SparseHurdleDEAdapterOutput]
    control_zero_rate: torch.Tensor
    target_zero_rate: torch.Tensor
    target_count: torch.Tensor
    de_labels: Optional[torch.Tensor]
    condition_ids: torch.Tensor
    active_mask: torch.Tensor
    observed_projection: Optional[SparseHurdleProjection]


def _cfg(lightning, name: str, default):
    model_cfg = getattr(lightning, "model_cfg", None)
    return getattr(model_cfg, name, default) if model_cfg is not None else default


def _optional_path(value) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(Path(value).expanduser().resolve())


def _finite_nonnegative(lightning, name: str, default: float) -> float:
    value = float(_cfg(lightning, name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def initialize_end_to_end_sparse_hurdle(lightning) -> None:
    """Load only pinned train statistics and install no state when disabled."""

    hurdle_enabled = bool(
        lightning.parent_residual_enabled
        and _cfg(lightning, "parent_residual_hurdle_geometry_enabled", False)
    )
    sparse_enabled = bool(
        lightning.parent_residual_enabled
        and _cfg(lightning, "parent_residual_sparse_support_enabled", False)
    )
    lightning.parent_residual_hurdle_geometry_enabled = hurdle_enabled
    lightning.parent_residual_sparse_support_enabled = sparse_enabled
    lightning.parent_residual_hurdle_bank = None
    if not hurdle_enabled and not sparse_enabled:
        return
    if not lightning.parent_residual_locked_flow_enabled:
        raise ValueError("end-to-end hurdle support requires Parent-locked flow")
    if (
        sparse_enabled
        and not lightning.parent_residual_de_aware_enabled
        and not bool(_cfg(lightning, "parent_residual_no_flow_enabled", False))
    ):
        raise ValueError("S requires the common train-only DE-aware label bank")
    if sparse_enabled and getattr(
        lightning.model, "sparse_hurdle_adapter", None
    ) is None:
        raise AssertionError("S config did not construct its online adapter")

    alpha = _finite_nonnegative(
        lightning, "parent_residual_hurdle_alpha", 1.0
    )
    if alpha > 1.0:
        raise ValueError("parent_residual_hurdle_alpha must lie in [0,1]")
    for name, default in (
        ("parent_residual_sparse_head_weight", 1.0e-3),
        ("parent_residual_sparse_endpoint_consistency_weight", 5.0e-4),
        ("parent_residual_sparse_signature_weight", 1.0e-3),
        ("parent_residual_sparse_signature_support_weight", 1.0),
        ("parent_residual_sparse_signature_null_weight", 1.0),
        ("parent_residual_sparse_signature_pds_l1_weight", 1.0),
        ("parent_residual_sparse_signature_pds_l2_weight", 1.0),
        ("parent_residual_sparse_signature_pds_cos_weight", 1.0),
        ("parent_residual_sparse_signature_no_reg_weight", 1.0),
        ("parent_residual_sparse_signature_pds_margin", 5.0e-3),
        ("parent_residual_sparse_hard_rank_weight", 0.0),
        ("parent_residual_sparse_hard_rank_margin", 0.0),
        ("parent_residual_sparse_gradient_cap_eps", 1.0e-12),
        ("parent_residual_sparse_zero_temperature", 3.0e-3),
        ("parent_residual_sparse_zero_rate_confidence_tau", 64.0),
    ):
        _finite_nonnegative(lightning, name, default)
    cap = _finite_nonnegative(
        lightning, "parent_residual_sparse_gradient_cap_ratio", 0.2
    )
    if cap > 1.0:
        raise ValueError("end-to-end sparse auxiliary gradient cap cannot exceed 1.0")
    hard_rank_temperature = float(
        _cfg(lightning, "parent_residual_sparse_hard_rank_temperature", 0.5)
    )
    if not math.isfinite(hard_rank_temperature) or hard_rank_temperature <= 0.0:
        raise ValueError(
            "parent_residual_sparse_hard_rank_temperature must be finite and positive"
        )
    for name in (
        "parent_residual_sparse_hard_rank_max_positive",
        "parent_residual_sparse_hard_rank_max_negative",
    ):
        raw_value = _cfg(lightning, name, 256)
        if (
            isinstance(raw_value, bool)
            or int(raw_value) < 1
            or float(raw_value) != float(int(raw_value))
        ):
            raise ValueError(f"{name} must be a positive integer")
    joint_context = _cfg(
        lightning, "parent_residual_sparse_joint_context_enabled", False
    )
    if not isinstance(joint_context, bool):
        raise TypeError(
            "parent_residual_sparse_joint_context_enabled must be boolean"
        )
    warmup = int(_cfg(lightning, "parent_residual_sparse_warmup_steps", 500))
    if warmup < 0:
        raise ValueError("parent_residual_sparse_warmup_steps must be nonnegative")

    artifact_path = _optional_path(
        _cfg(lightning, "parent_residual_hurdle_artifact_path", None)
    )
    artifact_sha = str(
        _cfg(lightning, "parent_residual_hurdle_artifact_sha256", "") or ""
    ).lower()
    if artifact_path is None or len(artifact_sha) != 64:
        raise ValueError("H/S requires a pinned TrainHurdleBank path and SHA256")
    source_sha = str(
        _cfg(lightning, "parent_residual_hurdle_source_h5ad_sha256", "") or ""
    ).lower()
    if len(source_sha) != 64:
        raise ValueError(
            "H/S requires an independently pinned source H5AD SHA256"
        )

    bank = TrainHurdleBank(
        artifact_path=artifact_path,
        expected_artifact_sha256=artifact_sha,
        parent_artifact_path=_cfg(
            lightning, "parent_residual_artifact_path", None
        ),
        # Matrix preflight hashes the 22GB source once; each worker pins it.
        source_h5ad_path=None,
        expected_source_h5ad_sha256=source_sha,
        expected_gene_dim=int(lightning.model_cfg.input_dim),
        expected_normalization_divisor=float(
            _cfg(lightning, "parent_residual_normalization_divisor", 10.0)
        ),
    )
    if sparse_enabled:
        de_bank = lightning.parent_residual_de_aware_bank
        de_mask = de_bank.train_condition_mask.detach().cpu().numpy()
        if not np.array_equal(de_mask, bank.train_condition_mask):
            raise ValueError("DE and hurdle effective train masks differ")

    lightning.parent_residual_hurdle_bank = bank
    lightning.register_buffer(
        "parent_residual_hurdle_control_zero_rate",
        torch.from_numpy(
            np.array(bank.control_zero_rate, dtype=np.float32, copy=True)
        ),
        persistent=False,
    )
    lightning.py_logger.info(
        "Loaded train-only hurdle bank %s sha256=%s source_sha256=%s H=%s S=%s",
        bank.artifact_path,
        bank.artifact_sha256,
        bank.source_h5ad_sha256,
        hurdle_enabled,
        sparse_enabled,
    )


def _control_zero_rate(lightning, line_ids: torch.Tensor) -> torch.Tensor:
    table = lightning.parent_residual_hurdle_control_zero_rate
    ids = line_ids.to(device=table.device, dtype=torch.long)
    if (ids < 0).any() or (ids >= table.shape[0]).any():
        raise ValueError("artifact cell-line ID is out of hurdle range")
    return table.index_select(0, ids)


def _strict_hurdle_targets(
    lightning,
    line_ids: torch.Tensor,
    perturbation_ids: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bank = lightning.parent_residual_hurdle_bank
    batch_size = int(line_ids.numel())
    genes = int(lightning.model_cfg.input_dim)
    target_q = torch.zeros(
        batch_size, genes, device=line_ids.device, dtype=torch.float32
    )
    target_count = torch.zeros(
        batch_size, device=line_ids.device, dtype=torch.long
    )
    lines = line_ids.detach().cpu().tolist()
    perturbations = perturbation_ids.detach().cpu().tolist()
    active_cpu = active.detach().cpu().tolist()
    for row, is_active in enumerate(active_cpu):
        if not is_active:
            continue
        lookup = bank.lookup(int(lines[row]), int(perturbations[row]))
        target_q[row] = torch.as_tensor(
            np.array(lookup.treated_zero_rate, copy=True),
            device=line_ids.device,
            dtype=torch.float32,
        )
        target_count[row] = int(lookup.treated_total_count)
    return target_q, target_count


def _expanded_groups(
    group_ids: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    batch_size, set_size = reference.shape[:2]
    if group_ids.ndim == 1 and group_ids.shape[0] == batch_size:
        return group_ids[:, None].expand(-1, set_size).to(torch.long)
    if tuple(group_ids.shape) != (batch_size, set_size):
        raise ValueError("group_ids must have shape [B] or [B,S]")
    return group_ids.to(torch.long)


def _observed_support_projection(
    expression: torch.Tensor,
    parent_mean: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float,
) -> SparseHurdleProjection:
    """Retain observed exact-zero support inside the fixed-Parent simplex."""

    if float(alpha) == 0.0:
        raise AssertionError("alpha=0 must return before hurdle projection")
    groups = _expanded_groups(group_ids, expression)
    valid = valid_mask
    if valid.ndim == 1:
        valid = valid[:, None].expand(-1, expression.shape[1])
    flat_valid = valid.reshape(-1)
    positions = torch.nonzero(flat_valid, as_tuple=False).flatten()
    raw_groups = groups.reshape(-1).index_select(0, positions)
    _, inverse = torch.unique(raw_groups, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse).to(torch.float32)
    values = expression.reshape(-1, expression.shape[-1]).index_select(
        0, positions
    )
    zero_sum = torch.zeros(
        counts.numel(),
        expression.shape[-1],
        device=expression.device,
        dtype=torch.float32,
    ).index_add_(0, inverse, (values == 0).to(torch.float32))
    q = zero_sum / counts[:, None]
    eps = 1.0e-6
    logits = torch.logit(q.clamp(min=eps, max=1.0 - eps))
    valid_logits = logits.index_select(0, inverse)
    flat_logits = torch.zeros(
        expression.shape[0] * expression.shape[1],
        expression.shape[-1],
        device=expression.device,
        dtype=torch.float32,
    ).index_copy_(0, positions, valid_logits)
    return project_ranked_hard_zero_fixed_parent_mean(
        expression,
        parent_mean,
        zero_rate_logits=flat_logits.reshape(expression.shape),
        alpha=float(alpha),
        rank_scores=expression,
        group_ids=group_ids,
        valid_mask=valid_mask,
    )


def prepare_end_to_end_hurdle_training_state(
    lightning,
    *,
    batch,
    prepared,
    context,
    supervision_mask: torch.Tensor,
    group_ids: torch.Tensor,
) -> EndToEndHurdleState | None:
    """Gate train labels before forming either H targets or S objectives."""

    hurdle_enabled = bool(
        getattr(lightning, "parent_residual_hurdle_geometry_enabled", False)
    )
    sparse_enabled = bool(
        getattr(lightning, "parent_residual_sparse_support_enabled", False)
    )
    alpha = float(_cfg(lightning, "parent_residual_hurdle_alpha", 1.0))
    if not lightning.training:
        return None
    if not sparse_enabled and (not hurdle_enabled or alpha == 0.0):
        return None

    active = supervision_mask[:, 0].to(torch.bool)
    line_ids = prepared.parent.lookup.artifact_cellline_ids.to(
        device=batch["pert_emb"].device, dtype=torch.long
    )
    perturbation_ids = prepared.parent.lookup.artifact_perturbation_ids.to(
        device=batch["pert_emb"].device, dtype=torch.long
    )
    # This lookup is deliberately first: audit/validation/test rows fail before
    # observed target support or any treated statistic is accessed.
    target_q, target_count = _strict_hurdle_targets(
        lightning, line_ids, perturbation_ids, active
    )
    control_q = _control_zero_rate(lightning, line_ids).to(
        batch["pert_emb"].device
    )

    adapter_output = None
    de_labels = None
    if sparse_enabled:
        de_lookup = lightning.parent_residual_de_aware_bank.lookup(
            line_ids, perturbation_ids, active_mask=active
        )
        de_labels = de_lookup.labels
        adapter_output = lightning.model.predict_sparse_support(
            context, prepared.parent, control_q
        )
        if not torch.equal(
            adapter_output.adjusted_parent_delta,
            adapter_output.base_parent_delta,
        ):
            raise AssertionError("S changed Parent delta despite alpha_mu=0")

    observed_projection = None
    target_expression = batch["pert_emb"]
    if hurdle_enabled and alpha > 0.0:
        observed_projection = _observed_support_projection(
            batch["pert_emb"],
            prepared.parent.parent_mean,
            group_ids=group_ids,
            valid_mask=supervision_mask,
            alpha=alpha,
        )
        target_expression = observed_projection.value.detach()

    condition_ids = (
        line_ids * int(lightning.parent_residual_hurdle_bank.train_condition_mask.shape[1])
        + perturbation_ids
    )
    return EndToEndHurdleState(
        target_expression=target_expression,
        adapter_output=adapter_output,
        control_zero_rate=control_q,
        target_zero_rate=target_q,
        target_count=target_count,
        de_labels=de_labels,
        condition_ids=condition_ids,
        active_mask=active,
        observed_projection=observed_projection,
    )


def _row_groups(
    compact_group_ids: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if compact_group_ids.ndim == 1:
        groups = compact_group_ids.to(torch.long)
    elif compact_group_ids.ndim == 2 and compact_group_ids.shape[0] == active.shape[0]:
        groups = torch.full(
            (compact_group_ids.shape[0],),
            -1,
            device=compact_group_ids.device,
            dtype=torch.long,
        )
        for row in torch.nonzero(active, as_tuple=False).flatten():
            valid_groups = compact_group_ids[row][compact_group_ids[row] >= 0]
            unique = torch.unique(valid_groups)
            if unique.numel() != 1:
                raise ValueError("each active H/S row must contain exactly one condition group")
            groups[row] = unique[0].to(torch.long)
    else:
        raise ValueError("compact_group_ids must have shape [B] or [B,S]")
    if (groups[active] < 0).any():
        raise ValueError("active H/S row has an invalid compact group")
    return groups


def _macro_rows(
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = value[active]
    selected_groups = groups[active]
    unique, inverse = torch.unique(
        selected_groups, sorted=True, return_inverse=True
    )
    result = torch.zeros(
        unique.numel(), value.shape[1], device=value.device, dtype=value.dtype
    ).index_add_(0, inverse, selected)
    counts = torch.bincount(inverse).to(value.dtype)
    return result / counts[:, None], unique, inverse


def _group_first(
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
    unique: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for group in unique:
        positions = torch.nonzero(
            active & (groups == group), as_tuple=False
        ).flatten()
        rows.append(value[positions[0]])
        if not torch.equal(
            value.index_select(0, positions),
            value[positions[0]].expand_as(value.index_select(0, positions)),
        ):
            raise ValueError("one condition group contains inconsistent targets")
    return torch.stack(rows)


def _hard_pairwise_rank_loss(
    score: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    temperature: float,
    max_positive: int,
    max_negative: int,
) -> torch.Tensor:
    positives = torch.nonzero(labels, as_tuple=False).flatten()
    negatives = torch.nonzero(~labels, as_tuple=False).flatten()
    if positives.numel() == 0 or negatives.numel() == 0:
        return score.sum() * 0.0
    positive_scores = score.index_select(0, positives)
    negative_scores = score.index_select(0, negatives)
    if positive_scores.numel() > int(max_positive):
        keep = torch.topk(
            -positive_scores.detach(), int(max_positive)
        ).indices
        positive_scores = positive_scores.index_select(0, keep)
    if negative_scores.numel() > int(max_negative):
        keep = torch.topk(
            negative_scores.detach(), int(max_negative)
        ).indices
        negative_scores = negative_scores.index_select(0, keep)
    pairwise = (
        negative_scores[:, None]
        - positive_scores[None, :]
        + float(margin)
    ) / float(temperature)
    return F.softplus(pairwise).mean().to(score)


def _condition_head_loss(
    state: EndToEndHurdleState,
    groups: torch.Tensor,
    *,
    false_positive_weight: float = 4.0,
    confidence_tau: float = 64.0,
    hard_rank_weight: float = 0.0,
    hard_rank_margin: float = 0.0,
    hard_rank_temperature: float = 0.5,
    hard_rank_max_positive: int = 256,
    hard_rank_max_negative: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    output = state.adapter_output
    if output is None or state.de_labels is None:
        raise AssertionError("S head loss requires adapter output and DE labels")
    rank_logits, unique, _ = _macro_rows(
        output.ranking_support_logits.float(), groups, state.active_mask
    )
    calibrated, _, _ = _macro_rows(
        output.calibrated_support_probability.float(), groups, state.active_mask
    )
    zero_logits, _, _ = _macro_rows(
        output.predicted_zero_logits.float(), groups, state.active_mask
    )
    labels = _group_first(
        state.de_labels, groups, state.active_mask, unique
    ).float()
    target_q = _group_first(
        state.target_zero_rate, groups, state.active_mask, unique
    ).float()
    target_count = _group_first(
        state.target_count[:, None], groups, state.active_mask, unique
    )[:, 0].float()

    bce = F.binary_cross_entropy_with_logits(
        rank_logits, labels, reduction="none"
    )
    effective_false_positive_weight = (
        1.0
        if bool(getattr(output, "uses_unified_gene_program", False))
        else float(false_positive_weight)
    )
    fp_weight = torch.where(
        labels > 0.5,
        torch.ones_like(labels),
        torch.full_like(labels, effective_false_positive_weight),
    )
    de_bce = (bce * fp_weight).mean(dim=1).mean()
    count_budget = F.smooth_l1_loss(
        calibrated.mean(dim=1), labels.mean(dim=1)
    )
    eps = 1.0e-6
    entropy = -(
        target_q * torch.log(target_q.clamp_min(eps))
        + (1.0 - target_q) * torch.log((1.0 - target_q).clamp_min(eps))
    )
    zero_kl = (
        F.softplus(zero_logits) - target_q * zero_logits - entropy
    ).clamp_min(0.0)
    confidence = target_count / (target_count + float(confidence_tau))
    zero_loss = (confidence * zero_kl.mean(dim=1)).mean()
    hard_rank = rank_logits.sum() * 0.0
    if float(hard_rank_weight) > 0.0:
        rank_terms = [
            _hard_pairwise_rank_loss(
                row,
                row_labels.to(torch.bool),
                margin=hard_rank_margin,
                temperature=hard_rank_temperature,
                max_positive=hard_rank_max_positive,
                max_negative=hard_rank_max_negative,
            )
            for row, row_labels in zip(rank_logits, labels)
        ]
        if rank_terms:
            hard_rank = torch.stack(rank_terms).mean()
    hard_rank_weighted = float(hard_rank_weight) * hard_rank
    total = de_bce + 0.25 * count_budget + zero_loss + hard_rank_weighted
    return (
        total,
        {
            "de_bce": de_bce,
            "de_count_budget": count_budget,
            "zero_count_kl": zero_loss,
            "hard_pairwise_rank": hard_rank,
            "hard_pairwise_rank_weighted": hard_rank_weighted,
            "mean_qhat": torch.sigmoid(zero_logits).mean().detach(),
        },
        torch.sigmoid(zero_logits).detach(),
        unique,
    )


def _endpoint_zero_consistency(
    prediction: torch.Tensor,
    compact_group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    qhat: torch.Tensor,
    unique_groups: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    valid = valid_mask
    if valid.ndim == 1:
        valid = valid[:, None].expand(-1, prediction.shape[1])
    flat_valid = valid.reshape(-1)
    positions = torch.nonzero(flat_valid, as_tuple=False).flatten()
    groups = _expanded_groups(compact_group_ids, prediction)
    selected_groups = groups.reshape(-1).index_select(0, positions)
    values = prediction.reshape(-1, prediction.shape[-1]).index_select(
        0, positions
    ).float()
    soft_zero = torch.exp(
        -values.clamp_min(0.0) / float(temperature)
    )
    rows = []
    for group in unique_groups:
        local = soft_zero[selected_groups == group].mean(dim=0)
        rows.append(local)
    predicted_q = torch.stack(rows).clamp(1.0e-6, 1.0 - 1.0e-6)
    teacher = qhat.clamp(1.0e-6, 1.0 - 1.0e-6)
    kl = (
        teacher * (torch.log(teacher) - torch.log(predicted_q))
        + (1.0 - teacher)
        * (torch.log1p(-teacher) - torch.log1p(-predicted_q))
    )
    return kl.mean(dim=1).mean().to(prediction)


def _distance_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = prediction - target
    l1 = difference.abs().mean(dim=1)
    l2 = (difference.square().mean(dim=1) + 1.0e-8).sqrt()
    cosine = 1.0 - F.cosine_similarity(
        prediction, target, dim=1, eps=1.0e-8
    )
    return l1, l2, cosine


def _signature_pds_no_reg(
    lightning,
    *,
    prepared,
    state: EndToEndHurdleState,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = state.active_mask
    correction_storage = prepared.parent.correction
    scale = float(lightning.model.correction_max_scale)
    # Quantize the scale in raw_correction's computation dtype before comparing
    # in the correction storage dtype. This accepts bf16's rounded boundary,
    # but rejects the next bf16 value even after promotion to float32.
    bound_dtype = prepared.parent.raw_correction.dtype
    representable_scale = torch.tensor(
        scale,
        dtype=bound_dtype,
        device=correction_storage.device,
    ).abs().to(dtype=correction_storage.dtype)
    if (correction_storage[active].abs() > representable_scale).any():
        raise AssertionError("bounded Parent correction exceeded configured scale")
    correction, unique, _ = _macro_rows(
        correction_storage.float(), groups, active
    )
    prior_delta, _, _ = _macro_rows(
        prepared.parent.lookup.prior_delta.to(correction).float(),
        groups,
        active,
    )
    target_delta, _, _ = _macro_rows(
        prepared.parent.lookup.target_delta.to(correction).float(),
        groups,
        active,
    )
    if state.adapter_output is None:
        raise AssertionError("signature loss requires calibrated S support")
    gate, _, _ = _macro_rows(
        state.adapter_output.calibrated_support_probability.float(),
        groups,
        active,
    )
    condition_ids = _group_first(
        state.condition_ids[:, None], groups, active, unique
    )[:, 0]
    detached_gate = gate.detach()
    target_signature = detached_gate * (target_delta - prior_delta).clamp(
        min=-scale, max=scale
    ).detach()

    support_fit = torch.stack(
        _distance_rows(correction, target_signature), dim=0
    ).mean()
    null_sparse = ((1.0 - detached_gate) * correction).abs().mean()
    pds_terms = [[], [], []]
    candidate_counts = []
    margin = float(
        _cfg(lightning, "parent_residual_sparse_signature_pds_margin", 5.0e-3)
    )
    predicted_delta = prior_delta + correction
    positive = _distance_rows(predicted_delta, target_delta.detach())
    prior_bank = lightning.model.prior_bank
    train_mask = prior_bank.train_target_mask
    train_delta = prior_bank.train_target_delta
    perturbation_count = int(train_mask.shape[1])
    for row in range(correction.shape[0]):
        condition = condition_ids[row].to(torch.long)
        line = torch.div(
            condition, perturbation_count, rounding_mode="floor"
        )
        perturbation = torch.remainder(condition, perturbation_count)
        candidate_mask = train_mask[line].clone()
        candidate_mask[perturbation] = False
        candidate_indices = torch.nonzero(
            candidate_mask, as_tuple=False
        ).flatten()
        if candidate_indices.numel() == 0:
            continue
        candidate_counts.append(candidate_indices.numel())
        minima = [None, None, None]
        for start in range(0, candidate_indices.numel(), 256):
            indices = candidate_indices[start : start + 256]
            negative_targets = train_delta[line].index_select(
                0, indices
            ).to(correction).detach()
            repeated = predicted_delta[row : row + 1].expand_as(
                negative_targets
            )
            distances = _distance_rows(repeated, negative_targets)
            for index in range(3):
                local = distances[index].min()
                minima[index] = (
                    local
                    if minima[index] is None
                    else torch.minimum(minima[index], local)
                )
        for index in range(3):
            pds_terms[index].append(
                F.relu(
                    positive[index][row] - minima[index] + margin
                )
            )
    zero = correction.sum() * 0.0
    pds_components = [
        torch.stack(values).mean() if values else zero for values in pds_terms
    ]
    pds = torch.stack(
        (
            float(
                _cfg(
                    lightning,
                    "parent_residual_sparse_signature_pds_l1_weight",
                    1.0,
                )
            )
            * pds_components[0],
            float(
                _cfg(
                    lightning,
                    "parent_residual_sparse_signature_pds_l2_weight",
                    1.0,
                )
            )
            * pds_components[1],
            float(
                _cfg(
                    lightning,
                    "parent_residual_sparse_signature_pds_cos_weight",
                    1.0,
                )
            )
            * pds_components[2],
        )
    ).mean()

    predicted_dist = _distance_rows(predicted_delta, target_delta.detach())
    base_dist = _distance_rows(prior_delta.detach(), target_delta.detach())
    no_reg = torch.stack(
        [
            F.relu(predicted_dist[index] - base_dist[index])
            for index in range(3)
        ],
        dim=0,
    ).mean()
    candidate_tensor = correction.new_tensor(
        candidate_counts, dtype=torch.float32
    )
    total = (
        float(
            _cfg(
                lightning,
                "parent_residual_sparse_signature_support_weight",
                1.0,
            )
        )
        * support_fit
        + float(
            _cfg(
                lightning,
                "parent_residual_sparse_signature_null_weight",
                1.0,
            )
        )
        * null_sparse
        + pds
        + float(
            _cfg(
                lightning,
                "parent_residual_sparse_signature_no_reg_weight",
                1.0,
            )
        )
        * no_reg
    )
    return total, {
        "pds_l1": pds_components[0],
        "pds_l2": pds_components[1],
        "pds_cosine": pds_components[2],
        "no_regression": no_reg,
        "support_fit": support_fit,
        "null_sparse": null_sparse,
        "support_correction_rms": (
            detached_gate * correction
        ).square().mean().sqrt(),
        "null_correction_rms": (
            (1.0 - detached_gate) * correction
        ).square().mean().sqrt(),
        "pds_bank_coverage": correction.new_tensor(
            len(pds_terms[0]) / max(1, correction.shape[0])
        ),
        "pds_candidate_count_mean": (
            candidate_tensor.mean() if candidate_counts else zero
        ),
        "pds_candidate_count_min": (
            candidate_tensor.min() if candidate_counts else zero
        ),
    }


def augment_locked_result_with_end_to_end_sparse_hurdle(
    lightning,
    *,
    result,
    flow_output,
    prepared,
    supervision_mask,
) -> None:
    state = flow_output.get("end_to_end_hurdle_state")
    if state is None:
        return
    if state.observed_projection is not None:
        result.update(
            {
                "parent_residual_hurdle_target_used": flow_output["loss"].new_tensor(1.0),
                "parent_residual_hurdle_target_zero_fraction": state.observed_projection.realized_zero_fraction.mean().detach(),
                "parent_residual_hurdle_target_parent_drift": state.observed_projection.group_mean_drift_max_abs.detach(),
            }
        )
    if not bool(
        getattr(lightning, "parent_residual_sparse_support_enabled", False)
    ):
        return

    groups = _row_groups(
        flow_output["pair"].target_projection.group_ids, state.active_mask
    )
    head, head_logs, qhat, unique = _condition_head_loss(
        state,
        groups,
        confidence_tau=float(
            _cfg(
                lightning,
                "parent_residual_sparse_zero_rate_confidence_tau",
                64.0,
            )
        ),
        hard_rank_weight=float(
            _cfg(lightning, "parent_residual_sparse_hard_rank_weight", 0.0)
        ),
        hard_rank_margin=float(
            _cfg(lightning, "parent_residual_sparse_hard_rank_margin", 0.0)
        ),
        hard_rank_temperature=float(
            _cfg(
                lightning,
                "parent_residual_sparse_hard_rank_temperature",
                0.5,
            )
        ),
        hard_rank_max_positive=int(
            _cfg(
                lightning,
                "parent_residual_sparse_hard_rank_max_positive",
                256,
            )
        ),
        hard_rank_max_negative=int(
            _cfg(
                lightning,
                "parent_residual_sparse_hard_rank_max_negative",
                256,
            )
        ),
    )
    endpoint = _endpoint_zero_consistency(
        flow_output["prediction"],
        flow_output["pair"].target_projection.group_ids,
        supervision_mask,
        qhat,
        unique,
        temperature=float(
            _cfg(lightning, "parent_residual_sparse_zero_temperature", 3.0e-3)
        ),
    )
    fraction = warmup_fraction(
        int(lightning.global_step),
        int(_cfg(lightning, "parent_residual_sparse_warmup_steps", 500)),
    )
    head_weighted = (
        float(_cfg(lightning, "parent_residual_sparse_head_weight", 1.0e-3))
        * float(fraction)
        * head
    )
    endpoint_weighted = (
        float(
            _cfg(
                lightning,
                "parent_residual_sparse_endpoint_consistency_weight",
                5.0e-4,
            )
        )
        * float(fraction)
        * endpoint
    )
    endpoint_capped, gradient_logs = cap_auxiliary_gradient(
        base_loss=flow_output.get("centered_flow_loss", flow_output["loss"]),
        auxiliary_loss=endpoint_weighted,
        proxy=flow_output["raw_velocity"],
        maximum_ratio=(0.0 if flow_output.get("no_flow", False) else float(
            _cfg(lightning, "parent_residual_sparse_gradient_cap_ratio", 0.2)
        )),
        eps=float(
            _cfg(lightning, "parent_residual_sparse_gradient_cap_eps", 1.0e-12)
        ),
    )
    result["loss"] = result["loss"] + head_weighted + endpoint_capped
    result.update(
        {
            "parent_residual_sparse_total_raw": (
                head_weighted + endpoint_weighted
            ).detach(),
            "parent_residual_sparse_total_capped": (
                head_weighted + endpoint_capped
            ).detach(),
            "parent_residual_sparse_warmup_fraction": endpoint.new_tensor(fraction).detach(),
            "parent_residual_sparse_head": head.detach(),
            "parent_residual_sparse_head_weighted": head_weighted.detach(),
            "parent_residual_sparse_endpoint_consistency": endpoint.detach(),
            "parent_residual_sparse_endpoint_capped": endpoint_capped.detach(),
            "parent_residual_sparse_de_bce": head_logs["de_bce"].detach(),
            "parent_residual_sparse_de_count_budget": head_logs["de_count_budget"].detach(),
            "parent_residual_sparse_zero_count_kl": head_logs["zero_count_kl"].detach(),
            "parent_residual_sparse_hard_pairwise_rank": head_logs["hard_pairwise_rank"].detach(),
            "parent_residual_sparse_hard_pairwise_rank_weighted": head_logs["hard_pairwise_rank_weighted"].detach(),
            "parent_residual_sparse_mean_qhat": head_logs["mean_qhat"],
            "parent_residual_sparse_endpoint_base_gradient_norm": gradient_logs["base_gradient_norm"],
            "parent_residual_sparse_endpoint_gradient_norm_pre_cap": gradient_logs["aux_gradient_norm_pre_cap"],
            "parent_residual_sparse_endpoint_gradient_ratio_pre_cap": gradient_logs["aux_gradient_ratio_pre_cap"],
            "parent_residual_sparse_endpoint_gradient_cap_scale": gradient_logs["aux_gradient_cap_scale"],
            "parent_residual_sparse_endpoint_gradient_ratio_post_cap": gradient_logs["aux_gradient_ratio_post_cap"],
        }
    )


def augment_parent_signature_with_sparse_support(
    lightning,
    *,
    result,
    flow_output,
    prepared,
    parent_base_loss: torch.Tensor,
) -> None:
    """Cap the support-aware Parent signature only against Parent training."""

    if flow_output is None or not bool(
        getattr(lightning, "parent_residual_sparse_support_enabled", False)
    ):
        return
    state = flow_output.get("end_to_end_hurdle_state")
    if state is None:
        return
    groups = _row_groups(
        flow_output["pair"].target_projection.group_ids, state.active_mask
    )
    signature, logs = _signature_pds_no_reg(
        lightning, prepared=prepared, state=state, groups=groups
    )
    fraction = warmup_fraction(
        int(lightning.global_step),
        int(_cfg(lightning, "parent_residual_sparse_warmup_steps", 500)),
    )
    weighted = (
        float(
            _cfg(
                lightning,
                "parent_residual_sparse_signature_weight",
                1.0e-3,
            )
        )
        * float(fraction)
        * signature
    )
    capped, gradient_logs = cap_auxiliary_gradient(
        base_loss=parent_base_loss,
        auxiliary_loss=weighted,
        proxy=prepared.parent.correction,
        maximum_ratio=float(
            _cfg(lightning, "parent_residual_sparse_gradient_cap_ratio", 0.2)
        ),
        eps=float(
            _cfg(lightning, "parent_residual_sparse_gradient_cap_eps", 1.0e-12)
        ),
    )
    result["loss"] = result["loss"] + capped
    result["parent_residual_sparse_total_raw"] = (
        result["parent_residual_sparse_total_raw"] + weighted.detach()
    )
    result["parent_residual_sparse_total_capped"] = (
        result["parent_residual_sparse_total_capped"] + capped.detach()
    )
    result.update(
        {
            "parent_residual_sparse_signature": signature.detach(),
            "parent_residual_sparse_signature_capped": capped.detach(),
            "parent_residual_sparse_pds_l1": logs["pds_l1"].detach(),
            "parent_residual_sparse_pds_l2": logs["pds_l2"].detach(),
            "parent_residual_sparse_pds_cosine": logs["pds_cosine"].detach(),
            "parent_residual_sparse_no_regression": logs["no_regression"].detach(),
            "parent_residual_sparse_support_fit": logs["support_fit"].detach(),
            "parent_residual_sparse_null_sparse": logs["null_sparse"].detach(),
            "parent_residual_sparse_support_correction_rms": logs["support_correction_rms"].detach(),
            "parent_residual_sparse_null_correction_rms": logs["null_correction_rms"].detach(),
            "parent_residual_sparse_pds_bank_coverage": logs["pds_bank_coverage"].detach(),
            "parent_residual_sparse_pds_candidate_count_mean": logs["pds_candidate_count_mean"].detach(),
            "parent_residual_sparse_pds_candidate_count_min": logs["pds_candidate_count_min"].detach(),
            "parent_residual_sparse_signature_base_gradient_norm": gradient_logs["base_gradient_norm"],
            "parent_residual_sparse_signature_gradient_norm_pre_cap": gradient_logs["aux_gradient_norm_pre_cap"],
            "parent_residual_sparse_signature_gradient_ratio_pre_cap": gradient_logs["aux_gradient_ratio_pre_cap"],
            "parent_residual_sparse_signature_gradient_cap_scale": gradient_logs["aux_gradient_cap_scale"],
            "parent_residual_sparse_signature_gradient_ratio_post_cap": gradient_logs["aux_gradient_ratio_post_cap"],
        }
    )


@torch.no_grad()
def attach_end_to_end_hurdle_inference_fields(
    lightning,
    *,
    batch,
    prepared,
    context,
) -> dict[str, torch.Tensor]:
    """Compute target-free control-q/qhat once and reuse it at the terminal."""

    hurdle_enabled = bool(
        getattr(lightning, "parent_residual_hurdle_geometry_enabled", False)
    )
    sparse_enabled = bool(
        getattr(lightning, "parent_residual_sparse_support_enabled", False)
    )
    alpha = float(_cfg(lightning, "parent_residual_hurdle_alpha", 1.0))
    if not hurdle_enabled and not sparse_enabled:
        return {}
    if not sparse_enabled and alpha == 0.0:
        return {}
    line_ids = prepared.parent.lookup.artifact_cellline_ids.to(
        device=prepared.parent.parent_mean.device, dtype=torch.long
    )
    control_q = _control_zero_rate(lightning, line_ids).to(
        prepared.parent.parent_mean
    )
    eps = 1.0e-6
    control_logits = torch.logit(control_q.clamp(eps, 1.0 - eps))
    fields = {
        "parent_residual_hurdle_control_zero_rate": control_q,
        "parent_residual_hurdle_control_zero_logits": control_logits,
    }
    if sparse_enabled:
        output = lightning.model.predict_sparse_support(
            context, prepared.parent, control_q
        )
        if not torch.equal(
            output.adjusted_parent_delta, output.base_parent_delta
        ):
            raise AssertionError("inference S changed Parent at alpha_mu=0")
        fields.update(
            {
                "parent_residual_sparse_ranking_probability": output.ranking_probability,
                "parent_residual_sparse_gate_probability": output.calibrated_support_probability,
                "parent_residual_sparse_predicted_zero_rate": output.predicted_zero_rate,
                "parent_residual_sparse_predicted_zero_logits": output.predicted_zero_logits,
            }
        )
    if hurdle_enabled and alpha > 0.0:
        fields["parent_residual_hurdle_terminal_zero_logits"] = (
            fields["parent_residual_sparse_predicted_zero_logits"]
            if sparse_enabled
            else control_logits
        )
    batch.update(fields)
    return fields


def _lexicographic_cell_rank_scores(
    expression: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    cell_keys: Optional[torch.Tensor],
) -> torch.Tensor:
    """Unique ordinal ranks for (expression primary, stable cell key secondary)."""

    if cell_keys is None:
        raise ValueError("enabled terminal H requires stable cell_key values")
    if not torch.is_tensor(cell_keys) or cell_keys.device != expression.device:
        raise ValueError("cell_keys must be a tensor on the expression device")
    if cell_keys.dtype not in (torch.int32, torch.int64):
        raise TypeError("cell_keys must be an integer tensor")
    if tuple(cell_keys.shape) != tuple(expression.shape[:2]):
        raise ValueError("cell_keys must have shape [B,S]")
    groups = _expanded_groups(group_ids, expression)
    valid = valid_mask
    if valid.ndim == 1:
        valid = valid[:, None].expand(-1, expression.shape[1])
    if tuple(valid.shape) != tuple(expression.shape[:2]):
        raise ValueError("valid_mask must have shape [B] or [B,S]")

    genes = expression.shape[-1]
    flat_value = expression.detach().reshape(-1, genes)
    flat_groups = groups.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_keys = cell_keys.reshape(-1)
    scores = torch.zeros_like(flat_value)
    for group in torch.unique(flat_groups[flat_valid], sorted=True):
        positions = torch.nonzero(
            flat_valid & (flat_groups == group), as_tuple=False
        ).flatten()
        keys = flat_keys.index_select(0, positions)
        if (keys < 0).any() or torch.unique(keys).numel() != keys.numel():
            raise ValueError(
                "terminal H requires unique nonnegative cell_key per group"
            )
        values = flat_value.index_select(0, positions)
        key_order = torch.argsort(keys, stable=True)
        key_sorted_values = values.index_select(0, key_order)
        value_order = torch.argsort(
            key_sorted_values, dim=0, stable=True
        )
        original_order = key_order[:, None].expand(-1, genes).gather(
            0, value_order
        )
        ordinal = torch.arange(
            positions.numel(), device=expression.device, dtype=expression.dtype
        )[:, None].expand(-1, genes)
        group_scores = torch.empty_like(values).scatter(
            0, original_order, ordinal
        )
        scores = scores.index_copy(0, positions, group_scores)
    return scores.reshape_as(expression)


def apply_end_to_end_terminal_hurdle(
    lightning,
    *,
    expression: torch.Tensor,
    parent_mean: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    zero_rate_logits: Optional[torch.Tensor],
    cell_keys: Optional[torch.Tensor] = None,
    reference_projection: Optional[NonnegativeFixedMeanProjection] = None,
) -> tuple[torch.Tensor, Optional[SparseHurdleProjection]]:
    """Apply H exactly once; disabled/alpha0 returns the original object."""

    if not bool(
        getattr(lightning, "parent_residual_hurdle_geometry_enabled", False)
    ):
        return expression, None
    alpha = float(_cfg(lightning, "parent_residual_hurdle_alpha", 1.0))
    if alpha == 0.0:
        return expression, None
    if zero_rate_logits is None:
        raise ValueError("enabled terminal H is missing target-free zero logits")
    rank_scores = _lexicographic_cell_rank_scores(
        expression,
        group_ids=group_ids,
        valid_mask=valid_mask,
        cell_keys=cell_keys,
    )
    projection = project_ranked_hard_zero_fixed_parent_mean(
        expression,
        parent_mean,
        zero_rate_logits=zero_rate_logits,
        alpha=alpha,
        rank_scores=rank_scores,
        group_ids=group_ids,
        valid_mask=valid_mask,
        reference_projection=reference_projection,
    )
    if int(projection.tie_expansion_count.detach().cpu()) != 0:
        raise RuntimeError(
            "terminal hurdle found tied positive-cell ranks; refusing an "
            "ambiguous zero-count expansion"
        )
    projection = repair_terminal_hard_zero_fixed_support(
        expression,
        projection,
        repair_rank=rank_scores,
    )
    if not torch.equal(
        projection.requested_zero_fraction,
        projection.realized_zero_fraction,
    ):
        count_scale = projection.counts.to(
            device=projection.value.device,
            dtype=projection.requested_zero_fraction.dtype,
        )[:, None]
        requested_count = torch.round(
            projection.requested_zero_fraction * count_scale
        ).to(torch.long)
        realized_count = torch.round(
            projection.realized_zero_fraction * count_scale
        ).to(torch.long)
        count_delta = realized_count - requested_count
        mismatch = count_delta != 0
        first_group, first_gene = torch.nonzero(
            mismatch, as_tuple=False
        )[0].detach().cpu().tolist()
        raise RuntimeError(
            "terminal hurdle realized a non-exact zero budget: "
            f"mismatch_columns={int(mismatch.sum().item())}, "
            f"extra_zero_cells={int(count_delta.clamp_min(0).sum().item())}, "
            f"missing_zero_cells={int((-count_delta).clamp_min(0).sum().item())}, "
            f"max_abs_count_delta={int(count_delta.abs().max().item())}, "
            f"first_group={first_group}, first_gene={first_gene}, "
            f"first_requested={int(requested_count[first_group, first_gene].item())}, "
            f"first_realized={int(realized_count[first_group, first_gene].item())}"
        )
    return projection.value, projection


__all__ = [
    "EndToEndHurdleState",
    "apply_end_to_end_terminal_hurdle",
    "attach_end_to_end_hurdle_inference_fields",
    "augment_locked_result_with_end_to_end_sparse_hurdle",
    "augment_parent_signature_with_sparse_support",
    "initialize_end_to_end_sparse_hurdle",
    "prepare_end_to_end_hurdle_training_state",
]

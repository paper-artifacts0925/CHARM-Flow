"""Condition-macro objectives for the opt-in sparse hurdle adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .output import SparseHurdleDEAdapterOutput


@dataclass(frozen=True)
class SparseHurdleDELossConfig:
    """Weights and stable margins for the independently gated objective."""

    false_positive_weight: float = 4.0
    de_bce_weight: float = 1.0
    de_count_budget_weight: float = 0.25
    zero_rate_count_nll_weight: float = 1.0
    zero_rate_confidence_tau: float = 64.0
    virtual_delta_l1_weight: float = 0.25
    virtual_delta_l2_weight: float = 0.25
    virtual_delta_cosine_weight: float = 0.1
    pds_l1_weight: float = 0.1
    pds_l2_weight: float = 0.1
    pds_cosine_weight: float = 0.1
    no_regression_weight: float = 0.25
    pds_margin: float = 0.05
    no_regression_tolerance: float = 0.0
    eps: float = 1.0e-6

    def validate(self) -> "SparseHurdleDELossConfig":
        for name, value in (
            ("false_positive_weight", self.false_positive_weight),
            ("de_bce_weight", self.de_bce_weight),
            ("de_count_budget_weight", self.de_count_budget_weight),
            ("zero_rate_count_nll_weight", self.zero_rate_count_nll_weight),
            ("virtual_delta_l1_weight", self.virtual_delta_l1_weight),
            ("virtual_delta_l2_weight", self.virtual_delta_l2_weight),
            (
                "virtual_delta_cosine_weight",
                self.virtual_delta_cosine_weight,
            ),
            ("pds_l1_weight", self.pds_l1_weight),
            ("pds_l2_weight", self.pds_l2_weight),
            ("pds_cosine_weight", self.pds_cosine_weight),
            ("no_regression_weight", self.no_regression_weight),
            ("pds_margin", self.pds_margin),
            ("no_regression_tolerance", self.no_regression_tolerance),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.false_positive_weight) <= 0.0:
            raise ValueError("false_positive_weight must be positive")
        if (
            not math.isfinite(float(self.zero_rate_confidence_tau))
            or float(self.zero_rate_confidence_tau) <= 0.0
        ):
            raise ValueError(
                "zero_rate_confidence_tau must be finite and positive"
            )
        if (
            not math.isfinite(float(self.eps))
            or not 0.0 < float(self.eps) < 0.5
        ):
            raise ValueError("eps must lie in (0, 0.5)")
        return self


@dataclass(frozen=True)
class SparseHurdleDELossOutput:
    """Total and unweighted components of the adapter objective."""

    total: torch.Tensor
    de_bce: torch.Tensor
    de_count_budget: torch.Tensor
    zero_rate_count_nll: torch.Tensor
    virtual_delta_l1: torch.Tensor
    virtual_delta_l2: torch.Tensor
    virtual_delta_cosine: torch.Tensor
    pds_l1: torch.Tensor
    pds_l2: torch.Tensor
    pds_cosine: torch.Tensor
    no_regression_hinge: torch.Tensor
    predicted_de_rate: torch.Tensor
    target_de_rate: torch.Tensor
    soft_false_positive_rate: torch.Tensor
    mean_ranking_probability: torch.Tensor
    mean_negative_calibrated_probability: torch.Tensor
    mean_zero_rate_confidence: torch.Tensor
    valid_pds_conditions: torch.Tensor


def _validate_gene_target(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.device != device:
        raise ValueError(f"{name} must share the adapter output device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _row_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _macro_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _row_mean(value, mask).mean()


def _cosine_distance_rows(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weight = mask.to(left.dtype)
    left_masked = left * weight
    right_masked = right * weight
    dot = (left_masked * right_masked).sum(dim=1)
    left_norm = left_masked.square().sum(dim=1).sqrt()
    right_norm = right_masked.square().sum(dim=1).sqrt()
    denominator = left_norm * right_norm
    cosine = dot / denominator.clamp_min(float(eps))
    both_zero = (left_norm <= float(eps)) & (right_norm <= float(eps))
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return 1.0 - cosine.clamp(min=-1.0, max=1.0)


def _matched_distances(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = prediction - target
    l1 = _row_mean(difference.abs(), mask)
    mean_square = _row_mean(difference.square(), mask)
    l2 = (mean_square + float(eps)).sqrt() - math.sqrt(float(eps))
    cosine = _cosine_distance_rows(prediction, target, mask, eps)
    return l1, l2.clamp_min(0.0), cosine


def _pds_hard_negative_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    condition_ids: torch.Tensor,
    *,
    margin: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    losses_l1 = []
    losses_l2 = []
    losses_cosine = []
    batch_size = prediction.shape[0]
    for row in range(batch_size):
        negative = condition_ids != condition_ids[row]
        if not negative.any():
            continue
        selected = mask[row]
        local_prediction = prediction[row, selected]
        local_targets = target[:, selected]
        difference = local_targets - local_prediction[None, :]
        distance_l1 = difference.abs().mean(dim=1)
        distance_l2 = (
            difference.square().mean(dim=1) + float(eps)
        ).sqrt() - math.sqrt(float(eps))
        prediction_norm = local_prediction.square().sum().sqrt()
        target_norm = local_targets.square().sum(dim=1).sqrt()
        denominator = prediction_norm * target_norm
        cosine = (
            (local_targets * local_prediction[None, :]).sum(dim=1)
            / denominator.clamp_min(float(eps))
        )
        cosine = torch.where(
            (prediction_norm <= float(eps)) & (target_norm <= float(eps)),
            torch.ones_like(cosine),
            cosine,
        )
        distance_cosine = 1.0 - cosine.clamp(min=-1.0, max=1.0)
        for distances, destination in (
            (distance_l1, losses_l1),
            (distance_l2, losses_l2),
            (distance_cosine, losses_cosine),
        ):
            positive_distance = distances[row]
            hard_negative = distances.masked_select(negative).min()
            destination.append(
                F.relu(positive_distance - hard_negative + float(margin))
            )

    zero = prediction.sum() * 0.0
    if not losses_l1:
        return zero, zero, zero, zero.detach()
    return (
        torch.stack(losses_l1).mean(),
        torch.stack(losses_l2).mean(),
        torch.stack(losses_cosine).mean(),
        prediction.new_tensor(float(len(losses_l1))).detach(),
    )


def sparse_hurdle_de_loss(
    output: SparseHurdleDEAdapterOutput,
    *,
    de_labels: torch.Tensor,
    target_zero_rate: torch.Tensor,
    target_virtual_delta: torch.Tensor,
    gene_mask: Optional[torch.Tensor] = None,
    zero_rate_counts: Optional[torch.Tensor] = None,
    condition_ids: Optional[torch.Tensor] = None,
    config: Optional[SparseHurdleDELossConfig] = None,
) -> SparseHurdleDELossOutput:
    """Compute false-positive-aware losses without touching the base graph."""

    cfg = (
        SparseHurdleDELossConfig() if config is None else config
    ).validate()
    logits = output.ranking_support_logits
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("adapter ranking logits must have shape [B,G]")
    batch_size, genes = logits.shape
    shape = (batch_size, genes)
    device = logits.device
    if not torch.is_tensor(de_labels):
        raise TypeError("de_labels must be a tensor")
    if de_labels.dtype == torch.bool:
        labels = de_labels.to(dtype=torch.float32)
    else:
        _validate_gene_target("de_labels", de_labels, shape, device)
        labels = de_labels.float()
    if tuple(de_labels.shape) != shape or de_labels.device != device:
        raise ValueError("de_labels must have shape [B,G] on the output device")
    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("de_labels must lie in [0, 1]")
    _validate_gene_target(
        "target_zero_rate", target_zero_rate, shape, device
    )
    _validate_gene_target(
        "target_virtual_delta", target_virtual_delta, shape, device
    )
    if ((target_zero_rate < 0) | (target_zero_rate > 1)).any():
        raise ValueError("target_zero_rate must lie in [0, 1]")
    if gene_mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        if (
            not torch.is_tensor(gene_mask)
            or gene_mask.dtype != torch.bool
            or tuple(gene_mask.shape) != shape
            or gene_mask.device != device
        ):
            raise ValueError("gene_mask must be boolean [B,G] on the output device")
        mask = gene_mask
    if not mask.any(dim=1).all():
        raise ValueError("each condition must retain at least one gene")

    if zero_rate_counts is None:
        counts = torch.ones(shape, device=device, dtype=torch.float32)
    else:
        if not torch.is_tensor(zero_rate_counts) or zero_rate_counts.dtype == torch.bool:
            raise TypeError("zero_rate_counts must be a numeric tensor")
        if zero_rate_counts.device != device:
            raise ValueError("zero_rate_counts must share the output device")
        if tuple(zero_rate_counts.shape) == (batch_size,):
            counts = zero_rate_counts[:, None].expand(-1, genes).float()
        elif tuple(zero_rate_counts.shape) == shape:
            counts = zero_rate_counts.float()
        else:
            raise ValueError("zero_rate_counts must have shape [B] or [B,G]")
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("zero_rate_counts must be finite and nonnegative")
        if not ((counts * mask).sum(dim=1) > 0).all():
            raise ValueError("each condition needs positive zero-rate count mass")
    counts = counts.detach()
    if condition_ids is None:
        ids = torch.arange(batch_size, device=device)
    else:
        if (
            not torch.is_tensor(condition_ids)
            or tuple(condition_ids.shape) != (batch_size,)
            or condition_ids.device != device
        ):
            raise ValueError("condition_ids must have shape [B] on the output device")
        if condition_ids.is_floating_point():
            raise TypeError("condition_ids must be discrete")
        ids = condition_ids

    labels = labels.detach()
    target_q = target_zero_rate.detach().float()
    target_delta = target_virtual_delta.detach().float()
    support_logits = logits.float()
    ranking_probability = output.ranking_probability.float()
    calibrated_probability = output.calibrated_support_probability.float()
    predicted_q = output.predicted_zero_rate.float().clamp(
        min=float(cfg.eps), max=1.0 - float(cfg.eps)
    )
    predicted_zero_logits = output.predicted_zero_logits.float()
    adjusted_delta = output.adjusted_parent_delta.float()
    base_delta = output.base_parent_delta.detach().float()
    uses_unified_program = bool(
        getattr(output, "uses_unified_gene_program", False)
    )
    expected_offset = (
        0.0
        if uses_unified_program
        else math.log(float(cfg.false_positive_weight))
    )
    offset = output.ranking_logit_calibration_offset
    if (
        not torch.is_tensor(offset)
        or offset.numel() != 1
        or not torch.isfinite(offset).all()
        or not math.isclose(
            float(offset.detach().cpu()),
            expected_offset,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise ValueError(
            "adapter ranking calibration offset disagrees with "
            "false_positive_weight"
        )
    for name, value in (
        ("ranking_probability", ranking_probability),
        ("calibrated_support_probability", calibrated_probability),
        ("predicted_zero_rate", predicted_q),
        ("predicted_zero_logits", predicted_zero_logits),
        ("adjusted_parent_delta", adjusted_delta),
        ("base_parent_delta", base_delta),
    ):
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"adapter {name} must be finite [B,G]")

    de_element = F.binary_cross_entropy_with_logits(
        support_logits, labels, reduction="none"
    )
    effective_false_positive_weight = (
        1.0
        if uses_unified_program
        else float(cfg.false_positive_weight)
    )
    fp_weight = torch.where(
        labels > 0.5,
        torch.ones_like(labels),
        torch.full_like(labels, effective_false_positive_weight),
    )
    de_bce = _macro_mean(de_element * fp_weight, mask)
    predicted_de_rate = _row_mean(calibrated_probability, mask)
    target_de_rate = _row_mean(labels, mask)
    de_count_budget = F.smooth_l1_loss(
        predicted_de_rate, target_de_rate
    )

    # Beta-binomial overdispersion gives an effective sample size that
    # saturates with n. With tau=(1-rho)/rho, normalized reliability is
    # c(n)=n/(n+tau). It is strictly increasing but bounded by one. We do not
    # divide by a sum of c(n), so count cannot algebraically cancel; the outer
    # row mean still gives every condition exactly one macro contribution.
    tau = float(cfg.zero_rate_confidence_tau)
    zero_rate_confidence = counts / (counts + tau)
    eps = float(cfg.eps)
    target_entropy = -(
        target_q * torch.log(target_q.clamp_min(eps))
        + (1.0 - target_q)
        * torch.log((1.0 - target_q).clamp_min(eps))
    )
    zero_rate_kl = (
        F.softplus(predicted_zero_logits)
        - target_q * predicted_zero_logits
        - target_entropy
    ).clamp_min(0.0)
    zero_rate_count_nll = _macro_mean(
        zero_rate_confidence * zero_rate_kl, mask
    )
    mean_zero_rate_confidence = _macro_mean(
        zero_rate_confidence, mask
    )

    virtual_l1_rows, virtual_l2_rows, virtual_cosine_rows = _matched_distances(
        adjusted_delta, target_delta, mask, cfg.eps
    )
    virtual_delta_l1 = virtual_l1_rows.mean()
    virtual_delta_l2 = virtual_l2_rows.mean()
    virtual_delta_cosine = virtual_cosine_rows.mean()
    pds_l1, pds_l2, pds_cosine, valid_pds = _pds_hard_negative_losses(
        adjusted_delta,
        target_delta,
        mask,
        ids,
        margin=cfg.pds_margin,
        eps=cfg.eps,
    )

    base_l1, base_l2, base_cosine = _matched_distances(
        base_delta, target_delta, mask, cfg.eps
    )
    tolerance = float(cfg.no_regression_tolerance)
    no_regression_hinge = torch.stack(
        (
            F.relu(virtual_l1_rows - base_l1 - tolerance),
            F.relu(virtual_l2_rows - base_l2 - tolerance),
            F.relu(virtual_cosine_rows - base_cosine - tolerance),
        ),
        dim=0,
    ).mean()

    negative_mask = mask & (labels <= 0.5)
    negative_counts = negative_mask.sum(dim=1)
    valid_negative = negative_counts > 0
    if valid_negative.any():
        denominator = negative_counts.clamp_min(1).to(
            ranking_probability.dtype
        )
        ranking_negative_rows = (
            (ranking_probability * negative_mask).sum(dim=1)
            / denominator
        )
        calibrated_negative_rows = (
            (calibrated_probability * negative_mask).sum(dim=1)
            / denominator
        )
        soft_false_positive_rate = ranking_negative_rows[
            valid_negative
        ].mean()
        mean_negative_calibrated_probability = calibrated_negative_rows[
            valid_negative
        ].mean()
    else:
        soft_false_positive_rate = ranking_probability.sum() * 0.0
        mean_negative_calibrated_probability = (
            calibrated_probability.sum() * 0.0
        )

    total = (
        float(cfg.de_bce_weight) * de_bce
        + float(cfg.de_count_budget_weight) * de_count_budget
        + float(cfg.zero_rate_count_nll_weight) * zero_rate_count_nll
        + float(cfg.virtual_delta_l1_weight) * virtual_delta_l1
        + float(cfg.virtual_delta_l2_weight) * virtual_delta_l2
        + float(cfg.virtual_delta_cosine_weight) * virtual_delta_cosine
        + float(cfg.pds_l1_weight) * pds_l1
        + float(cfg.pds_l2_weight) * pds_l2
        + float(cfg.pds_cosine_weight) * pds_cosine
        + float(cfg.no_regression_weight) * no_regression_hinge
    )
    components = (
        total,
        de_bce,
        de_count_budget,
        zero_rate_count_nll,
        virtual_delta_l1,
        virtual_delta_l2,
        virtual_delta_cosine,
        pds_l1,
        pds_l2,
        pds_cosine,
        no_regression_hinge,
    )
    if not all(torch.isfinite(value) for value in components):
        raise FloatingPointError("sparse hurdle DE loss became non-finite")
    return SparseHurdleDELossOutput(
        total=total,
        de_bce=de_bce,
        de_count_budget=de_count_budget,
        zero_rate_count_nll=zero_rate_count_nll,
        virtual_delta_l1=virtual_delta_l1,
        virtual_delta_l2=virtual_delta_l2,
        virtual_delta_cosine=virtual_delta_cosine,
        pds_l1=pds_l1,
        pds_l2=pds_l2,
        pds_cosine=pds_cosine,
        no_regression_hinge=no_regression_hinge,
        predicted_de_rate=predicted_de_rate.mean().detach(),
        target_de_rate=target_de_rate.mean().detach(),
        soft_false_positive_rate=soft_false_positive_rate.detach(),
        mean_ranking_probability=ranking_probability.mean().detach(),
        mean_negative_calibrated_probability=(
            mean_negative_calibrated_probability.detach()
        ),
        mean_zero_rate_confidence=(
            mean_zero_rate_confidence.detach()
        ),
        valid_pds_conditions=valid_pds,
    )


__all__ = [
    "SparseHurdleDELossConfig",
    "SparseHurdleDELossOutput",
    "sparse_hurdle_de_loss",
]

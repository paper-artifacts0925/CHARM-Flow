"""Permutation-invariant, DE-aware objectives on homogeneous cell sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import DEAwareResidualConfig


@dataclass(frozen=True)
class DEAwareResidualLoss:
    total: torch.Tensor
    signed_wilcoxon: torch.Tensor
    zero_rate: torch.Tensor
    log_variance: torch.Tensor
    quantile: torch.Tensor
    rank: torch.Tensor
    call: torch.Tensor
    signed_wilcoxon_correlation: torch.Tensor
    predicted_de_rate: torch.Tensor
    target_de_rate: torch.Tensor
    soft_false_positive_rate: torch.Tensor
    valid_groups: torch.Tensor
    mean_group_size: torch.Tensor


def _validate_expression(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,S,G]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _soft_signed_wilcoxon(
    values: torch.Tensor,
    controls: torch.Tensor,
    *,
    temperature: torch.Tensor,
    gene_chunk_size: int,
) -> torch.Tensor:
    """Soft two-sample Mann-Whitney effect in [-1,1], gene chunked."""

    if values.ndim != 2 or controls.ndim != 2 or values.shape[1] != controls.shape[1]:
        raise ValueError("values and controls must have shape [N,G] and [M,G]")
    if temperature.shape != (values.shape[1],):
        raise ValueError("temperature must have shape [G]")
    outputs = []
    for start in range(0, values.shape[1], int(gene_chunk_size)):
        stop = min(start + int(gene_chunk_size), values.shape[1])
        differences = (
            values[:, None, start:stop].float()
            - controls[None, :, start:stop].float()
        )
        probability = torch.sigmoid(
            differences / temperature[start:stop].float()[None, None, :]
        ).mean(dim=(0, 1))
        outputs.append(2.0 * probability - 1.0)
    return torch.cat(outputs).to(values)


def _safe_correlation(left: torch.Tensor, right: torch.Tensor, eps: float) -> torch.Tensor:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return (left * right).sum() / denominator.clamp_min(float(eps))


def _hard_pairwise_rank_loss(
    score: torch.Tensor,
    labels: torch.Tensor,
    config: DEAwareResidualConfig,
) -> torch.Tensor:
    positives = torch.nonzero(labels, as_tuple=False).flatten()
    negatives = torch.nonzero(~labels, as_tuple=False).flatten()
    if positives.numel() == 0 or negatives.numel() == 0:
        return score.sum() * 0.0
    positive_scores = score.index_select(0, positives)
    negative_scores = score.index_select(0, negatives)
    if positive_scores.numel() > int(config.rank_max_positive):
        # Lowest-scoring positives are the useful hard positives.
        keep = torch.topk(
            -positive_scores.detach(), int(config.rank_max_positive)
        ).indices
        positive_scores = positive_scores.index_select(0, keep)
    if negative_scores.numel() > int(config.rank_max_negative):
        keep = torch.topk(
            negative_scores.detach(), int(config.rank_max_negative)
        ).indices
        negative_scores = negative_scores.index_select(0, keep)
    margins = (
        negative_scores[:, None] - positive_scores[None, :]
    ) / float(config.rank_temperature)
    return F.softplus(margins).mean().to(score)


def grouped_de_aware_residual_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    control: torch.Tensor,
    group_ids: torch.Tensor,
    de_labels: torch.Tensor,
    target_counts: torch.Tensor,
    control_counts: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    config: Optional[DEAwareResidualConfig] = None,
) -> DEAwareResidualLoss:
    """Compare generated and true train-only cell distributions.

    ``de_labels`` is indexed per batch row and must originate from the audited
    train-only label bank.  Callers must perform the train-mask gate *before*
    invoking this function; this function intentionally has no fallback for a
    held-out condition.
    """

    config = DEAwareResidualConfig() if config is None else config
    config.validate()
    for name, value in (("prediction", prediction), ("target", target), ("control", control)):
        _validate_expression(name, value)
    if prediction.shape != target.shape or prediction.shape != control.shape:
        raise ValueError("prediction, target and control must share [B,S,G]")
    batch_size, set_size, genes = prediction.shape
    if group_ids.ndim == 1 and group_ids.shape[0] == batch_size:
        group_ids = group_ids[:, None].expand(-1, set_size)
    if tuple(group_ids.shape) != (batch_size, set_size):
        raise ValueError("group_ids must have shape [B] or [B,S]")
    if group_ids.device != prediction.device:
        raise ValueError("group_ids must share the expression device")
    if de_labels.shape != (batch_size, genes) or de_labels.dtype != torch.bool:
        raise ValueError("de_labels must be boolean [B,G]")
    if target_counts.shape != (batch_size,) or control_counts.shape != (batch_size,):
        raise ValueError("condition counts must have shape [B]")
    if valid_mask is None:
        valid_mask = torch.ones(
            batch_size, set_size, dtype=torch.bool, device=prediction.device
        )
    elif valid_mask.ndim == 1 and valid_mask.shape[0] == batch_size:
        valid_mask = valid_mask[:, None].expand(-1, set_size)
    if tuple(valid_mask.shape) != (batch_size, set_size) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B] or [B,S]")

    flat_valid = valid_mask.reshape(-1)
    flat_groups = group_ids.reshape(-1)[flat_valid]
    flat_prediction = prediction.reshape(-1, genes)[flat_valid]
    flat_target = target.reshape(-1, genes)[flat_valid]
    flat_control = control.reshape(-1, genes)[flat_valid]
    row_index = (
        torch.arange(batch_size, device=prediction.device)[:, None]
        .expand(-1, set_size)
        .reshape(-1)[flat_valid]
    )
    if flat_prediction.shape[0] == 0:
        raise ValueError("DE-aware objective received no valid cells")
    _, inverse = torch.unique(flat_groups, sorted=True, return_inverse=True)

    signed_terms = []
    zero_terms = []
    variance_terms = []
    quantile_terms = []
    rank_terms = []
    call_terms = []
    correlations = []
    predicted_rates = []
    target_rates = []
    false_positive_rates = []
    group_sizes = []
    quantile_levels = torch.tensor(
        config.quantile_levels, device=prediction.device, dtype=torch.float32
    )

    for group_index in range(int(inverse.max().item()) + 1):
        positions = torch.nonzero(inverse == group_index, as_tuple=False).flatten()
        if positions.numel() < int(config.min_cells_per_group):
            continue
        rows = row_index.index_select(0, positions)
        first_row = rows[0]
        labels = de_labels[first_row]
        if not torch.equal(de_labels.index_select(0, rows), labels[None, :].expand(rows.numel(), -1)):
            raise ValueError("one DE-aware group contains multiple condition labels")
        local_target_counts = target_counts.index_select(0, rows)
        local_control_counts = control_counts.index_select(0, rows)
        if not torch.equal(local_target_counts, local_target_counts[:1].expand_as(local_target_counts)):
            raise ValueError("one DE-aware group contains multiple target counts")
        if not torch.equal(local_control_counts, local_control_counts[:1].expand_as(local_control_counts)):
            raise ValueError("one DE-aware group contains multiple control counts")

        pred_group = flat_prediction.index_select(0, positions)
        target_group = flat_target.index_select(0, positions).detach()
        control_group = flat_control.index_select(0, positions).detach()
        control_scale = control_group.float().std(dim=0, unbiased=False)
        temperature = (
            float(config.wilcoxon_temperature_scale) * control_scale
        ).clamp(
            min=float(config.wilcoxon_temperature_min),
            max=float(config.wilcoxon_temperature_max),
        )
        predicted_u = _soft_signed_wilcoxon(
            pred_group,
            control_group,
            temperature=temperature,
            gene_chunk_size=config.gene_chunk_size,
        )
        target_u = _soft_signed_wilcoxon(
            target_group,
            control_group,
            temperature=temperature,
            gene_chunk_size=config.gene_chunk_size,
        ).detach()
        signed_terms.append(
            F.smooth_l1_loss(predicted_u.float(), target_u.float(), beta=0.1).to(prediction)
        )
        correlations.append(
            _safe_correlation(predicted_u, target_u, config.variance_eps).detach().to(prediction)
        )

        predicted_detection = torch.sigmoid(
            (pred_group.float() - float(config.zero_threshold))
            / float(config.zero_temperature)
        ).mean(dim=0)
        target_detection = torch.sigmoid(
            (target_group.float() - float(config.zero_threshold))
            / float(config.zero_temperature)
        ).mean(dim=0).detach()
        predicted_logit = torch.logit(predicted_detection.clamp(1e-4, 1.0 - 1e-4))
        target_logit = torch.logit(target_detection.clamp(1e-4, 1.0 - 1e-4))
        zero_terms.append(
            F.smooth_l1_loss(predicted_logit, target_logit).to(prediction)
        )

        predicted_variance = pred_group.float().var(dim=0, unbiased=False)
        target_variance = target_group.float().var(dim=0, unbiased=False)
        variance_terms.append(
            F.smooth_l1_loss(
                (predicted_variance + float(config.variance_eps)).log(),
                (target_variance + float(config.variance_eps)).log(),
            ).to(prediction)
        )
        predicted_quantiles = torch.quantile(
            pred_group.float(), quantile_levels, dim=0
        )
        target_quantiles = torch.quantile(
            target_group.float(), quantile_levels, dim=0
        ).detach()
        quantile_scale = target_variance.sqrt().add(
            float(config.quantile_scale_floor)
        ).clamp(
            min=float(config.quantile_scale_floor),
            max=float(config.quantile_scale_ceiling),
        )
        quantile_terms.append(
            F.smooth_l1_loss(
                predicted_quantiles / quantile_scale[None, :],
                target_quantiles / quantile_scale[None, :],
            ).to(prediction)
        )

        full_target_count = local_target_counts[0].float().clamp_min(2.0)
        full_control_count = local_control_counts[0].float().clamp_min(2.0)
        z_scale = (
            3.0
            * full_target_count
            * full_control_count
            / (full_target_count + full_control_count + 1.0)
        ).sqrt()
        score = predicted_u.float().abs() * z_scale
        rank_terms.append(_hard_pairwise_rank_loss(score, labels, config))
        call_logit = (
            score - float(config.call_z_threshold)
        ) / float(config.call_temperature)
        call_weights = torch.where(
            labels,
            torch.ones_like(call_logit),
            torch.full_like(call_logit, float(config.call_false_positive_weight)),
        )
        call_terms.append(
            (
                F.binary_cross_entropy_with_logits(
                    call_logit, labels.float(), reduction="none"
                )
                * call_weights
            ).mean().to(prediction)
        )
        call_probability = torch.sigmoid(call_logit)
        predicted_rates.append((score >= float(config.call_z_threshold)).float().mean().detach().to(prediction))
        target_rates.append(labels.float().mean().detach().to(prediction))
        negative = (~labels).float()
        false_positive_rates.append(
            ((call_probability * negative).sum() / negative.sum().clamp_min(1.0)).detach().to(prediction)
        )
        group_sizes.append(prediction.new_tensor(float(positions.numel())))

    zero = prediction.sum() * 0.0
    if not signed_terms:
        return DEAwareResidualLoss(
            total=zero,
            signed_wilcoxon=zero,
            zero_rate=zero,
            log_variance=zero,
            quantile=zero,
            rank=zero,
            call=zero,
            signed_wilcoxon_correlation=zero.detach(),
            predicted_de_rate=zero.detach(),
            target_de_rate=zero.detach(),
            soft_false_positive_rate=zero.detach(),
            valid_groups=zero.detach(),
            mean_group_size=zero.detach(),
        )
    signed = torch.stack(signed_terms).mean()
    zero_rate = torch.stack(zero_terms).mean()
    log_variance = torch.stack(variance_terms).mean()
    quantile = torch.stack(quantile_terms).mean()
    rank = torch.stack(rank_terms).mean()
    call = torch.stack(call_terms).mean()
    total = (
        float(config.signed_wilcoxon_weight) * signed
        + float(config.zero_rate_weight) * zero_rate
        + float(config.log_variance_weight) * log_variance
        + float(config.quantile_weight) * quantile
        + float(config.rank_weight) * rank
        + float(config.call_weight) * call
    )
    return DEAwareResidualLoss(
        total=total,
        signed_wilcoxon=signed,
        zero_rate=zero_rate,
        log_variance=log_variance,
        quantile=quantile,
        rank=rank,
        call=call,
        signed_wilcoxon_correlation=torch.stack(correlations).mean(),
        predicted_de_rate=torch.stack(predicted_rates).mean(),
        target_de_rate=torch.stack(target_rates).mean(),
        soft_false_positive_rate=torch.stack(false_positive_rates).mean(),
        valid_groups=prediction.new_tensor(float(len(signed_terms))),
        mean_group_size=torch.stack(group_sizes).mean(),
    )


def warmup_fraction(step: int, warmup_steps: int) -> float:
    if int(step) < 0 or int(warmup_steps) < 0:
        raise ValueError("step and warmup_steps must be non-negative")
    if int(warmup_steps) == 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


def cap_auxiliary_gradient(
    *,
    base_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    proxy: torch.Tensor,
    maximum_ratio: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cap auxiliary gradient norm relative to FM on the raw-velocity proxy."""

    if float(maximum_ratio) < 0.0:
        raise ValueError("maximum_ratio must be non-negative")
    if not auxiliary_loss.requires_grad or float(maximum_ratio) == 0.0:
        scale = auxiliary_loss.new_zeros(())
        zero = auxiliary_loss.detach().new_zeros(())
        return auxiliary_loss * scale, {
            "base_gradient_norm": zero,
            "aux_gradient_norm_pre_cap": zero,
            "aux_gradient_ratio_pre_cap": zero,
            "aux_gradient_cap_scale": scale.detach(),
            "aux_gradient_ratio_post_cap": zero,
        }
    base_gradient = torch.autograd.grad(
        base_loss,
        proxy,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    aux_gradient = torch.autograd.grad(
        auxiliary_loss,
        proxy,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    zero = auxiliary_loss.detach().new_zeros(())
    base_norm = (
        zero if base_gradient is None else base_gradient.detach().float().square().sum().sqrt()
    )
    aux_norm = (
        zero if aux_gradient is None else aux_gradient.detach().float().square().sum().sqrt()
    )
    ratio = aux_norm / base_norm.clamp_min(float(eps))
    allowed = auxiliary_loss.new_tensor(float(maximum_ratio))
    scale = torch.minimum(
        auxiliary_loss.new_ones(()),
        allowed * base_norm / aux_norm.clamp_min(float(eps)),
    ).detach()
    if base_norm <= float(eps) and aux_norm > float(eps):
        scale = scale.new_zeros(())
    return auxiliary_loss * scale, {
        "base_gradient_norm": base_norm,
        "aux_gradient_norm_pre_cap": aux_norm,
        "aux_gradient_ratio_pre_cap": ratio,
        "aux_gradient_cap_scale": scale,
        "aux_gradient_ratio_post_cap": ratio * scale,
    }


__all__ = [
    "DEAwareResidualLoss",
    "cap_auxiliary_gradient",
    "grouped_de_aware_residual_loss",
    "warmup_fraction",
]

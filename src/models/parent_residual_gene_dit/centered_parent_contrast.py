"""Train-only centered contrast for a condition-level Parent prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CenteredParentContrastResult:
    """Differentiable loss components and detached coverage diagnostics."""

    loss: torch.Tensor
    listwise: torch.Tensor
    listwise_l1: torch.Tensor
    listwise_l2: torch.Tensor
    listwise_cosine: torch.Tensor
    positive_huber: torch.Tensor
    centered_l1_norm_calibration: torch.Tensor
    centered_l2_norm_calibration: torch.Tensor
    positive_l1: torch.Tensor
    positive_l2: torch.Tensor
    positive_cosine: torch.Tensor
    prediction_centered_l1_norm: torch.Tensor
    target_centered_l1_norm: torch.Tensor
    prediction_centered_l2_norm: torch.Tensor
    target_centered_l2_norm: torch.Tensor
    loo_centroid_l1_norm: torch.Tensor
    loo_centroid_l2_norm: torch.Tensor
    norm_calibration_gate_mean: torch.Tensor
    norm_calibration_gate_min: torch.Tensor
    norm_calibration_gate_max: torch.Tensor
    query_count: torch.Tensor
    candidate_count_mean: torch.Tensor
    candidate_count_min: torch.Tensor
    excluded_target_gene_count: torch.Tensor


def _nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer_ids(name: str, value: torch.Tensor, size: int) -> torch.Tensor:
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape [B]")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must contain integer IDs")
    return value.to(dtype=torch.long)


def _feature_mask(
    *,
    gene_count: int,
    perturbation_id: int,
    target_gene_index: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    mask = torch.ones(gene_count, dtype=torch.bool, device=device)
    if target_gene_index is None:
        return mask, False
    excluded = int(target_gene_index[perturbation_id].item())
    if excluded >= 0:
        if gene_count == 1:
            raise ValueError("target-gene exclusion must retain at least one gene")
        mask[excluded] = False
        return mask, True
    return mask, False


def _distances(
    prediction: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances from one prediction to one or more targets."""

    prediction = prediction[feature_mask]
    target = target[..., feature_mask]
    difference = target - prediction
    l1 = difference.abs().mean(dim=-1)
    l2 = (difference.square().mean(dim=-1) + eps).sqrt()
    cosine = 1.0 - F.cosine_similarity(
        target,
        prediction.expand_as(target),
        dim=-1,
        eps=eps,
    )
    return l1, l2, cosine


def _centered_norms(
    value: torch.Tensor,
    feature_mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = value[feature_mask]
    return selected.abs().mean(), (selected.square().mean() + eps).sqrt()


def centered_parent_contrast_loss(
    *,
    parent_prediction: torch.Tensor,
    cell_line_ids: torch.Tensor,
    perturbation_ids: torch.Tensor,
    train_signature_bank: torch.Tensor,
    train_signature_mask: torch.Tensor,
    target_gene_index: torch.Tensor | None = None,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    cosine_weight: float = 1.0,
    positive_huber_weight: float = 1.0,
    centered_l1_norm_weight: float = 0.25,
    centered_l2_norm_weight: float = 0.25,
    l1_temperature: float = 0.05,
    l2_temperature: float = 0.05,
    cosine_temperature: float = 0.05,
    huber_delta: float = 0.1,
    norm_target_scale: float = 1.0,
    norm_huber_beta: float = 0.1,
    chunk_size: int = 256,
    eps: float = 1.0e-8,
) -> CenteredParentContrastResult:
    """Contrast Parent predictions against the complete same-line train bank.

    Args:
        parent_prediction: Parent perturbation signature, shape ``[B,G]``.
        cell_line_ids: Training-bank cell-line index for each query, ``[B]``.
        perturbation_ids: Correct training perturbation index, ``[B]``.
        train_signature_bank: Train-only truth signatures, ``[L,P,G]``.
        train_signature_mask: Availability of train signatures, ``[L,P]``.
        target_gene_index: Optional CellEval target-gene mapping, ``[P]``;
            ``-1`` keeps every gene.  No other gene selection is supported.

    Each query uses a leave-one-out centroid: the equally weighted mean of the
    same-line train signatures excluding its positive. Listwise denominators
    still contain every available candidate, including the positive once.
    """

    if parent_prediction.ndim != 2 or not torch.is_floating_point(
        parent_prediction
    ):
        raise TypeError("parent_prediction must be a floating-point [B,G] tensor")
    batch_size, gene_count = parent_prediction.shape
    if batch_size < 1 or gene_count < 1:
        raise ValueError("parent_prediction must have nonempty B and G dimensions")
    if train_signature_bank.ndim != 3 or not torch.is_floating_point(
        train_signature_bank
    ):
        raise TypeError(
            "train_signature_bank must be a floating-point [L,P,G] tensor"
        )
    line_count, perturbation_count, bank_gene_count = train_signature_bank.shape
    if bank_gene_count != gene_count:
        raise ValueError("parent_prediction and train_signature_bank must share G")
    if train_signature_mask.shape != (line_count, perturbation_count):
        raise ValueError("train_signature_mask must have shape [L,P]")
    if train_signature_mask.dtype != torch.bool:
        raise TypeError("train_signature_mask must be boolean")

    line_ids = _integer_ids("cell_line_ids", cell_line_ids, batch_size)
    perturb_ids = _integer_ids("perturbation_ids", perturbation_ids, batch_size)
    if ((line_ids < 0) | (line_ids >= line_count)).any():
        raise ValueError("cell_line_ids contains an out-of-range ID")
    if ((perturb_ids < 0) | (perturb_ids >= perturbation_count)).any():
        raise ValueError("perturbation_ids contains an out-of-range ID")
    if target_gene_index is not None:
        if target_gene_index.shape != (perturbation_count,):
            raise ValueError("target_gene_index must have shape [P]")
        if target_gene_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("target_gene_index must contain integer indices")
        if ((target_gene_index < -1) | (target_gene_index >= gene_count)).any():
            raise ValueError("target_gene_index entries must be -1 or in [0,G)")

    weights = tuple(
        _nonnegative(name, value)
        for name, value in (
            ("l1_weight", l1_weight),
            ("l2_weight", l2_weight),
            ("cosine_weight", cosine_weight),
            ("positive_huber_weight", positive_huber_weight),
            ("centered_l1_norm_weight", centered_l1_norm_weight),
            ("centered_l2_norm_weight", centered_l2_norm_weight),
        )
    )
    temperatures = tuple(
        _positive(name, value)
        for name, value in (
            ("l1_temperature", l1_temperature),
            ("l2_temperature", l2_temperature),
            ("cosine_temperature", cosine_temperature),
        )
    )
    huber_delta = _positive("huber_delta", huber_delta)
    norm_target_scale = _positive("norm_target_scale", norm_target_scale)
    norm_huber_beta = _positive("norm_huber_beta", norm_huber_beta)
    eps = _positive("eps", eps)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    device = parent_prediction.device
    work_prediction = parent_prediction.float()
    bank = train_signature_bank.detach().to(device=device, dtype=torch.float32)
    available = train_signature_mask.detach().to(device=device)
    line_ids = line_ids.to(device=device)
    perturb_ids = perturb_ids.to(device=device)
    target_mapping = (
        None
        if target_gene_index is None
        else target_gene_index.detach().to(device=device, dtype=torch.long)
    )

    available_counts = available.sum(dim=1)
    if not available[line_ids, perturb_ids].all():
        raise ValueError("every query's correct signature must exist in the train bank")
    if (available_counts.index_select(0, line_ids) < 2).any():
        raise ValueError("every query cell line must contain at least two train signatures")

    metric_terms: list[list[torch.Tensor]] = [[], [], []]
    huber_terms: list[torch.Tensor] = []
    l1_calibration_terms: list[torch.Tensor] = []
    l2_calibration_terms: list[torch.Tensor] = []
    positive_distances: list[list[torch.Tensor]] = [[], [], []]
    prediction_norms: list[list[torch.Tensor]] = [[], []]
    target_norms: list[list[torch.Tensor]] = [[], []]
    centroid_norms: list[list[torch.Tensor]] = [[], []]
    calibration_gates: list[torch.Tensor] = []
    candidate_counts: list[int] = []
    excluded_count = 0

    for row in range(batch_size):
        line = int(line_ids[row].item())
        perturbation = int(perturb_ids[row].item())
        feature_mask, excluded = _feature_mask(
            gene_count=gene_count,
            perturbation_id=perturbation,
            target_gene_index=target_mapping,
            device=device,
        )
        excluded_count += int(excluded)
        line_mask = available[line].clone()
        line_mask[perturbation] = False
        loo_count = line_mask.sum()
        centroid = (bank[line][line_mask].sum(dim=0) / loo_count.to(bank.dtype)).detach()
        prediction = work_prediction[row]
        prediction_centered = prediction - centroid
        positive = bank[line, perturbation]
        positive_centered = positive - centroid

        distances = _distances(
            prediction_centered, positive_centered, feature_mask, eps
        )
        for metric, distance in enumerate(distances):
            positive_distances[metric].append(distance)

        huber_terms.append(
            F.huber_loss(
                prediction[feature_mask],
                positive[feature_mask],
                reduction="mean",
                delta=huber_delta,
            )
        )
        prediction_l1, prediction_l2 = _centered_norms(
            prediction_centered, feature_mask, eps
        )
        target_l1, target_l2 = _centered_norms(
            positive_centered, feature_mask, eps
        )
        prediction_norms[0].append(prediction_l1)
        prediction_norms[1].append(prediction_l2)
        target_norms[0].append(target_l1)
        target_norms[1].append(target_l2)
        centroid_l1, centroid_l2 = _centered_norms(centroid, feature_mask, eps)
        centroid_norms[0].append(centroid_l1)
        centroid_norms[1].append(centroid_l2)
        l1_gate = target_l1 / (target_l1 + norm_target_scale)
        l2_gate = target_l2 / (target_l2 + norm_target_scale)
        calibration_gates.append(torch.stack((l1_gate, l2_gate)))
        l1_calibration_terms.append(
            l1_gate * F.smooth_l1_loss(
                torch.log(prediction_l1 + eps), torch.log(target_l1 + eps),
                reduction="none", beta=norm_huber_beta,
            )
        )
        l2_calibration_terms.append(
            l2_gate * F.smooth_l1_loss(
                torch.log(prediction_l2 + eps), torch.log(target_l2 + eps),
                reduction="none", beta=norm_huber_beta,
            )
        )

        candidate_indices = torch.nonzero(
            available[line], as_tuple=False
        ).flatten()
        candidate_counts.append(int(candidate_indices.numel()))
        denominators: list[torch.Tensor | None] = [None, None, None]
        for start in range(0, candidate_indices.numel(), chunk_size):
            selected = candidate_indices[start : start + chunk_size]
            candidates_centered = bank[line].index_select(0, selected) - centroid
            chunk_distances = _distances(
                prediction_centered, candidates_centered, feature_mask, eps
            )
            for metric, candidate_distance in enumerate(chunk_distances):
                chunk_lse = torch.logsumexp(
                    -candidate_distance / temperatures[metric], dim=0
                )
                denominators[metric] = (
                    chunk_lse
                    if denominators[metric] is None
                    else torch.logaddexp(denominators[metric], chunk_lse)
                )
        for metric, positive_distance in enumerate(distances):
            denominator = denominators[metric]
            if denominator is None:  # guarded by available_counts, for typing
                raise RuntimeError("empty train candidate bank")
            metric_terms[metric].append(
                denominator + positive_distance / temperatures[metric]
            )

    listwise_components = tuple(
        torch.stack(metric_terms[index]).mean() for index in range(3)
    )
    listwise = (
        weights[0] * listwise_components[0]
        + weights[1] * listwise_components[1]
        + weights[2] * listwise_components[2]
    )
    positive_huber = torch.stack(huber_terms).mean()
    l1_calibration = torch.stack(l1_calibration_terms).mean()
    l2_calibration = torch.stack(l2_calibration_terms).mean()
    loss = (
        listwise
        + weights[3] * positive_huber
        + weights[4] * l1_calibration
        + weights[5] * l2_calibration
    )

    counts = parent_prediction.new_tensor(candidate_counts, dtype=torch.float32)
    prediction_l1_values = torch.stack(prediction_norms[0])
    prediction_l2_values = torch.stack(prediction_norms[1])
    target_l1_values = torch.stack(target_norms[0])
    target_l2_values = torch.stack(target_norms[1])
    centroid_l1_values = torch.stack(centroid_norms[0])
    centroid_l2_values = torch.stack(centroid_norms[1])
    gate_values = torch.stack(calibration_gates)
    return CenteredParentContrastResult(
        loss=loss,
        listwise=listwise,
        listwise_l1=listwise_components[0],
        listwise_l2=listwise_components[1],
        listwise_cosine=listwise_components[2],
        positive_huber=positive_huber,
        centered_l1_norm_calibration=l1_calibration,
        centered_l2_norm_calibration=l2_calibration,
        positive_l1=torch.stack(positive_distances[0]).mean(),
        positive_l2=torch.stack(positive_distances[1]).mean(),
        positive_cosine=torch.stack(positive_distances[2]).mean(),
        prediction_centered_l1_norm=prediction_l1_values.mean(),
        target_centered_l1_norm=target_l1_values.mean(),
        prediction_centered_l2_norm=prediction_l2_values.mean(),
        target_centered_l2_norm=target_l2_values.mean(),
        loo_centroid_l1_norm=centroid_l1_values.mean().detach(),
        loo_centroid_l2_norm=centroid_l2_values.mean().detach(),
        norm_calibration_gate_mean=gate_values.mean().detach(),
        norm_calibration_gate_min=gate_values.min().detach(),
        norm_calibration_gate_max=gate_values.max().detach(),
        query_count=counts.new_tensor(float(batch_size)).detach(),
        candidate_count_mean=counts.mean().detach(),
        candidate_count_min=counts.min().detach(),
        excluded_target_gene_count=counts.new_tensor(float(excluded_count)).detach(),
    )


__all__ = [
    "CenteredParentContrastResult",
    "centered_parent_contrast_loss",
]

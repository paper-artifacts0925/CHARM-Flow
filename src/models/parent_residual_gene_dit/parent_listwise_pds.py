"""Full-bank PDS supervision for a condition-level Parent endpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


RankingMode = Literal["pairwise", "listwise"]


@dataclass(frozen=True)
class ParentListwisePDSResult:
    """Loss components and full-bank coverage diagnostics."""

    loss: torch.Tensor
    pds_l1: torch.Tensor
    pds_l2: torch.Tensor
    pds_cosine: torch.Tensor
    positive_fit: torch.Tensor
    no_regression: torch.Tensor
    bank_coverage: torch.Tensor
    condition_count: torch.Tensor
    candidate_count_mean: torch.Tensor
    candidate_count_min: torch.Tensor


def _validate_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_temperature(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_chunk_size(chunk_size: int) -> int:
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")


def _require_integer(name: str, value: torch.Tensor) -> None:
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must contain integer IDs")


def _row_condition_ids(
    compact_group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Resolve exactly one condition ID for every active endpoint row."""

    batch_size = active.shape[0]
    _require_integer("compact_group_ids", compact_group_ids)
    if compact_group_ids.ndim == 1:
        if compact_group_ids.shape != (batch_size,):
            raise ValueError("compact_group_ids must have shape [B] or [B,S]")
        groups = compact_group_ids.to(device=active.device, dtype=torch.long)
    elif compact_group_ids.ndim == 2:
        if compact_group_ids.shape[0] != batch_size:
            raise ValueError("compact_group_ids must have shape [B] or [B,S]")
        if valid_mask.ndim == 2 and valid_mask.shape != compact_group_ids.shape:
            raise ValueError(
                "2D valid_mask and compact_group_ids must have the same shape"
            )
        groups = torch.full(
            (batch_size,), -1, device=active.device, dtype=torch.long
        )
        ids = compact_group_ids.to(active.device)
        mask = (
            valid_mask.to(active.device)
            if valid_mask.ndim == 2
            else torch.ones_like(ids, dtype=torch.bool)
        )
        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            selected = ids[row][mask[row] & (ids[row] >= 0)]
            unique = torch.unique(selected)
            if unique.numel() != 1:
                raise ValueError(
                    "each active Parent row must resolve to exactly one condition"
                )
            groups[row] = unique[0]
    else:
        raise ValueError("compact_group_ids must have shape [B] or [B,S]")
    if (groups[active] < 0).any():
        raise ValueError("an active Parent row has an invalid condition ID")
    return groups


def _macro_mean(
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = value[active]
    selected_groups = groups[active]
    unique, inverse = torch.unique(selected_groups, sorted=True, return_inverse=True)
    total = value.new_zeros((unique.numel(), value.shape[1])).index_add_(
        0, inverse, selected
    )
    counts = torch.bincount(inverse, minlength=unique.numel()).to(value.dtype)
    return total / counts[:, None], unique


def _group_constant(
    name: str,
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
    unique: torch.Tensor,
) -> torch.Tensor:
    """Select one value per condition and reject inconsistent supervision."""

    rows = []
    for group in unique:
        positions = torch.nonzero(active & (groups == group), as_tuple=False).flatten()
        selected = value.index_select(0, positions.to(value.device))
        if not torch.equal(selected, selected[:1].expand_as(selected)):
            raise ValueError(f"one condition contains inconsistent {name}")
        rows.append(selected[0])
    return torch.stack(rows)


def _distance_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape [N,G]")
    if feature_mask.shape != prediction.shape or feature_mask.dtype != torch.bool:
        raise ValueError("feature_mask must be boolean [N,G]")
    numeric = feature_mask.to(dtype=prediction.dtype)
    count = numeric.sum(dim=1)
    if (count < 1).any():
        raise ValueError("every query must retain at least one evaluated gene")
    difference = prediction - target
    l1 = (difference.abs() * numeric).sum(dim=1) / count
    l2 = ((difference.square() * numeric).sum(dim=1) / count + 1.0e-8).sqrt()
    cosine = 1.0 - F.cosine_similarity(
        prediction * numeric,
        target * numeric,
        dim=1,
        eps=1.0e-8,
    )
    return l1, l2, cosine


def _pairwise_chunk_term(
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    return F.softplus((positive - negatives + margin) / temperature) * temperature


def _ranking_term(
    *,
    positive: torch.Tensor,
    negative_chunks: list[torch.Tensor],
    mode: RankingMode,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    if mode == "pairwise":
        total = positive.new_zeros(())
        count = 0
        for negatives in negative_chunks:
            total = total + _pairwise_chunk_term(
                positive, negatives, margin=margin, temperature=temperature
            ).sum()
            count += negatives.numel()
        return total / count

    # Cross entropy over one positive and the complete negative bank.  The
    # negative margin raises every negative logit, enforcing
    # d_negative >= d_positive + margin.  logaddexp provides a streaming,
    # numerically stable reduction without concatenating the full bank.
    positive_logit = -positive / temperature
    denominator = positive_logit
    for negatives in negative_chunks:
        chunk_logsumexp = torch.logsumexp(
            (margin - negatives) / temperature, dim=0
        )
        denominator = torch.logaddexp(denominator, chunk_logsumexp)
    return temperature * (denominator - positive_logit)


def parent_endpoint_listwise_pds_loss(
    *,
    parent_endpoint: torch.Tensor,
    control_mean: torch.Tensor,
    target_delta: torch.Tensor,
    base_parent_endpoint: torch.Tensor,
    target_available: torch.Tensor,
    artifact_cellline_ids: torch.Tensor,
    artifact_perturbation_ids: torch.Tensor,
    compact_group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    train_target_delta: torch.Tensor,
    train_target_mask: torch.Tensor,
    target_gene_index: torch.Tensor | None = None,
    ranking_mode: RankingMode = "listwise",
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    cosine_weight: float = 1.0,
    l1_temperature: float = 0.05,
    l2_temperature: float = 0.05,
    cosine_temperature: float = 0.05,
    margin: float = 0.0,
    positive_fit_weight: float = 0.0,
    no_regression_weight: float = 0.0,
    chunk_size: int = 256,
) -> ParentListwisePDSResult:
    """Compute condition-macro Parent PDS loss against a same-line train bank.

    Args:
        parent_endpoint: Proposed expression endpoint, shape ``[B,G]``.
        control_mean: Matching unperturbed expression endpoint, shape ``[B,G]``.
        target_delta: Train-only true perturbation delta, shape ``[B,G]``.
        base_parent_endpoint: Frozen reference endpoint used by the optional
            no-regression penalty, shape ``[B,G]``.
        compact_group_ids: Condition IDs as ``[B]`` or projected IDs ``[B,S]``.
        valid_mask: Boolean row mask ``[B]`` or projection mask ``[B,S]``.
        train_target_delta: Full train bank ``[L,P,G]``.
        train_target_mask: Boolean availability bank ``[L,P]``.
        target_gene_index: Optional exact target-gene index for every
            perturbation ``[P]``.  ``-1`` denotes unavailable mapping.  When
            supplied, that dimension is excluded from all three PDS metrics.

    The positive perturbation is always removed from its own negative bank.
    Conditions without any remaining same-line negative contribute to the
    positive-fit/no-regression terms but not to the ranking terms.
    """

    if parent_endpoint.ndim != 2:
        raise ValueError("parent_endpoint must have shape [B,G]")
    if not torch.is_floating_point(parent_endpoint):
        raise TypeError("parent_endpoint must be floating point")
    batch_size, gene_dim = parent_endpoint.shape
    if gene_dim < 1:
        raise ValueError("parent_endpoint must contain at least one gene")
    row_shape = (batch_size, gene_dim)
    for name, value in {
        "control_mean": control_mean,
        "target_delta": target_delta,
        "base_parent_endpoint": base_parent_endpoint,
    }.items():
        _require_float_matrix(name, value, row_shape)

    for name, value in {
        "target_available": target_available,
        "artifact_cellline_ids": artifact_cellline_ids,
        "artifact_perturbation_ids": artifact_perturbation_ids,
    }.items():
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
    if target_available.dtype != torch.bool:
        raise TypeError("target_available must be boolean")
    _require_integer("artifact_cellline_ids", artifact_cellline_ids)
    _require_integer("artifact_perturbation_ids", artifact_perturbation_ids)

    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if valid_mask.ndim == 1:
        if valid_mask.shape != (batch_size,):
            raise ValueError("valid_mask must have shape [B] or [B,S]")
        valid_rows = valid_mask.to(parent_endpoint.device)
    elif valid_mask.ndim == 2 and valid_mask.shape[0] == batch_size:
        valid_rows = valid_mask.to(parent_endpoint.device).any(dim=1)
    else:
        raise ValueError("valid_mask must have shape [B] or [B,S]")

    if train_target_delta.ndim != 3 or train_target_delta.shape[2] != gene_dim:
        raise ValueError("train_target_delta must have shape [L,P,G]")
    if not torch.is_floating_point(train_target_delta):
        raise TypeError("train_target_delta must be floating point")
    if train_target_mask.shape != train_target_delta.shape[:2]:
        raise ValueError("train_target_mask must have shape [L,P]")
    if train_target_mask.dtype != torch.bool:
        raise TypeError("train_target_mask must be boolean")
    line_count, perturbation_count = train_target_mask.shape

    if target_gene_index is not None:
        if target_gene_index.shape != (perturbation_count,):
            raise ValueError("target_gene_index must have shape [P]")
        _require_integer("target_gene_index", target_gene_index)
        if ((target_gene_index < -1) | (target_gene_index >= gene_dim)).any():
            raise ValueError("target_gene_index entries must be -1 or in [0,G)")

    if ranking_mode not in {"pairwise", "listwise"}:
        raise ValueError("ranking_mode must be 'pairwise' or 'listwise'")
    weights = tuple(
        _validate_nonnegative(name, value)
        for name, value in (
            ("l1_weight", l1_weight),
            ("l2_weight", l2_weight),
            ("cosine_weight", cosine_weight),
            ("positive_fit_weight", positive_fit_weight),
            ("no_regression_weight", no_regression_weight),
        )
    )
    temperatures = tuple(
        _validate_temperature(name, value)
        for name, value in (
            ("l1_temperature", l1_temperature),
            ("l2_temperature", l2_temperature),
            ("cosine_temperature", cosine_temperature),
        )
    )
    margin = _validate_nonnegative("margin", margin)
    chunk_size = _validate_chunk_size(chunk_size)

    available = target_available.to(parent_endpoint.device)
    active = valid_rows & available
    zero = parent_endpoint.sum() * 0.0
    if not active.any():
        detached_zero = zero.detach()
        return ParentListwisePDSResult(
            loss=zero,
            pds_l1=zero,
            pds_l2=zero,
            pds_cosine=zero,
            positive_fit=zero,
            no_regression=zero,
            bank_coverage=detached_zero,
            condition_count=detached_zero,
            candidate_count_mean=detached_zero,
            candidate_count_min=detached_zero,
        )

    groups = _row_condition_ids(compact_group_ids, valid_mask, active)
    endpoint, unique = _macro_mean(parent_endpoint.float(), groups, active)
    control, _ = _macro_mean(
        control_mean.detach().to(parent_endpoint.device).float(), groups, active
    )
    base_endpoint, _ = _macro_mean(
        base_parent_endpoint.detach().to(parent_endpoint.device).float(), groups, active
    )
    predicted_delta = endpoint - control
    base_delta = base_endpoint - control
    positive_target = _group_constant(
        "target delta",
        target_delta.detach().to(parent_endpoint.device).float(),
        groups,
        active,
        unique,
    )
    line_ids = _group_constant(
        "cell-line ID",
        artifact_cellline_ids[:, None].to(parent_endpoint.device),
        groups,
        active,
        unique,
    )[:, 0].to(torch.long)
    perturbation_ids = _group_constant(
        "perturbation ID",
        artifact_perturbation_ids[:, None].to(parent_endpoint.device),
        groups,
        active,
        unique,
    )[:, 0].to(torch.long)

    if ((line_ids < 0) | (line_ids >= line_count)).any():
        raise ValueError("artifact_cellline_ids contains an out-of-range ID")
    if ((perturbation_ids < 0) | (perturbation_ids >= perturbation_count)).any():
        raise ValueError("artifact_perturbation_ids contains an out-of-range ID")

    feature_mask = torch.ones_like(predicted_delta, dtype=torch.bool)
    if target_gene_index is not None and gene_dim > 1:
        mapping = target_gene_index.to(parent_endpoint.device, dtype=torch.long)
        excluded = mapping.index_select(0, perturbation_ids)
        rows = torch.nonzero(excluded >= 0, as_tuple=False).flatten()
        if rows.numel() > 0:
            feature_mask[rows, excluded.index_select(0, rows)] = False

    positive_distances = _distance_rows(
        predicted_delta, positive_target, feature_mask
    )
    base_distances = _distance_rows(base_delta, positive_target, feature_mask)

    per_metric_terms: list[list[torch.Tensor]] = [[], [], []]
    candidate_counts: list[int] = []
    bank = train_target_delta.detach().to(parent_endpoint.device).float()
    bank_mask = train_target_mask.to(parent_endpoint.device)
    for row in range(predicted_delta.shape[0]):
        line = int(line_ids[row].item())
        perturbation = int(perturbation_ids[row].item())
        candidate_mask = bank_mask[line].clone()
        candidate_mask[perturbation] = False
        indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        candidate_counts.append(indices.numel())
        negative_chunks: list[list[torch.Tensor]] = [[], [], []]
        for start in range(0, indices.numel(), chunk_size):
            selected = indices[start : start + chunk_size]
            negative_targets = bank[line].index_select(0, selected)
            repeated_prediction = predicted_delta[row : row + 1].expand_as(
                negative_targets
            )
            repeated_mask = feature_mask[row : row + 1].expand_as(negative_targets)
            distances = _distance_rows(
                repeated_prediction, negative_targets, repeated_mask
            )
            for metric in range(3):
                negative_chunks[metric].append(distances[metric])
        for metric in range(3):
            per_metric_terms[metric].append(
                _ranking_term(
                    positive=positive_distances[metric][row],
                    negative_chunks=negative_chunks[metric],
                    mode=ranking_mode,
                    margin=margin,
                    temperature=temperatures[metric],
                )
            )

    components = tuple(
        torch.stack(values).mean() if values else zero
        for values in per_metric_terms
    )
    positive_fit = torch.stack(positive_distances, dim=0).mean()
    no_regression = torch.stack(
        [
            F.relu(positive_distances[index] - base_distances[index])
            for index in range(3)
        ],
        dim=0,
    ).mean()
    loss = (
        weights[0] * components[0]
        + weights[1] * components[1]
        + weights[2] * components[2]
        + weights[3] * positive_fit
        + weights[4] * no_regression
    )
    counts = parent_endpoint.new_tensor(candidate_counts, dtype=torch.float32)
    condition_count = parent_endpoint.new_tensor(
        predicted_delta.shape[0], dtype=torch.float32
    )
    return ParentListwisePDSResult(
        loss=loss,
        pds_l1=components[0],
        pds_l2=components[1],
        pds_cosine=components[2],
        positive_fit=positive_fit,
        no_regression=no_regression,
        bank_coverage=parent_endpoint.new_tensor(
            len(candidate_counts) / predicted_delta.shape[0], dtype=torch.float32
        ),
        condition_count=condition_count,
        candidate_count_mean=(counts.mean() if candidate_counts else zero.detach()),
        candidate_count_min=(counts.min() if candidate_counts else zero.detach()),
    )


__all__ = [
    "ParentListwisePDSResult",
    "parent_endpoint_listwise_pds_loss",
]

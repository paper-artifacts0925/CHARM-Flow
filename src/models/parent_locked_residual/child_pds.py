"""Train-only PDS supervision for the mean-carrying Child endpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ChildEndpointPDSResult:
    loss: torch.Tensor
    pds_l1: torch.Tensor
    pds_l2: torch.Tensor
    pds_cosine: torch.Tensor
    positive_fit: torch.Tensor
    no_regression: torch.Tensor
    bank_coverage: torch.Tensor
    candidate_count_mean: torch.Tensor
    candidate_count_min: torch.Tensor


def _cfg(lightning, name: str, default):
    model_cfg = getattr(lightning, "model_cfg", None)
    return getattr(model_cfg, name, default) if model_cfg is not None else default


def _finite_nonnegative(lightning, name: str, default: float) -> float:
    value = float(_cfg(lightning, name, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_integer(lightning, name: str, default: int) -> int:
    value = _cfg(lightning, name, default)
    if isinstance(value, bool) or int(value) < 1 or float(value) != float(int(value)):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _row_group_ids(
    compact_group_ids: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Resolve one compact condition id for every active batch row."""

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
            valid = compact_group_ids[row][compact_group_ids[row] >= 0]
            unique = torch.unique(valid)
            if unique.numel() != 1:
                raise ValueError(
                    "each active Child endpoint row must contain exactly one "
                    "condition group"
                )
            groups[row] = unique[0]
    else:
        raise ValueError("compact_group_ids must have shape [B] or [B,S]")
    if (groups[active] < 0).any():
        raise ValueError("active Child endpoint row has an invalid condition group")
    return groups


def _macro_rows(
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = value[active]
    selected_groups = groups[active]
    unique, inverse = torch.unique(selected_groups, sorted=True, return_inverse=True)
    result = torch.zeros(
        unique.numel(), value.shape[1], device=value.device, dtype=value.dtype
    ).index_add_(0, inverse, selected)
    counts = torch.bincount(inverse).to(value.dtype)
    return result / counts[:, None], unique


def _group_first(
    value: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
    unique: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for group in unique:
        positions = torch.nonzero(active & (groups == group), as_tuple=False).flatten()
        selected = value.index_select(0, positions)
        if not torch.equal(selected, selected[:1].expand_as(selected)):
            raise ValueError("one Child endpoint group contains inconsistent targets")
        rows.append(selected[0])
    return torch.stack(rows)


def _distance_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PDS distances with an optional per-query excluded gene dimension."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [N,G]")
    if feature_mask is None:
        feature_mask = torch.ones_like(prediction, dtype=torch.bool)
    if feature_mask.shape != prediction.shape or feature_mask.dtype != torch.bool:
        raise ValueError("feature_mask must be boolean [N,G]")
    numeric = feature_mask.to(prediction)
    count = numeric.sum(dim=1).clamp_min(1.0)
    difference = prediction - target
    l1 = (difference.abs() * numeric).sum(dim=1) / count
    l2 = ((difference.square() * numeric).sum(dim=1) / count + 1.0e-8).sqrt()
    masked_prediction = prediction * numeric
    masked_target = target * numeric
    cosine = 1.0 - F.cosine_similarity(
        masked_prediction, masked_target, dim=1, eps=1.0e-8
    )
    return l1, l2, cosine


def _pairwise_logistic(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    # Smaller distance is better.  softplus is a smooth full-bank surrogate
    # for P(d_positive < d_negative), and unlike nearest-negative hinge it
    # keeps a learning signal from every available training signature.
    return (
        F.softplus((positive[:, None] - negative + margin) / temperature)
        * temperature
    ).mean()


def child_endpoint_pds_loss(
    *,
    child_endpoint_mean: torch.Tensor,
    control_mean: torch.Tensor,
    target_delta: torch.Tensor,
    prior_delta: torch.Tensor,
    target_available: torch.Tensor,
    artifact_cellline_ids: torch.Tensor,
    artifact_perturbation_ids: torch.Tensor,
    compact_group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    train_target_delta: torch.Tensor,
    train_target_mask: torch.Tensor,
    target_gene_index: torch.Tensor | None = None,
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
) -> ChildEndpointPDSResult:
    """Rank a Child endpoint against every train signature from its line."""

    if child_endpoint_mean.ndim != 2:
        raise ValueError("child_endpoint_mean must have shape [B,G]")
    batch_size, gene_dim = child_endpoint_mean.shape
    row_tensors = {
        "control_mean": control_mean,
        "target_delta": target_delta,
        "prior_delta": prior_delta,
    }
    for name, value in row_tensors.items():
        if value.shape != child_endpoint_mean.shape:
            raise ValueError(f"{name} must have shape [B,G]")
    for name, value in {
        "target_available": target_available,
        "artifact_cellline_ids": artifact_cellline_ids,
        "artifact_perturbation_ids": artifact_perturbation_ids,
    }.items():
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
    if train_target_delta.ndim != 3 or train_target_delta.shape[2] != gene_dim:
        raise ValueError("train_target_delta must have shape [L,P,G]")
    if train_target_mask.shape != train_target_delta.shape[:2]:
        raise ValueError("train_target_mask must have shape [L,P]")
    if valid_mask.ndim == 1:
        active = valid_mask & target_available.to(torch.bool)
    elif valid_mask.ndim == 2 and valid_mask.shape[0] == batch_size:
        active = valid_mask.any(dim=1) & target_available.to(torch.bool)
    else:
        raise ValueError("valid_mask must have shape [B] or [B,S]")

    zero = child_endpoint_mean.sum() * 0.0
    if not active.any():
        return ChildEndpointPDSResult(
            loss=zero,
            pds_l1=zero,
            pds_l2=zero,
            pds_cosine=zero,
            positive_fit=zero,
            no_regression=zero,
            bank_coverage=zero.detach(),
            candidate_count_mean=zero.detach(),
            candidate_count_min=zero.detach(),
        )

    groups = _row_group_ids(compact_group_ids, active)
    predicted_mean, unique = _macro_rows(child_endpoint_mean.float(), groups, active)
    control, _ = _macro_rows(control_mean.detach().to(predicted_mean).float(), groups, active)
    predicted_delta = predicted_mean - control
    positive_target = _group_first(
        target_delta.detach().to(predicted_delta).float(), groups, active, unique
    )
    baseline = _group_first(
        prior_delta.detach().to(predicted_delta).float(), groups, active, unique
    )
    line_ids = _group_first(
        artifact_cellline_ids[:, None], groups, active, unique
    )[:, 0].to(torch.long)
    perturbation_ids = _group_first(
        artifact_perturbation_ids[:, None], groups, active, unique
    )[:, 0].to(torch.long)

    feature_mask = torch.ones_like(predicted_delta, dtype=torch.bool)
    if target_gene_index is not None:
        if target_gene_index.ndim != 1 or target_gene_index.shape[0] != train_target_mask.shape[1]:
            raise ValueError("target_gene_index must have shape [P]")
        excluded = target_gene_index.to(perturbation_ids.device).index_select(
            0, perturbation_ids
        )
        reliable = (excluded >= 0) & (excluded < gene_dim)
        rows = torch.nonzero(reliable, as_tuple=False).flatten()
        if rows.numel() > 0 and gene_dim > 1:
            feature_mask[rows, excluded.index_select(0, rows)] = False

    positive_distances = _distance_rows(
        predicted_delta, positive_target, feature_mask
    )
    baseline_distances = _distance_rows(
        baseline, positive_target, feature_mask
    )
    metric_terms: list[list[torch.Tensor]] = [[], [], []]
    candidate_counts = []
    temperatures = (l1_temperature, l2_temperature, cosine_temperature)
    for row in range(predicted_delta.shape[0]):
        line = line_ids[row]
        perturbation = perturbation_ids[row]
        if line < 0 or line >= train_target_mask.shape[0]:
            raise ValueError("artifact_cellline_ids contains an out-of-range ID")
        if perturbation < 0 or perturbation >= train_target_mask.shape[1]:
            raise ValueError("artifact_perturbation_ids contains an out-of-range ID")
        candidate_mask = train_target_mask[line].clone()
        # The positive perturbation is never allowed into its own negative bank.
        candidate_mask[perturbation] = False
        indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        candidate_counts.append(indices.numel())
        negative_parts: list[list[torch.Tensor]] = [[], [], []]
        for start in range(0, indices.numel(), int(chunk_size)):
            selected = indices[start : start + int(chunk_size)]
            negative_targets = train_target_delta[line].index_select(
                0, selected
            ).detach().to(predicted_delta).float()
            repeated = predicted_delta[row : row + 1].expand_as(negative_targets)
            row_mask = feature_mask[row : row + 1].expand_as(negative_targets)
            distances = _distance_rows(repeated, negative_targets, row_mask)
            for metric in range(3):
                negative_parts[metric].append(distances[metric])
        for metric in range(3):
            negatives = torch.cat(negative_parts[metric], dim=0)[None, :]
            positive = positive_distances[metric][row : row + 1]
            metric_terms[metric].append(
                _pairwise_logistic(
                    positive,
                    negatives,
                    margin=float(margin),
                    temperature=float(temperatures[metric]),
                )
            )

    components = [
        torch.stack(values).mean() if values else zero
        for values in metric_terms
    ]
    positive_fit = torch.stack(positive_distances, dim=0).mean()
    no_regression = torch.stack(
        [
            F.relu(positive_distances[index] - baseline_distances[index])
            for index in range(3)
        ],
        dim=0,
    ).mean()
    loss = (
        float(l1_weight) * components[0]
        + float(l2_weight) * components[1]
        + float(cosine_weight) * components[2]
        + float(positive_fit_weight) * positive_fit
        + float(no_regression_weight) * no_regression
    )
    count_tensor = child_endpoint_mean.new_tensor(
        candidate_counts, dtype=torch.float32
    )
    return ChildEndpointPDSResult(
        loss=loss,
        pds_l1=components[0],
        pds_l2=components[1],
        pds_cosine=components[2],
        positive_fit=positive_fit,
        no_regression=no_regression,
        bank_coverage=child_endpoint_mean.new_tensor(
            len(candidate_counts) / max(1, predicted_delta.shape[0])
        ),
        candidate_count_mean=(count_tensor.mean() if candidate_counts else zero.detach()),
        candidate_count_min=(count_tensor.min() if candidate_counts else zero.detach()),
    )


def augment_locked_result_with_child_endpoint_pds(
    lightning,
    *,
    result: dict,
    flow_output: dict,
    prepared,
) -> None:
    """Add opt-in Child PDS supervision to the existing flow result in-place."""

    master_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_pds_weight", 0.0
    )
    # This early return is the compatibility contract: the default path does
    # not inspect or transform any existing training tensor.
    if master_weight == 0.0:
        return
    child_enabled = bool(
        getattr(
            lightning,
            "parent_residual_child_mean_flow_enabled",
            getattr(
                getattr(lightning, "parent_locked_residual_flow", None),
                "child_mean_flow_enabled",
                False,
            ),
        )
    )
    if not child_enabled:
        raise ValueError(
            "parent_residual_child_endpoint_pds_weight > 0 requires the "
            "mean-carrying Child flow"
        )
    endpoint = flow_output.get("child_endpoint_mean")
    if endpoint is None:
        raise AssertionError("Child flow did not return child_endpoint_mean")

    l1_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_pds_l1_weight", 1.0
    )
    l2_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_pds_l2_weight", 1.0
    )
    cosine_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_pds_cos_weight", 1.0
    )
    positive_fit_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_positive_fit_weight", 0.0
    )
    no_regression_weight = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_no_reg_weight", 0.0
    )
    margin = _finite_nonnegative(
        lightning, "parent_residual_child_endpoint_pds_margin", 0.0
    )
    temperatures = []
    for name in (
        "parent_residual_child_endpoint_pds_l1_temperature",
        "parent_residual_child_endpoint_pds_l2_temperature",
        "parent_residual_child_endpoint_pds_cos_temperature",
    ):
        value = _finite_nonnegative(lightning, name, 0.05)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        temperatures.append(value)
    chunk_size = _positive_integer(
        lightning, "parent_residual_child_endpoint_pds_chunk_size", 256
    )

    lookup = prepared.parent.lookup
    prior_bank = lightning.model.prior_bank
    supervision_mask = flow_output.get("child_mean_supervision_mask")
    if supervision_mask is None:
        supervision_mask = flow_output["pair"].target_projection.valid_mask
    auxiliary = child_endpoint_pds_loss(
        child_endpoint_mean=endpoint,
        control_mean=lookup.control_mean,
        target_delta=lookup.target_delta,
        prior_delta=lookup.prior_delta,
        target_available=lookup.target_available,
        artifact_cellline_ids=lookup.artifact_cellline_ids,
        artifact_perturbation_ids=lookup.artifact_perturbation_ids,
        compact_group_ids=flow_output["pair"].target_projection.group_ids,
        valid_mask=supervision_mask,
        train_target_delta=prior_bank.train_target_delta,
        train_target_mask=prior_bank.train_target_mask,
        target_gene_index=getattr(prior_bank, "target_gene_index", None),
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        cosine_weight=cosine_weight,
        l1_temperature=temperatures[0],
        l2_temperature=temperatures[1],
        cosine_temperature=temperatures[2],
        margin=margin,
        positive_fit_weight=positive_fit_weight,
        no_regression_weight=no_regression_weight,
        chunk_size=chunk_size,
    )
    weighted = master_weight * auxiliary.loss
    result["loss"] = result["loss"] + weighted

    gradient_proxy = endpoint.new_zeros((), dtype=torch.float32)
    if torch.is_grad_enabled() and endpoint.requires_grad and weighted.requires_grad:
        gradient = torch.autograd.grad(
            weighted,
            endpoint,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is not None:
            gradient_proxy = gradient.detach().float().square().mean().sqrt()
    result.update(
        {
            "parent_residual_child_endpoint_pds": auxiliary.loss.detach(),
            "parent_residual_child_endpoint_pds_weighted": weighted.detach(),
            "parent_residual_child_endpoint_pds_l1": auxiliary.pds_l1.detach(),
            "parent_residual_child_endpoint_pds_l2": auxiliary.pds_l2.detach(),
            "parent_residual_child_endpoint_pds_cosine": auxiliary.pds_cosine.detach(),
            "parent_residual_child_endpoint_positive_fit": auxiliary.positive_fit.detach(),
            "parent_residual_child_endpoint_no_regression": auxiliary.no_regression.detach(),
            "parent_residual_child_endpoint_pds_bank_coverage": auxiliary.bank_coverage.detach(),
            "parent_residual_child_endpoint_pds_candidate_count_mean": auxiliary.candidate_count_mean.detach(),
            "parent_residual_child_endpoint_pds_candidate_count_min": auxiliary.candidate_count_min.detach(),
            "parent_residual_child_endpoint_pds_gradient_proxy_rms": gradient_proxy,
        }
    )


__all__ = [
    "ChildEndpointPDSResult",
    "augment_locked_result_with_child_endpoint_pds",
    "child_endpoint_pds_loss",
]

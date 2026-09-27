"""Opt-in Lightning-result adapter for full-bank Parent PDS supervision."""

from __future__ import annotations

import math

import torch

from .parent_listwise_pds import parent_endpoint_listwise_pds_loss


_PREFIX = "parent_residual_parent_listwise_pds"


def _cfg(lightning, suffix: str, default):
    model_cfg = getattr(lightning, "model_cfg", None)
    name = f"{_PREFIX}_{suffix}"
    return getattr(model_cfg, name, default) if model_cfg is not None else default


def _finite_nonnegative(lightning, suffix: str, default: float) -> float:
    name = f"{_PREFIX}_{suffix}"
    try:
        value = float(_cfg(lightning, suffix, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_temperature(lightning, suffix: str, default: float) -> float:
    value = _finite_nonnegative(lightning, suffix, default)
    if value <= 0.0:
        raise ValueError(f"{_PREFIX}_{suffix} must be positive")
    return value


def _positive_integer(lightning, suffix: str, default: int) -> int:
    name = f"{_PREFIX}_{suffix}"
    value = _cfg(lightning, suffix, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer < 1 or not math.isfinite(numeric) or numeric != float(integer):
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _ranking_mode(lightning) -> str:
    value = _cfg(lightning, "ranking_mode", "listwise")
    if not isinstance(value, str) or value not in {"pairwise", "listwise"}:
        raise ValueError(
            f"{_PREFIX}_ranking_mode must be 'pairwise' or 'listwise'"
        )
    return value


def augment_result_with_parent_listwise_pds(
    lightning,
    *,
    result: dict,
    prepared,
    compact_group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Add opt-in full-bank Parent PDS supervision to ``result`` in place.

    All target and bank tensors are detached inside the core loss.  Gradients
    therefore reach only the final ``prepared.parent.parent_mean`` endpoint.
    """

    master_weight = _finite_nonnegative(lightning, "weight", 0.0)
    # Compatibility contract: existing configurations do not inspect any
    # training object and preserve the exact original loss tensor identity.
    if master_weight == 0.0:
        return

    if not isinstance(result, dict) or "loss" not in result:
        raise ValueError("result must be a dict containing 'loss'")
    if not isinstance(result["loss"], torch.Tensor):
        raise TypeError("result['loss'] must be a torch.Tensor")

    ranking_mode = _ranking_mode(lightning)
    l1_weight = _finite_nonnegative(lightning, "l1_weight", 1.0)
    l2_weight = _finite_nonnegative(lightning, "l2_weight", 1.0)
    cosine_weight = _finite_nonnegative(lightning, "cos_weight", 1.0)
    positive_fit_weight = _finite_nonnegative(
        lightning, "positive_fit_weight", 0.0
    )
    no_regression_weight = _finite_nonnegative(
        lightning, "no_reg_weight", 0.0
    )
    margin = _finite_nonnegative(lightning, "margin", 0.0)
    l1_temperature = _positive_temperature(lightning, "l1_temperature", 0.05)
    l2_temperature = _positive_temperature(lightning, "l2_temperature", 0.05)
    cosine_temperature = _positive_temperature(
        lightning, "cos_temperature", 0.05
    )
    chunk_size = _positive_integer(lightning, "chunk_size", 256)

    parent = prepared.parent
    lookup = parent.lookup
    endpoint = parent.parent_mean
    if not isinstance(endpoint, torch.Tensor):
        raise TypeError("prepared.parent.parent_mean must be a torch.Tensor")
    base_endpoint = getattr(parent, "base_parent_mean", None)
    if base_endpoint is None:
        base_endpoint = lookup.control_mean + lookup.prior_delta

    model = getattr(lightning, "model", None)
    prior_bank = getattr(model, "prior_bank", None)
    if prior_bank is None:
        raise ValueError("lightning.model.prior_bank is required")

    auxiliary = parent_endpoint_listwise_pds_loss(
        parent_endpoint=endpoint,
        control_mean=lookup.control_mean,
        target_delta=lookup.target_delta,
        base_parent_endpoint=base_endpoint,
        target_available=lookup.target_available,
        artifact_cellline_ids=lookup.artifact_cellline_ids,
        artifact_perturbation_ids=lookup.artifact_perturbation_ids,
        compact_group_ids=compact_group_ids,
        valid_mask=valid_mask,
        train_target_delta=prior_bank.train_target_delta,
        train_target_mask=prior_bank.train_target_mask,
        target_gene_index=getattr(prior_bank, "target_gene_index", None),
        ranking_mode=ranking_mode,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        cosine_weight=cosine_weight,
        l1_temperature=l1_temperature,
        l2_temperature=l2_temperature,
        cosine_temperature=cosine_temperature,
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
            f"{_PREFIX}": auxiliary.loss.detach(),
            f"{_PREFIX}_weighted": weighted.detach(),
            f"{_PREFIX}_l1": auxiliary.pds_l1.detach(),
            f"{_PREFIX}_l2": auxiliary.pds_l2.detach(),
            f"{_PREFIX}_cosine": auxiliary.pds_cosine.detach(),
            f"{_PREFIX}_positive_fit": auxiliary.positive_fit.detach(),
            f"{_PREFIX}_no_regression": auxiliary.no_regression.detach(),
            f"{_PREFIX}_bank_coverage": auxiliary.bank_coverage.detach(),
            f"{_PREFIX}_condition_count": auxiliary.condition_count.detach(),
            f"{_PREFIX}_candidate_count_mean": auxiliary.candidate_count_mean.detach(),
            f"{_PREFIX}_candidate_count_min": auxiliary.candidate_count_min.detach(),
            f"{_PREFIX}_gradient_proxy_rms": gradient_proxy,
        }
    )


__all__ = ["augment_result_with_parent_listwise_pds"]

"""Minimal opt-in wiring for DE-aware Parent-locked training."""

from __future__ import annotations

import torch

from .artifact import TrainDELabelBank
from .config import DEAwareResidualConfig
from .losses import (
    cap_auxiliary_gradient,
    grouped_de_aware_residual_loss,
    warmup_fraction,
)


def _cfg(lightning, name, default):
    return getattr(lightning.model_cfg, name, default)


def _build_config(lightning) -> DEAwareResidualConfig:
    quantiles = _cfg(
        lightning,
        "parent_residual_de_aware_quantile_levels",
        [0.0, 0.25, 0.5, 0.75, 0.9],
    )
    return DEAwareResidualConfig(
        signed_wilcoxon_weight=float(
            _cfg(lightning, "parent_residual_de_aware_signed_wilcoxon_weight", 2e-4)
        ),
        zero_rate_weight=float(
            _cfg(lightning, "parent_residual_de_aware_zero_rate_weight", 2e-4)
        ),
        log_variance_weight=float(
            _cfg(lightning, "parent_residual_de_aware_log_variance_weight", 1e-4)
        ),
        quantile_weight=float(
            _cfg(lightning, "parent_residual_de_aware_quantile_weight", 1e-4)
        ),
        rank_weight=float(
            _cfg(lightning, "parent_residual_de_aware_rank_weight", 2e-5)
        ),
        call_weight=float(
            _cfg(lightning, "parent_residual_de_aware_call_weight", 1e-4)
        ),
        warmup_steps=int(
            _cfg(lightning, "parent_residual_de_aware_warmup_steps", 500)
        ),
        min_cells_per_group=int(
            _cfg(lightning, "parent_residual_de_aware_min_cells_per_group", 8)
        ),
        gene_chunk_size=int(
            _cfg(lightning, "parent_residual_de_aware_gene_chunk_size", 256)
        ),
        wilcoxon_temperature_scale=float(
            _cfg(lightning, "parent_residual_de_aware_wilcoxon_temperature_scale", 0.2)
        ),
        wilcoxon_temperature_min=float(
            _cfg(lightning, "parent_residual_de_aware_wilcoxon_temperature_min", 3e-3)
        ),
        wilcoxon_temperature_max=float(
            _cfg(lightning, "parent_residual_de_aware_wilcoxon_temperature_max", 3e-2)
        ),
        zero_threshold=float(
            _cfg(lightning, "parent_residual_de_aware_zero_threshold", 0.0)
        ),
        zero_temperature=float(
            _cfg(lightning, "parent_residual_de_aware_zero_temperature", 3e-3)
        ),
        variance_eps=float(
            _cfg(lightning, "parent_residual_de_aware_variance_eps", 1e-6)
        ),
        quantile_scale_floor=float(
            _cfg(lightning, "parent_residual_de_aware_quantile_scale_floor", 3e-3)
        ),
        quantile_scale_ceiling=float(
            _cfg(lightning, "parent_residual_de_aware_quantile_scale_ceiling", 1e-1)
        ),
        quantile_levels=tuple(float(value) for value in quantiles),
        rank_temperature=float(
            _cfg(lightning, "parent_residual_de_aware_rank_temperature", 0.5)
        ),
        rank_max_positive=int(
            _cfg(lightning, "parent_residual_de_aware_rank_max_positive", 256)
        ),
        rank_max_negative=int(
            _cfg(lightning, "parent_residual_de_aware_rank_max_negative", 256)
        ),
        call_z_threshold=float(
            _cfg(lightning, "parent_residual_de_aware_call_z_threshold", 4.0)
        ),
        call_temperature=float(
            _cfg(lightning, "parent_residual_de_aware_call_temperature", 0.5)
        ),
        call_false_positive_weight=float(
            _cfg(lightning, "parent_residual_de_aware_call_false_positive_weight", 2.0)
        ),
        gradient_cap_ratio=float(
            _cfg(lightning, "parent_residual_de_aware_gradient_cap_ratio", 0.3)
        ),
        gradient_cap_eps=float(
            _cfg(lightning, "parent_residual_de_aware_gradient_cap_eps", 1e-12)
        ),
    ).validate()


def initialize_de_aware_residual(lightning) -> None:
    """Install the train-only label bank if and only if the feature is enabled."""

    enabled = bool(
        lightning.parent_residual_enabled
        and _cfg(lightning, "parent_residual_de_aware_enabled", False)
    )
    lightning.parent_residual_de_aware_enabled = enabled
    lightning.parent_residual_de_aware_config = None
    lightning.parent_residual_de_aware_bank = None
    if not enabled:
        return
    if not lightning.parent_residual_locked_flow_enabled:
        raise ValueError(
            "DE-aware residual objectives require parent_residual_locked_flow_enabled"
        )
    artifact_path = _cfg(
        lightning, "parent_residual_de_aware_artifact_path", None
    )
    artifact_sha = _cfg(
        lightning, "parent_residual_de_aware_artifact_sha256", None
    )
    if artifact_path in (None, "", "null"):
        raise ValueError("enabled DE-aware training requires an artifact path")
    lightning.parent_residual_de_aware_config = _build_config(lightning)
    lightning.parent_residual_de_aware_bank = TrainDELabelBank(
        artifact_path,
        _cfg(lightning, "parent_residual_artifact_path", None),
        expected_sha256=str(artifact_sha or ""),
        expected_parent_sha256=_cfg(
            lightning, "parent_residual_artifact_sha256", None
        ),
        expected_gene_dim=int(lightning.model_cfg.input_dim),
        expected_normalization_divisor=float(
            _cfg(lightning, "parent_residual_normalization_divisor", 10.0)
        ),
        expected_parent_calibration=str(
            _cfg(lightning, "parent_residual_calibration", "no_intercept")
        ),
    )
    lightning.py_logger.info(
        "Loaded train-only DE-aware label artifact %s sha256=%s "
        "audit_sha256=%s parent_calibration=%s calibration_support_sha256=%s",
        lightning.parent_residual_de_aware_bank.artifact_path,
        lightning.parent_residual_de_aware_bank.artifact_sha256,
        lightning.parent_residual_de_aware_bank.audit_split_sha256,
        lightning.parent_residual_de_aware_bank.parent_calibration_mode,
        lightning.parent_residual_de_aware_bank.parent_calibration_support_sha256,
    )


def augment_locked_result_with_de_aware(
    lightning,
    *,
    result,
    flow_output,
    batch,
    prepared,
    supervision_mask,
):
    """Add P2 only during training and only after a strict train-mask gate."""

    if not lightning.parent_residual_de_aware_enabled:
        return None
    # validation_step is no_grad and may contain validation/test treated cells.
    # Do not even form their statistics; P2 is a training objective only.
    if not lightning.training:
        return None
    bank = lightning.parent_residual_de_aware_bank
    row_valid = supervision_mask[:, 0].to(dtype=torch.bool)
    # lookup is strict: audit/validation/test rows raise before any labels,
    # treated expression, or MAD/scale statistic can be accessed.
    lookup = bank.lookup(
        prepared.parent.lookup.artifact_cellline_ids,
        prepared.parent.lookup.artifact_perturbation_ids,
        active_mask=row_valid,
    )
    if bool(
        getattr(lightning, "parent_residual_strict_parent_only_enabled", False)
    ):
        control = prepared.parent.lookup.control_mean.to(batch["pert_emb"])
        control = control[:, None, :].expand_as(batch["pert_emb"])
    else:
        if "cont_emb" not in batch:
            raise ValueError(
                "DE-aware residual objective requires real control rows in "
                "cont_emb"
            )
        control = batch["cont_emb"]
    objective = grouped_de_aware_residual_loss(
        prediction=flow_output["prediction"],
        target=batch["pert_emb"],
        control=control,
        group_ids=flow_output["pair"].target_projection.group_ids,
        de_labels=lookup.labels,
        target_counts=lookup.target_count,
        control_counts=lookup.control_count,
        valid_mask=supervision_mask,
        config=lightning.parent_residual_de_aware_config,
    )
    fraction = warmup_fraction(
        int(lightning.global_step),
        lightning.parent_residual_de_aware_config.warmup_steps,
    )
    warmed = objective.total * float(fraction)
    capped, gradient_logs = cap_auxiliary_gradient(
        base_loss=flow_output.get("centered_flow_loss", flow_output["loss"]),
        auxiliary_loss=warmed,
        proxy=flow_output["raw_velocity"],
        maximum_ratio=lightning.parent_residual_de_aware_config.gradient_cap_ratio,
        eps=lightning.parent_residual_de_aware_config.gradient_cap_eps,
    )
    result["loss"] = result["loss"] + capped
    config = lightning.parent_residual_de_aware_config
    weighted_components = {
        "signed_wilcoxon": objective.signed_wilcoxon
        * float(config.signed_wilcoxon_weight),
        "zero_rate": objective.zero_rate * float(config.zero_rate_weight),
        "log_variance": objective.log_variance
        * float(config.log_variance_weight),
        "quantile": objective.quantile * float(config.quantile_weight),
        "rank": objective.rank * float(config.rank_weight),
        "call": objective.call * float(config.call_weight),
    }
    effective_scale = float(fraction) * gradient_logs[
        "aux_gradient_cap_scale"
    ]
    active_target_counts = lookup.target_count[row_valid]
    active_control_counts = lookup.control_count[row_valid]
    result.update(
        {
            "parent_residual_de_aware_total_raw": objective.total.detach(),
            "parent_residual_de_aware_total_warmed": warmed.detach(),
            "parent_residual_de_aware_total_capped": capped.detach(),
            "parent_residual_de_aware_warmup_fraction": capped.new_tensor(fraction).detach(),
            "parent_residual_de_aware_signed_wilcoxon": objective.signed_wilcoxon.detach(),
            "parent_residual_de_aware_zero_rate": objective.zero_rate.detach(),
            "parent_residual_de_aware_log_variance": objective.log_variance.detach(),
            "parent_residual_de_aware_quantile": objective.quantile.detach(),
            "parent_residual_de_aware_rank": objective.rank.detach(),
            "parent_residual_de_aware_call": objective.call.detach(),
            "parent_residual_de_aware_weighted_signed_wilcoxon": weighted_components["signed_wilcoxon"].detach(),
            "parent_residual_de_aware_weighted_zero_rate": weighted_components["zero_rate"].detach(),
            "parent_residual_de_aware_weighted_log_variance": weighted_components["log_variance"].detach(),
            "parent_residual_de_aware_weighted_quantile": weighted_components["quantile"].detach(),
            "parent_residual_de_aware_weighted_rank": weighted_components["rank"].detach(),
            "parent_residual_de_aware_weighted_call": weighted_components["call"].detach(),
            "parent_residual_de_aware_effective_signed_wilcoxon": (weighted_components["signed_wilcoxon"] * effective_scale).detach(),
            "parent_residual_de_aware_effective_zero_rate": (weighted_components["zero_rate"] * effective_scale).detach(),
            "parent_residual_de_aware_effective_log_variance": (weighted_components["log_variance"] * effective_scale).detach(),
            "parent_residual_de_aware_effective_quantile": (weighted_components["quantile"] * effective_scale).detach(),
            "parent_residual_de_aware_effective_rank": (weighted_components["rank"] * effective_scale).detach(),
            "parent_residual_de_aware_effective_call": (weighted_components["call"] * effective_scale).detach(),
            "parent_residual_de_aware_signed_wilcoxon_correlation": objective.signed_wilcoxon_correlation.detach(),
            "parent_residual_de_aware_predicted_de_rate": objective.predicted_de_rate.detach(),
            "parent_residual_de_aware_target_de_rate": objective.target_de_rate.detach(),
            "parent_residual_de_aware_soft_false_positive_rate": objective.soft_false_positive_rate.detach(),
            "parent_residual_de_aware_valid_groups": objective.valid_groups.detach(),
            "parent_residual_de_aware_mean_group_size": objective.mean_group_size.detach(),
            "parent_residual_de_aware_mean_target_condition_count": active_target_counts.float().mean().detach(),
            "parent_residual_de_aware_min_target_condition_count": active_target_counts.min().to(capped).detach(),
            "parent_residual_de_aware_mean_control_count": active_control_counts.float().mean().detach(),
            "parent_residual_de_aware_base_gradient_norm": gradient_logs["base_gradient_norm"],
            "parent_residual_de_aware_aux_gradient_norm_pre_cap": gradient_logs["aux_gradient_norm_pre_cap"],
            "parent_residual_de_aware_aux_gradient_ratio_pre_cap": gradient_logs["aux_gradient_ratio_pre_cap"],
            "parent_residual_de_aware_aux_gradient_cap_scale": gradient_logs["aux_gradient_cap_scale"],
            "parent_residual_de_aware_aux_gradient_ratio_post_cap": gradient_logs["aux_gradient_ratio_post_cap"],
        }
    )
    return objective


__all__ = [
    "augment_locked_result_with_de_aware",
    "initialize_de_aware_residual",
]

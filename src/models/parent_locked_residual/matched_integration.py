"""Hungarian posterior routing and grouped set loss for Parent-locked FM."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from src.models.matched_set_transport import (
    PosteriorPriorMixConfig,
    ResidualSetLossConfig,
    condition_level_posterior_prior_diagnostics,
    grouped_residual_set_loss,
    select_mixed_child_indices,
)


def _finite_nonnegative(name, value):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _load_set_projection(path, gene_dim, expected_components=None):
    artifact = Path(str(path)).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Parent residual set PCA artifact not found: {artifact}")
    with np.load(str(artifact), allow_pickle=False) as payload:
        if "raw_components" in payload.files:
            components = np.asarray(payload["raw_components"], dtype=np.float32)
        elif "components" in payload.files:
            components = np.asarray(payload["components"], dtype=np.float32)
        else:
            raise ValueError("set PCA artifact needs raw_components or components")
        if "explained_variance" not in payload.files:
            raise ValueError("set PCA artifact needs explained_variance")
        variance = np.asarray(payload["explained_variance"], dtype=np.float32)
    if components.ndim != 2 or components.shape[1] != int(gene_dim):
        raise ValueError(
            "set PCA components must have shape [D, model.input_dim]"
        )
    if variance.shape != (components.shape[0],):
        raise ValueError("explained_variance must have shape [D]")
    if not np.isfinite(components).all() or not np.isfinite(variance).all():
        raise ValueError("set PCA artifact contains non-finite values")
    if (variance <= 0).any():
        raise ValueError("set PCA explained_variance must be positive")
    if expected_components not in (None, 0) and components.shape[0] != int(
        expected_components
    ):
        raise ValueError(
            f"set PCA component count {components.shape[0]} does not match "
            f"configured {int(expected_components)}"
        )
    projection = torch.from_numpy(components.T.copy())
    scale = torch.from_numpy(np.sqrt(variance).astype(np.float32, copy=False))
    return artifact, projection, scale


def initialize_matched_set_transport(lightning):
    """Initialize optional routing and grouped residual-set supervision."""

    matched_set_requested = bool(
        lightning.parent_residual_enabled
        and getattr(
            lightning.model_cfg,
            "parent_residual_matched_set_enabled",
            False,
        )
    )
    independent_set_loss_requested = bool(
        lightning.parent_residual_enabled
        and getattr(
            lightning.model_cfg,
            "parent_residual_grouped_set_loss_enabled",
            False,
        )
    )
    grouped_set_loss_enabled = bool(
        matched_set_requested or independent_set_loss_requested
    )
    strict_parent_only_enabled = bool(
        getattr(
            getattr(lightning, "model", None),
            "strict_parent_only_enabled",
            False,
        )
    )
    routing_enabled = bool(
        matched_set_requested and not strict_parent_only_enabled
    )
    lightning.parent_residual_matched_set_enabled = routing_enabled
    lightning.parent_residual_grouped_set_loss_enabled = (
        grouped_set_loss_enabled
    )
    lightning.parent_residual_posterior_mix_config = None
    lightning.parent_residual_set_loss_config = None
    lightning.parent_residual_parent_grouping = _validate_parent_grouping(
        getattr(
            lightning.model_cfg,
            "parent_residual_parent_grouping",
            "celltype",
        )
    )
    if strict_parent_only_enabled and matched_set_requested:
        raise ValueError(
            "strict Parent-only mode forbids "
            "parent_residual_matched_set_enabled=true; use "
            "parent_residual_grouped_set_loss_enabled=true for set loss"
        )
    ot_enabled = bool(
        lightning.parent_residual_enabled
        and getattr(
            lightning.model_cfg,
            "parent_residual_semi_balanced_ot_enabled",
            False,
        )
    )
    if ot_enabled and not routing_enabled:
        raise ValueError("semi-balanced OT matching requires matched-set training")
    if not grouped_set_loss_enabled:
        return
    if not lightning.parent_residual_locked_flow_enabled:
        raise ValueError(
            "matched-set transport requires parent_residual_locked_flow_enabled"
        )

    if routing_enabled:
        resume_path = getattr(lightning.model_cfg, "ckpt_path", None)
        resume_opt_in = bool(
            getattr(
                lightning.model_cfg,
                "parent_residual_matched_resume_opt_in",
                False,
            )
        )
        if resume_path not in (None, "", "null") and not resume_opt_in:
            raise ValueError(
                "matched posterior schedules require a fresh optimizer step; use "
                "partial_weight_ckpt_path unless an audited same-run full resume "
                "explicitly opts in"
            )

        mix_config = PosteriorPriorMixConfig(
            start_fraction=float(
                getattr(
                    lightning.model_cfg,
                    "parent_residual_posterior_start_fraction",
                    1.0,
                )
            ),
            end_fraction=float(
                getattr(
                    lightning.model_cfg,
                    "parent_residual_posterior_end_fraction",
                    0.0,
                )
            ),
            transition_start_step=int(
                getattr(
                    lightning.model_cfg,
                    "parent_residual_posterior_transition_start_step",
                    500,
                )
            ),
            transition_end_step=int(
                getattr(
                    lightning.model_cfg,
                    "parent_residual_posterior_transition_end_step",
                    5000,
                )
            ),
            curve=str(
                getattr(
                    lightning.model_cfg,
                    "parent_residual_posterior_curve",
                    "cosine",
                )
            ),
        ).validate()
        source_scale = (
            float(lightning.model.real_control_residual_scale)
            if getattr(lightning.model, "real_control_source_enabled", False)
            else float(lightning.model.child_beta)
        )
        if (
            max(
                float(mix_config.start_fraction),
                float(mix_config.end_fraction),
            )
            > 0.0
            and source_scale <= 0.0
        ):
            raise ValueError(
                "nonzero matched posterior source requires a positive Child "
                "centroid beta or real-control residual scale"
            )
        lightning.parent_residual_posterior_mix_config = mix_config

    set_config = ResidualSetLossConfig(
        energy_weight=float(
            getattr(lightning.model_cfg, "parent_residual_set_energy_weight", 1.0)
        ),
        log_variance_weight=float(
            getattr(
                lightning.model_cfg,
                "parent_residual_set_log_variance_weight",
                0.25,
            )
        ),
        correlation_weight=float(
            getattr(
                lightning.model_cfg,
                "parent_residual_set_correlation_weight",
                0.05,
            )
        ),
        mean_weight=float(
            getattr(lightning.model_cfg, "parent_residual_set_mean_weight", 0.0)
        ),
        min_cells_per_group=int(
            getattr(
                lightning.model_cfg,
                "parent_residual_set_min_cells_per_group",
                8,
            )
        ),
        max_cells_per_group=int(
            getattr(
                lightning.model_cfg,
                "parent_residual_set_max_cells_per_group",
                256,
            )
        ),
        normalize_projection_distance=bool(
            getattr(
                lightning.model_cfg,
                "parent_residual_set_normalize_projection_distance",
                True,
            )
        ),
        eps=float(
            getattr(lightning.model_cfg, "parent_residual_set_eps", 1e-6)
        ),
    ).validate()
    outer_weight = _finite_nonnegative(
        "parent_residual_set_loss_weight",
        getattr(lightning.model_cfg, "parent_residual_set_loss_weight", 1e-4),
    )
    pca_path = getattr(lightning.model_cfg, "parent_residual_set_pca_path", None)
    if pca_path in (None, "", "null"):
        raise ValueError(
            "matched-set diagnostics require parent_residual_set_pca_path"
        )
    artifact, projection, scale = _load_set_projection(
        pca_path,
        int(lightning.model_cfg.input_dim),
        getattr(
            lightning.model_cfg,
            "parent_residual_set_pca_components",
            None,
        ),
    )
    lightning.register_buffer(
        "parent_residual_set_projection", projection, persistent=True
    )
    lightning.register_buffer(
        "parent_residual_set_scale", scale, persistent=True
    )
    lightning.parent_residual_set_loss_config = set_config
    lightning.parent_residual_set_loss_weight = outer_weight
    if artifact is not None:
        lightning.py_logger.info(
            "Loaded Parent residual set PCA %s projection=%s scale=%s",
            artifact,
            tuple(projection.shape),
            tuple(scale.shape),
        )


def prepare_matched_training_source(
    lightning,
    *,
    batch,
    context,
    match,
    child_stds,
):
    """Use scheduled posterior/prior indices to construct the actual FM source."""

    source_probabilities = getattr(
        lightning.model,
        "source_selection_probabilities",
        lightning.model.inference_probabilities,
    )
    prior = source_probabilities(context)
    selection = select_mixed_child_indices(
        posterior=match.child_distribution.detach(),
        prior=prior,
        child_mask=context.anchor_mask,
        step=int(lightning.global_step),
        config=lightning.parent_residual_posterior_mix_config,
        num_samples=int(batch["pert_emb"].shape[1]),
        stochastic=bool(
            getattr(
                lightning.model_cfg,
                "parent_residual_posterior_source_stochastic",
                True,
            )
        ),
    )
    prepared = lightning.model.build_prior_source(
        context=context,
        cov_celltype=batch["cov_celltype"],
        cov_pert=batch["cov_pert"],
        cov_batch=batch.get("cov_batch"),
        child_stds=child_stds,
        selected_indices=selection.indices,
    )
    return prepared, selection


def _validate_parent_grouping(parent_grouping="celltype"):
    grouping = str(parent_grouping).strip().lower()
    if grouping not in {"celltype", "donor", "donor_celltype"}:
        raise ValueError(
            "parent_grouping must be 'celltype', 'donor', or 'donor_celltype'"
        )
    return grouping


def _parent_group_value(batch, parent_grouping="celltype"):
    grouping = _validate_parent_grouping(parent_grouping)
    key = "cov_celltype" if grouping == "celltype" else "cov_batch"
    if key not in batch or batch[key] is None:
        raise ValueError(f"parent_grouping={grouping!r} requires batch[{key!r}]")
    return batch[key]


def parent_condition_components(batch, parent_grouping="celltype"):
    """Return tensors forming the exact dataset/Parent/perturbation key."""

    grouping = _validate_parent_grouping(parent_grouping)
    dataset_id = batch.get("dataset_id")
    if dataset_id is None:
        dataset_id = torch.zeros_like(batch["cov_pert"])
    if grouping == "donor_celltype":
        for key in ("cov_batch", "cov_celltype"):
            if key not in batch or batch[key] is None:
                raise ValueError(
                    f"parent_grouping={grouping!r} requires batch[{key!r}]"
                )
        return (
            dataset_id,
            batch["cov_batch"],
            batch["cov_celltype"],
            batch["cov_pert"],
        )
    parent_group = _parent_group_value(batch, grouping)
    return dataset_id, parent_group, batch["cov_pert"]


def _set_group_ids(batch, set_size, parent_grouping="celltype"):
    components = []
    for value in parent_condition_components(batch, parent_grouping):
        if value.ndim == 1:
            value = value[:, None].expand(-1, int(set_size))
        elif value.ndim == 2 and value.shape[1] == 1 and int(set_size) > 1:
            value = value.expand(-1, int(set_size))
        elif value.ndim != 2 or value.shape[1] != int(set_size):
            raise ValueError("condition IDs must have shape [B] or [B,S]")
        components.append(value.to(torch.long))
    return torch.stack(components, dim=-1)


def condition_group_ids(batch, set_size=1, parent_grouping="celltype"):
    """Return collision-free dataset/Parent-axis/perturbation tuples.

    ``celltype`` preserves the historical grouping exactly. ``donor`` uses
    ``cov_batch`` as the Parent axis while cell type remains available to the
    model as a semantic covariate.
    """

    return _set_group_ids(
        batch,
        int(set_size),
        parent_grouping=parent_grouping,
    )


def augment_locked_result_with_grouped_set_loss(
    lightning,
    *,
    result,
    flow_output,
    batch,
    supervision_mask,
    parent_grouping="celltype",
):
    """Add grouped residual-set supervision without routing diagnostics."""

    set_size = int(batch["pert_emb"].shape[1])
    set_loss = grouped_residual_set_loss(
        prediction=flow_output["prediction"],
        target=batch["pert_emb"],
        group_ids=_set_group_ids(
            batch,
            set_size,
            parent_grouping=parent_grouping,
        ),
        projection=lightning.parent_residual_set_projection,
        whitening_scale=lightning.parent_residual_set_scale,
        valid_mask=supervision_mask,
        config=lightning.parent_residual_set_loss_config,
    )
    weighted_set_loss = float(
        lightning.parent_residual_set_loss_weight
    ) * set_loss.total
    result["loss"] = result["loss"] + weighted_set_loss
    raw_velocity_rms = (
        flow_output["raw_velocity"].detach().float().square().mean().sqrt()
    )
    removed_rms = flow_output["removed_common_velocity_rms"].detach()
    result.update(
        {
            "parent_residual_set_loss_raw": set_loss.total.detach(),
            "parent_residual_set_loss_weighted": weighted_set_loss.detach(),
            "parent_residual_set_energy": set_loss.energy.detach(),
            "parent_residual_set_log_variance": set_loss.log_variance.detach(),
            "parent_residual_set_correlation": set_loss.correlation.detach(),
            "parent_residual_set_mean": set_loss.mean.detach(),
            "parent_residual_set_variance_ratio": set_loss.variance_ratio.detach(),
            "parent_residual_set_valid_groups": set_loss.valid_groups.detach(),
            "parent_residual_set_mean_group_size": set_loss.mean_group_size.detach(),
            "parent_residual_raw_velocity_rms": raw_velocity_rms,
            "parent_residual_common_drift_ratio": (
                removed_rms / raw_velocity_rms.clamp_min(1e-8)
            ).detach(),
        }
    )
    return set_loss


def augment_locked_result_with_matched_set(
    lightning,
    *,
    result,
    flow_output,
    batch,
    supervision_mask,
    selection,
    parent_grouping="celltype",
):
    """Add grouped set supervision and historical routing diagnostics."""

    set_loss = augment_locked_result_with_grouped_set_loss(
        lightning,
        result=result,
        flow_output=flow_output,
        batch=batch,
        supervision_mask=supervision_mask,
        parent_grouping=parent_grouping,
    )
    condition_routing = condition_level_posterior_prior_diagnostics(
        selection.posterior_distribution,
        selection.prior_distribution,
        group_ids=_set_group_ids(
            batch,
            1,
            parent_grouping=parent_grouping,
        )[:, 0],
        row_mask=supervision_mask[:, 0],
    )
    result.update(
        {
            "parent_residual_posterior_schedule_step": result["loss1"].new_tensor(
                float(lightning.global_step)
            ),
            "parent_residual_posterior_fraction_scheduled": selection.posterior_fraction.detach(),
            "parent_residual_posterior_fraction_realized": selection.realized_posterior_fraction.detach(),
            "parent_residual_posterior_eligible_fraction": selection.posterior_eligible.float().mean().detach(),
            "parent_residual_condition_qbar_prior_kl": condition_routing.kl_mean,
            "parent_residual_condition_qbar_prior_js": condition_routing.js_mean,
            "parent_residual_condition_qbar_prior_js_max": condition_routing.js_max,
            "parent_residual_condition_qbar_entropy": condition_routing.qbar_entropy_mean,
            "parent_residual_condition_prior_entropy": condition_routing.prior_entropy_mean,
            "parent_residual_condition_routing_valid_groups": condition_routing.valid_groups,
            "parent_residual_condition_routing_mean_group_size": condition_routing.mean_group_size,
        }
    )
    return set_loss


def matched_teacher_auxiliary(lightning, match):
    """Small explicit supervision that makes the Hungarian teacher meaningful."""

    weights = {
        "delta": _finite_nonnegative(
            "parent_residual_teacher_delta_weight",
            getattr(
                lightning.model_cfg,
                "parent_residual_teacher_delta_weight",
                0.0,
            ),
        ),
        "direction": _finite_nonnegative(
            "parent_residual_teacher_direction_weight",
            getattr(
                lightning.model_cfg,
                "parent_residual_teacher_direction_weight",
                0.0,
            ),
        ),
        "magnitude": _finite_nonnegative(
            "parent_residual_teacher_magnitude_weight",
            getattr(
                lightning.model_cfg,
                "parent_residual_teacher_magnitude_weight",
                0.0,
            ),
        ),
        "latent": _finite_nonnegative(
            "parent_residual_teacher_latent_weight",
            getattr(
                lightning.model_cfg,
                "parent_residual_teacher_latent_weight",
                0.0,
            ),
        ),
    }
    weighted = {
        "delta": weights["delta"] * match.delta_loss,
        "direction": weights["direction"] * match.direction_loss,
        "magnitude": weights["magnitude"] * match.magnitude_loss,
        "latent": weights["latent"] * match.latent_match_loss,
    }
    total = sum(weighted.values(), match.delta_loss.new_zeros(()))
    logs = {
        "parent_residual_teacher_auxiliary": total.detach(),
        "parent_residual_teacher_delta_weighted": weighted["delta"].detach(),
        "parent_residual_teacher_direction_weighted": weighted["direction"].detach(),
        "parent_residual_teacher_magnitude_weighted": weighted["magnitude"].detach(),
        "parent_residual_teacher_latent_weighted": weighted["latent"].detach(),
        "parent_residual_teacher_delta": match.delta_loss.detach(),
        "parent_residual_teacher_direction": match.direction_loss.detach(),
        "parent_residual_teacher_magnitude": match.magnitude_loss.detach(),
        "parent_residual_teacher_latent": match.latent_match_loss.detach(),
    }
    return total, logs


__all__ = [
    "augment_locked_result_with_grouped_set_loss",
    "augment_locked_result_with_matched_set",
    "initialize_matched_set_transport",
    "condition_group_ids",
    "matched_teacher_auxiliary",
    "parent_condition_components",
    "prepare_matched_training_source",
]

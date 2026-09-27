"""Narrow integration helpers for the existing PerturbDiff Lightning module."""

from __future__ import annotations

import torch

from .centering import condition_group_ids as encode_condition_group_ids
from .matched_integration import parent_condition_components


def _batch_condition_group_ids(batch, parent_grouping="celltype"):
    return encode_condition_group_ids(
        *parent_condition_components(
            batch,
            parent_grouping=parent_grouping,
        )
    )


def _sample_locked_flow_time_and_weights(
    lightning,
    *,
    group_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    """Sample one flow time per condition in Child mean-carrying mode.

    The common/centred decomposition is a condition-level operation.  Giving
    cells from the same condition different times would make ``t * mean_shift``
    cease to be a common mode and would create a train/sampling mismatch.
    Legacy Parent-locked flow deliberately keeps its original per-row draw.
    """

    if not bool(
        getattr(lightning, "parent_residual_child_mean_flow_enabled", False)
    ):
        return lightning.schedule_sampler.sample(int(batch_size), device)
    if group_ids.ndim == 1:
        row_group_ids = group_ids
    elif group_ids.ndim == 2:
        if group_ids.shape[1] < 1 or not torch.equal(
            group_ids,
            group_ids[:, :1].expand_as(group_ids),
        ):
            raise ValueError(
                "Child mean-carrying flow requires one condition group per "
                "batch row"
            )
        row_group_ids = group_ids[:, 0]
    else:
        raise ValueError("condition group_ids must have shape [B] or [B,S]")
    if row_group_ids.shape != (int(batch_size),):
        raise ValueError("condition group_ids do not match the flow batch size")
    _, inverse = torch.unique(
        row_group_ids.to(torch.long),
        sorted=True,
        return_inverse=True,
    )
    group_time, group_weights = lightning.schedule_sampler.sample(
        int(inverse.max().item()) + 1,
        device,
    )
    return (
        group_time.index_select(0, inverse),
        group_weights.index_select(0, inverse),
    )


def compute_locked_flow_base_result(
    lightning,
    *,
    batch,
    prepared,
    context,
    supervision_mask,
    device,
    selection=None,
    parent_grouping="celltype",
):
    """Replace only the legacy expression-space base loss when enabled."""

    group_ids = _batch_condition_group_ids(batch, parent_grouping)
    from src.models.sparse_hurdle_de.lightning_integration import (
        prepare_end_to_end_hurdle_training_state,
    )

    hurdle_state = prepare_end_to_end_hurdle_training_state(
        lightning,
        batch=batch,
        prepared=prepared,
        context=context,
        supervision_mask=supervision_mask,
        group_ids=group_ids,
    )
    target_condition_mean = (
        prepared.parent.lookup.control_mean
        + prepared.parent.lookup.target_delta
    ).to(prepared.parent.parent_mean)
    time, weights = _sample_locked_flow_time_and_weights(
        lightning,
        group_ids=group_ids,
        batch_size=batch["pert_emb"].shape[0],
        device=device,
    )
    self_condition = {
        "batch_emb": batch["batch_emb"],
        "cont_emb": prepared.source.source.detach(),
        "cov_celltype": batch["cov_celltype"],
        "cov_pert": batch["cov_pert"],
        "ds_name": batch.get("ds_name"),
    }
    conditioning_tensors = getattr(
        lightning.model,
        "conditioning_tensors",
        None,
    )
    if conditioning_tensors is None:
        if bool(
            getattr(lightning.model, "strict_parent_only_enabled", False)
        ):
            raise ValueError(
                "strict Parent-only flow requires model.conditioning_tensors"
            )
        # Compatibility for minimal legacy velocity adapters used independently
        # of ParentResidualGeneDiTModel.
        model_conditioning = {
            "parent_residual_parent_token": context.parent_token,
            "parent_residual_child_tokens": prepared.selected_child_tokens,
            "parent_residual_routed_token": context.routed_token,
        }
        for key in (
            "parent_residual_all_child_tokens",
            "parent_residual_child_mask",
            "parent_residual_perturbation_query",
        ):
            if key in batch:
                model_conditioning[key] = batch[key]
    else:
        model_conditioning = conditioning_tensors(context, prepared)
    self_condition.update(model_conditioning)
    flow_output = lightning.parent_locked_residual_flow.training_losses(
        lightning.model,
        target_expression=(
            batch["pert_emb"]
            if hurdle_state is None
            else hurdle_state.target_expression
        ),
        source_expression=prepared.source.source.detach(),
        parent_mean=prepared.parent.parent_mean,
        target_condition_mean=target_condition_mean,
        time=time,
        self_condition=self_condition,
        group_ids=group_ids,
        valid_mask=supervision_mask,
        target_projection_override=(
            None if hurdle_state is None else hurdle_state.observed_projection
        ),
        target_condition_mean_mask=getattr(
            prepared.parent.lookup,
            "target_available",
            None,
        ),
    )
    flow_output["end_to_end_hurdle_state"] = hurdle_state
    group_counts = flow_output["pair"].target_projection.counts
    required_group_cells = int(
        getattr(
            lightning.model_cfg,
            "parent_residual_locked_min_group_cells",
            8,
        )
    )
    if lightning.training and int(group_counts.min()) < required_group_cells:
        raise ValueError(
            "Parent-locked residual flow requires homogeneous condition groups "
            f"with at least {required_group_cells} valid cells; observed "
            f"minimum={int(group_counts.min())}. Enable the CombinationEpisode "
            "sampler or provide a larger cell set."
        )

    zero = flow_output["loss"].new_zeros(())
    result = {
        "loss": flow_output["loss_per_batch"],
        "loss1": flow_output["loss"],
        "loss1_per_sample": flow_output["loss_per_batch"],
        "mmd1": zero,
        "mmd1_per_sample": torch.zeros_like(weights),
        "weights": weights,
        "parent_residual_locked_flow_loss": flow_output["loss"].detach(),
        "parent_residual_centered_flow_loss": flow_output.get(
            "centered_flow_loss", flow_output["loss"]
        ).detach(),
        "parent_residual_removed_common_velocity_rms": flow_output[
            "removed_common_velocity_rms"
        ].detach(),
        "parent_residual_locked_group_min_cells": group_counts.min()
        .to(zero)
        .detach(),
        "parent_residual_locked_group_count": zero.new_tensor(
            group_counts.numel()
        ),
    }
    if "common_velocity_loss" in flow_output:
        result.update(
            {
                "parent_residual_common_velocity_loss": flow_output[
                    "common_velocity_loss"
                ].detach(),
                "parent_residual_weighted_common_velocity_loss": flow_output[
                    "weighted_common_velocity_loss"
                ].detach(),
            }
        )
    if "child_mean_loss" in flow_output:
        result.update(
            {
                "parent_residual_child_mean_loss": flow_output[
                    "child_mean_loss"
                ].detach(),
                "parent_residual_child_common_velocity_rms": flow_output[
                    "common_velocity_rms"
                ].detach(),
                "parent_residual_child_integrated_mean_shift_rms": flow_output[
                    "integrated_mean_shift"
                ].detach()
                .float()
                .square()
                .mean()
                .sqrt(),
            }
        )
    if not bool(
        getattr(lightning, "parent_residual_strict_parent_only_enabled", False)
    ):
        from .child_pds import augment_locked_result_with_child_endpoint_pds

        augment_locked_result_with_child_endpoint_pds(
            lightning,
            result=result,
            flow_output=flow_output,
            prepared=prepared,
        )
    grouped_set_loss_enabled = bool(
        getattr(
            lightning,
            "parent_residual_grouped_set_loss_enabled",
            getattr(
                lightning,
                "parent_residual_matched_set_enabled",
                False,
            ),
        )
    )
    if grouped_set_loss_enabled:
        matched_routing_enabled = bool(
            getattr(
                lightning,
                "parent_residual_matched_set_enabled",
                False,
            )
        )
        if matched_routing_enabled:
            if selection is None:
                raise ValueError(
                    "matched-set locked flow requires a source selection"
                )
            from src.models.parent_locked_residual.matched_integration import (
                augment_locked_result_with_matched_set,
            )

            augment_locked_result_with_matched_set(
                lightning,
                result=result,
                flow_output=flow_output,
                batch=batch,
                supervision_mask=supervision_mask,
                selection=selection,
                parent_grouping=parent_grouping,
            )
        else:
            from src.models.parent_locked_residual.matched_integration import (
                augment_locked_result_with_grouped_set_loss,
            )

            augment_locked_result_with_grouped_set_loss(
                lightning,
                result=result,
                flow_output=flow_output,
                batch=batch,
                supervision_mask=supervision_mask,
                parent_grouping=parent_grouping,
            )
    return result, flow_output


def attach_locked_inference_fields(
    lightning, batch, prepared, parent_grouping="celltype"
):
    """Persist everything the sampling hook needs on the current batch."""

    if not lightning.parent_residual_locked_flow_enabled:
        return None
    group_ids = _batch_condition_group_ids(batch, parent_grouping)
    batch["parent_residual_locked_parent_mean"] = prepared.parent.parent_mean
    batch["parent_residual_locked_group_ids"] = group_ids
    return group_ids


__all__ = [
    "attach_locked_inference_fields",
    "compute_locked_flow_base_result",
]

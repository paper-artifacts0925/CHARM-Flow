"""Sampling generation core split from rawdata_diffusion_sample.py (logic-preserving)."""

import gc
import time

import anndata
import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics import r2_score

from src.apps.sampling.sampling_io import save_adata
from src.apps.sampling.sampling_provenance import (
    add_provenance_metadata,
    assert_paired_provenance_equal,
    attach_paired_output_provenance,
    collect_treated_provenance,
)
from src.apps.sampling.sampling_units import scale_saved_expression_units
from src.apps.sampling.sampling_generation_helpers import (
    build_obs_data,
    build_self_condition,
    build_x_ctrl,
    build_gene_embedding_cache,
    collect_batch_covariates,
    load_ctrl_adata,
    load_selected_genes,
    resolve_sampling_runner,
)
from src.models.parent_locked_residual import (
    project_group_nonnegative_fixed_mean,
    project_group_zero_mean,
)


def _expand_parent_locked_mean(parent_mean, reference):
    """Expand a Parent mean to ``[B,S,G]`` without changing its values."""

    if not torch.is_tensor(parent_mean) or not parent_mean.is_floating_point():
        raise TypeError("Parent-locked parent_mean must be a floating-point tensor")
    if not torch.is_tensor(reference) or reference.ndim != 3:
        raise ValueError("Parent-locked reference must have shape [B,S,G]")
    if parent_mean.device != reference.device:
        raise ValueError("Parent-locked parent_mean must share the source device")
    batch_size, set_size, genes = reference.shape
    parent_mean = parent_mean.to(dtype=reference.dtype)
    if tuple(parent_mean.shape) == (batch_size, genes):
        parent_mean = parent_mean[:, None, :].expand(-1, set_size, -1)
    elif tuple(parent_mean.shape) == (batch_size, 1, genes):
        parent_mean = parent_mean.expand(-1, set_size, -1)
    elif tuple(parent_mean.shape) != tuple(reference.shape):
        raise ValueError(
            "Parent-locked parent_mean must have shape [B,G], [B,1,G], "
            "or [B,S,G]"
        )
    if not torch.isfinite(parent_mean).all():
        raise ValueError("Parent-locked parent_mean must be finite")
    return parent_mean.detach()


def _parent_locked_group_mean_max_abs(value, projection, minimum_count=1):
    """Maximum absolute condition mean in ``value`` over selected groups."""

    flat_mask = projection.valid_mask.reshape(-1)
    valid_positions = torch.nonzero(flat_mask, as_tuple=False).flatten()
    flat_ids = projection.group_ids.reshape(-1).index_select(0, valid_positions)
    feature_size = int(np.prod(value.shape[2:]))
    flat_value = value.reshape(-1, feature_size).index_select(0, valid_positions)
    accumulation_dtype = (
        torch.float32
        if flat_value.dtype in (torch.float16, torch.bfloat16)
        else flat_value.dtype
    )
    sums = torch.zeros(
        projection.counts.numel(),
        feature_size,
        dtype=accumulation_dtype,
        device=value.device,
    ).index_add_(0, flat_ids, flat_value.to(accumulation_dtype))
    means = sums / projection.counts.to(accumulation_dtype)[:, None]
    selected = projection.counts >= int(minimum_count)
    if not selected.any():
        return 0.0
    return float(means[selected].abs().max().detach().cpu())


def _sample_parent_locked_with_scientific_gate(
    model,
    *,
    routing,
    self_condition,
    group_ids,
    valid_mask,
    guidance_strength,
    endpoint_mean_drift_tolerance,
    cell_keys=None,
):
    """Sample, then enforce the feasible nonnegative Parent simplex."""

    tolerance = float(endpoint_mean_drift_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "sampling.parent_locked_endpoint_mean_drift_tolerance must be "
            "finite and non-negative"
        )
    source = routing["source"]
    parent = _expand_parent_locked_mean(routing["parent_mean"], source)
    source_projection = project_group_zero_mean(
        source - parent,
        group_ids=group_ids,
        valid_mask=valid_mask,
        singleton_policy="zero",
    )
    safe_source = parent + source_projection.value
    locked_sample = model.parent_locked_residual_flow.sample(
        model.model,
        source_expression=safe_source,
        parent_mean=routing["parent_mean"],
        self_condition=self_condition,
        group_ids=group_ids,
        valid_mask=valid_mask,
        guidance_strength=float(guidance_strength),
    )

    raw_expression = locked_sample.expression
    child_endpoint_mean = getattr(locked_sample, "endpoint_mean", None)
    endpoint_mean_reference = (
        parent
        if child_endpoint_mean is None
        else _expand_parent_locked_mean(child_endpoint_mean, raw_expression)
    )
    endpoint_projection = project_group_nonnegative_fixed_mean(
        raw_expression,
        endpoint_mean_reference,
        group_ids=group_ids,
        valid_mask=valid_mask,
    )
    expression = endpoint_projection.value
    from src.models.sparse_hurdle_de.lightning_integration import (
        apply_end_to_end_terminal_hurdle,
    )

    expression, hurdle_projection = apply_end_to_end_terminal_hurdle(
        model,
        expression=expression,
        parent_mean=endpoint_mean_reference,
        group_ids=group_ids,
        valid_mask=valid_mask,
        zero_rate_logits=routing.get("parent_residual_hurdle_terminal_zero_logits"),
        cell_keys=cell_keys,
    )
    if hurdle_projection is not None:
        endpoint_projection = hurdle_projection
    raw_endpoint_residual = raw_expression - endpoint_mean_reference
    raw_non_singleton_drift = _parent_locked_group_mean_max_abs(
        raw_endpoint_residual,
        endpoint_projection,
        minimum_count=2,
    )
    endpoint_drift = float(
        endpoint_projection.group_mean_drift_max_abs.detach().cpu()
    )
    if not np.isfinite(endpoint_drift) or endpoint_drift > tolerance:
        raise AssertionError(
            "Parent-locked feasible endpoint mean drift exceeded the scientific "
            f"gate: {endpoint_drift:.8g} > {tolerance:.8g}"
        )

    valid_values = expression[endpoint_projection.valid_mask]
    minimum_expression = float(valid_values.min().detach().cpu())
    negative_after = int((valid_values < 0).sum().item())
    if minimum_expression < 0.0 or negative_after:
        raise AssertionError(
            "Parent-locked nonnegative projection produced a negative endpoint"
        )
    valid_cells = int(endpoint_projection.valid_mask.sum().item())
    singleton_cells = int(endpoint_projection.singleton_mask.sum().item())
    if valid_cells < 1 or not 0 <= singleton_cells <= valid_cells:
        raise AssertionError("invalid Parent-locked singleton accounting")
    singleton_fraction = singleton_cells / valid_cells
    if not 0.0 <= singleton_fraction <= 1.0:
        raise AssertionError("Parent-locked singleton_fraction is outside [0,1]")
    diagnostics = {
        "valid_cells": valid_cells,
        "singleton_cells": singleton_cells,
        "singleton_fraction": singleton_fraction,
        "endpoint_mean_drift_max_abs": endpoint_drift,
        "raw_non_singleton_endpoint_mean_drift_max_abs": raw_non_singleton_drift,
        "minimum_expression": minimum_expression,
        "negative_values_before_projection": int(
            endpoint_projection.negative_values_before.detach().cpu()
        ),
        "negative_values_after_projection": negative_after,
        "raw_parent_adjustment_max_abs": float(
            endpoint_projection.raw_parent_adjustment_max_abs.detach().cpu()
        ),
        "infeasible_negative_parent_group_genes": int(
            (endpoint_projection.raw_group_parent_mean < 0).sum().item()
        ),
        "endpoint_mean_policy": (
            "parent"
            if child_endpoint_mean is None
            else "integrated_child_mean"
        ),
    }
    return expression, diagnostics, endpoint_projection


def _summarize_parent_locked_sampling_gate(
    batch_diagnostics,
    *,
    max_singleton_fraction,
    endpoint_mean_drift_tolerance,
):
    """Aggregate and validate the formal feasible-Parent sampling gate."""

    maximum_fraction = float(max_singleton_fraction)
    tolerance = float(endpoint_mean_drift_tolerance)
    if not np.isfinite(maximum_fraction) or not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError(
            "sampling.parent_locked_max_singleton_fraction must lie in [0,1]"
        )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "sampling.parent_locked_endpoint_mean_drift_tolerance must be "
            "finite and non-negative"
        )
    if not batch_diagnostics:
        raise AssertionError("Parent-locked sampling produced no gate diagnostics")
    required = {
        "minimum_expression",
        "negative_values_before_projection",
        "negative_values_after_projection",
        "raw_parent_adjustment_max_abs",
        "infeasible_negative_parent_group_genes",
    }
    for item in batch_diagnostics:
        missing = sorted(required.difference(item))
        if missing:
            raise AssertionError(f"Parent-lock batch diagnostics are missing {missing}")

    valid_cells = sum(item["valid_cells"] for item in batch_diagnostics)
    singleton_cells = sum(item["singleton_cells"] for item in batch_diagnostics)
    if valid_cells < 1:
        raise AssertionError("Parent-locked sampling produced no valid cells")
    singleton_fraction = singleton_cells / valid_cells
    endpoint_drift = max(
        item["endpoint_mean_drift_max_abs"] for item in batch_diagnostics
    )
    raw_endpoint_drift = max(
        item["raw_non_singleton_endpoint_mean_drift_max_abs"]
        for item in batch_diagnostics
    )
    minimum_expression = min(item["minimum_expression"] for item in batch_diagnostics)
    negative_before = sum(
        item["negative_values_before_projection"] for item in batch_diagnostics
    )
    negative_after = sum(
        item["negative_values_after_projection"] for item in batch_diagnostics
    )
    parent_adjustment = max(
        item["raw_parent_adjustment_max_abs"] for item in batch_diagnostics
    )
    infeasible_parent = sum(
        item["infeasible_negative_parent_group_genes"]
        for item in batch_diagnostics
    )
    endpoint_mean_policies = {
        item.get("endpoint_mean_policy", "parent")
        for item in batch_diagnostics
    }
    if len(endpoint_mean_policies) != 1:
        raise AssertionError(
            "sampling mixed Parent and Child endpoint-mean policies"
        )
    endpoint_mean_policy = endpoint_mean_policies.pop()
    if not 0.0 <= singleton_fraction <= 1.0:
        raise AssertionError("Parent-locked singleton_fraction is outside [0,1]")
    if singleton_fraction > maximum_fraction:
        raise AssertionError(
            "Parent-locked singleton_fraction exceeded the scientific gate: "
            f"{singleton_fraction:.8g} > {maximum_fraction:.8g}. Increase the "
            "main sampling batch size or use a condition-grouped sampler; do "
            "not switch to fixed-source row replication."
        )
    if not np.isfinite(endpoint_drift) or endpoint_drift > tolerance:
        raise AssertionError(
            "Parent-locked feasible endpoint mean drift exceeded the scientific "
            f"gate: {endpoint_drift:.8g} > {tolerance:.8g}"
        )
    if not np.isfinite(minimum_expression) or minimum_expression < 0.0 or negative_after:
        raise AssertionError("Parent-locked endpoint failed the nonnegative gate")
    return {
        "schema_version": 2,
        "singleton_policy": "effective_parent_zero_residual",
        "projection_policy": "euclidean_nonnegative_fixed_effective_parent_mean",
        "parent_mean_policy": (
            "clamp_condition_parent_mean_at_zero"
            if endpoint_mean_policy == "parent"
            else "clamp_condition_integrated_child_mean_at_zero"
        ),
        "endpoint_mean_policy": endpoint_mean_policy,
        "batch_count": len(batch_diagnostics),
        "valid_cells": valid_cells,
        "singleton_cells": singleton_cells,
        "singleton_fraction": float(singleton_fraction),
        "max_singleton_fraction": maximum_fraction,
        "endpoint_mean_drift_max_abs": float(endpoint_drift),
        "endpoint_mean_drift_tolerance": tolerance,
        "raw_non_singleton_endpoint_mean_drift_max_abs": float(raw_endpoint_drift),
        "minimum_expression": float(minimum_expression),
        "negative_values_before_projection": int(negative_before),
        "negative_values_after_projection": int(negative_after),
        "raw_parent_adjustment_max_abs": float(parent_adjustment),
        "infeasible_negative_parent_group_genes": int(infeasible_parent),
        "saved_h5ad_audit_required": True,
        "passed": True,
    }


def build_initial_sampling_state(diffusion, cfg, batch_data, shape, device, logger):
    """Return optional x_t initialization for reverse sampling."""
    mode = str(getattr(cfg.sampling, "initial_state", "gaussian") or "gaussian").lower()
    has_parent_residual_source = "parent_residual_source" in batch_data
    if mode == "parent_residual" or has_parent_residual_source:
        if not has_parent_residual_source:
            raise ValueError(
                "Parent-Residual initialization requires "
                "batch_data['parent_residual_source']"
            )
        initial = batch_data["parent_residual_source"]
        if not torch.is_tensor(initial):
            raise TypeError("batch_data['parent_residual_source'] must be a tensor")
        initial = initial.to(device=device, dtype=torch.float32)
        if tuple(initial.shape) != tuple(shape):
            raise ValueError(
                f"Parent-Residual initial state shape {tuple(initial.shape)} "
                f"!= requested shape {tuple(shape)}"
            )
        if not torch.isfinite(initial).all():
            raise ValueError("Parent-Residual initial state must be finite")
        logger.info(
            "Using Parent-Residual source as the exact rectified-flow t=0 state"
        )
        return initial
    has_anchor_bridge_source = "anchor_bridge_source" in batch_data
    if mode == "anchor_bridge" or has_anchor_bridge_source:
        if not has_anchor_bridge_source:
            raise ValueError(
                "Anchor-Bridge initialization requires "
                "batch_data['anchor_bridge_source']"
            )
        initial = batch_data["anchor_bridge_source"]
        if not torch.is_tensor(initial):
            raise TypeError("batch_data['anchor_bridge_source'] must be a tensor")
        initial = initial.to(device=device, dtype=torch.float32)
        if tuple(initial.shape) != tuple(shape):
            raise ValueError(
                f"Anchor-Bridge initial state shape {tuple(initial.shape)} "
                f"!= requested shape {tuple(shape)}"
            )
        if not torch.isfinite(initial).all():
            raise ValueError("Anchor-Bridge initial state must be finite")
        logger.info(
            "Using Anchor-Bridge source as the exact rectified-flow t=0 state"
        )
        return initial
    if mode in ("gaussian", "none", "null"):
        return None
    if mode not in ("control_context", "control_cluster", "cluster"):
        raise ValueError(f"Unknown sampling.initial_state={mode}")
    if "cont_emb" not in batch_data:
        raise ValueError("control-context initialization requires batch_data[cont_emb]")
    x_start = batch_data["cont_emb"].to(device=device, dtype=torch.float32)
    if tuple(x_start.shape) != tuple(shape):
        raise ValueError(
            f"control-context initial state shape {tuple(x_start.shape)} != requested shape {tuple(shape)}"
        )
    t = torch.full(
        (shape[0],),
        max(int(cfg.sampling.start_time) - 1, 0),
        device=device,
        dtype=torch.long,
    )
    initial = diffusion.q_sample(x_start, t, noise=torch.randn_like(x_start))
    logger.info("Using %s initial sampling state from noised control context", mode)
    return initial


def generate_samples(
    model,
    diffusion,
    cfg, 
    device, 
    logger, 
    datamodule, 
    pca_for_decode=None,
    store_noised_truth=False,
):
    """
    Dispatcher that preserves original behavior:
      - If `pca_for_decode` is None -> sample directly in **original gene space**.
      - If `pca_for_decode` is provided -> delegate to `generate_samples_pca` and return its outputs.

    Gene-space mode:
      - pert_emb and samples are [B, S, G] (G ~ 2000)
      - R^2 and MMD are computed in gene space (this is the working space, not a 'decoded' space)
      - Returns (truths, samples, trajectories, decoded=None)
    """
    split = str(getattr(cfg.sampling, "split", "test"))
    split_names = list(
        getattr(datamodule, "evaluation_split_names", datamodule.all_split_names)
    )
    if split not in split_names:
        raise ValueError(f"Unknown sampling split {split!r}; available: {split_names}")
    dataloader = datamodule.val_dataloader()[split_names.index(split)]

    if getattr(cfg.sampling, "fixed_cells_per_perturbation", None) is not None:
        return generate_fixed_source_samples(
            model, diffusion, cfg, device, logger, datamodule, dataloader
        )

    dataloader_iter = iter(dataloader)
    model = model.to(device)
    model.eval()

    # Model type helpers
    model_type = getattr(getattr(model, "model_cfg", None), "model_type", None)
    if model_type is None:
        model_type = getattr(cfg.model, "model_type", None)

    # Set up sampling parameters
    batch_size = cfg.sampling.batch_size
    if cfg.data.use_cell_set is not None:
        batch_size = int(batch_size // cfg.data.use_cell_set)
    num_sampled_batches = cfg.sampling.num_sampled_batches
    if num_sampled_batches is None:
        num_sampled_batches = len(dataloader)
    num_samples = len(dataloader.sampler)
    
    use_ddim = cfg.sampling.use_ddim
    clip_denoised = cfg.sampling.clip_denoised
    progress = cfg.sampling.progress

    # Resolve sampling dimensionality from config if available.
    input_dim = (
        getattr(cfg.model, "input_dim", None)
        or getattr(cfg.model, "output_size", None)
        or getattr(cfg.model, "gene_dim", None)
    )

    logger.info(f"Generating {num_samples} samples with batch size {batch_size}")
    if getattr(model, "cell_detr_direct_enabled", False):
        logger.info("Using Direct-Match one-pass sampling (no t, no FM solver)")
    else:
        logger.info(f"Using {'DDIM' if use_ddim else 'DDPM'} sampling")
    
    all_truths = []
    all_samples = []
    all_trajectories = []
    all_covariates = []
    all_treated_provenance = []
    all_projection_group_ids = []
    all_projection_references = []
    projection_group_offset = 0
    parent_locked_gate_batches = []
    parent_locked_gate_summary = None
    parent_locked_enabled = bool(
        getattr(model, "parent_residual_locked_flow_enabled", False)
    )
    if parent_locked_enabled:
        parent_locked_max_singleton_fraction = float(
            getattr(cfg.sampling, "parent_locked_max_singleton_fraction", 1.0)
        )
        parent_locked_endpoint_mean_drift_tolerance = float(
            getattr(
                cfg.sampling,
                "parent_locked_endpoint_mean_drift_tolerance",
                1.0e-5,
            )
        )
    else:
        # Do not even parse Parent-lock-only controls on legacy sampling paths.
        parent_locked_max_singleton_fraction = 1.0
        parent_locked_endpoint_mean_drift_tolerance = 1.0e-5
    
    # Generate samples in batches

    cell_set_number = cfg.data.use_cell_set if hasattr(cfg.data, 'use_cell_set') and cfg.data.use_cell_set is not None else 1

    data_args = getattr(dataloader.dataset, "data_args", None)
    normalize_counts = getattr(data_args, "normalize_counts", None) if data_args is not None else None

    assert store_noised_truth == False
    
    _genes = load_selected_genes(cfg)
    assert len(_genes) == 2000

    with torch.no_grad():
        for batch_idx in range(num_sampled_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            logger.info(f"Generating batch {batch_idx + 1}/{num_sampled_batches} (size: {current_batch_size})")
            
            direct_match = bool(
                getattr(model, "cell_detr_direct_enabled", False)
            )
            sample_fn = sampling_kwargs = None
            if not direct_match:
                sample_fn, sampling_kwargs = resolve_sampling_runner(
                    cfg, diffusion, use_ddim
                )

            batch_data = next(dataloader_iter)
            batch_data = move_data_to_device(batch_data, device)
            use_latent_prior = bool(
                direct_match
                or getattr(model, "match_fm_enabled", False)
                or getattr(cfg.model, "latent_control_mixture", False)
                or getattr(model, "hungarian_flow_enabled", False)
                or getattr(model, "cell_detr_enabled", False)
                or getattr(model, "parent_residual_enabled", False)
            )
            has_control_bank = (
                "cell_detr_control_bank" in batch_data
                or "latent_control_emb" in batch_data
            )
            strict_parent_direct = bool(
                getattr(model, "parent_residual_strict_direct_flow_enabled", False)
            )

            routing = None
            if use_latent_prior and (has_control_bank or strict_parent_direct):
                if direct_match:
                    routing = model.apply_cell_detr_direct_prediction(batch_data)
                    label = "Direct-Match"
                elif getattr(model, "parent_residual_enabled", False):
                    routing = model.apply_parent_residual_inference_context(batch_data)
                    label = "Parent-Residual Gene-DiT"
                elif getattr(model, "cell_detr_enabled", False):
                    routing = model.apply_cell_detr_inference_context(batch_data)
                    label = "Cell-DETR"
                elif getattr(model, "hungarian_flow_enabled", False):
                    routing = model.apply_hungarian_flow_inference_context(batch_data)
                    label = "Hungarian-flow"
                else:
                    routing = model.apply_match_fm_inference_prior(batch_data)
                    label = "MATCH-FM"
                prior = routing.get("prior")
                if prior is None:
                    logger.info("%s inference uses no Child prior", label)
                else:
                    logger.info(
                        "%s inference prior entropy=%.4f",
                        label,
                        float((-(prior * prior.clamp_min(1e-8).log()).sum(dim=-1).mean()).cpu()),
                    )


            batch_data["batch_emb"] = model._encode_covariates(batch_data)
            pert_emb = batch_data["pert_emb"]
            # Fallback to runtime batch shape when config does not expose a sampling dim.
            resolved_input_dim = int(input_dim) if input_dim is not None else int(pert_emb.shape[-1])

            if len(pert_emb) != current_batch_size:
                logger.warning("batch size mismatch detected; this may be the last batch")
            current_batch_size = len(pert_emb)
            device = pert_emb.device
            gene_emb = build_gene_embedding_cache(model, batch_data, device)
            self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
            # Generate samples
            start_time = time.time()
            mmd_loss = SamplesLoss(loss="energy", blur=0.05, scaling=0.5).to(device)

            mask = ~batch_data["is_padded_list"].bool()
            all_treated_provenance.append(
                collect_treated_provenance(batch_data, dataloader.dataset, mask)
            )
            sample_shape = (current_batch_size, cell_set_number, resolved_input_dim)
            endpoint_projection = None
            if direct_match:
                if routing is None:
                    raise ValueError("Direct-Match sampling has no control bank")
                sample = routing["prediction"]
                traj = None
            elif parent_locked_enabled:
                if routing is None:
                    raise ValueError("Parent-locked sampling has no prepared prior")
                sample, gate_diagnostics, endpoint_projection = _sample_parent_locked_with_scientific_gate(
                    model,
                    routing=routing,
                    self_condition=self_condition,
                    group_ids=batch_data["parent_residual_locked_group_ids"],
                    valid_mask=mask,
                    guidance_strength=float(
                        getattr(cfg.sampling, "guidance_strength", 0.0)
                    ),
                    endpoint_mean_drift_tolerance=(
                        parent_locked_endpoint_mean_drift_tolerance
                    ),
                    cell_keys=batch_data.get("cell_key"),
                )
                parent_locked_gate_batches.append(gate_diagnostics)
                logger.info(
                    "Parent-lock scientific gate batch=%d "
                    "singleton_fraction=%.8f singleton_cells=%d/%d "
                    "endpoint_mean_drift_max_abs=%.8g "
                    "raw_non_singleton_endpoint_mean_drift_max_abs=%.8g",
                    batch_idx + 1,
                    gate_diagnostics["singleton_fraction"],
                    gate_diagnostics["singleton_cells"],
                    gate_diagnostics["valid_cells"],
                    gate_diagnostics["endpoint_mean_drift_max_abs"],
                    gate_diagnostics[
                        "raw_non_singleton_endpoint_mean_drift_max_abs"
                    ],
                )
                traj = None
            else:
                initial_state = build_initial_sampling_state(
                    diffusion, cfg, batch_data, sample_shape, device, logger
                )
                sample, traj = sample_fn(
                    model.model,
                    sample_shape,
                    self_condition=self_condition,
                    noise=initial_state,
                    clip_denoised=clip_denoised,
                    device=device,
                    progress=progress,
                    **sampling_kwargs
                )

            # remove duplicated sample that are used for padding
            pert_emb = pert_emb[mask]
            if sample is not None:
                sample = sample[mask]

            logger.debug("sample: %s", sample)
            logger.debug("pert_emb: %s", pert_emb)

            pert_emb_cpu = pert_emb.detach().cpu()
            if sample is not None:
                sample_cpu = sample.detach().cpu()
            
            # Store samples
            np_mask = np.isin(batch_data["col_genes"][0], _genes)
            truth_np = pert_emb_cpu.numpy()[:, np_mask]
            if sample is not None:
                sample_np = sample_cpu.numpy()[:, np_mask]
            
            if sample is not None:

                r2_metric = r2_score(truth_np.mean(0), sample_np.mean(0))
                torch_mask = torch.as_tensor(np_mask, device=device, dtype=torch.bool)
                sample_eval = sample[:, torch_mask]
                truth_eval = pert_emb[:, torch_mask]
                mmd_metric = mmd_loss(sample_eval, truth_eval).item()

                batch_time = time.time() - start_time
                logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s, r2_metric for this batch: {r2_metric}, mmd_metric for this batch: {mmd_metric}")
            
            all_truths.append(truth_np)
            if sample is not None:
                all_samples.append(sample_np)
                if parent_locked_enabled:
                    if endpoint_projection is None:
                        raise AssertionError("Parent-lock batch has no endpoint projection")
                    local_ids = (
                        endpoint_projection.group_ids[mask].detach().cpu().numpy()
                    )
                    if local_ids.size == 0 or local_ids.min() < 0:
                        raise AssertionError("invalid compact Parent-lock projection IDs")
                    reference = (
                        endpoint_projection.effective_group_parent_mean
                        .detach()
                        .cpu()
                        .numpy()[:, np_mask]
                    )
                    if reference.shape[0] != endpoint_projection.counts.numel():
                        raise AssertionError("Parent-lock reference/group count mismatch")
                    all_projection_group_ids.append(
                        local_ids.astype(np.int64, copy=False) + projection_group_offset
                    )
                    all_projection_references.append(reference.astype(np.float32, copy=False))
                    projection_group_offset += reference.shape[0]

            # covariates
            all_covariates.extend(collect_batch_covariates(batch_data, dataloader, datamodule, mask))

    if parent_locked_enabled:
        parent_locked_gate_summary = _summarize_parent_locked_sampling_gate(
            parent_locked_gate_batches,
            max_singleton_fraction=parent_locked_max_singleton_fraction,
            endpoint_mean_drift_tolerance=(
                parent_locked_endpoint_mean_drift_tolerance
            ),
        )
        logger.info(
            "Parent-lock scientific gate passed: singleton_fraction=%.8f "
            "(%d/%d; maximum=%.8f), endpoint_mean_drift_max_abs=%.8g "
            "(tolerance=%.8g)",
            parent_locked_gate_summary["singleton_fraction"],
            parent_locked_gate_summary["singleton_cells"],
            parent_locked_gate_summary["valid_cells"],
            parent_locked_gate_summary["max_singleton_fraction"],
            parent_locked_gate_summary["endpoint_mean_drift_max_abs"],
            parent_locked_gate_summary["endpoint_mean_drift_tolerance"],
        )

    # Concatenate all samples
    all_truths = np.concatenate(all_truths, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    if parent_locked_enabled:
        all_projection_group_ids = np.concatenate(all_projection_group_ids)
        all_projection_references = np.concatenate(all_projection_references, axis=0)
        if all_projection_group_ids.shape != (all_samples.shape[0],):
            raise AssertionError("Parent-lock saved group IDs do not align to samples")
        if all_projection_references.shape != (projection_group_offset, all_samples.shape[1]):
            raise AssertionError("Parent-lock saved reference shape is invalid")
    if not np.isfinite(all_samples).all():
        raise FloatingPointError("sampling produced non-finite values")
    if not np.any(all_samples):
        logger.warning("Sampling produced an all-zero prediction matrix; downstream metrics are not informative")

    logger.info(f"Generated {all_samples.shape[0]} samples with shape {all_samples.shape}")

    r2_metric = r2_score(all_truths.mean(0), all_samples.mean(0))
    logger.info(f"Overall r2_metric: {r2_metric}")


    all_pert = np.concatenate([x[0] for x in all_covariates], axis=0)
    all_celltype = np.concatenate([x[1] for x in all_covariates], axis=0)
    all_batch = np.concatenate([x[2] for x in all_covariates], axis=0)
        
    ctrl_adata, var_index = load_ctrl_adata(cfg)

    # Getting X_ctrl.
    var_index = _genes

    X_ctrl = build_x_ctrl(ctrl_adata, _genes, cfg)

    scale_saved_expression_units(
        all_samples,
        all_truths,
        normalize_counts,
        parent_projection_references=(
            all_projection_references if parent_locked_enabled else None
        ),
    )

    obs = build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata)
    obs = attach_paired_output_provenance(
        obs, all_treated_provenance, ctrl_adata
    )
    if parent_locked_enabled:
        obs["parent_locked_projection_group"] = np.concatenate(
            [
                all_projection_group_ids,
                np.full(X_ctrl.shape[0], -1, dtype=np.int64),
            ]
        )
    
    if len(np_mask) > 2000: # i.e., when cfg.data.data_name = "Tahoe100mPBMCPretrain"
        logger.info("Restricting to downstream data selected genes only...")
        assert (ctrl_adata.var.index[ctrl_adata.var.highly_variable] == _genes).all()

        # unmerged data setting, no need of using sel_mask
        sel_mask = np.isin(batch_data["col_genes"][0], _genes)
        if all_samples.shape[-1] != len(_genes):
        
            assert sel_mask.sum() == 2000
            all_samples = all_samples[:, sel_mask]
            all_truths = all_truths[:, sel_mask]

        # reorder
        cur_genes = np.array(batch_data["col_genes"][0])[sel_mask].tolist()
        sort_idx = [cur_genes.index(g) for g in _genes]
        assert (np.array(cur_genes)[sort_idx] == _genes).all()
        all_samples = all_samples[:, sort_idx]
        all_truths = all_truths[:, sort_idx]
        if parent_locked_enabled:
            all_projection_references = all_projection_references[:, sort_idx]

        var_index = _genes

    pred_adata = anndata.AnnData(
        X=np.concatenate([all_samples, X_ctrl]), obs=obs.copy()
    )
    true_adata = anndata.AnnData(
        X=np.concatenate([all_truths, X_ctrl]), obs=obs.copy()
    )
    add_provenance_metadata(pred_adata)
    add_provenance_metadata(true_adata)
    assert_paired_provenance_equal(true_adata, pred_adata)
    if parent_locked_gate_summary is not None:
        pred_adata.uns["parent_locked_sampling_gate"] = parent_locked_gate_summary
        true_adata.uns["parent_locked_sampling_gate"] = parent_locked_gate_summary
        pred_adata.uns["parent_locked_feasible_parent_reference"] = {
            "schema_version": 1,
            "projection_group_ids": np.arange(
                all_projection_references.shape[0], dtype=np.int64
            ),
            "effective_parent_mean": all_projection_references,
            "generated_cells": int(all_samples.shape[0]),
            "control_group_id": -1,
        }
    if var_index is not None:
        pred_adata.var.index = var_index
        true_adata.var.index = var_index

    # Use a shared timestamp so that auxiliary files align with the main outputs.
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    pred_path, _ = save_adata(
        pred_adata, true_adata, cfg, logger, timestamp=run_timestamp
    )
    if parent_locked_gate_summary is not None:
        from src.models.parent_locked_residual.saved_audit import (
            audit_saved_parent_locked_h5ad,
        )

        saved_record = audit_saved_parent_locked_h5ad(
            pred_path,
            max_singleton_fraction=parent_locked_max_singleton_fraction,
            max_endpoint_mean_drift=parent_locked_endpoint_mean_drift_tolerance,
        )
        logger.info(
            "Saved Parent-lock H5AD gate passed: min=%.8g drift=%.8g",
            saved_record["saved_h5ad_minimum_expression"],
            saved_record["saved_h5ad_endpoint_mean_drift_max_abs"],
        )

    gc.collect()

    return all_truths, all_samples, all_trajectories, None



def _slice_batch_row(batch_data, row, batch_size):
    selected = {}
    for key, value in batch_data.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            selected[key] = value[row : row + 1]
        elif isinstance(value, list) and len(value) == batch_size:
            selected[key] = value[row : row + 1]
        else:
            selected[key] = value
    return selected


def _repeat_single_row_batch(batch_data, repeats):
    repeated = {}
    for key, value in batch_data.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == 1:
            repeated[key] = value.repeat(repeats, *([1] * (value.ndim - 1)))
        elif isinstance(value, list) and len(value) == 1:
            repeated[key] = value * repeats
        else:
            repeated[key] = value
    return repeated


def generate_fixed_source_samples(model, diffusion, cfg, device, logger, datamodule, dataloader):
    """Generate an equal number of cells from every source for each perturbation."""
    if bool(
        getattr(model, "parent_residual_locked_flow_enabled", False)
        or getattr(
            model,
            "parent_residual_strict_direct_flow_enabled",
            False,
        )
    ):
        raise ValueError(
            "Parent-residual fixed-source sampling is disabled: repeating one "
            "dataloader row duplicates the same source cells and creates false "
            "heterogeneity. Use the main condition-grouped sampling path with "
            "sampling.fixed_cells_per_perturbation=null."
        )
    fixed_cells = int(cfg.sampling.fixed_cells_per_perturbation)
    cells_per_source = int(cfg.sampling.cells_per_source)
    if fixed_cells <= 0 or cells_per_source <= 0:
        raise ValueError("fixed sampling counts must be positive")
    if fixed_cells % cells_per_source:
        raise ValueError("fixed_cells_per_perturbation must divide by cells_per_source")
    repeats = fixed_cells // cells_per_source
    max_perturbations = getattr(cfg.sampling, "max_perturbations", None)
    max_perturbations = int(max_perturbations) if max_perturbations is not None else None
    if int(cfg.data.use_cell_set) != cells_per_source:
        raise ValueError("data.use_cell_set must equal sampling.cells_per_source")

    model = model.to(device)
    model.eval()
    selected_genes = load_selected_genes(cfg)
    input_dim = int(getattr(cfg.model, "input_dim", len(selected_genes)))
    direct_match = bool(getattr(model, "cell_detr_direct_enabled", False))
    sample_fn = sampling_kwargs = None
    if not direct_match:
        sample_fn, sampling_kwargs = resolve_sampling_runner(
            cfg, diffusion, cfg.sampling.use_ddim
        )
    seen_perturbations = set()
    all_truths, all_samples = [], []
    true_covariates, pred_covariates = [], []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = move_data_to_device(batch_data, device)
            use_latent_prior = bool(
                getattr(model, "match_fm_enabled", False)
                or getattr(cfg.model, "latent_control_mixture", False)
                or getattr(model, "hungarian_flow_enabled", False)
                or getattr(model, "cell_detr_enabled", False)
                or getattr(model, "parent_residual_enabled", False)
            )
            strict_parent_direct = bool(
                getattr(model, "parent_residual_strict_direct_flow_enabled", False)
            )
            has_control_bank = (
                "latent_control_emb" in batch_data
                or "cell_detr_control_bank" in batch_data
            )
            if use_latent_prior and (has_control_bank or strict_parent_direct):
                if getattr(model, "parent_residual_enabled", False):
                    routing = model.apply_parent_residual_inference_context(batch_data)
                    label = "Parent-Residual Gene-DiT"
                elif getattr(model, "cell_detr_enabled", False):
                    routing = model.apply_cell_detr_inference_context(batch_data)
                    label = "Cell-DETR"
                elif getattr(model, "hungarian_flow_enabled", False):
                    routing = model.apply_hungarian_flow_inference_context(batch_data)
                    label = "Hungarian-flow"
                else:
                    routing = model.apply_match_fm_inference_prior(batch_data)
                    label = "MATCH-FM"
                prior = routing.get("prior")
                if prior is None:
                    logger.info("%s inference uses no Child prior", label)
                else:
                    logger.info(
                        "%s inference prior entropy=%.4f",
                        label,
                        float((-(prior * prior.clamp_min(1e-8).log()).sum(dim=-1).mean()).cpu()),
                    )
            batch_size = len(batch_data["pert_emb"])
            batch_data["batch_emb"] = model._encode_covariates(batch_data)
            gene_emb = build_gene_embedding_cache(model, batch_data, device)
            covariates = collect_batch_covariates(
                batch_data,
                dataloader,
                datamodule,
                ~batch_data["is_padded_list"].bool(),
            )
            gene_mask = np.isin(batch_data["col_genes"][0], selected_genes)

            for row, (pert, cell_line, batch) in enumerate(covariates):
                valid = ~batch_data["is_padded_list"][row].bool()
                all_truths.append(
                    batch_data["pert_emb"][row, valid]
                    .detach()
                    .cpu()
                    .numpy()[:, gene_mask]
                )
                true_covariates.append((pert, cell_line, batch))
                perturbation = str(pert[0])
                if perturbation in seen_perturbations:
                    continue
                if max_perturbations is not None and len(seen_perturbations) >= max_perturbations:
                    continue
                seen_perturbations.add(perturbation)

                row_data = _slice_batch_row(batch_data, row, batch_size)
                row_data = _repeat_single_row_batch(row_data, repeats)
                row_gene_emb = None
                if gene_emb is not None:
                    row_gene_emb = gene_emb[row : row + 1].repeat(
                        repeats, *([1] * (gene_emb.ndim - 1))
                    )
                if direct_match:
                    row_data["batch_emb"] = model._encode_covariates(row_data)
                    direct = model.apply_cell_detr_direct_prediction(row_data)
                    sample = direct["prediction"]
                    source_values = (
                        direct["child_indices"].detach().cpu().numpy().reshape(-1)
                    )
                else:
                    condition = build_self_condition(cfg, model, row_data, row_gene_emb)
                    if getattr(model, "parent_residual_locked_flow_enabled", False):
                        locked_sample = model.parent_locked_residual_flow.sample(
                            model.model,
                            source_expression=row_data["parent_residual_source"],
                            parent_mean=row_data[
                                "parent_residual_locked_parent_mean"
                            ],
                            self_condition=condition,
                            group_ids=row_data[
                                "parent_residual_locked_group_ids"
                            ],
                            valid_mask=~row_data["is_padded_list"].bool(),
                            guidance_strength=float(
                                getattr(cfg.sampling, "guidance_strength", 0.0)
                            ),
                        )
                        sample = locked_sample.expression
                        if bool(
                            getattr(
                                model,
                                "parent_residual_hurdle_geometry_enabled",
                                False,
                            )
                        ):
                            legacy_projection = project_group_nonnegative_fixed_mean(
                                sample,
                                row_data["parent_residual_locked_parent_mean"],
                                group_ids=row_data["parent_residual_locked_group_ids"],
                                valid_mask=~row_data["is_padded_list"].bool(),
                            )
                            from src.models.sparse_hurdle_de.lightning_integration import (
                                apply_end_to_end_terminal_hurdle,
                            )

                            sample, _ = apply_end_to_end_terminal_hurdle(
                                model,
                                expression=legacy_projection.value,
                                parent_mean=row_data["parent_residual_locked_parent_mean"],
                                group_ids=row_data["parent_residual_locked_group_ids"],
                                valid_mask=~row_data["is_padded_list"].bool(),
                                zero_rate_logits=row_data.get("parent_residual_hurdle_terminal_zero_logits"),
                                cell_keys=row_data.get("cell_key"),
                            )
                        source_values = (
                            row_data["parent_residual_child_indices"]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                    else:
                        condition["source_indices"] = (
                            torch.arange(repeats, device=device)
                            % model.model.num_sources
                        )
                        sample, _ = sample_fn(
                            model.model,
                            (repeats, cells_per_source, input_dim),
                            self_condition=condition,
                            clip_denoised=cfg.sampling.clip_denoised,
                            device=device,
                            progress=cfg.sampling.progress,
                            **sampling_kwargs,
                        )
                        source_values = np.repeat(
                            np.arange(repeats) % model.model.num_sources,
                            cells_per_source,
                        )
                all_samples.append(
                    sample.detach().cpu().numpy()[:, :, gene_mask].reshape(
                        fixed_cells, int(gene_mask.sum())
                    )
                )
                pred_covariates.append(
                    (
                        np.repeat(perturbation, fixed_cells),
                        np.repeat(str(cell_line[0]), fixed_cells),
                        np.repeat(str(batch[0]), fixed_cells),
                        source_values,
                    )
                )
            logger.info(
                "Fixed-source validation batch %d/%d: %d perturbations collected",
                batch_idx + 1,
                len(dataloader),
                len(seen_perturbations),
            )

    all_truths = np.concatenate(all_truths)
    all_samples = np.concatenate(all_samples)
    if not np.isfinite(all_samples).all():
        raise FloatingPointError("sampling produced non-finite values")
    if not np.any(all_samples):
        logger.warning("Sampling produced an all-zero prediction matrix; downstream metrics are not informative")
    if len(all_samples) != len(seen_perturbations) * fixed_cells:
        raise AssertionError("fixed-source sampling did not produce the requested cell count")

    true_pert = np.concatenate([item[0] for item in true_covariates])
    true_line = np.concatenate([item[1] for item in true_covariates])
    true_batch = np.concatenate([item[2] for item in true_covariates])
    truth_keep = np.isin(true_pert, list(seen_perturbations))
    all_truths = all_truths[truth_keep]
    true_pert = true_pert[truth_keep]
    true_line = true_line[truth_keep]
    true_batch = true_batch[truth_keep]
    pred_pert = np.concatenate([item[0] for item in pred_covariates])
    pred_line = np.concatenate([item[1] for item in pred_covariates])
    pred_batch = np.concatenate([item[2] for item in pred_covariates])
    pred_source = np.concatenate([item[3] for item in pred_covariates])

    normalize_counts = getattr(dataloader.dataset.data_args, "normalize_counts", None)
    scale_saved_expression_units(all_samples, all_truths, normalize_counts)
    ctrl_adata, _ = load_ctrl_adata(cfg)
    x_ctrl = build_x_ctrl(ctrl_adata, selected_genes, cfg)
    pred_obs = build_obs_data(cfg, pred_pert, pred_line, pred_batch, ctrl_adata)
    true_obs = build_obs_data(cfg, true_pert, true_line, true_batch, ctrl_adata)
    pred_obs["source_local_index"] = np.concatenate(
        [pred_source, np.full(len(ctrl_adata), -1, dtype=np.int64)]
    )
    true_obs["source_local_index"] = -1
    pred_adata = anndata.AnnData(
        X=np.concatenate([all_samples, x_ctrl]), obs=pred_obs
    )
    true_adata = anndata.AnnData(
        X=np.concatenate([all_truths, x_ctrl]), obs=true_obs
    )
    pred_adata.var.index = selected_genes
    true_adata.var.index = selected_genes
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_adata(pred_adata, true_adata, cfg, logger, timestamp=timestamp)
    logger.info(
        "Fixed-source sampling complete: %d perturbations, %d predictions each",
        len(seen_perturbations),
        fixed_cells,
    )
    return all_truths, all_samples, [], None

"""Batch-level unique control banks for Cell-DETR."""

from __future__ import annotations

import numpy as np
import torch


def _cfg_get(cfg, name, default=None):
    try:
        return getattr(cfg, name)
    except Exception:
        return default


def resolve_control_members_per_token(dataset, response_set_size):
    """Resolve control-token width independently of the response set.

    Old configs keep using ``data.use_cell_set`` (passed as
    ``response_set_size``).  A positive ``control_members_per_token`` opts into
    an independent width without changing the shape of the response tensor.
    """
    configured = _cfg_get(
        dataset.data_args, "control_members_per_token", None
    )
    if configured in (None, "", "null"):
        configured = response_set_size
    configured = int(configured)
    if configured <= 0:
        configured = int(response_set_size)
    if configured <= 0:
        raise ValueError("control_members_per_token must resolve to a positive integer")
    return configured


def control_context_statistics(context):
    """Return validated per-prototype means, standard deviations, and counts.

    Fixed-anchor v1 artifacts contain all three arrays.  Older artifacts did
    not always persist dispersion/support metadata, so they receive neutral,
    shape-safe defaults: zero standard deviation and weight-derived or unit
    support counts.
    """
    means = np.asarray(context["prototype_means"], dtype=np.float32)
    if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError("prototype_means must have shape [K,G] with K,G > 0")
    if not np.isfinite(means).all():
        raise ValueError("prototype_means contain non-finite values")

    raw_stds = context.get("prototype_stds")
    if raw_stds is None:
        stds = np.zeros_like(means)
    else:
        stds = np.asarray(raw_stds, dtype=np.float32)
        if stds.shape != means.shape:
            raise ValueError(
                "prototype_stds must have the same [K,G] shape as prototype_means"
            )
        if not np.isfinite(stds).all() or (stds < 0).any():
            raise ValueError("prototype_stds must be finite and non-negative")

    raw_counts = context.get("cluster_cell_counts")
    if raw_counts is None:
        raw_counts = context.get("token_counts")
    if raw_counts is None:
        weights = np.asarray(
            context.get("weights", np.ones(len(means))), dtype=np.float64
        )
        total = context.get("diagnostics", {}).get("num_cells")
        if (
            weights.shape == (len(means),)
            and np.isfinite(weights).all()
            and (weights >= 0).all()
            and weights.sum() > 0
            and total is not None
            and int(total) > 0
        ):
            counts = np.maximum(
                np.rint(weights / weights.sum() * int(total)), 1
            ).astype(np.float32)
        else:
            counts = np.ones((len(means),), dtype=np.float32)
    else:
        counts = np.asarray(raw_counts, dtype=np.float32)
        if counts.shape != (len(means),):
            raise ValueError("cluster_cell_counts must have shape [K]")
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(
                "cluster_cell_counts must be finite and non-negative"
            )

    return means, stds, counts


def _parent_ids(states, num_states):
    if states is None:
        return np.arange(num_states, dtype=np.int64)
    states = list(states)
    if len(states) != num_states:
        raise ValueError("control context states do not align with prototypes")
    parent_ids = []
    fallback_ids = {}
    for state in states:
        prefix = str(state).split(".", 1)[0]
        try:
            parent_id = int(prefix)
        except ValueError:
            if prefix not in fallback_ids:
                fallback_ids[prefix] = len(fallback_ids)
            parent_id = fallback_ids[prefix]
        parent_ids.append(parent_id)
    return np.asarray(parent_ids, dtype=np.int64)


def _runtime_control_group(ds_name, cell_line, batch_name, grouping):
    dataset_prefix = f"{ds_name}_"
    donor = (
        batch_name[len(dataset_prefix):]
        if batch_name.startswith(dataset_prefix)
        else batch_name
    )
    if grouping == "celltype":
        return cell_line
    if grouping == "donor":
        return donor
    if grouping == "donor_celltype":
        return f"{donor}::{cell_line}"
    raise ValueError(f"unsupported control_bank_grouping {grouping!r}")


def build_unique_cell_line_control_bank(
    dataset,
    ds_name_list,
    cell_type_vars,
    batch_vars,
    set_size,
):
    """Build one fixed KxS bank per configured biological control group."""
    if bool(_cfg_get(dataset.data_args, "latent_control_use_cluster_sets", False)):
        raise ValueError(
            "cell_detr_unique_control_bank requires fixed cluster centroids; "
            "set latent_control_use_cluster_sets=false"
        )

    grouping = str(
        _cfg_get(dataset.data_args, "control_bank_grouping", "celltype")
        or "celltype"
    )
    if grouping not in {"celltype", "donor", "donor_celltype"}:
        raise ValueError("invalid control_bank_grouping")
    celltype_prior_weight = float(
        _cfg_get(dataset.data_args, "control_bank_celltype_prior_weight", 0.0)
        or 0.0
    )
    if not np.isfinite(celltype_prior_weight) or celltype_prior_weight < 0:
        raise ValueError("control_bank_celltype_prior_weight must be non-negative")
    if grouping != "donor" and celltype_prior_weight != 0.0:
        raise ValueError("cell-type soft priors require donor control grouping")

    context_bank = getattr(dataset, "control_distribution_context", None)
    if not context_bank:
        raise ValueError("Cell-DETR unique control-bank mode needs a context bank")
    by_cell_line = context_bank.get("context_by_cell_line", {})
    inverse_cell_type = {
        int(value): str(name)
        for name, value in dataset.meta_cache.cell_type_dict.items()
    }
    inverse_batch = None
    if grouping != "celltype":
        batch_dict = getattr(dataset.meta_cache, "batch_dict", None)
        if batch_dict is None:
            raise ValueError("donor control grouping requires batch_dict metadata")
        inverse_batch = {
            int(value): str(name) for name, value in batch_dict.items()
        }
    cell_type_values = cell_type_vars[:, 0].detach().cpu().tolist()
    batch_values = batch_vars[:, 0].detach().cpu().tolist()
    if not (len(cell_type_values) == len(batch_values) == len(ds_name_list)):
        raise ValueError("control grouping values must align with collated samples")

    group_to_bank = {}
    unique_groups = []
    bank_inverse = []
    fallback_flags = []
    for ds_name, cell_type_value, batch_value in zip(
        ds_name_list, cell_type_values, batch_values
    ):
        cell_line = inverse_cell_type[int(cell_type_value)]
        batch_name = (
            "" if inverse_batch is None else inverse_batch[int(batch_value)]
        )
        control_group = _runtime_control_group(
            str(ds_name), cell_line, batch_name, grouping
        )
        requested_control_group = control_group
        if control_group not in by_cell_line and grouping == "donor_celltype":
            suffix = f"::{cell_line}"
            candidates = [
                name for name in by_cell_line if str(name).endswith(suffix)
            ]
            if candidates:
                control_group = sorted(
                    candidates,
                    key=lambda name: (
                        -int(
                            np.asarray(
                                by_cell_line[name]["cluster_cell_counts"]
                            ).sum()
                        ),
                        str(name),
                    ),
                )[0]
        key = (str(ds_name), control_group)
        if celltype_prior_weight > 0:
            key = (*key, cell_line)
        bank_index = group_to_bank.get(key)
        if bank_index is None:
            if control_group not in by_cell_line:
                raise KeyError(f"missing control context for group {control_group!r}")
            bank_index = len(unique_groups)
            group_to_bank[key] = bank_index
            unique_groups.append((control_group, cell_line))
        bank_inverse.append(bank_index)
        fallback_flags.append(control_group != requested_control_group)

    candidate_strategy = str(
        _cfg_get(dataset.data_args, "latent_control_candidate_strategy", "all")
        or "all"
    )
    if candidate_strategy != "all":
        raise ValueError(
            "cell_detr_unique_control_bank currently requires "
            "latent_control_candidate_strategy=all"
        )
    top_k = int(_cfg_get(dataset.data_args, "latent_control_top_k", 0) or 0)
    min_prior = float(
        _cfg_get(dataset.data_args, "latent_control_min_prior", 1e-8) or 1e-8
    )
    fixed_max_k = bool(
        _cfg_get(dataset.data_args, "latent_control_fixed_max_k", False)
    )

    members_per_token = resolve_control_members_per_token(dataset, set_size)
    prepared = []
    for control_group, cell_line in unique_groups:
        context = by_cell_line[control_group]
        means, stds, counts = control_context_statistics(context)
        base_priors = np.asarray(context["weights"], dtype=np.float32)
        if base_priors.shape != (len(means),):
            raise ValueError("control context weights must have shape [K]")
        if (
            not np.isfinite(base_priors).all()
            or (base_priors < 0).any()
            or base_priors.sum() <= 0
        ):
            raise ValueError("control context weights must be finite and non-negative")
        # Child order must remain the global occupancy order used by the real
        # control reservoir. A soft cell-type prior changes mass, not IDs.
        order = np.argsort(-base_priors)
        if top_k > 0:
            order = order[:top_k]
        priors = base_priors[order].copy()
        if celltype_prior_weight > 0:
            auxiliary_names = np.asarray(
                context.get("auxiliary_label_names", []), dtype=str
            )
            matches = np.flatnonzero(auxiliary_names == cell_line)
            auxiliary_weights = np.asarray(
                context.get("child_auxiliary_weights"), dtype=np.float32
            )
            if len(matches) != 1 or auxiliary_weights.shape != (
                len(means), len(auxiliary_names)
            ):
                raise ValueError("donor bank lacks aligned cell-type compositions")
            compatibility = auxiliary_weights[order, int(matches[0])]
            priors *= np.maximum(compatibility, min_prior) ** celltype_prior_weight
        prepared.append(
            (
                means[order],
                stds[order],
                counts[order],
                priors,
                order.astype(np.int64),
                _parent_ids(context.get("states"), len(means))[order],
            )
        )

    if not prepared:
        raise ValueError("cannot build a unique control bank for an empty batch")
    max_k = max(len(item[3]) for item in prepared)
    if fixed_max_k:
        artifact_capacity = context_bank.get("anchor_capacity")
        if artifact_capacity in (None, "", "null"):
            artifact_capacity = context_bank.get("metadata", {}).get(
                "anchor_capacity"
            )
        if artifact_capacity not in (None, "", "null"):
            artifact_capacity = int(artifact_capacity)
            if artifact_capacity < max_k:
                raise ValueError(
                    "control context anchor_capacity is smaller than selected Children"
                )
            max_k = artifact_capacity
        else:
            # Backward compatibility for legacy pickle/NPZ banks that predate
            # an explicit fixed-capacity contract.
            max_k = max(
                max_k,
                max(len(v["prototype_means"]) for v in by_cell_line.values()),
            )

    banks = []
    bank_stds = []
    bank_counts = []
    bank_priors = []
    bank_masks = []
    cluster_ids = []
    parent_ids = []
    normalize_counts = _cfg_get(dataset.data_args, "normalize_counts", None)
    for means, stds, counts, priors, ids, parents in prepared:
        k_val, gene_dim = means.shape
        bank = np.zeros(
            (max_k, int(members_per_token), gene_dim), dtype=np.float32
        )
        bank_std = np.zeros((max_k, gene_dim), dtype=np.float32)
        bank_count = np.zeros((max_k,), dtype=np.float32)
        prior = np.zeros((max_k,), dtype=np.float32)
        mask = np.zeros((max_k,), dtype=np.float32)
        padded_ids = np.zeros((max_k,), dtype=np.int64)
        padded_parents = np.zeros((max_k,), dtype=np.int64)

        bank[:k_val] = np.repeat(
            means[:, None, :], int(members_per_token), axis=1
        )
        bank_std[:k_val] = stds
        bank_count[:k_val] = counts
        if normalize_counts:
            scale = float(normalize_counts)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("normalize_counts must be finite and positive")
            bank[:k_val] /= scale
            bank_std[:k_val] /= scale
        normalized_prior = np.maximum(priors, min_prior)
        normalized_prior = normalized_prior / normalized_prior.sum()
        prior[:k_val] = normalized_prior
        mask[:k_val] = 1.0
        padded_ids[:k_val] = ids
        padded_parents[:k_val] = parents

        banks.append(torch.from_numpy(bank))
        bank_stds.append(torch.from_numpy(bank_std))
        bank_counts.append(torch.from_numpy(bank_count))
        bank_priors.append(torch.from_numpy(prior))
        bank_masks.append(torch.from_numpy(mask))
        cluster_ids.append(torch.from_numpy(padded_ids))
        parent_ids.append(torch.from_numpy(padded_parents))

    return {
        "cell_detr_control_bank": torch.stack(banks),
        "cell_detr_control_stds": torch.stack(bank_stds),
        "cell_detr_control_counts": torch.stack(bank_counts),
        "cell_detr_control_prior": torch.stack(bank_priors),
        "cell_detr_control_mask": torch.stack(bank_masks),
        "cell_detr_control_cluster_id": torch.stack(cluster_ids),
        "cell_detr_control_parent_id": torch.stack(parent_ids),
        "cell_detr_control_group_fallback": torch.as_tensor(
            fallback_flags, dtype=torch.bool
        ),
        "cell_detr_bank_inverse": torch.as_tensor(bank_inverse, dtype=torch.long),
    }

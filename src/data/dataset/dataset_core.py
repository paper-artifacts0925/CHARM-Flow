"""Module `data/dataset_core.py`."""
import ctypes
import ctypes.util
import pickle
import zlib
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.dataset.cell_detr_control import (
    build_unique_cell_line_control_bank,
    control_context_statistics,
    resolve_control_members_per_token,
)
from src.data.dataset.dataset_grouping import (
    get_selected_gene_vars,
    register_mapping_indices,
    split_out_control,
)
from src.data.dataset.dataset_io import retrieve_counts
from src.data.file_handle import H5Store
from src.data.metadata_cache import GlobalH5MetadataCache
from src.models.hungarian_flow.context_artifact import load_control_context_bank

_libc_name = ctypes.util.find_library("c")
libc = ctypes.CDLL(_libc_name) if _libc_name else None


def compute_index(dataset, ds_name: str, idx: int):
    """Execute `compute_index` and return values used by downstream logic."""
    assert ds_name in dataset.data_indices and len(dataset.data_indices[ds_name]) > 0
    try:
        local = int(dataset.data_indices[ds_name][idx])
    except Exception:
        local = int(dataset.data_indices[ds_name]["perturb"][idx])
    return ds_name, int(local)


def _pert_mix_shuffle(counts, pert_vars, cell_vars, batch_vars, ds_names, col_genes, is_padded, use_cell_set):
    """Mix perturbation cells between samples in a batch.

    Randomly swaps half the perturbation cells between pairs of samples
    that share the same cell line but have different perturbations.
    This creates a contrastive signal for source routing.
    """
    B = len(counts)
    if B < 2:
        return counts, pert_vars, cell_vars, batch_vars, ds_names, col_genes, is_padded

    perm = np.random.permutation(B)
    for i in range(B):
        j = perm[i]
        if i == j:
            continue
        # Only mix same cell line, different perturbation
        if not torch.equal(cell_vars[i], cell_vars[j]):
            continue
        if torch.equal(pert_vars[i], pert_vars[j]):
            continue

        half = use_cell_set // 2
        counts_i = counts[i].clone()
        counts_j = counts[j].clone()
        counts[i] = torch.cat([counts_i[:half], counts_j[half:]])
        counts[j] = torch.cat([counts_j[:half], counts_i[half:]])

    return counts, pert_vars, cell_vars, batch_vars, ds_names, col_genes, is_padded


def collate_fn(dataset, batch):
    """Execute `collate_fn` and return values used by downstream logic."""
    bsz = len(batch)
    _ = bsz

    pca_mode = isinstance(dataset.data_args.embed_key, str) and dataset.data_args.embed_key.startswith("X_pca")
    _ = pca_mode

    assert dataset.data_args.pad_length == dataset.data_args.embed_shape, "Not supporting padding truncation anymore"
    assert len(batch[0][0]) == 1

    batch_counts, batch_mapped_counts = [], []
    pert_var_list, mapped_pert_var_list = [], []
    cell_type_var_list, batch_var_list = [], []
    is_padded_list, episode_token_list = [], []
    episode_local_id_list, episode_view_id_list = [], []
    cell_index_list, dataset_id_list = [], []
    ds_name_list = []
    col_gene_list = []

    for item in batch:
        # Historical dataset items had nine fields (no episode token) or ten
        # fields (fixed-episode token). Twelve-field ragged items from early
        # checkpoints are also accepted; missing cell identities stay -1.
        if len(item) == 9:
            item = (*item, -1, -1, -1, -1, -1)
        elif len(item) == 10:
            item = (*item, -1, -1, -1, -1)
        elif len(item) == 12:
            item = (*item, -1, -1)
        elif len(item) != 14:
            raise ValueError(
                f"collate item must have 9, 10, 12, or 14 fields, got {len(item)}"
            )
        (
            counts,
            mapped_counts,
            gene_vars,
            pert_var,
            mapped_pert_var,
            cell_type_var,
            batch_var,
            is_padded,
            tmp_ds_name,
            episode_token,
            episode_local_id,
            episode_view_id,
            cell_index,
            dataset_id,
        ) = item
        ds_name_list.append(tmp_ds_name)
        col_gene_list.append(gene_vars)
        episode_token_list.append(int(episode_token))
        episode_local_id_list.append(int(episode_local_id))
        episode_view_id_list.append(int(episode_view_id))
        cell_index_list.append(int(cell_index))
        dataset_id_list.append(int(dataset_id))

        pert_var_list.append(pert_var)
        mapped_pert_var_list.append(mapped_pert_var)
        cell_type_var_list.append(cell_type_var)
        batch_var_list.append(batch_var)
        is_padded_list.append(is_padded)

        batch_counts.append(counts[0])
        batch_mapped_counts.append(mapped_counts[0])

    batch_counts = torch.stack(batch_counts)
    batch_mapped_counts = torch.stack(batch_mapped_counts)

    if dataset.data_args.normalize_counts is not None and dataset.data_args.normalize_counts:
        batch_counts /= dataset.data_args.normalize_counts
        batch_mapped_counts /= dataset.data_args.normalize_counts

    pert_var_list = torch.as_tensor(pert_var_list, dtype=torch.long)
    mapped_pert_var_list = torch.as_tensor(mapped_pert_var_list, dtype=torch.long)
    cell_type_var_list = torch.as_tensor(cell_type_var_list, dtype=torch.long)
    batch_var_list = torch.as_tensor(batch_var_list, dtype=torch.long)
    is_padded_list = torch.as_tensor(is_padded_list, dtype=torch.long)
    episode_token_list = torch.as_tensor(episode_token_list, dtype=torch.long)
    episode_local_id_list = torch.as_tensor(episode_local_id_list, dtype=torch.long)
    episode_view_id_list = torch.as_tensor(episode_view_id_list, dtype=torch.long)
    cell_index_list = torch.as_tensor(cell_index_list, dtype=torch.long)
    dataset_id_list = torch.as_tensor(dataset_id_list, dtype=torch.long)

    if dataset.data_args.use_cell_set is not None:
        s_val = dataset.data_args.use_cell_set

        batch_counts = batch_counts.reshape(-1, s_val, batch_counts.shape[-1])
        batch_mapped_counts = batch_mapped_counts.reshape(-1, s_val, batch_mapped_counts.shape[-1])
        pert_var_list = pert_var_list.reshape(-1, s_val)
        mapped_pert_var_list = mapped_pert_var_list.reshape(-1, s_val)
        cell_type_var_list = cell_type_var_list.reshape(-1, s_val)
        batch_var_list = batch_var_list.reshape(-1, s_val)
        is_padded_list = is_padded_list.reshape(-1, s_val)
        episode_token_list = episode_token_list.reshape(-1, s_val)
        episode_local_id_list = episode_local_id_list.reshape(-1, s_val)
        episode_view_id_list = episode_view_id_list.reshape(-1, s_val)
        cell_index_list = cell_index_list.reshape(-1, s_val)
        dataset_id_list = dataset_id_list.reshape(-1, s_val)

        if dataset.data_args.use_cell_set > 1:
            assert ds_name_list[:: dataset.data_args.use_cell_set] == ds_name_list[1:: dataset.data_args.use_cell_set]
            assert col_gene_list[:: dataset.data_args.use_cell_set] == col_gene_list[1:: dataset.data_args.use_cell_set]

        ds_name_list = ds_name_list[:: dataset.data_args.use_cell_set]
        col_gene_list = col_gene_list[:: dataset.data_args.use_cell_set]

    valid_cell_identity = (cell_index_list >= 0) & (dataset_id_list >= 0)
    if (
        (cell_index_list[valid_cell_identity] >= 2 ** 48).any()
        or (dataset_id_list[valid_cell_identity] >= 2 ** 15).any()
    ):
        raise ValueError("cell_index/dataset_id exceed the stable cell-key encoding")
    cell_key = torch.full_like(cell_index_list, -1)
    cell_key[valid_cell_identity] = (
        dataset_id_list[valid_cell_identity] * (2 ** 48)
        + cell_index_list[valid_cell_identity]
    )

    extra_fields = {
        "valid_cell_mask": ~is_padded_list.bool(),
        "meta_episode_token": episode_token_list,
        "cell_index": cell_index_list,
        "dataset_id": dataset_id_list,
        "cell_key": cell_key,
    }
    if dataset.data_args.use_cell_set is not None:
        extra_fields.update(
            latent_control_candidates(
                dataset,
                ds_name_list,
                pert_var_list,
                cell_type_var_list,
                batch_var_list,
                dataset.data_args.use_cell_set,
                episode_ids=episode_token_list,
            )
        )
        extra_fields.update(
            source_transport_batch_fields(dataset, cell_type_var_list)
        )

        episode_enabled = bool(
            _cfg_get(dataset.data_args, "meta_batch_episode_enabled", False)
        ) and str(getattr(dataset, "stage", "")).lower() == "train"
        if episode_enabled:
            num_sets = int(batch_counts.shape[0])
            parent_grouping = str(
                _cfg_get(
                    dataset.data_args, "meta_batch_parent_grouping", "celltype"
                )
            ).strip().lower()
            if parent_grouping not in {"celltype", "donor", "donor_celltype"}:
                raise ValueError("invalid meta_batch_parent_grouping")
            if not extra_fields["valid_cell_mask"].all():
                raise ValueError(
                    "meta-batch sampler must emit full sets without replacement padding"
                )
            dynamic_views = bool(
                _cfg_get(dataset.data_args, "meta_batch_dynamic_views", False)
            )
            if dynamic_views:
                if not (
                    episode_local_id_list
                    == episode_local_id_list[:, :1]
                ).all():
                    raise ValueError("one cell set contains multiple episode IDs")
                if not (
                    episode_view_id_list
                    == episode_view_id_list[:, :1]
                ).all():
                    raise ValueError("one cell set contains multiple view IDs")
                episode_ids = episode_local_id_list[:, 0]
                view_ids = episode_view_id_list[:, 0]
                if (episode_ids < 0).any() or (view_ids < 0).any():
                    raise ValueError(
                        "dynamic meta-batch sampler must provide episode/view IDs"
                    )
                unique_ids, view_counts = torch.unique_consecutive(
                    episode_ids, return_counts=True
                )
                expected_ids = torch.arange(
                    len(unique_ids), dtype=torch.long
                )
                if not torch.equal(unique_ids, expected_ids):
                    raise ValueError(
                        "dynamic episodes must be contiguous and batch-local IDs "
                        "must start at zero"
                    )
                episode_ptr = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long),
                        view_counts.cumsum(dim=0),
                    ]
                )
                min_views = int(
                    _cfg_get(dataset.data_args, "meta_batch_min_views", 2)
                )
                max_views = int(
                    _cfg_get(dataset.data_args, "meta_batch_max_views", 5)
                )
                if (view_counts < min_views).any() or (
                    view_counts > max_views
                ).any():
                    raise ValueError(
                        f"dynamic episode view counts must be in "
                        f"[{min_views}, {max_views}]"
                    )
                for episode_id in range(len(unique_ids)):
                    start = int(episode_ptr[episode_id])
                    end = int(episode_ptr[episode_id + 1])
                    expected_views = torch.arange(
                        end - start, dtype=torch.long
                    )
                    if not torch.equal(view_ids[start:end], expected_views):
                        raise ValueError(
                            "view IDs must be contiguous and start at zero "
                            "within every episode"
                        )
                    if not (
                        pert_var_list[start:end]
                        == pert_var_list[start, 0]
                    ).all():
                        raise ValueError(
                            "all views in an episode must share one perturbation"
                        )
                    if parent_grouping == "donor_celltype":
                        if not (
                            batch_var_list[start:end]
                            == batch_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all views in a donor/cell-type episode must "
                                "share one donor"
                            )
                        if not (
                            cell_type_var_list[start:end]
                            == cell_type_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all views in a donor/cell-type episode must "
                                "share one cell type"
                            )
                    elif parent_grouping == "donor":
                        if not (
                            batch_var_list[start:end]
                            == batch_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all views in a donor-parent episode must share "
                                "one donor"
                            )
                        if not (
                            cell_type_var_list[start:end]
                            == cell_type_var_list[start:end, :1]
                        ).all():
                            raise ValueError(
                                "each view in a donor-parent episode must contain "
                                "one cell type"
                            )
                    elif not (
                        cell_type_var_list[start:end]
                        == cell_type_var_list[start, 0]
                    ).all():
                        raise ValueError(
                            "all views in an episode must share one cell line"
                        )
                    if not (
                        episode_token_list[start:end]
                        == episode_token_list[start, 0]
                    ).all():
                        raise ValueError(
                            "all views in an episode must share one episode token"
                        )
                    if not bool(
                        _cfg_get(dataset.data_args, "hungarian_flow_combination_sampler", False)
                    ) and not (
                        batch_var_list[start:end]
                        == batch_var_list[start:end, :1]
                    ).all():
                        raise ValueError(
                            "a dynamic view must come from one raw batch"
                        )
            else:
                num_views = int(
                    _cfg_get(dataset.data_args, "meta_batch_num_views", 3)
                )
                if num_sets % num_views != 0:
                    raise ValueError(
                        f"meta-batch has {num_sets} sets, not divisible by "
                        f"{num_views} views"
                    )
                num_episodes = num_sets // num_views
                episode_ids = torch.arange(
                    num_episodes, dtype=torch.long
                ).repeat_interleave(num_views)
                view_ids = torch.arange(
                    num_views, dtype=torch.long
                ).repeat(num_episodes)
                view_counts = torch.full(
                    (num_episodes,), num_views, dtype=torch.long
                )
                episode_ptr = torch.arange(
                    0, num_sets + 1, num_views, dtype=torch.long
                )
                for episode_id in range(num_episodes):
                    start = int(episode_ptr[episode_id])
                    end = int(episode_ptr[episode_id + 1])
                    if not (
                        pert_var_list[start:end]
                        == pert_var_list[start, 0]
                    ).all():
                        raise ValueError(
                            "all fixed views must share one perturbation"
                        )
                    if parent_grouping == "donor_celltype":
                        if not (
                            batch_var_list[start:end]
                            == batch_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all fixed donor/cell-type views must share one donor"
                            )
                        if not (
                            cell_type_var_list[start:end]
                            == cell_type_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all fixed donor/cell-type views must share one cell type"
                            )
                    elif parent_grouping == "donor":
                        if not (
                            batch_var_list[start:end]
                            == batch_var_list[start, 0]
                        ).all():
                            raise ValueError(
                                "all fixed views in a donor-parent episode must "
                                "share one donor"
                            )
                        if not (
                            cell_type_var_list[start:end]
                            == cell_type_var_list[start:end, :1]
                        ).all():
                            raise ValueError(
                                "each fixed view in a donor-parent episode must "
                                "contain one cell type"
                            )
                    elif not (
                        cell_type_var_list[start:end]
                        == cell_type_var_list[start, 0]
                    ).all():
                        raise ValueError(
                            "all fixed views must share one cell line"
                        )

            extra_fields["meta_episode_id"] = episode_ids
            extra_fields["meta_view_id"] = view_ids
            extra_fields["meta_episode_ptr"] = episode_ptr
            extra_fields["meta_episode_num_views"] = view_counts[episode_ids]


    # ---- Perturbation batch mixing (new) ----
    pert_mix_enabled = bool(_cfg_get(dataset.data_args, "pert_mix_enabled", False))
    pert_mix_prob = float(_cfg_get(dataset.data_args, "pert_mix_prob", 0.3))
    s_val = int(dataset.data_args.use_cell_set) if dataset.data_args.use_cell_set is not None else 1
    if pert_mix_enabled and np.random.random() < pert_mix_prob:
        batch_counts, pert_var_list, cell_type_var_list, batch_var_list, ds_name_list, col_gene_list, is_padded_list = _pert_mix_shuffle(
            batch_counts, pert_var_list, cell_type_var_list, batch_var_list,
            ds_name_list, col_gene_list, is_padded_list, s_val,
        )

    return {
        **extra_fields,
        "pert_emb": batch_counts,
        "col_genes": col_gene_list,
        "cont_emb": batch_mapped_counts,
        "cov_pert": pert_var_list,
        "cov_mapped_pert": mapped_pert_var_list,
        "cov_celltype": cell_type_var_list,
        "cov_batch": batch_var_list,
        "is_padded_list": is_padded_list,
        "ds_name": ds_name_list,
    }


def _cfg_get(cfg, name, default=None):
    """Read optional OmegaConf fields without requiring them in every config."""
    try:
        return getattr(cfg, name)
    except Exception:
        return default



def load_source_transport_bank(dataset):
    """Load G3 metadata and enforce source-bank/dataset gene alignment."""
    dataset.source_transport_bank = None
    dataset.source_transport_cell_line_to_id = None
    bank_path = _cfg_get(dataset.data_args, "source_transport_bank_path", None)
    if bank_path in (None, "", "null"):
        return
    bank = torch.load(str(bank_path), map_location="cpu", weights_only=True)
    metadata = bank.get("metadata", {})
    required_scope = str(
        _cfg_get(dataset.data_args, "source_transport_scope", "within_cell_line")
    )
    if metadata.get("scope") != required_scope:
        raise ValueError(
            f"source bank scope {metadata.get('scope')!r} does not match "
            f"data.source_transport_scope={required_scope!r}"
        )
    if required_scope != "within_cell_line":
        raise ValueError("dataset source lookup is reserved for within_cell_line banks")
    cell_lines = bank.get("cell_lines", {})
    if not cell_lines:
        raise ValueError("within-cell-line source bank has no cell_lines")
    bank_genes = metadata.get("gene_names")
    dataset_genes = dataset.selected_genes
    if isinstance(dataset_genes, dict):
        unique_orders = {tuple(value) for value in dataset_genes.values()}
        if len(unique_orders) != 1:
            raise ValueError("G3 requires one shared gene order across datasets")
        dataset_genes = list(next(iter(unique_orders)))
    if bank_genes is None or list(bank_genes) != list(dataset_genes):
        raise ValueError("source_bank_gene_names != dataset_gene_names")
    dataset.source_transport_bank = bank
    dataset.source_transport_cell_line_to_id = {
        name: index for index, name in enumerate(cell_lines)
    }
    dataset.py_logger.info("Loaded source bank: %s", Path(str(bank_path)).name)


def source_transport_batch_fields(dataset, cell_type_vars):
    """Attach local source-bank cell-line IDs and masks to one collated batch."""
    bank = getattr(dataset, "source_transport_bank", None)
    if bank is None:
        return {}
    if cell_type_vars.ndim != 2:
        raise ValueError("cov_celltype must have shape [B,S]")
    if not torch.all(cell_type_vars == cell_type_vars[:, :1]):
        raise ValueError("a target cell set contains multiple cell lines")
    inverse_cell_type = {
        int(value): str(name)
        for name, value in dataset.meta_cache.cell_type_dict.items()
    }
    names = [inverse_cell_type[int(value)] for value in cell_type_vars[:, 0]]
    mapping = dataset.source_transport_cell_line_to_id
    missing = sorted(set(names) - set(mapping))
    if missing:
        raise ValueError(f"target cell lines missing from source bank: {missing}")
    local_ids = torch.as_tensor([mapping[name] for name in names], dtype=torch.long)
    masks = torch.stack(
        [bank["cell_lines"][name]["valid_source_mask"].bool() for name in names]
    )
    if not masks.any(dim=-1).all():
        raise ValueError("source_mask has a sample with no valid source")
    source_names = [tuple(bank["cell_lines"])[index] for index in local_ids.tolist()]
    if source_names != names:
        raise ValueError("source_cell_line != target_cell_line")
    return {
        "source_cell_line_id": local_ids,
        "source_cell_line_name": names,
        "source_mask": masks,
    }

def load_control_bank(dataset):
    """Load an optional precomputed control-state bank for clustered control sampling."""
    dataset.control_bank = None
    dataset.control_bank_global_indices = None
    dataset.control_bank_sampler_type = _cfg_get(dataset.data_args, "control_bank_sampler_type", None)
    dataset.control_bank_stats = {"bank_hits": 0, "original_fallbacks": 0}
    seed = int(_cfg_get(dataset.data_args, "control_bank_seed", 0) or 0)
    dataset.control_bank_rng = np.random.default_rng(seed)

    bank_path = _cfg_get(dataset.data_args, "control_bank_path", None)
    if bank_path in (None, "", "null"):
        return

    with open(str(bank_path), "rb") as f:
        dataset.control_bank = pickle.load(f)

    if dataset.control_bank_sampler_type in (None, "", "auto"):
        dataset.control_bank_sampler_type = dataset.control_bank.get("metadata", {}).get("sampler_type")
    if dataset.control_bank_sampler_type == "global_random_mapping":
        arrays = [
            np.asarray(arr, dtype=np.int64)
            for arr in dataset.control_bank.get("by_cell_line_cluster", {}).values()
            if arr is not None and len(arr) > 0
        ]
        if arrays:
            dataset.control_bank_global_indices = np.unique(np.concatenate(arrays))
    dataset.py_logger.info(
        "Loaded control bank %s with sampler=%s",
        bank_path,
        dataset.control_bank_sampler_type,
    )


def load_control_distribution_context(dataset):
    """Load deterministic control-distribution contexts for M3/M4 experiments."""
    dataset.control_distribution_context = None
    dataset.control_context_mode = _cfg_get(dataset.data_args, "control_context_mode", "original")
    dataset.control_distribution_context_stats = {"context_hits": 0, "fallbacks": 0}

    context_path = _cfg_get(dataset.data_args, "control_distribution_context_path", None)
    if context_path in (None, "", "null"):
        return

    expected_sha256 = _cfg_get(
        dataset.data_args, "control_distribution_context_sha256", None
    )
    dataset.control_distribution_context = load_control_context_bank(
        context_path,
        expected_sha256=expected_sha256,
    )

    dataset.py_logger.info(
        "Loaded control distribution context %s with mode=%s",
        context_path,
        dataset.control_context_mode,
    )


def distribution_context_counts(dataset, ds_name, local_idx):
    """Return a deterministic control-distribution expression vector, if configured."""
    context_bank = getattr(dataset, "control_distribution_context", None)
    mode = getattr(dataset, "control_context_mode", "original")
    if not context_bank or mode in (None, "", "original", "latent_cluster_mixture"):
        return None

    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
    cell_line_key = int(cache.cell_type_codes[local_idx])
    cell_line = str(cache.cell_type_categories[cell_line_key])
    by_cell_line = context_bank.get("context_by_cell_line", {})
    context = by_cell_line.get(cell_line)
    if context is None:
        dataset.control_distribution_context_stats["fallbacks"] += 1
        return None

    if mode == "prototype_weighted_mean":
        values = context["weighted_mean"]
    elif mode == "prototype_token_set":
        tokens = context["prototype_tokens"]
        token_idx = int(local_idx) % int(tokens.shape[0])
        values = tokens[token_idx]
    elif mode == "prototype_prior_sample":
        means = np.asarray(context["prototype_means"], dtype=np.float32)
        priors = np.asarray(context["weights"], dtype=np.float64)
        priors = priors / priors.sum()
        token_idx = int(dataset.control_bank_rng.choice(len(priors), p=priors))
        values = means[token_idx]
    elif mode == "prototype_prior_hash":
        means = np.asarray(context["prototype_means"], dtype=np.float32)
        priors = np.asarray(context["weights"], dtype=np.float64)
        priors = priors / priors.sum()
        cdf = np.cumsum(priors)
        u_val = ((int(local_idx) * 1103515245 + 12345) % 1000003) / 1000003.0
        token_idx = int(np.searchsorted(cdf, u_val, side="right"))
        token_idx = min(token_idx, len(priors) - 1)
        values = means[token_idx]
    else:
        raise ValueError(f"Unknown control_context_mode={mode}")

    dataset.control_distribution_context_stats["context_hits"] += 1
    return torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)



def _lookup_local_category_code(categories, name):
    matches = np.where(categories == name)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _latent_perturb_group_size(dataset, ds_name, pert_var, cell_type_var, batch_var):
    try:
        inverse_pert = {int(v): str(k) for k, v in dataset.meta_cache.pert_dict.items()}
        inverse_cell_type = {int(v): str(k) for k, v in dataset.meta_cache.cell_type_dict.items()}
        inverse_batch = {int(v): str(k) for k, v in dataset.meta_cache.batch_dict.items()}
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

        pert_name = inverse_pert[int(pert_var)]
        cell_line = inverse_cell_type[int(cell_type_var)]
        batch_name = inverse_batch[int(batch_var)]
        prefix = f"{ds_name}_"
        if batch_name.startswith(prefix):
            batch_name = batch_name[len(prefix):]

        pert_code = _lookup_local_category_code(cache.pert_categories, pert_name)
        cell_code = _lookup_local_category_code(cache.cell_type_categories, cell_line)
        batch_code = _lookup_local_category_code(cache.batch_categories, batch_name)
        if pert_code is None or cell_code is None or batch_code is None:
            return 0
        return int(dataset.grouped_pert_num_cell.get(ds_name, {}).get((pert_code, cell_code, batch_code), 0))
    except Exception:
        return 0


def _latent_parent_ids(states, num_states):
    """Map hierarchical leaf labels such as 0.0/0.1 to stable parent IDs."""
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


def latent_control_candidates(
    dataset,
    ds_name_list,
    pert_vars,
    cell_type_vars,
    batch_vars,
    s_val,
    episode_ids=None,
):
    """Build per-sample latent control-cluster candidates from the distribution bank."""
    context_bank = getattr(dataset, "control_distribution_context", None)
    mode = getattr(dataset, "control_context_mode", "original")
    if not context_bank or mode != "latent_cluster_mixture":
        return {}

    if bool(_cfg_get(dataset.data_args, "cell_detr_unique_control_bank", False)):
        return build_unique_cell_line_control_bank(
            dataset=dataset,
            ds_name_list=ds_name_list,
            cell_type_vars=cell_type_vars,
            batch_vars=batch_vars,
            set_size=s_val,
        )

    members_per_token = resolve_control_members_per_token(dataset, s_val)
    top_k = int(_cfg_get(dataset.data_args, "latent_control_top_k", 2) or 0)
    min_prior = float(_cfg_get(dataset.data_args, "latent_control_min_prior", 1e-8) or 1e-8)
    candidate_strategy = str(_cfg_get(dataset.data_args, "latent_control_candidate_strategy", "all") or "all")
    num_candidate_samples = int(_cfg_get(dataset.data_args, "latent_control_num_candidate_samples", 0) or 0)
    include_top_k = int(_cfg_get(dataset.data_args, "latent_control_include_top_k", 0) or 0)
    use_cluster_sets = bool(_cfg_get(dataset.data_args, "latent_control_use_cluster_sets", False))
    inverse_cell_type = {int(v): str(k) for k, v in dataset.meta_cache.cell_type_dict.items()}
    by_cell_line = context_bank.get("context_by_cell_line", {})

    candidate_embs, candidate_stds, candidate_counts = [], [], []
    candidate_priors, candidate_masks = [], []
    candidate_cluster_ids, candidate_parent_ids = [], []
    perturb_group_sizes = []
    max_k = 0
    prepared = []
    cell_type_values = cell_type_vars[:, 0].detach().cpu().numpy().tolist()
    pert_values = pert_vars[:, 0].detach().cpu().numpy().tolist()
    batch_values = batch_vars[:, 0].detach().cpu().numpy().tolist()
    episode_sampling = bool(_cfg_get(dataset.data_args, "meta_batch_episode_enabled", False)) and str(getattr(dataset, "stage", "")).lower() == "train"
    base_seed = int(_cfg_get(dataset.data_args, "control_bank_seed", 0) or 0)
    for sample_index, (ds_name, pert_var, cell_type_var, batch_var) in enumerate(zip(ds_name_list, pert_values, cell_type_values, batch_values)):
        cell_line = inverse_cell_type[int(cell_type_var)]
        context = by_cell_line[cell_line]
        means, stds, counts = control_context_statistics(context)
        all_parent_ids = _latent_parent_ids(
            context.get("states"), len(means)
        )
        priors = np.asarray(context["weights"], dtype=np.float32)
        full_order = np.argsort(-priors)
        if candidate_strategy == "prior_sample" and num_candidate_samples > 0:
            sample_probs = np.maximum(priors, min_prior)
            sample_probs = sample_probs / sample_probs.sum()
            sampled = dataset.control_bank_rng.choice(
                np.arange(len(priors), dtype=np.int64),
                size=num_candidate_samples,
                replace=True,
                p=sample_probs,
            )
            if include_top_k > 0:
                order = np.concatenate([full_order[:include_top_k], sampled]).astype(np.int64)
            else:
                order = sampled.astype(np.int64)
        elif candidate_strategy == "all":
            order = full_order
            if top_k > 0:
                order = order[:top_k]
        else:
            raise ValueError(f"Unknown latent_control_candidate_strategy={candidate_strategy}")
        means = means[order]
        cluster_sets = None
        if use_cluster_sets:
            episode_token = (
                -1
                if episode_ids is None
                else int(episode_ids[sample_index, 0])
            )
            if (
                episode_sampling
                and episode_ids is not None
                and episode_token >= 0
            ):
                if not episode_ids[sample_index].eq(episode_token).all():
                    raise ValueError(
                        "all cells in a view must share one episode token"
                    )
                set_tokens = episode_ids[:, 0]
                member_indices = torch.nonzero(
                    set_tokens.eq(episode_token), as_tuple=False
                ).flatten()
                episode_tokens = episode_ids[member_indices]
                if not episode_tokens.eq(episode_token).all():
                    raise ValueError(
                        "all views in an episode must share one episode token"
                    )
                episode_batches = tuple(
                    int(batch_values[index])
                    for index in member_indices.tolist()
                )
            else:
                episode_batches = (int(batch_var),)
            seed_key = (ds_name, cell_line, int(pert_var), episode_token, episode_batches, tuple(order.tolist()), int(members_per_token))
            local_seed = (base_seed + zlib.crc32(repr(seed_key).encode("utf-8"))) % (2 ** 32)
            cluster_rng = np.random.default_rng(local_seed)
            cluster_cell_tokens = context.get("cluster_cell_tokens")
            selected_sets = []
            if cluster_cell_tokens is not None:
                pools = np.asarray(cluster_cell_tokens, dtype=np.float32)
                if pools.ndim != 3 or pools.shape[0] != len(context["prototype_means"]):
                    raise ValueError(
                        f"Invalid cluster cell-token pools for cell line {cell_line}"
                    )
                for cluster_index in order.tolist():
                    pool = pools[cluster_index]
                    if len(pool) == 0:
                        pool = np.asarray(context["prototype_means"][cluster_index])[None, :]
                    selected = cluster_rng.choice(
                        len(pool),
                        size=int(members_per_token),
                        replace=len(pool) < int(members_per_token),
                    )
                    selected_sets.append(pool[selected])
            else:
                prototype_tokens = np.asarray(context.get("prototype_tokens"), dtype=np.float32)
                token_counts = np.asarray(context.get("token_counts"), dtype=np.int64)
                if token_counts.ndim != 1 or token_counts.sum() != len(prototype_tokens):
                    raise ValueError(
                        f"Invalid prototype token partition for cell line {cell_line}"
                    )
                offsets = np.concatenate([[0], np.cumsum(token_counts)])
                for cluster_index, centroid in zip(order.tolist(), means):
                    tokens = prototype_tokens[
                        offsets[cluster_index]:offsets[cluster_index + 1]
                    ]
                    if len(tokens) == 0:
                        tokens = centroid[None, :]
                    repeats = int(
                        np.ceil(float(members_per_token) / float(len(tokens)))
                    )
                    selected_sets.append(
                        np.tile(tokens, (repeats, 1))[:members_per_token]
                    )
            cluster_sets = np.stack(selected_sets).astype(np.float32, copy=False)
        priors = priors[order]
        priors = np.maximum(priors, min_prior)
        priors = priors / priors.sum()
        prepared.append(
            (
                means,
                cluster_sets,
                stds[order],
                counts[order],
                priors,
                order.astype(np.int64),
                all_parent_ids[order].astype(np.int64),
            )
        )
        max_k = max(max_k, int(len(priors)))
        perturb_group_sizes.append(_latent_perturb_group_size(dataset, ds_name, pert_var, cell_type_var, batch_var))

    if bool(
        _cfg_get(dataset.data_args, "latent_control_fixed_max_k", False)
    ):
        max_k = max(
            max_k,
            max(
                len(context["prototype_means"])
                for context in by_cell_line.values()
            ),
        )

    for (
        means,
        cluster_sets,
        stds,
        counts,
        priors,
        cluster_ids,
        parent_ids,
    ) in prepared:
        k_val = int(len(priors))
        emb = np.zeros(
            (max_k, members_per_token, means.shape[-1]), dtype=np.float32
        )
        std = np.zeros((max_k, means.shape[-1]), dtype=np.float32)
        count = np.zeros((max_k,), dtype=np.float32)
        prior = np.zeros((max_k,), dtype=np.float32)
        mask = np.zeros((max_k,), dtype=np.float32)
        ids = np.zeros((max_k,), dtype=np.int64)
        parents = np.zeros((max_k,), dtype=np.int64)
        if cluster_sets is None:
            emb[:k_val] = np.repeat(
                means[:, None, :], members_per_token, axis=1
            )
        else:
            emb[:k_val] = cluster_sets
        std[:k_val] = stds
        count[:k_val] = counts
        normalize_counts = _cfg_get(dataset.data_args, "normalize_counts", None)
        if normalize_counts:
            scale = float(normalize_counts)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("normalize_counts must be finite and positive")
            emb[:k_val] /= scale
            std[:k_val] /= scale
        prior[:k_val] = priors
        mask[:k_val] = 1.0
        ids[:k_val] = cluster_ids
        parents[:k_val] = parent_ids
        candidate_embs.append(torch.from_numpy(emb))
        candidate_stds.append(torch.from_numpy(std))
        candidate_counts.append(torch.from_numpy(count))
        candidate_priors.append(torch.from_numpy(prior))
        candidate_masks.append(torch.from_numpy(mask))
        candidate_cluster_ids.append(torch.from_numpy(ids))
        candidate_parent_ids.append(torch.from_numpy(parents))

    return {
        "latent_control_emb": torch.stack(candidate_embs),
        "latent_control_stds": torch.stack(candidate_stds),
        "latent_control_counts": torch.stack(candidate_counts),
        "latent_control_prior": torch.stack(candidate_priors),
        "latent_control_mask": torch.stack(candidate_masks),
        "latent_control_cluster_id": torch.stack(candidate_cluster_ids),
        "latent_control_parent_id": torch.stack(candidate_parent_ids),
        "latent_control_perturb_group_size": torch.as_tensor(perturb_group_sizes, dtype=torch.float32),
    }


def _bank_choose_index(dataset, arrays, balanced):
    arrays = [arr for arr in arrays if arr is not None and len(arr) > 0]
    if not arrays:
        return None
    rng = dataset.control_bank_rng
    if balanced:
        arr = arrays[int(rng.integers(len(arrays)))]
    else:
        weights = np.asarray([len(arr) for arr in arrays], dtype=np.float64)
        weights = weights / weights.sum()
        arr = arrays[int(rng.choice(len(arrays), p=weights))]
    return int(arr[int(rng.integers(len(arr)))])


def mapping_cells_from_control_bank(dataset, ds_name, key1, key2):
    """Sample a control cell from a precomputed control-state bank, if configured."""
    bank = getattr(dataset, "control_bank", None)
    if not bank:
        return None

    sampler = getattr(dataset, "control_bank_sampler_type", None)
    if sampler in (None, "", "original"):
        return None

    if sampler == "global_random_mapping":
        arr = getattr(dataset, "control_bank_global_indices", None)
        if arr is None or len(arr) == 0:
            dataset.control_bank_stats["original_fallbacks"] += 1
            return None
        dataset.control_bank_stats["bank_hits"] += 1
        return int(arr[int(dataset.control_bank_rng.integers(len(arr)))])

    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
    cell_line = str(cache.cell_type_categories[key1])
    batch = str(cache.batch_categories[key2])

    by_context = bank.get("by_cell_line_batch_cluster", {})
    by_cell = bank.get("by_cell_line_cluster", {})

    context_arrays = [arr for (cl, ba, _cluster), arr in by_context.items() if cl == cell_line and ba == batch]
    fallback_arrays = [arr for (cl, _cluster), arr in by_cell.items() if cl == cell_line]

    if sampler == "random_mapping":
        arrays = context_arrays or fallback_arrays
        selected = _bank_choose_index(dataset, arrays, balanced=False)
    elif sampler == "cluster_proportional_mapping":
        arrays = context_arrays or fallback_arrays
        selected = _bank_choose_index(dataset, arrays, balanced=False)
    elif sampler in ("cluster_balanced_mapping", "random_label_balanced_mapping"):
        arrays = context_arrays or fallback_arrays
        selected = _bank_choose_index(dataset, arrays, balanced=True)
    else:
        raise ValueError(f"Unknown control_bank_sampler_type={sampler}")

    if selected is None:
        dataset.control_bank_stats["original_fallbacks"] += 1
        return None
    dataset.control_bank_stats["bank_hits"] += 1
    return selected


def mapping_cells(dataset, ds_name, local_idx):
    """Execute `mapping_cells` and return values used by downstream logic."""
    assert dataset.data_args.mapping_strategy == "random"
    randint = lambda high: int(np.random.randint(0, high, size=(1))[0])
    choice = lambda arr: np.random.choice(arr, 1)[0]

    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

    key1 = cache.cell_type_codes[local_idx]
    key2 = cache.batch_codes[local_idx]

    bank_selected_idx = mapping_cells_from_control_bank(dataset, ds_name, key1, key2)
    if bank_selected_idx is not None:
        return int(bank_selected_idx)

    n_cells = dataset.grouped_num_cell[ds_name][(key1, key2)]
    if n_cells == 0:
        select_key1 = []
        for (i_val, j_val), v in dataset.grouped_num_cell[ds_name].items():
            if j_val == key2 and v != 0:
                select_key1.append(i_val)
        try:
            assert len(select_key1) > 0
            key1 = choice(select_key1)
        except Exception:
            select_key2 = []
            for (i_val, j_val), v in dataset.grouped_num_cell[ds_name].items():
                if i_val == key1 and v != 0:
                    select_key2.append(j_val)
            assert len(select_key2) > 0
            key2 = choice(select_key2)

        n_cells = dataset.grouped_num_cell[ds_name][(key1, key2)]

    selected_idx = randint(n_cells)
    if dataset.control_type[ds_name] is not None:
        selected_local_idx = dataset.data_indices[ds_name]["control"][
            dataset.grouped_data_indices[ds_name][(key1, key2)][selected_idx]
        ]
    else:
        selected_local_idx = dataset.data_indices[ds_name][dataset.grouped_data_indices[ds_name][(key1, key2)][selected_idx]]

    return int(selected_local_idx)


def get_covarates(dataset, ds_name, local_idx):
    """Execute `get_covarates` and return values used by downstream logic."""
    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

    try:
        pert_idx = cache.pert_codes[local_idx]
        pert_type: str = cache.pert_categories[pert_idx]
        pert_var = dataset.meta_cache.pert_dict[pert_type]
    except Exception:
        pert_var = -1

    try:
        cell_type_idx = cache.cell_type_codes[local_idx]
        cell_type: str = cache.cell_type_categories[cell_type_idx]
        cell_type_var = dataset.meta_cache.cell_type_dict[cell_type]
    except Exception:
        dataset.py_logger.info(
            "[Error]: both datasets should have cell type infos "
            "and codes. It's cell_type column for RNA-seq, while cell_line_id column for Perturb-seq"
        )
        assert 0

    try:
        batch_idx = cache.batch_codes[local_idx]
        global_batch_type = ds_name + "_" + cache.batch_categories[batch_idx]
        assert global_batch_type in dataset.meta_cache.batch_dict
        batch_var = dataset.meta_cache.batch_dict[global_batch_type]
    except Exception:
        dataset.py_logger.info("[Error]: both datasets should have batch infos and codes.")
        assert 0

    return pert_var, cell_type_var, batch_var


def getitem_impl(dataset, index):
    """Execute `getitem_impl` and return values used by downstream logic."""
    episode_local_id = -1
    episode_view_id = -1
    if len(index) == 3:
        ds_name, local_idx, is_ignore = index
        episode_token = -1
    elif len(index) == 4:
        ds_name, local_idx, is_ignore, episode_token = index
    elif len(index) == 6:
        (
            ds_name,
            local_idx,
            is_ignore,
            episode_token,
            episode_local_id,
            episode_view_id,
        ) = index
    else:
        raise ValueError(
            f"dataset index must have 3, 4, or 6 fields, got {len(index)}"
        )
    stable_local_idx = int(local_idx)
    dataset_ids = getattr(dataset, "datasets_to_num", None)
    if dataset_ids is None:
        dataset_ids = {
            name: offset
            for offset, name in enumerate(dataset.dataset_path_map.keys())
        }
    stable_dataset_id = int(dataset_ids[ds_name])

    h5f = dataset.store.dataset_file(ds_name)

    local_idx_list = [local_idx]
    is_padded_list = [is_ignore]

    counts_list, mapped_counts_list = [], []
    gene_var_list = []
    pert_var_list, mapped_pert_var_list = [], []
    cell_type_var_list, batch_var_list = [], []

    for local_idx in local_idx_list:
        counts = dataset._retrieve_counts(h5f, ds_name, local_idx)

        if bool(_cfg_get(dataset.data_args, "disable_control_mapping", False)):
            mapped_local_idx = local_idx
            mapped_counts = torch.zeros_like(counts)
        elif dataset.control_type[ds_name] is None:
            mapped_local_idx = local_idx
            mapped_counts = counts.clone()
        else:
            mapped_local_idx = dataset._mapping_cells(ds_name, local_idx)
            distribution_counts = dataset._distribution_context_counts(ds_name, local_idx)
            if distribution_counts is None:
                mapped_counts = dataset._retrieve_counts(h5f, ds_name, mapped_local_idx)
            else:
                mapped_counts = distribution_counts

        counts_list.append(counts)
        mapped_counts_list.append(mapped_counts)
        if isinstance(dataset.selected_genes, dict):
            gene_var_list.append(dataset.selected_genes[ds_name])
        else:
            gene_var_list.append(dataset.selected_genes)

        pert_var, cell_type_var, batch_var = dataset._get_covarates(ds_name, local_idx)
        mapped_pert_var, mapped_cell_type_var, mapped_batch_var = dataset._get_covarates(ds_name, mapped_local_idx)
        _ = mapped_cell_type_var
        _ = mapped_batch_var

        pert_var_list.append(pert_var)
        mapped_pert_var_list.append(mapped_pert_var)
        cell_type_var_list.append(cell_type_var)
        batch_var_list.append(batch_var)

    if libc is not None and hasattr(libc, "malloc_trim") and np.random.rand() < 0.1:
        libc.malloc_trim(0)

    return (
        counts_list[0],
        mapped_counts_list[0],
        gene_var_list[0],
        pert_var_list[0],
        mapped_pert_var_list[0],
        cell_type_var_list[0],
        batch_var_list[0],
        is_padded_list[0],
        [ds_name][0],
        int(episode_token),
        int(episode_local_id),
        int(episode_view_id),
        stable_local_idx,
        stable_dataset_id,
    )


class H5adSentenceDataset(Dataset):
    """
    A virtual dataset containing the indices for each h5/h5ad dataset,
    and during indexing, it has to iterate through all indices to find the correct
    dataset file and local index.
    """

    def __init__(
        self,
        stage: str,
        meta_cache: GlobalH5MetadataCache,
        dataset_path_map: Dict[str, str],
        selected_genes_list: Dict[str, list],
        data_indices: Dict[str, np.ndarray],
        num_cell: Dict[str, int],
        control_type: Dict[str, str],
        data_args: DictConfig,
        py_logger: Logger,
        strict_nshot_gate=None,
    ) -> None:
        """
        Initialize the class instance.

        :param stage: Input `stage` value.
        :param meta_cache: Input `meta_cache` value.
        :param dataset_path_map: Input `dataset_path_map` value.
        :param selected_genes_list: List of values used in this step.
        :param data_indices: Input `data_indices` value.
        :param num_cell: Count used to control loop/shape behavior.
        :param control_type: Input `control_type` value.
        :param data_args: Input `data_args` value.
        :param py_logger: Input `py_logger` value.
        :return: None.
        """
        super(H5adSentenceDataset, self).__init__()
        """
        control_type: a dict for each dataset, specify the perturbation label for control cell
        """
        self.stage = stage
        self.meta_cache = meta_cache
        self.data_indices = data_indices
        self.num_cell = num_cell
        self.control_type = control_type
        self.data_args = data_args
        self.py_logger = py_logger

        # fix order for datasets to ensure reproducibility
        self._names = list(data_indices.keys())

        # filter needed dataset paths
        self.dataset_path_map = {k: v for k, v in dataset_path_map.items() if k in set(self._names)}
        self.selected_genes_list = {k: v for k, v in selected_genes_list.items() if k in set(self._names)}

        # File handle store (read-only, process-local)
        self.store = H5Store(self.dataset_path_map, max_open=data_args.max_open_files)

        # Strict N-shot filtering happens on raw H5 row indices, before control
        # splitting and before any grouped virtual-index cache is constructed.
        # The default None branch intentionally preserves the historical data
        # objects and indexing path.
        self.strict_nshot_gate = strict_nshot_gate
        if self.strict_nshot_gate is not None:
            filtered_indices = {}
            filtered_num_cell = dict(self.num_cell)
            for ds_name in self._names:
                cache = self.meta_cache._cache[self.dataset_path_map[ds_name]]
                filtered = self.strict_nshot_gate.filter_indices(
                    stage=self.stage,
                    dataset_name=ds_name,
                    indices=self.data_indices[ds_name],
                    cache=cache,
                    control_perturbation=self.control_type[ds_name],
                )
                filtered_indices[ds_name] = filtered
                filtered_num_cell[ds_name] = len(filtered)
                self.py_logger.warning(
                    "Strict-N-shot index coverage: %s",
                    self.strict_nshot_gate.reports[
                        (str(self.stage).lower(), str(ds_name))
                    ].to_dict(),
                )
            self.data_indices = filtered_indices
            self.num_cell = filtered_num_cell

        # split data_indices into perturb & control
        self.split_out_control()

        # group data_indices into different categories for perturb & control cell mapping
        self.register_mapping_indices()

        # Optional clustered control context sampler. Defaults to the original random mapper.
        load_control_bank(self)
        load_control_distribution_context(self)

        # cumulative number of cells for computing index
        self._cum = np.cumsum([self.num_cell[n] for n in self._names])
        self.total_num_cell = int(self._cum[-1]) if len(self._cum) else 0

        # mapping dataset_name -> integer id
        self.datasets_to_num = {n: i for i, n in enumerate(self._names)}

        # get gene variables for each dataset
        self._get_selected_gene_vars()
        load_source_transport_bank(self)

        self._obsm_dim = {}

    def split_out_control(self):
        """Execute `split_out_control` and return values used by downstream logic."""
        return split_out_control(self)

    def register_mapping_indices(self):
        """Execute `register_mapping_indices` and return values used by downstream logic."""
        return register_mapping_indices(self)

    def _get_selected_gene_vars(self):
        """Execute `_get_selected_gene_vars` and return values used by downstream logic."""
        return get_selected_gene_vars(self)

    def _compute_index(self, ds_name: str, idx: int):
        """Execute `_compute_index` and return values used by downstream logic."""
        return compute_index(self, ds_name, idx)

    def collate_fn(self, batch: List[Tuple[Any, ...]]):
        """Execute `collate_fn` and return values used by downstream logic."""
        return collate_fn(self, batch)

    def _mapping_cells(self, ds_name, local_idx):
        """Execute `_mapping_cells` and return values used by downstream logic."""
        return mapping_cells(self, ds_name, local_idx)

    def _retrieve_counts(self, h5f, ds_name, local_idx):
        """Execute `_retrieve_counts` and return values used by downstream logic."""
        return retrieve_counts(self, h5f, ds_name, local_idx)

    def _distribution_context_counts(self, ds_name, local_idx):
        """Execute `_distribution_context_counts` and return values used by downstream logic."""
        return distribution_context_counts(self, ds_name, local_idx)

    def _get_covarates(self, ds_name, local_idx):
        """Execute `_get_covarates` and return values used by downstream logic."""
        return get_covarates(self, ds_name, local_idx)

    def __getitem__(self, index):
        """Special method `__getitem__`."""
        return getitem_impl(self, index)

    def __len__(self) -> int:
        """Special method `__len__`."""
        return self.total_num_cell

    def close(self):
        """Execute `close` and return values used by downstream logic."""
        self.store.close_all()

    def __del__(self):
        """Special method `__del__`."""
        try:
            self.store.close_all()
        except Exception:
            pass

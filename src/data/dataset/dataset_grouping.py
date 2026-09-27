"""Module `data/dataset_grouping.py`."""
import os
import pickle
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from src.common.utils import safe_decode_array

def split_out_control(dataset):
    """
    Remove control cells from virtual indexing
    """

    for ds_name in dataset._names:
        # ignore if it's RNA-seq
        if dataset.control_type[ds_name] is None:
            continue

        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
        target_code = (dataset.control_type[ds_name] == cache.pert_categories).nonzero()[0]
        assert len(target_code) == 1
        if len(arr) > 0:
            control_mask = cache.pert_codes[arr] == target_code
            assert max(arr) < cache.pert_codes.shape[0]
        else:
            control_mask = np.ones_like(arr, dtype=bool)
        control_indices = arr[control_mask]
        if dataset.data_args.keep_control_cell:
            perturb_indices = arr
        else:
            perturb_indices = arr[~control_mask]
        dataset.data_indices[ds_name] = {
            "control": control_indices,
            "perturb": perturb_indices,
        }

        if dataset.data_args.keep_control_cell:
            pass
        else:
            dataset.num_cell[ds_name] -= len(control_indices)
        dataset.py_logger.info(
            f"For dataset {ds_name} (different split), control cells: "
            f"{len(control_indices)}, perturb cells: {len(perturb_indices)}"
        )

def group_by_one_keys(dataset, arr, cache, key1_name):
    """Execute `group_by_one_keys` and return values used by downstream logic."""
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))

    for i in range(num_key1):
        mask = key1_codes == i
        gp_indices[i] = mask.nonzero()[0]
        gp_cells[i] = len(gp_indices[i])

    return gp_indices, gp_cells

def group_by_two_keys(dataset, arr, cache, key1_name, key2_name):
    """Execute `group_by_two_keys` and return values used by downstream logic."""
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    key2_codes = getattr(cache, f"{key2_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))
    num_key2 = len(getattr(cache, f"{key2_name}_categories"))

    for i in range(num_key1):
        for j in range(num_key2):
            mask = (key1_codes == i) & (key2_codes == j)
            gp_indices[(i, j)] = mask.nonzero()[0]
            gp_cells[(i, j)] = len(gp_indices[(i, j)])

    return gp_indices, gp_cells

def group_by_three_keys(dataset, arr, cache, key1_name, key2_name, key3_name):
    """
    Group by three keys.

    :param dataset: Input `dataset` value.
    :param arr: Source 1-D array for indexed extraction.
    :param cache: Input `cache` value.
    :param key1_name: Input `key1_name` value.
    :param key2_name: Input `key2_name` value.
    :param key3_name: Input `key3_name` value.
    :return: Computed output(s) for this function.
    """
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    key2_codes = getattr(cache, f"{key2_name}_codes")[arr]
    key3_codes = getattr(cache, f"{key3_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))
    num_key2 = len(getattr(cache, f"{key2_name}_categories"))
    num_key3 = len(getattr(cache, f"{key3_name}_categories"))

    dataset.py_logger.info(
        f"group by three keys: {key1_name}: {num_key1}, {key2_name}: {num_key2}, {key3_name}: {num_key3}"
    )

    for i in tqdm(range(num_key1)):
        for j in range(num_key2):
            for k in range(num_key3):
                mask = (key1_codes == i) & (key2_codes == j) & (key3_codes == k)
                gp_indices[(i, j, k)] = mask.nonzero()[0]
                gp_cells[(i, j, k)] = len(gp_indices[(i, j, k)])

    return gp_indices, gp_cells

def group_by_three_keys_sparse(dataset, arr, cache, key1_name, key2_name, key3_name):
    """Group only observed tuples in O(N log N), for strict filtered indices."""

    arr = np.asarray(arr, dtype=np.int64)
    if not len(arr):
        return {}, {}
    key1 = np.asarray(getattr(cache, f"{key1_name}_codes")[arr], dtype=np.int64)
    key2 = np.asarray(getattr(cache, f"{key2_name}_codes")[arr], dtype=np.int64)
    key3 = np.asarray(getattr(cache, f"{key3_name}_codes")[arr], dtype=np.int64)
    order = np.lexsort((key3, key2, key1)).astype(np.int64, copy=False)
    sorted1, sorted2, sorted3 = key1[order], key2[order], key3[order]
    starts = np.flatnonzero(
        np.r_[
            True,
            (sorted1[1:] != sorted1[:-1])
            | (sorted2[1:] != sorted2[:-1])
            | (sorted3[1:] != sorted3[:-1]),
        ]
    )
    ends = np.r_[starts[1:], len(order)]
    grouped_indices, grouped_cells = {}, {}
    for start, end in zip(starts.tolist(), ends.tolist()):
        key = (
            int(sorted1[start]),
            int(sorted2[start]),
            int(sorted3[start]),
        )
        values = order[start:end].copy()
        grouped_indices[key] = values
        grouped_cells[key] = int(len(values))
    dataset.py_logger.info(
        "sparse group by three keys: %s/%s/%s observed groups=%d cells=%d",
        key1_name,
        key2_name,
        key3_name,
        len(grouped_indices),
        len(arr),
    )
    return grouped_indices, grouped_cells


def register_mapping_indices(dataset):
    """
    During indexing, it's default for indexing perturb-seq, and we select
        from mapped control cells for it. As for RNA-seq, we also select one
        cell that shares the same cell type and optionally batch.

    In other words, control cells in perturb-seq and the whole RNA-seq would be selected among.
    And we pre-group their indices for convenient mapping and selection
    """
    dataset.grouped_data_indices, dataset.grouped_num_cell = {}, {}
    for ds_name in dataset.data_indices.keys():
        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

        if dataset.control_type[ds_name] is not None:
            arr = arr["control"]
        else:
            # RNA-seq
            arr = np.array([], dtype=np.int64)
        if len(arr) > 0:
            gp_indices, gp_cells = group_by_two_keys(dataset, arr, cache, key1_name="cell_type", key2_name="batch")
        else:
            gp_indices, gp_cells = {}, {}
        dataset.grouped_data_indices[ds_name] = gp_indices
        dataset.grouped_num_cell[ds_name] = gp_cells

    if not hasattr(dataset.data_args, "use_cell_set") or dataset.data_args.use_cell_set is None:
        return

    dataset.grouped_pert_data_indices, dataset.grouped_pert_num_cell = {}, {}
    strict_gate = getattr(dataset, "strict_nshot_gate", None)
    strict_cache_tag = "" if strict_gate is None else f"_{strict_gate.cache_tag}"
    for ds_name in dataset.data_indices.keys():
        indices_cache_file = os.path.join(
            dataset.data_args.indices_cache_dir,
            f"grouped_pert_data_indices_{ds_name}_{dataset.stage}{strict_cache_tag}.pkl",
        )
        num_cell_cache_file = os.path.join(
            dataset.data_args.indices_cache_dir,
            f"grouped_pert_num_cell_{ds_name}_{dataset.stage}{strict_cache_tag}.pkl",
        )

        if (
            (not getattr(dataset.data_args, "skip_cached_indices", False))
            and os.path.exists(indices_cache_file)
            and os.path.exists(num_cell_cache_file)
        ):
            with open(indices_cache_file, "rb") as fin:
                gp_indices = pickle.load(fin)
            with open(num_cell_cache_file, "rb") as fin:
                gp_cells = pickle.load(fin)

            dataset.grouped_pert_data_indices[ds_name] = gp_indices
            dataset.grouped_pert_num_cell[ds_name] = gp_cells
            continue

        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

        if dataset.control_type[ds_name] is not None:
            arr = arr["perturb"]
            if len(arr) > 0:
                grouping = (
                    group_by_three_keys_sparse
                    if strict_gate is not None
                    else group_by_three_keys
                )
                gp_indices, gp_cells = grouping(
                    dataset,
                    arr,
                    cache,
                    key1_name="pert",
                    key2_name="cell_type",
                    key3_name="batch",
                )
            else:
                gp_indices, gp_cells = {}, {}
        else:
            # RNA-seq
            if "cellxgene" in ds_name.lower():
                gp_indices, gp_cells = group_by_one_keys(dataset, arr, cache, key1_name="cell_type")
            else:
                raise NotImplementedError

        dataset.grouped_pert_data_indices[ds_name] = gp_indices
        dataset.grouped_pert_num_cell[ds_name] = gp_cells

        Path(indices_cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(num_cell_cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(indices_cache_file, "wb") as fout:
            pickle.dump(dataset.grouped_pert_data_indices[ds_name], fout)
        with open(num_cell_cache_file, "wb") as fout:
            pickle.dump(dataset.grouped_pert_num_cell[ds_name], fout)

    if dataset.data_args.keep_control_cell:
        for ds_name in dataset.data_indices.keys():
            arr = dataset.data_indices[ds_name]
            cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
            if dataset.control_type[ds_name] is None:
                continue
            target_code = (dataset.control_type[ds_name] == cache.pert_categories).nonzero()[0]

            assert len(target_code) == 1
            target_code = target_code.item()
            for k, v in dataset.grouped_data_indices[ds_name].items():
                dataset.grouped_pert_data_indices[ds_name][(target_code, k[0], k[1])] = v
            for k, v in dataset.grouped_num_cell[ds_name].items():
                dataset.grouped_pert_num_cell[ds_name][(target_code, k[0], k[1])] = v

def get_selected_gene_vars(dataset):
    """
    Get selected gene vars.

    :param dataset: Input `dataset` value.
    :return: Requested object(s) for downstream use.
    """
    if dataset.data_args.selected_gene_file is None:
        dataset.selected_genes = {}
        for ds_name in dataset._names:
            dataset.selected_genes[ds_name] = dataset.selected_genes_list[ds_name]

    else:
        with open(dataset.data_args.selected_gene_file, "rb") as fin:
            dataset.selected_genes = pickle.load(fin)

        if isinstance(dataset.selected_genes, set):
            dataset.selected_genes = sorted(dataset.selected_genes)

    dataset.selected_gene_mask, dataset.gene_vars = {}, {}

    for ds_name in dataset._names:
        h5f = dataset.store.dataset_file(ds_name)
        try:
            all_genes = safe_decode_array((h5f["/var/_index"]))
        except:
            try:
                all_genes = safe_decode_array((h5f["/var/gene_name_index"]))
            except:
                try:
                    all_genes = safe_decode_array((h5f["/var/ensembl_id"]))
                except:
                    try:
                        all_genes = safe_decode_array((h5f["/var/gene_name"]))
                    except:
                        all_genes = safe_decode_array((h5f["/var/gene_short_name"]))

        if "cellxgene" in ds_name:
            gm = pd.read_csv(dataset.data_args.gene_mapping_file)
            gmap = {k: v for k, v in zip(gm["id"], gm["name"])}
            new_all_genes = []
            not_found = 0
            for x in all_genes:
                if x in gmap:
                    new_all_genes.append(gmap[x])
                else:
                    new_all_genes.append(x)
                    not_found += 1
            dataset.py_logger.info(
                f"[Warning]: {not_found} gene names in RNA-seq {ds_name} are not found in gene mapping dict"
            )
            dataset.gene_vars[ds_name] = np.array(new_all_genes)
            all_genes = new_all_genes
        else:
            dataset.gene_vars[ds_name] = all_genes

        if isinstance(dataset.selected_genes, dict):
            mask = np.isin(all_genes, dataset.selected_genes[ds_name])
        else:
            mask = np.isin(all_genes, dataset.selected_genes)
        dataset.selected_gene_mask[ds_name] = mask
        dataset.py_logger.info(
            f"Shared gene count in data {ds_name}: \t\t---- ({mask.sum()} / {len(dataset.selected_genes)}; original: {len(all_genes)})"
        )

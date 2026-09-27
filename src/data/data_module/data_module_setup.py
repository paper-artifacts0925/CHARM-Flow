"""Setup helpers extracted from data_module.py (logic-preserving)."""

import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.common.utils import safe_decode_array
from src.data.dataset.dataset_core import H5adSentenceDataset
from src.data.metadata_cache import GlobalH5MetadataCache


def _filter_requested_dataset_sources(dm):
    """Apply sample/skip dataset filters before opening source files."""
    names = list(dm.dataset_name_list)
    files = list(dm.file_list)
    data_types = list(dm.data_type_list)
    key_infos = list(dm.key_info_list)
    selected = list(dm.selected_genes_list)

    sample_pbmc = bool(getattr(dm.data_args, "sample_pbmc_only", False))
    sample_replogle = bool(getattr(dm.data_args, "sample_replogle_only", False))
    sample_tahoe = bool(getattr(dm.data_args, "sample_tahoe100m_only", False))
    skip_tahoe = bool(getattr(dm.data_args, "skip_tahoe100m", False))
    skip_pbmc = bool(getattr(dm.data_args, "skip_pbmc", False))
    skip_replogle = bool(getattr(dm.data_args, "skip_replogle", False))
    skip_cellxgene = bool(getattr(dm.data_args, "skip_cellxgene", False))

    if not any([sample_pbmc, sample_replogle, sample_tahoe, skip_tahoe, skip_pbmc, skip_replogle, skip_cellxgene]):
        return

    keep = []
    for name in names:
        lower = name.lower()
        flag = True
        if sample_pbmc:
            flag = "pbmc" in lower
        if sample_replogle:
            flag = "replogle" in lower
        if sample_tahoe:
            flag = "tahoe100m" in lower
        if skip_tahoe and "tahoe100m" in lower:
            flag = False
        if skip_pbmc and "pbmc" in lower:
            flag = False
        if skip_replogle and "replogle" in lower:
            flag = False
        if skip_cellxgene and "cellxgene" in lower:
            flag = False
        keep.append(flag)

    dm.dataset_name_list = [x for x, flag in zip(names, keep) if flag]
    dm.file_list = [x for x, flag in zip(files, keep) if flag]
    dm.data_type_list = [x for x, flag in zip(data_types, keep) if flag]
    dm.key_info_list = [x for x, flag in zip(key_infos, keep) if flag]
    dm.selected_genes_list = [x for x, flag in zip(selected, keep) if flag]

    if not dm.dataset_name_list:
        raise ValueError("Dataset filtering removed all sources")


def pretraining_setup_dataset(dm, stage=None):
    """
    Set up datasets by looping through all processed cellxgene / tahoe100m datasets.
    Split by different strategies: random and holdout
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(
        dm.pert_dict,
        dm.batch_dict,
        dm.cell_type_dict,
    )
    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = {x: y for x, y in zip(dm.dataset_name_list, dm.file_list)}
    selected_genes_list = {x: y for x, y in zip(dm.dataset_name_list, dm.selected_genes_list)}
    control_type = {
        x: None if y == "RNA-seq" else dm.data_args.control_pert
        for x, y in zip(dm.dataset_name_list, dm.data_type_list)
    }

    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Splitting into train/validation/test data (only for indices) ...",
    ):
        # RNA-seq from cellxgene only contains cell types
        # Perturb-seq from tahoe100m only contains perturbation conditions
        if data_type == "RNA-seq":
            assert 0
        elif data_type == "Perturb-seq":
            cache = meta_cache.get_cache(
                dataset_path,
                batch_col=dm.data_args.perturbseq_batch_col,
                pert_col=dm.data_args.pert_col,
                cell_type_key=dm.data_args.cell_line_key,
            )
            split_indices = dm._proceed_data_split(
                cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
            )
        else:
            raise NotImplementedError

        for split_stage in split_indices:
            tmp = getattr(dm, f"{split_stage}_indices")
            tmp[dataset_name] = split_indices[split_stage]

            tmp = getattr(dm, f"{split_stage}_num_cell")
            tmp[dataset_name] = len(split_indices[split_stage])

            dm.py_logger.info(f"Processed {split_stage}: {len(split_indices[split_stage])}")

    for split_stage in ["train"] + dm.all_split_names:
        dm.py_logger.info(">>>>> stage: %s", split_stage)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                dm.data_args,
                dm.py_logger,
                strict_nshot_gate=getattr(
                    dm, "strict_nshot_gate", None
                ),
            ),
        )


def pretraining_setup(dm, stage=None):
    """
    Pretraining setup.

    :param dm: Input `dm` value.
    :param stage: Input `stage` value.
    :return: Computed output(s) for this function.
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    all_perts, all_batches, all_celltypes = set(), set(), set()

    # search through data directory for all processed h5 / h5ad files
    (
        dm.dataset_name_list,
        dm.file_list,
        dm.data_type_list,
        dm.key_info_list,
        dm.selected_genes_list,
    ) = dm.get_dataset_names_and_paths(dm.data_args, dm.data_args.data_name)

    dm.original_dataset_name_list = list(dm.dataset_name_list)

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list), "Processing meta data ..."
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                assert 0

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{dm.data_args.cell_line_key}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{dm.data_args.cell_line_key}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{dm.data_args.pert_col}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{dm.data_args.perturbseq_batch_col}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{dm.data_args.perturbseq_batch_col}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.original_dataset_name_list = dm.dataset_name_list

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )


def perturbation_pretraining_setup_dataset(dm, stage=None):
    """
    Set up datasets by looping through all processed cellxgene / tahoe100m datasets.
    Split by different strategies: random and holdout
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(
        dm.pert_dict,
        dm.batch_dict,
        dm.cell_type_dict,
    )
    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = {x: y for x, y in zip(dm.dataset_name_list, dm.file_list)}
    selected_genes_list = {x: y for x, y in zip(dm.dataset_name_list, dm.selected_genes_list)}
    control_type = {
        x: None if y == "RNA-seq" else z["control_pert"]
        for x, y, z in zip(dm.dataset_name_list, dm.data_type_list, dm.key_info_list)
    }

    exist_flag = False
    if dm.data_args.data_name == "Tahoe100mPBMCReplogleCellxGenePretrain":
        exist_flag = True
        for split_stage in ["train"] + dm.all_split_names:
            split_indices_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            split_num_cell_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            if (not os.path.exists(split_indices_cache_file)) or (
                not os.path.exists(split_num_cell_cache_file)
            ):
                exist_flag = False
                break

    if not exist_flag or dm.data_args.skip_cached_indices:
        for dataset_name, dataset_path, data_type, key_info in tqdm(
            zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
            "Splitting into train/validation/test data (only for indices) ...",
        ):
            # RNA-seq from cellxgene only contains cell types
            # Perturb-seq from tahoe100m only contains perturbation conditions
            if data_type == "RNA-seq":
                cache = meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["rnaseq_batch_col"],
                    cell_type_key=key_info["cell_type_key"],
                )
                split_indices = dm._proceed_data_split_cellxgene(
                    cache, "celltype", key_info, "holdout_celltype", "random_RNAseq"
                )

            elif data_type == "Perturb-seq":
                cache = meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["perturbseq_batch_col"],
                    pert_col=key_info["pert_col"],
                    cell_type_key=key_info["cell_line_key"],
                )
                if "pbmc" in dataset_path.lower():
                    split_indices = dm._proceed_data_split_pbmc(
                        cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
                    )
                elif "tahoe100m" in dataset_path.lower() or "replogle" in dataset_path.lower():
                    split_indices = dm._proceed_data_split_tahoe100m(
                        cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            for split_stage in split_indices:
                tmp = getattr(dm, f"{split_stage}_indices")
                tmp[dataset_name] = split_indices[split_stage]

                tmp = getattr(dm, f"{split_stage}_num_cell")
                tmp[dataset_name] = len(split_indices[split_stage])

                dm.py_logger.info(f"Processed {split_stage}: {len(split_indices[split_stage])}")

        if not exist_flag and not dm.data_args.skip_cached_indices:
            for split_stage in ["train"] + dm.all_split_names:
                split_indices_cache_file = os.path.join(
                    dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
                )
                split_num_cell_cache_file = os.path.join(
                    dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
                )
                with open(split_indices_cache_file, "wb") as fout:
                    pickle.dump(getattr(dm, f"{split_stage}_indices"), fout)
                with open(split_num_cell_cache_file, "wb") as fout:
                    pickle.dump(getattr(dm, f"{split_stage}_num_cell"), fout)
                dm.py_logger.info(f"Processed {split_stage}: writing split indices to caches")
    else:
        # store cache still
        for dataset_name, dataset_path, data_type, key_info in tqdm(
            zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
            "Splitting into train/validation/test data (only for indices) ...",
        ):
            # RNA-seq from cellxgene only contains cell types
            # Perturb-seq from tahoe100m only contains perturbation conditions
            if data_type == "RNA-seq":
                meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["rnaseq_batch_col"],
                    cell_type_key=key_info["cell_type_key"],
                )
            elif data_type == "Perturb-seq":
                meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["perturbseq_batch_col"],
                    pert_col=key_info["pert_col"],
                    cell_type_key=key_info["cell_line_key"],
                )
            else:
                raise NotImplementedError

        for split_stage in ["train"] + dm.all_split_names:
            split_indices_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            split_num_cell_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            dm.py_logger.info(f"Processed {split_stage}: loading from cached split indices")
            with open(split_indices_cache_file, "rb") as fin:
                split_indices = pickle.load(fin)
            setattr(dm, f"{split_stage}_indices", split_indices)
            with open(split_num_cell_cache_file, "rb") as fin:
                split_num_cell = pickle.load(fin)
            setattr(dm, f"{split_stage}_num_cell", split_num_cell)

    for split_stage in ["train"] + dm.all_split_names:
        dm.py_logger.info(">>>>> stage: %s", split_stage)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                dm.data_args,
                dm.py_logger,
                strict_nshot_gate=getattr(
                    dm, "strict_nshot_gate", None
                ),
            ),
        )


def build_new_pert_dict(dm):
    """
    Build new pert dict.

    :param dm: Input `dm` value.
    :return: Requested object(s) for downstream use.
    """
    dm.py_logger.info("Calling new_pert_dict ...")

    all_perts = set()

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                pass
            elif data_type == "Perturb-seq":
                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

    pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    return pert_dict


def build_new_all_dict(dm):
    """
    Build new all dict.

    :param dm: Input `dm` value.
    :return: Requested object(s) for downstream use.
    """
    all_perts, all_batches, all_celltypes = set(), set(), set()

    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                # cell type
                try:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )
    return pert_dict, batch_dict, cell_type_dict


def perturbation_pretraining_setup(dm, stage=None):
    """
    Perturbation pretraining setup.

    :param dm: Input `dm` value.
    :param stage: Input `stage` value.
    :return: Computed output(s) for this function.
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    all_perts, all_batches, all_celltypes = set(), set(), set()

    # search through data directory for all processed h5 / h5ad files
    (
        dm.dataset_name_list,
        dm.file_list,
        dm.data_type_list,
        dm.key_info_list,
        dm.selected_genes_list,
    ) = dm.get_dataset_names_and_paths(dm.data_args)

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                # cell type
                try:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )

    # only sample pbmc
    dm.original_dataset_name_list = dm.dataset_name_list
    if dm.data_args.sample_pbmc_only:
        mask = np.array([True if "pbmc" in x.lower() else False for x in dm.dataset_name_list])
        assert mask[0] and (not mask[1:].any())
        dm.dataset_name_list = dm.dataset_name_list[:1]
        dm.file_list = dm.file_list[:1]
        dm.data_type_list = dm.data_type_list[:1]
        dm.key_info_list = dm.key_info_list[:1]

    if hasattr(dm.data_args, "sample_replogle_only") and dm.data_args.sample_replogle_only:
        mask = np.array([True if "replogle" in x.lower() else False for x in dm.dataset_name_list])
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        dm.selected_genes_list = [
            genes for genes, keep in zip(dm.selected_genes_list, mask) if keep
        ]

    if hasattr(dm.data_args, "sample_tahoe100m_only") and dm.data_args.sample_tahoe100m_only:
        mask = np.array([True if "tahoe100m" in x.lower() else False for x in dm.dataset_name_list])
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        dm.selected_genes_list = [
            genes for genes, keep in zip(dm.selected_genes_list, mask) if keep
        ]

    skip_tahoe100m = hasattr(dm.data_args, "skip_tahoe100m") and dm.data_args.skip_tahoe100m
    skip_pbmc = hasattr(dm.data_args, "skip_pbmc") and dm.data_args.skip_pbmc
    skip_replogle = hasattr(dm.data_args, "skip_replogle") and dm.data_args.skip_replogle
    skip_cellxgene = hasattr(dm.data_args, "skip_cellxgene") and dm.data_args.skip_cellxgene

    if skip_tahoe100m or skip_pbmc or skip_replogle or skip_cellxgene:
        dm.py_logger.info("Skipping datasets according to user settings ...")
        mask = []
        for x in dm.dataset_name_list:
            flag = True
            if skip_tahoe100m and ("tahoe100m" in x.lower()):
                flag = False
            if skip_pbmc and ("pbmc" in x.lower()):
                flag = False
            if skip_replogle and ("replogle" in x.lower()):
                flag = False
            if skip_cellxgene and ("cellxgene" in x.lower()):
                flag = False
            mask.append(flag)
        mask = np.array(mask)
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        dm.selected_genes_list = [
            genes for genes, keep in zip(dm.selected_genes_list, mask) if keep
        ]

    if dm.replace_pert_dict:
        dm.pert_dict = build_new_pert_dict(dm)
        dm.py_logger.info(f"Replacing new #perturbation: {len(dm.pert_dict)}")

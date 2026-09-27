"""Module `data/data_module.py`."""
import os
import sys
from glob import glob
import numpy as np
import h5py
from pathlib import Path
from typing import Literal, Set, Dict
import omegaconf
import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.distributed as dist

from tqdm import tqdm

from src.data.dataset.dataset_core import H5adSentenceDataset
from src.data.metadata_cache import GlobalH5MetadataCache
from src.common.utils import safe_decode_array

from src.data.sampler import (
    CellSetBatchSampler,
    DistributedCellSetFixPairingBatchSampler,
    DistributedMetaBatchEpisodeBatchSampler,
)
from src.data.split_strategy import split_cellxgene, split_pbmc, split_tahoe100m
from src.data.data_module.data_module_setup import (
    build_new_all_dict,
    build_new_pert_dict,
    perturbation_pretraining_setup,
    perturbation_pretraining_setup_dataset,
    pretraining_setup,
    pretraining_setup_dataset,
)

import tracemalloc
import gc


def build_train_dataloader(dm):
    """
    Build the training dataloader for a datamodule instance.

    :param dm: Datamodule instance.
    :return: Training dataloader.

    """
    if not hasattr(dm, "train_dataset"):
        dm.setup_dataset()

    if bool(
        getattr(dm.data_args, "hungarian_flow_combination_sampler", False)
    ):
        from src.models.hungarian_flow.loader import build_combination_dataloader
        return build_combination_dataloader(dm)

    if bool(getattr(dm.data_args, "meta_batch_episode_enabled", False)):
        if bool(getattr(dm.data_args, "pert_mix_enabled", False)):
            raise ValueError("pert_mix_enabled must be false for meta-batch episodes")
        num_views = int(getattr(dm.data_args, "meta_batch_num_views", 3))
        episodes_per_batch = int(
            getattr(dm.data_args, "meta_batch_episodes_per_batch", 1)
        )
        dynamic_views = bool(
            getattr(dm.data_args, "meta_batch_dynamic_views", False)
        )
        set_size = int(dm.data_args.use_cell_set)
        if dynamic_views:
            configured_budget = getattr(
                dm.data_args, "meta_batch_set_views_per_batch", None
            )
            if configured_budget in (None, 0):
                if int(dm.micro_batch_size) % set_size:
                    raise ValueError(
                        "optimization.micro_batch_size must be divisible by "
                        "data.use_cell_set for dynamic meta-batch episodes"
                    )
                set_views_per_batch = int(dm.micro_batch_size) // set_size
            else:
                set_views_per_batch = int(configured_budget)
            expected_cells = set_size * set_views_per_batch
            expected_description = (
                "data.use_cell_set * data.meta_batch_set_views_per_batch"
            )
        else:
            set_views_per_batch = num_views * episodes_per_batch
            expected_cells = set_size * set_views_per_batch
            expected_description = (
                "data.use_cell_set * data.meta_batch_num_views * "
                "data.meta_batch_episodes_per_batch"
            )
        if int(dm.micro_batch_size) != expected_cells:
            raise ValueError(
                "optimization.micro_batch_size must equal "
                f"{expected_description} ({expected_cells})"
            )
        batch_sampler = DistributedMetaBatchEpisodeBatchSampler(
            dm.train_dataset,
            episodes_per_batch=episodes_per_batch,
            num_views=num_views,
            shuffle=True,
            seed=dm.seed,
            dynamic_views=dynamic_views,
            set_views_per_batch=set_views_per_batch,
            min_views=int(
                getattr(dm.data_args, "meta_batch_min_views", 2)
            ),
            max_views=int(
                getattr(dm.data_args, "meta_batch_max_views", 5)
            ),
        )
        loader = DataLoader(
            dm.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=dm.train_dataset.collate_fn,
            num_workers=dm.data_args.num_workers,
            persistent_workers=dm.data_args.persistent_workers,
            pin_memory=dm.data_args.pin_memory,
            prefetch_factor=dm.data_args.prefetch_factor,
        )
        dm.py_logger.info(
            "Finished loading meta-batch training data: %d episodes",
            batch_sampler.num_episodes,
        )
        return loader
    if dist.is_initialized() and dm.data_args.use_fixed_pairing:
        sampler = DistributedCellSetFixPairingBatchSampler(
            dm.train_dataset,
            shuffle=True,
            drop_last=True,
            seed=dm.seed,
        )
    else:
        sampler = CellSetBatchSampler(
            dm.train_dataset,
            shuffle=True,
            seed=dm.seed,
        )

    loader = DataLoader(
        dm.train_dataset,
        batch_size=dm.micro_batch_size,
        collate_fn=dm.train_dataset.collate_fn,
        sampler=sampler,
        num_workers=dm.data_args.num_workers,
        persistent_workers=dm.data_args.persistent_workers,
        pin_memory=dm.data_args.pin_memory,
        drop_last=True,
        prefetch_factor=dm.data_args.prefetch_factor,
    )
    dm.py_logger.info(f"Finished loading training data: {len(dm.train_dataset)} samples")
    return loader


def build_val_dataloaders(dm):
    """
    Build validation/test dataloaders for all splits.

    :param dm: Datamodule instance.
    :return: List of dataloaders in split order.
    """
    loaders = []

    split_names = getattr(dm, "evaluation_split_names", dm.all_split_names)
    if any(not hasattr(dm, f"{split}_dataset") for split in split_names):
        dm.setup_dataset(stage="validate")
    for split in split_names:
        assert hasattr(dm, f"{split}_dataset")

        cur_dataset = getattr(dm, f"{split}_dataset")
        collate_fn = cur_dataset.collate_fn

        if dist.is_initialized() and dm.data_args.use_fixed_pairing:
            sampler = DistributedCellSetFixPairingBatchSampler(
                cur_dataset,
                shuffle=False,
                drop_last=True,
                seed=dm.seed,
            )
        else:
            sampler = CellSetBatchSampler(
                cur_dataset,
                shuffle=False,
                seed=dm.seed,
            )

        loader = DataLoader(
            cur_dataset,
            batch_size=dm.micro_batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=dm.data_args.num_workers,
            persistent_workers=dm.data_args.persistent_workers,
            pin_memory=dm.data_args.pin_memory,
            drop_last=False,
            prefetch_factor=dm.data_args.prefetch_factor,
        )
        dm.py_logger.info(f"Finished loading {split} data: {len(cur_dataset)} samples")
        loaders.append(loader)
    return loaders


class PretrainingDataModule(pl.LightningDataModule):

    #def __init__(self, 
    #    seed: int, 
    #    micro_batch_size: int, 
    #    data_args, 
    #    py_logger
    #):
    #    super().__init__()
    #
    #    self.seed = seed
    #    self.micro_batch_size = micro_batch_size
    #    self.data_args = data_args
    #    self.py_logger = py_logger
    #
    #    self.all_split_names = ["holdout_celltype", "holdout_pert", "random_RNAseq", "random_Perturbseq"]
    
    """Pretrainingdatamodule implementation used by the PerturbDiff pipeline."""
    def train_dataloader(self):
        """This will be run every epoch."""
        return build_train_dataloader(self)

    def val_dataloader(self):
        """Prepare all split sets for pretraining validation here"""
        return build_val_dataloaders(self)
    
    #@profile
    def setup_dataset(self, stage=None):
        """Execute `setup_dataset` and return values used by downstream logic."""
        return pretraining_setup_dataset(self, stage=stage)
    #@profile
    def setup(self, stage=None):
        """Execute `setup` and return values used by downstream logic."""
        return pretraining_setup(self, stage=stage)

class PBMCPerturbationDataModule(PretrainingDataModule):
    """
    Inherent from PretrainingDataModule for func: 
        - setup() 
        - setup_dataset()
        - train_dataloader()
        - val_dataloader()
    
    """

    def __init__(self, 
        seed: int, 
        micro_batch_size: int, 
        data_args, 
        py_logger
    ):
        """Special method `__init__`."""
        super(PretrainingDataModule, self).__init__()

        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger

        self.all_split_names = ["validation", "test"]

    def get_dataset_names_and_paths(self, cfg, data_name):

        """
        Get dataset names and paths.

        :param cfg: Runtime configuration object.
        :param data_name: Input `data_name` value.
        :return: Requested object(s) for downstream use.
        """
        key_info = {
            "pert_col": cfg.pert_col,
            "rnaseq_batch_col": cfg.rnaseq_batch_col,
            "perturbseq_batch_col": cfg.perturbseq_batch_col,
            "cell_type_key": cfg.cell_type_key,
            "cell_line_key": cfg.cell_line_key,
            "control_pert": cfg.control_pert,
            "holdout_celltype": cfg.holdout_celltype,
            "holdout_batches": cfg.holdout_batches,
            "holdout_pert": cfg.holdout_pert,
        }

        with open(cfg.selected_gene_file, "rb") as fin:
            selected_genes = pickle.load(fin)
        # it's a set, because it's merged from multiple datasets
        if isinstance(selected_genes, set):
            selected_genes = sorted(list(selected_genes))

        if cfg.dataset_name is None:
            file_list = glob(os.path.join(cfg.dataset_path, "*.h5ad"))
            file_list = sorted(file_list)[::-1]
            
            if "cellxgene" in cfg.dataset_path.lower():
                data_type = "RNA-seq"
                assert len(file_list) == 23
            elif "tahoe100m" in cfg.dataset_path.lower():
                data_type = "Perturb-seq"
                assert len(file_list) == 14 # total 14 files for tahoe100m
            else:
                raise NotImplementedError
            dataset_name_list = [os.path.splitext(os.path.basename(x))[0] for x in file_list]
            return dataset_name_list, file_list, [data_type] * len(file_list), [key_info] * len(file_list), [selected_genes] * len(file_list)
        else:
            return [cfg.dataset_name], [cfg.dataset_path], ["Perturb-seq"], [key_info], [selected_genes]
    
    def _proceed_data_split(self, 
        cache, target_type, key_info, holdout_setname=None, random_setname=None):
        """Execute `_proceed_data_split` and return values used by downstream logic."""
        return self._proceed_data_split_pbmc(cache, target_type, key_info, holdout_setname, random_setname)
    
    def _proceed_data_split_pbmc(self, 
        cache, target_type, key_info, holdout_setname=None, random_setname=None
    ):
        """Execute `_proceed_data_split_pbmc` and return values used by downstream logic."""
        return split_pbmc(self, cache, target_type, key_info, holdout_setname, random_setname)


    
class Tahoe100mPerturbationDataModule(PBMCPerturbationDataModule):

    """Tahoe100Mperturbationdatamodule implementation used by the PerturbDiff pipeline."""
    def _proceed_data_split(self, 
        cache, target_type, key_info, holdout_setname=None, random_setname=None):
        """Execute `_proceed_data_split` and return values used by downstream logic."""
        return self._proceed_data_split_tahoe100m(cache, target_type, key_info, holdout_setname, random_setname)

    def _proceed_data_split_tahoe100m(self, 
        cache, target_type, key_info, holdout_setname=None, random_setname=None
    ):
        """Execute `_proceed_data_split_tahoe100m` and return values used by downstream logic."""
        return split_tahoe100m(self, cache, target_type, key_info, holdout_setname, random_setname)


class CellxGeneDataModule(Tahoe100mPerturbationDataModule):

    """Cellxgenedatamodule implementation used by the PerturbDiff pipeline."""
    def _proceed_data_split_cellxgene(self, 
        cache, target_type, key_info, holdout_setname=None, random_setname=None
    ):
        """Execute `_proceed_data_split_cellxgene` and return values used by downstream logic."""
        return split_cellxgene(self, cache, target_type, key_info, holdout_setname, random_setname)



class PerturbationPretrainingDataModule(CellxGeneDataModule):
        

    """Perturbationpretrainingdatamodule implementation used by the PerturbDiff pipeline."""
    def get_dataset_names_and_paths(self, cfg):
        """Execute `get_dataset_names_and_paths` and return values used by downstream logic."""
        assert self.data_args == cfg
        assert isinstance(self.data_args.dataset_name, omegaconf.dictconfig.DictConfig)

        dataset_name_list, file_list, data_type_list, key_info_list, selected_genes_list = [], [], [], [], []
        for key in self.data_args.dataset_name:
            key_lower = str(key).lower()
            if getattr(self.data_args, "sample_pbmc_only", False) and "pbmc" not in key_lower:
                continue
            if getattr(self.data_args, "sample_replogle_only", False) and "replogle" not in key_lower:
                continue
            if getattr(self.data_args, "sample_tahoe100m_only", False) and "tahoe100m" not in key_lower:
                continue
            if getattr(self.data_args, "skip_pbmc", False) and "pbmc" in key_lower:
                continue
            if getattr(self.data_args, "skip_replogle", False) and "replogle" in key_lower:
                continue
            if getattr(self.data_args, "skip_tahoe100m", False) and "tahoe100m" in key_lower:
                continue
            if getattr(self.data_args, "skip_cellxgene", False) and "cellxgene" in key_lower:
                continue
            ret = super().get_dataset_names_and_paths(self.data_args.dataset_name[key], self.data_args.data_name)
            dataset_name_list.extend(ret[0])
            file_list.extend(ret[1])
            data_type_list.extend(ret[2])
            key_info_list.extend(ret[3])
            selected_genes_list.extend(ret[4])

        return dataset_name_list, file_list, data_type_list, key_info_list, selected_genes_list

    def setup_dataset(self, stage=None):
        """Execute `setup_dataset` and return values used by downstream logic."""
        return perturbation_pretraining_setup_dataset(self, stage=stage)

    def get_new_pert_dict(self):
        """Execute `get_new_pert_dict` and return values used by downstream logic."""
        return build_new_pert_dict(self)
    
    def get_new_all_dict(self,):
        """Execute `get_new_all_dict` and return values used by downstream logic."""
        return build_new_all_dict(self)
    
    def setup(self, stage=None):
        """Execute `setup` and return values used by downstream logic."""
        return perturbation_pretraining_setup(self, stage=stage)

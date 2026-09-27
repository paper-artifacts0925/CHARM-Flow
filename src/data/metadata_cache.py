"""Module `data/metadata_cache.py`."""
import logging
import warnings

import anndata
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from typing import Literal, Set, Dict, List, Any, Tuple, Optional

from src.common.utils import safe_decode_array

class Singleton(type):
    """
    Ensures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Special method `__call__`."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""

    def __init__(
        self,
        h5_path: str,
        pert_col: str = None,
        cell_type_key: str = None,
        control_pert: str = None,
        batch_col: str = None,
    ):
        """
        Args:
            h5_path: Path to the .h5ad or .h5 file
            pert_col: obs column name for perturbation
            cell_type_key: obs column name for cell type
            control_pert: the perturbation to treat as control
            batch_col: obs column name for batch/plate
        """

        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]

            # -- Categories --
            if pert_col is not None:
                self.pert_categories = safe_decode_array(obs[pert_col]["categories"][:])
            
            if cell_type_key is not None:
                self.cell_type_categories = safe_decode_array(
                    obs[cell_type_key]["categories"][:]
                )

            # -- Batch: handle categorical vs numeric storage --
            if batch_col is not None:
                
                batch_ds = obs[batch_col]
                if "categories" in batch_ds:
                    self.batch_is_categorical = True
                    self.batch_categories = safe_decode_array(batch_ds["categories"][:])
                    self.batch_codes = batch_ds["codes"][:].astype(np.int32)
                else:
                    # only for replogle
                    assert "replogle" in h5_path
                    self.batch_is_categorical = False
                    raw = batch_ds[:]
                    self.batch_categories = np.array(np.sort(list(set(raw))).astype(str))
                    assert (self.batch_categories == (np.arange(len(self.batch_categories)) + 1).astype(str)).all()
                    self.batch_codes = raw.astype(np.int32) - 1

            # -- Codes for pert & cell type --
            if pert_col is not None:
                self.pert_codes = obs[pert_col]["codes"][:].astype(np.int32)
                self.n_cells = len(self.pert_codes)
            if cell_type_key is not None:
                self.cell_type_codes = obs[cell_type_key]["codes"][:].astype(np.int32)
                self.n_cells = len(self.cell_type_codes)

            # -- Control mask & counts --
            if control_pert is not None:
                idx = np.where(self.pert_categories == control_pert)[0]
                if idx.size == 0:
                    raise ValueError(
                        f"control_pert='{control_pert}' not found in {pert_col} categories"
                    )
                self.control_pert_code = int(idx[0])
                self.control_mask = self.pert_codes == self.control_pert_code


    def get_batch_names(self, indices: np.ndarray) -> np.ndarray:
        """Return batch labels for the provided cell indices."""
        return self.batch_categories[indices]

    def get_cell_type_names(self, indices: np.ndarray) -> np.ndarray:
        """Return cell‐type labels for the provided cell indices."""
        return self.cell_type_categories[indices]

    def get_pert_names(self, indices: np.ndarray) -> np.ndarray:
        """Return perturbation labels for the provided cell indices."""
        return self.pert_categories[indices]


class GlobalH5MetadataCache(metaclass=Singleton):
    """
    Singleton managing a shared dict of H5MetadataCache instances.
    Keys by h5_path only (same as before).
    """

    def __init__(self):
        """Special method `__init__`."""
        self._cache: Dict[str, H5MetadataCache] = {}

    def register_covariate_dict(
        self, 
        pert_dict: Optional[Dict[str, int]] = None,
        batch_dict: Optional[Dict[str, int]] = None,
        cell_type_dict: Optional[Dict[str, int]] = None,
    ):
        """Execute `register_covariate_dict` and return values used by downstream logic."""
        self.pert_dict = pert_dict
        self.batch_dict = batch_dict
        self.cell_type_dict = cell_type_dict

    def get_cache(
        self,
        h5_path: str,
        pert_col: str = None,
        cell_type_key: str = None,
        control_pert: str = None,
        batch_col: str = None,
    ) -> H5MetadataCache:
        """
        If a cache for this file doesn’t yet exist, create it with the
        given parameters; otherwise return the existing one.
        """
        if h5_path not in self._cache:
            self._cache[h5_path] = H5MetadataCache(
                h5_path, pert_col, cell_type_key, control_pert, batch_col
            )
        return self._cache[h5_path]

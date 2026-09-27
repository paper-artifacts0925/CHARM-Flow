"""Module `data/file_handle.py`."""
import os
from tqdm import tqdm
import logging
from pathlib import Path
from omegaconf import DictConfig
import glob
import re
import functools
from functools import partial
from typing import Literal, Set, Dict, Any, Tuple
import atexit
import threading
import h5py
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class H5Store:
    """
    Process-safe, LRU-capped cache of h5py.File handles (read-only).
    - Never shares open handles across processes (DDP ranks / workers).
    - Lazily opens per process/worker and reuses within that process.
    """
    def __init__(self, 
        dataset_path_map: Dict[str, str], 
        max_open: int = 16, 
        #locking: bool | None = False
    ):
        """Special method `__init__`."""
        self.dataset_path_map = dict(dataset_path_map)
        self.max_open = int(max_open)
        self._handles: "OrderedDict[str, h5py.File]" = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        ## Read-only, yet still enables HDF5 file locking (esp. for NFS/Lustre/GPFS)
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        atexit.register(self.close_all)

    def _ensure_process_local(self):
        """Execute `_ensure_process_local` and return values used by downstream logic."""
        current_pid = os.getpid()
        if current_pid != self._pid:
            # Close any inherited file handles before clearing
            # Now clear and reset PID
            self._handles.clear()
            self._pid = current_pid

    def _open(self, name: str) -> h5py.File:
        """Execute `_open` and return values used by downstream logic."""
        path = self.dataset_path_map[name]

        kwargs = {}
        return h5py.File(path, "r", rdcc_nbytes=0, rdcc_nslots=0, rdcc_w0=1.0, driver="sec2", libver='latest', **kwargs)
        

    def dataset_file(self, name: str) -> h5py.File:
        """
        Dataset file.

        :param name: Name key used to select behavior.
        :return: Computed output(s) for this function.
        """
        self._ensure_process_local()
        with self._lock:
            f = self._handles.pop(name, None)
            if f is not None:
                try:
                    if f.id and f.id.valid:
                        self._handles[name] = f  # mark MRU
                        return f
                except Exception:
                    pass
            f = self._open(name)
            self._handles[name] = f
            while len(self._handles) > self.max_open:
                logger.warning("removing file handles, please increase max_open_files")
                old_name, old_f = self._handles.popitem(last=False)
                try:
                    if old_f.id and old_f.id.valid:
                        old_f.close()
                except Exception:
                    pass
            return f

    def close_all(self):
        """Execute `close_all` and return values used by downstream logic."""
        with self._lock:
            for _, f in list(self._handles.items()):
                try:
                    if f.id and f.id.valid:
                        f.close()
                except Exception:
                    pass
            self._handles.clear()

    # Ensure we never pickle open file handles into workers/ranks
    def __getstate__(self):
        """Special method `__getstate__`."""
        d = self.__dict__.copy()
        d["_handles"] = OrderedDict()
        d["_pid"] = os.getpid()
        return d

    def __setstate__(self, d):
        """Special method `__setstate__`."""
        self.__dict__.update(d)
        self._handles = OrderedDict()
        self._pid = os.getpid()


__all__ = ["H5Store"]

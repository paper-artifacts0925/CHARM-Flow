"""Module `common/utils.py`."""
import os
import sys
import warnings
import random
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from tqdm import tqdm
import logging
import math
import torch as th

import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font_properties = font_manager.FontProperties(family="Times New Roman", style="normal", )
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib import rcParams
from sklearn.manifold import TSNE
import seaborn as sns

def get_short_dsname(ds_name):
    """Execute `get_short_dsname` and return values used by downstream logic."""
    if "tahoe" in ds_name.lower():
        return "tahoe100m"
    if "pbmc" in ds_name.lower():
        return "pbmc"
    if "replogle" in ds_name.lower():
        return "replogle"
    if "cellxgene" in ds_name.lower():
        return "cellxgene"
    logging.getLogger(__name__).error("Unknown dataset name: %s", ds_name)
    assert 0

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_ratio: float = 0.0,
    plateau_ratio: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        min_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum ratio a learning rate would decay to.
        plateau_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio for plateau phase.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        """Execute `lr_lambda` and return values used by downstream logic."""
        plateau_steps = int(plateau_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + plateau_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps - plateau_steps) / float(
            max(1, num_training_steps - num_warmup_steps - plateau_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return max(
            min_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def setup_loggings(cfg):
    """
    Setup loggings.

    :param cfg: Runtime configuration object.
    :return: Computed output(s) for this function.
    """
    import torch
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("Configuration args")
    logger.info(cfg)

    # compute
    computed_total_batch_size = (
        cfg.trainer.num_nodes * cfg.optimization.micro_batch_size
    )
    computed_total_batch_size *= torch.cuda.device_count()
    logging.info(
        f"Training with {cfg.trainer.num_nodes} nodes "
        f"micro-batch size {cfg.optimization.micro_batch_size} "
        f"total batch size {computed_total_batch_size} "
        f"and {torch.cuda.device_count()} devices per-node"
    )

    # set save directory path
    cfg.save_dir_path = os.path.join(cfg.trainer.default_root_dir, cfg.run_name)

    return logger

def safe_decode_array(arr) -> np.ndarray: 
    """
    Decode any byte-strings in arr to UTF-8 and cast all entries to Python str. 

    Refer to https://github.com/ArcInstitute/cell-load/blob/d96d1abcde360b9695cf819a7cfc40e9c56983d1/src/cell_load/utils/data_utils.py#L114
    Args: arr: array-like of bytes or other objects 
    Returns: np.ndarray[str]: decoded strings 
    """ 
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            # decode bytes, ignoring errors 
            decoded.append(x.decode("utf-8", errors="ignore")) 
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Gather timestep-indexed values and broadcast to a target tensor shape.

    :param arr: 1-D numpy array of per-timestep values.
    :param timesteps: Tensor of indices into `arr`.
    :param broadcast_shape: Output shape with matching batch dimension.
    :return: Broadcast tensor aligned with `broadcast_shape`.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def maybe_add_mask(scores: th.Tensor, attn_mask=None):
    """
    Apply additive attention mask if an attention mask is provided.

    :param scores: Attention score tensor.
    :param attn_mask: Optional additive mask tensor.
    :return: Masked or original score tensor.
    """
    return scores if attn_mask is None else scores + attn_mask


def reshape_concat_to_tokens(t: th.Tensor, gene_dim: int = 2000):
    """
    Convert concatenated gene features into token-major layout.

    :param t: Input tensor shaped [B, 1, T].
    :param gene_dim: Number of genes per token block.
    :return: Tensor shaped [B, gene_dim, X].
    """
    squeezed = t.squeeze(1)
    batch_size, total_dim = squeezed.shape
    assert total_dim % gene_dim == 0
    token_dim = total_dim // gene_dim
    reshaped = squeezed.view(batch_size, token_dim, gene_dim)
    return reshaped.transpose(1, 2)


def reshape_tokens_to_concat(t: th.Tensor, gene_dim: int = 2000):
    """
    Convert token-major layout back to concatenated gene features.

    :param t: Input tensor shaped [B, gene_dim, X].
    :param gene_dim: Number of genes (kept for API compatibility).
    :return: Tensor shaped [B, 1, gene_dim * X].
    """
    _ = gene_dim
    batch_size, num_genes, token_dim = t.shape
    transposed = t.transpose(1, 2).contiguous()
    return transposed.view(batch_size, 1, num_genes * token_dim)

"""Sampling output helpers split from rawdata_diffusion_sample.py (logic-preserving)."""

import time
from pathlib import Path
from typing import Optional

import anndata


def save_adata(pred_adata, true_adata, cfg, logger, timestamp: Optional[str] = None):

    # Set up output directory
    """
    Save adata.

    :param pred_adata: Input `pred_adata` value.
    :param true_adata: Input `true_adata` value.
    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param timestamp: Input `timestamp` value.
    :return: Computed output(s) for this function.
    """
    if cfg.sampling.output_dir is None:
        output_dir = Path(cfg.save_dir_path) / "samples"
    else:
        output_dir = Path(cfg.sampling.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_format = cfg.sampling.output_format.lower()

    pred_path = output_dir / f"diffusion_predict_{timestamp}.h5ad"
    pred_adata.write_h5ad(pred_path)
    logger.info(f"Samples saved to: {pred_path}")
    true_path = output_dir / f"diffusion_true_{timestamp}.h5ad"
    true_adata.write_h5ad(true_path)
    logger.info(f"Truths saved to: {true_path}")
    return pred_path, true_path

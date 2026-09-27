"""Sampling utility helpers used by sampling entrypoints."""

import torch


def setup_device(cfg, logger):
    """Setup device for sampling."""
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")
    return device

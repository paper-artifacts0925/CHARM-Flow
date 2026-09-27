"""Module `common/paths.py`."""

import os
import sys
import yaml
import logging
from pathlib import Path

# Prefer refactored config path; keep fallback for compatibility.
_cfg_candidates = [
    Path("configs/path/trixie_path.yaml"),
    Path("configs/path/path.yaml"),
]
_cfg_path = next((p for p in _cfg_candidates if p.exists()), None)
if _cfg_path is None:
    _paths = {}
else:
    with open(_cfg_path, "r") as f:
        _paths = yaml.safe_load(f) or {}
logger = logging.getLogger(__name__)

CELLFLOW_DATA_CACHE_DIR = None
CELLFLOW_CKPT_SAVE_DIR = None
STATE_CKPT_CACHE_DIR = None

if STATE_CKPT_CACHE_DIR is None:
    STATE_INFERENCE_CACHE_DIR = None
    STATE_INFERENCE_PROTEIN_EMBED_DIR = None
else:
    STATE_INFERENCE_CACHE_DIR = os.path.join(STATE_CKPT_CACHE_DIR, "cache_pbmc")
    logger.warning("temporary state inference caching dir is set to: %s", STATE_INFERENCE_CACHE_DIR)
    STATE_INFERENCE_PROTEIN_EMBED_DIR = os.path.join(STATE_CKPT_CACHE_DIR, "protein_embeddings.pt")
    logger.warning(
        "see STATE protein embedding under: %s; consult the STATE repository "
        "documentation for the original model location",
        STATE_INFERENCE_PROTEIN_EMBED_DIR,
    )

WANDB_LOGGING_DIR = _paths.get("wandb", {}).get("logging_dir")
if WANDB_LOGGING_DIR is not None:
    logger.warning("log wandb to: %s", WANDB_LOGGING_DIR)

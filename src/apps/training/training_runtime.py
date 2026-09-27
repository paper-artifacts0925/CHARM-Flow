"""Training runtime utilities split from rawdata_diffusion_training.py (logic-preserving)."""

import os
import sys
import logging

import hydra
import omegaconf
import pytorch_lightning as pl

# local imports
exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)

from src.apps.training.training_runtime_helpers import build_callbacks, build_profiler, resolve_trainer_strategy

def setup_trainer(cfg):
    """
    Setup trainer.

    :param cfg: Runtime configuration object.
    :return: Computed output(s) for this function.
    """
    trainer_logger = hydra.utils.instantiate(cfg.lightning.logger)
    if getattr(cfg, "disable_logger", False):
        trainer_logger = False
    strategy = resolve_trainer_strategy(cfg)
    callbacks = build_callbacks(cfg, trainer_logger)

    trainer_kwargs = omegaconf.OmegaConf.to_container(cfg.trainer, resolve=True)
    profiler = build_profiler(cfg)

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        plugins=[],
        strategy=strategy,
        logger=trainer_logger,
        profiler=profiler,
    )

    return trainer

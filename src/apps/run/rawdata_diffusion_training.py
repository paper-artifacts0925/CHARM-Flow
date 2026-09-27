"""Training entrypoint split from rawdata_diffusion_training.py (logic-preserving)."""

import os
import sys
import logging
import hydra
import omegaconf
import pytorch_lightning as pl

# local imports
exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)


from src.common.utils import setup_loggings
from src.apps.training.training_datamodule_builder import build_datamodule, populate_covariate_cfg
from src.apps.training.training_model_builder import (
    build_model,
    maybe_compare_model_representation,
    maybe_load_and_patch_checkpoint_model,
    maybe_reinitialize_from_scratch,
)
from src.apps.training.training_pipeline import run_pipeline
from src.apps.training.training_runtime import setup_trainer

@hydra.main(version_base=None, config_path="../../../configs", config_name="rawdata_diffusion_training")
def main(cfg):
    """
    Launch supervised fine-tuning using a hydra config, for protein classification
    """
    omegaconf.OmegaConf.resolve(cfg)

    logger = setup_loggings(cfg)

    if cfg.model.ckpt_path:
        logger.info(f"Resuming from checkpoint {cfg.model.ckpt_path}. ")

    pl.seed_everything(cfg.optimization.seed)

    trainer = setup_trainer(cfg)

    datamodule = build_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, datamodule)

    model = build_model(cfg, logger, datamodule)
    maybe_compare_model_representation(cfg, logger, datamodule)
    model = maybe_load_and_patch_checkpoint_model(cfg, model, logger)
    maybe_reinitialize_from_scratch(cfg, model, logger)
    logger.info("Model:\n%s", model)
    run_pipeline(trainer, model, datamodule, cfg, logger)

if __name__ == "__main__":
    main()

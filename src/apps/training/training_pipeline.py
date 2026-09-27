"""Train/validate run pipeline split from rawdata_diffusion_training.py (logic-preserving)."""

import torch


def run_pipeline(trainer, model, datamodule, cfg, logger):
    """
    Run pipeline.

    :param trainer: Input `trainer` value.
    :param model: Model instance for forward/sampling/training.
    :param datamodule: Data module providing datasets and loaders.
    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Computed output(s) for this function.
    """
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n\n")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not getattr(cfg, "skip_final_validate", False):
            trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path,
        )

"""Setup helpers for sampling entrypoint."""

import torch

from src.data import data_module
from src.models.lightning.lightning_module import PlModel
from src.apps.training.training_model_checkpoint import (
    _override_covariate_embedding_paths,
    allow_omegaconf_checkpoint_unpickling,
)


def build_sampling_datamodule(cfg, logger):
    """
    Build sampling datamodule.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Requested object(s) for downstream use.
    """
    if cfg.data.data_name in ["PBMCFinetune"]:
        datamodule = data_module.PBMCPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.sampling.batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["Tahoe100mFinetune", "ReplogleFinetune"]:
        datamodule = data_module.Tahoe100mPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in [
        "Tahoe100mPBMCPretrain",
        "CellxGenePretrain",
        "PBMCReploglePretrain",
        "Tahoe100mPBMCReplogleCellxGenePretrain",
        "Tahoe100mPBMCCellxGenePretrain",
        "PBMCReplogleCellxGenePretrain",
        "ReplogleCellxGenePretrain",
        "Tahoe100mCellxGenePretrain",
    ]:
        datamodule = data_module.PerturbationPretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    else:
        assert not cfg.data.data_name.endswith("Finetune")
        datamodule = data_module.PretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.sampling.batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    datamodule.replace_pert_dict = cfg.cov_encoding.get("replace_pert_dict", False)
    datamodule.setup()
    return datamodule


def populate_covariate_cfg(cfg, datamodule):
    """Execute `populate_covariate_cfg` and return values used by downstream logic."""
    cfg.cov_encoding.num_pert = len(datamodule.pert_dict)
    cfg.cov_encoding.num_celltype = len(datamodule.cell_type_dict)
    cfg.cov_encoding.num_batch = len(datamodule.batch_dict)

    cfg.cov_encoding.pert_dict = datamodule.pert_dict
    cfg.cov_encoding.cell_type_dict = datamodule.cell_type_dict
    cfg.cov_encoding.batch_dict = datamodule.batch_dict
    cfg.model.dataset_dict = datamodule.original_dataset_name_list


def load_sampling_model(cfg, logger, datamodule):
    """
    Load sampling model.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param datamodule: Data module providing datasets and loaders.
    :return: Requested object(s) for downstream use.
    """
    allow_omegaconf_checkpoint_unpickling()
    try:
        ckpt = torch.load(
            cfg.model_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to inspect sampling checkpoint with safe weights-only "
            f"deserialization: {cfg.model_checkpoint_path}. Unsafe pickle "
            "fallback is disabled."
        ) from exc
    hparams = ckpt.get("hyper_parameters", {})

    #needs_cov = "cov_encoding_cfg" in hparams
    #needs_model = "model_cfg" in hparams
    #needs_opt = "optimizer_cfg" in hparams
    
    # using the training setting
    needs_cov = needs_model = needs_opt = True
    hparams["cov_encoding_cfg"]["celltype_encoding"] = cfg.cov_encoding.celltype_encoding
    hparams["cov_encoding_cfg"] = _override_covariate_embedding_paths(
        hparams["cov_encoding_cfg"], cfg.cov_encoding
    )

    try:
        model = PlModel.load_from_checkpoint(
            cfg.model_checkpoint_path,
            cov_encoding_cfg=hparams["cov_encoding_cfg"] if needs_cov else cfg.cov_encoding,
            model_cfg=hparams["model_cfg"] if needs_model else cfg.model,
            optimizer_cfg=hparams["optimizer_cfg"] if needs_opt else cfg.optimization,
            py_logger=logger,
            trainer_cfg=cfg.trainer,
            all_split_names=datamodule.all_split_names,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to construct sampling model with safe weights-only "
            f"deserialization: {cfg.model_checkpoint_path}. Unsafe pickle "
            "fallback is disabled."
        ) from exc
    return model

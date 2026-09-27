"""DataModule builders split from rawdata_diffusion_training.py (logic-preserving)."""

from src.data import data_module
from src.data.strict_nshot import maybe_build_strict_nshot_gate

def build_datamodule(cfg, logger):
    """
    Build datamodule.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Requested object(s) for downstream use.
    """
    if cfg.data.data_name in ["PBMCFinetune"]:
        datamodule = data_module.PBMCPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
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
    elif cfg.data.data_name in ["Tahoe100mPBMCReplogleCellxGenePretrain"]:
        datamodule = data_module.PerturbationPretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    else:
        assert 0

    datamodule.replace_pert_dict = cfg.cov_encoding.get("replace_pert_dict", False)
    datamodule.setup()
    datamodule.strict_nshot_gate = maybe_build_strict_nshot_gate(
        cfg.model,
        runtime_cell_type_dict=datamodule.cell_type_dict,
        runtime_perturbation_dict=datamodule.pert_dict,
    )
    if datamodule.strict_nshot_gate is not None:
        logger.warning(
            "Strict-N-shot index gate enabled: %s",
            datamodule.strict_nshot_gate.summary(),
        )
    evaluation_splits = getattr(cfg.data, "evaluation_splits", None)
    if evaluation_splits:
        evaluation_splits = [str(split) for split in evaluation_splits]
        unknown = set(evaluation_splits) - set(datamodule.all_split_names)
        if unknown:
            raise ValueError(f"Unknown evaluation splits: {sorted(unknown)}")
        datamodule.evaluation_split_names = evaluation_splits
    else:
        datamodule.evaluation_split_names = list(datamodule.all_split_names)
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

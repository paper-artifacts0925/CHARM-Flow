"""Checkpoint load and patch flows extracted from training_model_builder."""

from collections import defaultdict
from logging import getLogger
from typing import Any

import numpy as np
import torch
from numpy._core.multiarray import scalar as numpy_scalar
from omegaconf import DictConfig, ListConfig
from omegaconf.base import Box, Container, ContainerMetadata, Metadata, Node, SCMode
from omegaconf.nodes import (
    AnyNode,
    BooleanNode,
    BytesNode,
    EnumNode,
    FloatNode,
    IntegerNode,
    InterpolationResultNode,
    PathNode,
    StringNode,
    ValueNode,
)

from src.models.lightning.lightning_module import PlModel


def allow_omegaconf_checkpoint_unpickling():
    """PyTorch 2.6 defaults to weights_only=True and blocks unknown globals."""
    try:
        torch.serialization.add_safe_globals(
            [
                DictConfig,
                ListConfig,
                Box,
                Container,
                ContainerMetadata,
                Metadata,
                Node,
                SCMode,
                AnyNode,
                BooleanNode,
                BytesNode,
                EnumNode,
                FloatNode,
                IntegerNode,
                InterpolationResultNode,
                PathNode,
                StringNode,
                ValueNode,
                (Any, "typing.Any"),
                int,
                dict,
                list,
                defaultdict,
                numpy_scalar,
                np.dtype,
                np.dtypes.StrDType,
                getLogger,
            ]
        )
    except Exception:
        pass


def _override_covariate_embedding_paths(cov_cfg, runtime_cov_cfg):
    """Patch ckpt covariate path fields with runtime config values when provided."""
    if cov_cfg is None or runtime_cov_cfg is None:
        return cov_cfg
    path_keys = [
        "celltype_embedding_path",
        "gene_embedding_path",
        "pert_embedding_path",
        "drug_embedding_path",
        "replogle_gene_embedding_path",
    ]
    for key in path_keys:
        val = runtime_cov_cfg.get(key, None)
        if val is not None:
            cov_cfg[key] = val
    return cov_cfg


def load_plmodel_checkpoint(
    ckpt_path: str,
    strict: bool,
    logger,
    runtime_cfg=None,
    runtime_all_split_names=None,
):
    """Execute `load_plmodel_checkpoint` and return values used by downstream logic."""
    allow_omegaconf_checkpoint_unpickling()
    try:
        return PlModel.load_from_checkpoint(ckpt_path, strict=strict, weights_only=True)
    except Exception as exc:
        msg = str(exc)
        should_retry_compat = (
            ("weights_only" in msg)
            or ("Unsupported global" in msg)
            or ("_SpecialForm" in msg)
            or isinstance(exc, FileNotFoundError)
            or ("No such file or directory" in msg)
        )
        if not should_retry_compat:
            raise
        logger.warning("Checkpoint fast load failed; retrying with compatibility path-aware loader.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    cov_cfg = hparams.get("cov_encoding_cfg", None)
    model_cfg = hparams.get("model_cfg", None)
    optimizer_cfg = hparams.get("optimizer_cfg", None)
    trainer_cfg = hparams.get("trainer_cfg", None)
    all_split_names = hparams.get("all_split_names", None)

    if runtime_cfg is not None:
        cov_cfg = _override_covariate_embedding_paths(cov_cfg, runtime_cfg.get("cov_encoding", None))
        model_cfg = model_cfg if model_cfg is not None else runtime_cfg.get("model", None)
        optimizer_cfg = optimizer_cfg if optimizer_cfg is not None else runtime_cfg.get("optimization", None)
        trainer_cfg = runtime_cfg.get("trainer", trainer_cfg)
        if all_split_names is None:
            all_split_names = runtime_all_split_names

    return PlModel.load_from_checkpoint(
        ckpt_path,
        strict=strict,
        weights_only=False,
        cov_encoding_cfg=cov_cfg,
        model_cfg=model_cfg,
        optimizer_cfg=optimizer_cfg,
        py_logger=logger,
        trainer_cfg=trainer_cfg,
        all_split_names=all_split_names if all_split_names is not None else [],
    )


def _basic_init(module):
    """Execute `_basic_init` and return values used by downstream logic."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


def maybe_load_and_patch_checkpoint_model(cfg, model, logger):
    """
    Conditionally load and patch checkpoint model.

    :param cfg: Runtime configuration object.
    :param model: Model instance for forward/sampling/training.
    :param logger: Logger instance.
    :return: Computed output(s) for this function.
    """
    if not cfg.model.model_weight_ckpt_path:
        return model

    new_model = load_plmodel_checkpoint(
        cfg.model.model_weight_ckpt_path,
        strict=True,
        logger=logger,
        runtime_cfg=cfg,
        runtime_all_split_names=getattr(model, "all_split_names", None),
    )
    assert cfg.model.ckpt_path is None

    new_model.__dict__["_modules"]["model"].final_layer.apply(_basic_init)
    new_model.__dict__["_modules"]["model"].control_final_layer.apply(_basic_init)
    torch.nn.init.constant_(new_model.__dict__["_modules"]["model"].final_layer.adaLN_modulation[-1].weight, 0)
    torch.nn.init.constant_(new_model.__dict__["_modules"]["model"].final_layer.adaLN_modulation[-1].bias, 0)
    torch.nn.init.constant_(new_model.__dict__["_modules"]["model"].control_final_layer.adaLN_modulation[-1].weight, 0)
    torch.nn.init.constant_(new_model.__dict__["_modules"]["model"].control_final_layer.adaLN_modulation[-1].bias, 0)

    new_model.__dict__["_modules"]["cov_encoder"].pert_encoder = model.__dict__["_modules"]["cov_encoder"].pert_encoder
    if model.__dict__["cov_encoding_cfg"]["celltype_encoding"] != new_model.__dict__["cov_encoding_cfg"]["celltype_encoding"]:
        new_model.__dict__["_modules"]["cov_encoder"].celltype_encoder = model.__dict__["_modules"]["cov_encoder"].celltype_encoder
    if cfg.model.replace_batch_encoder:
        new_model.__dict__["_modules"]["cov_encoder"].batch_encoder = model.__dict__["_modules"]["cov_encoder"].batch_encoder
    model.__dict__["_modules"] = new_model.__dict__["_modules"]
    model.__dict__["_modules"]["model"].model_cfg = model.__dict__["model_cfg"]
    model.__dict__["_modules"]["cov_encoder"].cov_cfg = model.__dict__["cov_encoding_cfg"]
    model.model_cfg.ckpt_path = None

    if cfg.model.reinitial_all:
        model.model.initialize_weights()
        logger.info("Re-initialize all layers")
    if cfg.model.replace_2kgene_layer:
        model.model.replace_2kgene_layer(new_input_size=2000)
        logger.info("Replace 2k gene layers")
    if cfg.model.replace_1w2gene_layer:
        model.model.replace_2kgene_layer(new_input_size=12626)

    return model

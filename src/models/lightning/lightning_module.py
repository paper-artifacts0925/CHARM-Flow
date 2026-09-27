"""Lightning training module split from lightning_module (logic-preserving)."""

import csv
import copy
import gc
import json
import math
import os
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from pytorch_lightning.utilities import grad_norm
from transformers.trainer_pt_utils import get_parameter_names

from src.common.utils import get_short_dsname
from src.loss.source_transport_loss import source_transport_loss
from src.models.covariate_encoding import CovEncoder
from src.models.hungarian_flow.episode import episode_distribution_consistency

from src.models.lightning.lightning_factories import (
    create_diffusion,
    create_named_schedule_sampler,
    get_optimizer,
    model_init_fn,
)


def _responsibility_weighted_mmd(source_mmd, assignment):
    """Weight per-source MMD with detached cell-averaged responsibilities."""
    source_q = assignment.detach().mean(dim=1)
    return (source_q * source_mmd).sum(dim=-1).mean()


def _augment_parent_listwise_pds_result(
    lightning,
    *,
    result,
    prepared,
    batch,
    supervision_mask,
    parent_grouping,
):
    """Wire the opt-in Parent endpoint ranking loss without touching defaults."""

    from src.models.parent_residual_gene_dit.parent_listwise_integration import (
        augment_result_with_parent_listwise_pds,
    )

    raw_weight = getattr(
        getattr(lightning, "model_cfg", None),
        "parent_residual_parent_listwise_pds_weight",
        0.0,
    )
    try:
        disabled = float(raw_weight) == 0.0
    except (TypeError, ValueError, OverflowError):
        disabled = False
    if disabled:
        # The adapter returns before inspecting any other argument.  Keeping
        # these as None makes the default-path compatibility contract testable.
        augment_result_with_parent_listwise_pds(
            lightning,
            result=result,
            prepared=None,
            compact_group_ids=None,
            valid_mask=None,
        )
        return

    from src.models.parent_locked_residual.centering import (
        condition_group_ids as encode_condition_group_ids,
    )
    from src.models.parent_locked_residual.matched_integration import (
        condition_group_ids,
    )

    tuple_group_ids = condition_group_ids(
        batch,
        set_size=1,
        parent_grouping=parent_grouping,
    )
    if tuple_group_ids.ndim != 3 or tuple_group_ids.shape[1] != 1:
        raise ValueError("Parent listwise tuple group IDs must have shape [B,1,C]")
    compact_group_ids = encode_condition_group_ids(
        *tuple_group_ids.unbind(dim=-1)
    )
    augment_result_with_parent_listwise_pds(
        lightning,
        result=result,
        prepared=prepared,
        compact_group_ids=compact_group_ids,
        valid_mask=supervision_mask,
    )


def _augment_joint_parent_child_result(
    lightning,
    result,
    prepared,
    match,
    batch,
    supervision_mask,
    parent_grouping,
    parent_reference_loss,
):
    """Single opt-in wiring point for the joint Parent/Child objective."""

    from src.models.parent_residual_gene_dit.joint_parent_child_integration import (
        augment_result_with_joint_parent_child_objective,
    )

    augment_result_with_joint_parent_child_objective(
        lightning,
        result,
        prepared,
        match,
        batch,
        supervision_mask,
        parent_grouping,
        parent_reference_loss,
    )


class PlModel(pl.LightningModule):
    """Plmodel implementation used by the PerturbDiff pipeline."""

    def on_load_checkpoint(self, checkpoint):
        """Remember checkpoint progress for Trainer-free sampling runtimes."""

        has_global_step = "global_step" in checkpoint
        loaded_step = checkpoint.get("global_step", 0)
        if (
            isinstance(loaded_step, bool)
            or not isinstance(loaded_step, int)
            or loaded_step < 0
        ):
            raise ValueError(
                "checkpoint global_step must be a non-negative integer"
            )
        self._loaded_checkpoint_global_step = int(loaded_step)
        logger = getattr(self, "py_logger", None)
        if logger is not None:
            logger.info(
                "Loaded checkpoint inference global_step=%d source=%s",
                self._loaded_checkpoint_global_step,
                "checkpoint" if has_global_step else "missing_default_zero",
            )

    def _inference_child_guidance_step(self):
        """Use Trainer progress when attached, else checkpoint progress."""

        if getattr(self, "_trainer", None) is not None:
            return int(self.global_step)
        return int(getattr(self, "_loaded_checkpoint_global_step", 0))

    def __init__(self, cov_encoding_cfg, model_cfg, py_logger, optimizer_cfg, trainer_cfg, all_split_names):
        """
        Initialize the class instance.

        :param cov_encoding_cfg: Configuration object for this component.
        :param model_cfg: Model configuration.
        :param py_logger: Input `py_logger` value.
        :param optimizer_cfg: Optimizer configuration.
        :param trainer_cfg: Trainer configuration.
        :param all_split_names: Input `all_split_names` value.
        :return: None.
        """
        super().__init__()
        self.strict_loading = False
        self._loaded_checkpoint_global_step = 0
        self.cov_encoding_cfg = cov_encoding_cfg
        self.model_cfg = model_cfg
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.trainer_cfg = trainer_cfg
        self.all_split_names = all_split_names

        for split in self.all_split_names:
            setattr(self, f"validation_{split}_step_outputs", [])

        self.cov_encoder = CovEncoder(self.cov_encoding_cfg)
        self.gene_embedding = {}
        if self.model_cfg.use_gene_embedding:
            for gene_emb_file in self.cov_encoding_cfg.gene_embedding_path:
                # read pickle file
                with open(gene_emb_file, "rb") as f:
                    gene_emb = pickle.load(f)
                    # transform to tensor
                    gene_emb = {k: torch.tensor(v, dtype=torch.float32) for k, v in gene_emb.items()}
                    self.gene_embedding.update(gene_emb)
        else:
            self.gene_embedding = None
        self.gene_name_embedding_cache = {}

        self.model = model_init_fn(self.model_cfg, self.cov_encoding_cfg)
        self.anchor_bridge_enabled = (
            str(self.model_cfg.model_type).lower() == "anchor_bridge_cross_dit"
        )
        self.parent_residual_enabled = (
            str(self.model_cfg.model_type).lower()
            == "parent_residual_gene_dit"
        )
        self.parent_residual_strict_parent_only_enabled = bool(
            self.parent_residual_enabled
            and getattr(self.model, "strict_parent_only_enabled", False)
        )
        self.parent_residual_strict_direct_flow_enabled = bool(
            self.parent_residual_strict_parent_only_enabled
            and getattr(self.model, "strict_direct_flow_enabled", False)
        )

        self.parent_residual_matching_guidance_enabled = bool(
            self.parent_residual_enabled
            and getattr(self.model, "matching_guidance_enabled", True)
        )
        if self.parent_residual_enabled:
            self.py_logger.info(
                "parent_residual_child_context_mode=%s "
                "real_control_sampling_mode=%s strict_parent_only=%s",
                self.model.child_context_mode,
                self.model.real_control_sampling_mode,
                self.parent_residual_strict_parent_only_enabled,
            )
            self.py_logger.info(
                "parent_residual_matching_guidance_enabled=%d "
                "parent_residual_match_skipped=%d",
                int(self.parent_residual_matching_guidance_enabled),
                int(not self.parent_residual_matching_guidance_enabled),
            )
        if self.parent_residual_strict_parent_only_enabled:
            self.py_logger.info(
                "parent_residual_strict_parent_only=1 parent_residual_match_skipped=1"
            )
        if self.parent_residual_strict_direct_flow_enabled:
            self.py_logger.info(
                "parent_residual_strict_direct_flow=1 parent_residual_locked_flow=0"
            )
        self.parent_residual_locked_flow_enabled = bool(
            self.parent_residual_enabled
            and getattr(
                self.model_cfg,
                "parent_residual_locked_flow_enabled",
                False,
            )
        )
        self.parent_residual_child_mean_flow_enabled = bool(
            self.parent_residual_locked_flow_enabled
            and not self.parent_residual_strict_parent_only_enabled
            and getattr(
                self.model_cfg,
                "parent_residual_child_mean_flow_enabled",
                False,
            )
        )
        if bool(
            getattr(
                self.model_cfg,
                "parent_residual_child_mean_flow_enabled",
                False,
            )
        ) and not self.parent_residual_locked_flow_enabled:
            raise ValueError(
                "Child-mean flow requires parent_residual_locked_flow_enabled"
            )
        if self.parent_residual_child_mean_flow_enabled:
            if not bool(
                getattr(
                    self.model_cfg,
                    "parent_residual_matched_set_enabled",
                    False,
                )
            ):
                raise ValueError(
                    "Child-mean flow requires matched-set Child routing"
                )
            posterior_max = max(
                float(
                    getattr(
                        self.model_cfg,
                        "parent_residual_posterior_start_fraction",
                        1.0,
                    )
                ),
                float(
                    getattr(
                        self.model_cfg,
                        "parent_residual_posterior_end_fraction",
                        0.0,
                    )
                ),
            )
            if posterior_max != 0.0:
                raise ValueError(
                    "Child-mean flow requires target-free Router sources: set "
                    "parent_residual_posterior_start_fraction and "
                    "parent_residual_posterior_end_fraction to zero"
                )
        self.parent_locked_residual_flow = None
        if self.parent_residual_locked_flow_enabled:
            from src.models.parent_locked_residual import ParentLockedResidualFlow

            self.parent_locked_residual_flow = ParentLockedResidualFlow(
                sampling_steps=int(
                    getattr(self.model_cfg, "flow_sampling_steps", 50)
                ),
                time_scale=float(
                    getattr(self.model_cfg, "flow_time_scale", 1000.0)
                ),
                child_mean_flow_enabled=(
                    self.parent_residual_child_mean_flow_enabled
                ),
                child_mean_loss_weight=float(
                    getattr(
                        self.model_cfg,
                        "parent_residual_child_mean_loss_weight",
                        1.0,
                    )
                ),
                child_mean_endpoint_blend=float(
                    getattr(
                        self.model_cfg,
                        "parent_residual_child_mean_endpoint_blend",
                        1.0,
                    )
                ),
                common_velocity_loss_weight=float(
                    getattr(
                        self.model_cfg,
                        "parent_residual_common_velocity_loss_weight",
                        0.0,
                    )
                ),
            )
        self.parent_residual_matched_set_enabled = False
        if self.parent_residual_enabled:
            from src.models.parent_locked_residual.matched_integration import (
                initialize_matched_set_transport,
            )

            initialize_matched_set_transport(self)
        self.parent_residual_de_aware_enabled = False
        if self.parent_residual_enabled:
            from src.models.de_aware_residual.lightning_integration import (
                initialize_de_aware_residual,
            )

            initialize_de_aware_residual(self)
            from src.models.sparse_hurdle_de.lightning_integration import (
                initialize_end_to_end_sparse_hurdle,
            )

            initialize_end_to_end_sparse_hurdle(self)
        self.hungarian_flow_enabled = bool(
            getattr(self.model_cfg, "hungarian_flow_enabled", False)
            or str(self.model_cfg.model_type).lower() == "hungarian_flow_cross_dit"
        )
        self.cell_detr_enabled = bool(
            getattr(self.model_cfg, "cell_detr_enabled", False)
            or str(self.model_cfg.model_type).lower() == "cell_detr_cross_dit"
            or self.anchor_bridge_enabled
        )
        self.cell_detr_direct_enabled = bool(
            getattr(self.model_cfg, "cell_detr_direct_enabled", False)
            or str(self.model_cfg.model_type).lower() == "cell_detr_direct"
        )

        if self.cell_detr_direct_enabled:
            self.diffusion = None
            self.schedule_sampler = None
        else:
            self.diffusion = create_diffusion(self.model_cfg)
            sampler_name = getattr(self.model_cfg, "schedule_sampler", "uniform")
            self.schedule_sampler = create_named_schedule_sampler(
                sampler_name, self.diffusion
            )

        self.model_cfg.p_drop_cond = getattr(self.model_cfg, "p_drop_cond", 0.0)

        self.latent_control_learned_scoring = bool(getattr(self.model_cfg, "latent_control_learned_scoring", False))
        if self.latent_control_learned_scoring:
            num_clusters = int(getattr(self.model_cfg, "latent_control_num_cluster_embeddings", 64) or 64)
            embed_dim = int(getattr(self.model_cfg, "latent_control_cluster_embedding_dim", 16) or 16)
            self.latent_control_cluster_embedding = torch.nn.Embedding(num_clusters, embed_dim)
            self.latent_control_score = torch.nn.Sequential(
                torch.nn.Linear(embed_dim + 1, embed_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim, 1),
            )
        else:
            self.latent_control_cluster_embedding = None
            self.latent_control_score = None
        self.adaptive_anchor_enabled = bool(
            getattr(self.model_cfg, "adaptive_anchor_enabled", False)
        )
        if self.adaptive_anchor_enabled:
            parent_memory_mode = str(
                getattr(self.model_cfg, "adaptive_anchor_parent_memory_mode", "legacy")
            ).lower()
            parent_memory_num_patches = 0
            if bool(
                getattr(self.model_cfg, "adaptive_anchor_parent_kv_enabled", False)
            ) and parent_memory_mode == "gene_program":
                parent_memory_num_patches = int(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_parent_memory_num_patches",
                        16,
                    )
                )
            from src.models.adaptive_state_anchor import AdaptiveStateAnchor

            self.adaptive_state_anchor = AdaptiveStateAnchor(
                gene_dim=int(self.model_cfg.input_dim),
                num_perturbations=int(self.cov_encoding_cfg.num_pert),
                num_cell_types=int(self.cov_encoding_cfg.num_celltype),
                num_cluster_embeddings=int(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_num_cluster_embeddings",
                        64,
                    )
                ),
                hidden_dim=int(
                    getattr(self.model_cfg, "adaptive_anchor_hidden_dim", 128)
                ),
                offset_rank=int(
                    getattr(self.model_cfg, "adaptive_anchor_offset_rank", 16)
                ),
                split_factor=int(
                    getattr(self.model_cfg, "adaptive_anchor_split_factor", 2)
                ),
                offset_scale=float(
                    getattr(self.model_cfg, "adaptive_anchor_offset_scale", 0.1)
                ),
                objectness_threshold=float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_objectness_threshold",
                        0.5,
                    )
                ),
                memory_num_patches=parent_memory_num_patches,
                memory_program_rank=int(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_parent_memory_program_rank",
                        4,
                    )
                ),
            )
        else:
            self.adaptive_state_anchor = None


        self.match_fm_enabled = bool(getattr(self.model_cfg, "match_fm_enabled", False))
        if self.match_fm_enabled:
            from src.models.match_fm import PerturbationConditionedPrior

            self.match_fm_prior = PerturbationConditionedPrior(
                input_dim=int(self.model_cfg.input_dim),
                num_perturbations=int(self.cov_encoding_cfg.num_pert),
                num_cell_types=int(self.cov_encoding_cfg.num_celltype),
                num_cluster_embeddings=int(getattr(self.model_cfg, "match_fm_num_cluster_embeddings", 64) or 64),
                hidden_dim=int(getattr(self.model_cfg, "match_fm_prior_hidden_dim", 64) or 64),
                empirical_prior_scale=float(
                    getattr(self.model_cfg, "match_fm_empirical_prior_scale", 1.0)
                ),
                semantic_context_dim=int(self.model_cfg.input_dim),
                parent_context_dim=int(
                    getattr(self.model_cfg, "adaptive_anchor_hidden_dim", 128)
                ),
                use_cluster_identity=bool(
                    getattr(self.model_cfg, "match_fm_use_cluster_identity", False)
                ),
            )
        else:
            self.match_fm_prior = None
        self.match_fm_lagged_prior_enabled = self.match_fm_enabled and bool(
            getattr(self.model_cfg, "match_fm_lagged_prior", False)
        )
        if getattr(self, "match_fm_lagged_prior_enabled", False):
            self.match_fm_prior_teacher = copy.deepcopy(self.match_fm_prior)
            self.match_fm_prior_teacher.requires_grad_(False)
            self.match_fm_prior_teacher.eval()
        else:
            self.match_fm_prior_teacher = None
        pca_basis_path = getattr(self.model_cfg, "match_fm_pca_basis_path", None)
        if pca_basis_path:
            payload = np.load(str(pca_basis_path), allow_pickle=False)
            components = np.asarray(payload["components"], dtype=np.float32)
            if components.ndim != 2 or components.shape[1] != int(self.model_cfg.input_dim):
                raise ValueError(
                    "MATCH-FM PCA components must have shape [D, model.input_dim]"
                )
            pca_basis = torch.from_numpy(components)
            self.py_logger.info(
                "Loaded MATCH-FM control PCA basis %s with shape %s",
                pca_basis_path,
                tuple(pca_basis.shape),
            )
        else:
            pca_basis = torch.empty(0, int(self.model_cfg.input_dim))
        self.register_buffer("match_fm_pca_basis", pca_basis, persistent=True)

        self.online_anchor_em_enabled = (
            self.adaptive_anchor_enabled
            and bool(getattr(self.model_cfg, "online_anchor_em_enabled", False))
        )
        if self.online_anchor_em_enabled:
            from src.models.online_anchor_em import OnlineAnchorEM

            max_leaves = int(
                getattr(self.model_cfg, "online_anchor_em_max_leaves", 8)
            )
            split_factor = int(
                getattr(self.model_cfg, "adaptive_anchor_split_factor", 3)
            )
            online_parent_ids = torch.arange(max_leaves).repeat_interleave(
                split_factor
            )
            online_child_ids = torch.arange(split_factor).repeat(max_leaves)
            feature_dim = int(
                getattr(self.model_cfg, "online_anchor_em_feature_dim", 64)
            )
            if not pca_basis.numel() or pca_basis.shape[0] != feature_dim:
                raise ValueError(
                    "online anchor EM requires a MATCH-FM PCA basis whose row "
                    f"count equals feature_dim={feature_dim}"
                )
            self.online_anchor_em = OnlineAnchorEM(
                num_cells=int(
                    getattr(self.model_cfg, "online_anchor_em_num_cells", 643413)
                ),
                num_keys=int(self.cov_encoding_cfg.num_celltype)
                * int(self.cov_encoding_cfg.num_pert),
                parent_ids=online_parent_ids,
                child_ids=online_child_ids,
                feature_dim=feature_dim,
                burn_in_steps=int(getattr(self.model_cfg, "online_anchor_em_burn_in_steps", 4000)),
                history_ramp_steps=int(getattr(self.model_cfg, "online_anchor_em_history_ramp_steps", 20000)),
                history_max_weight=float(getattr(self.model_cfg, "online_anchor_em_history_max_weight", 0.35)),
                history_stale_half_life_steps=float(getattr(self.model_cfg, "online_anchor_em_stale_half_life_steps", 50000.0)),
                occupancy_half_life_cells=float(getattr(self.model_cfg, "online_anchor_em_occupancy_half_life_cells", 256.0)),
                prototype_half_life_cells=float(getattr(self.model_cfg, "online_anchor_em_prototype_half_life_cells", 128.0)),
                activate_threshold=float(getattr(self.model_cfg, "online_anchor_em_activate_threshold", 0.03)),
                deactivate_threshold=float(getattr(self.model_cfg, "online_anchor_em_deactivate_threshold", 0.005)),
                activate_patience=int(getattr(self.model_cfg, "online_anchor_em_activate_patience", 3)),
                deactivate_patience=int(getattr(self.model_cfg, "online_anchor_em_deactivate_patience", 20)),
                min_active_per_parent=1,
            )
        else:
            self.online_anchor_em = None
        self._pending_online_anchor_em_update = None

        self.match_fm_ema_teacher_enabled = self.match_fm_enabled and bool(
            getattr(self.model_cfg, "match_fm_ema_teacher", False)
        )
        if self.match_fm_ema_teacher_enabled:
            self.match_fm_teacher = copy.deepcopy(self.model)
            self.match_fm_teacher.requires_grad_(False)
            self.match_fm_teacher.eval()
        else:
            self.match_fm_teacher = None
        ema_start_step = int(getattr(self.model_cfg, "match_fm_ema_start_step", 0) or 0)
        self._match_fm_teacher_initialized = (
            self.match_fm_ema_teacher_enabled and ema_start_step <= 0
        )

        self._last_logged_batch_start_time = time.monotonic()
        self._curriculum_resource_time = time.monotonic()
        self._curriculum_resource_step = 0
        self._previous_top1_assignments = None
        self._validation_source_records = []
        self.validation_step_outputs = [] 

        blur = self.optimizer_cfg.get("blur", 0.05)
        self.loss_fn = SamplesLoss(loss="energy", blur=blur)

        self.save_hyperparameters()  # Save hparams for checkpointing
    
    def log_data(self,
                log_dict,
                train=True):
        """
        Log data.

        :param log_dict: Dictionary containing mapped values.
        :param train: Whether to run in training mode.
        :return: Computed output(s) for this function.
        """
        if train:
            self.log_dict(
                    log_dict,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=self.optimizer_cfg.micro_batch_size,
                    logger=True,
                    sync_dist=True,
                )
        else:
            self.log_dict(
                    log_dict,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=self.optimizer_cfg.micro_batch_size,
                    logger=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Index value used for lookup or slicing.
        :return: Computed output(s) for this function.
        """
        loss_dict = self._compute_loss(batch)
        loss = (loss_dict["loss"] * loss_dict["weights"]).mean()
        if not torch.isfinite(loss):
            nonfinite = {}
            for key, value in loss_dict.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    finite = value.detach()[torch.isfinite(value.detach())]
                    nonfinite[key] = {
                        "shape": tuple(value.shape),
                        "finite_count": int(finite.numel()),
                        "min": float(finite.min().cpu()) if finite.numel() else None,
                        "max": float(finite.max().cpu()) if finite.numel() else None,
                    }
            raise FloatingPointError(f"Non-finite loss components: {nonfinite}")

        log_dict = {"training_loss_step": loss}
        diagnostic_log = {}

        for k,v in loss_dict.items():
            if k.startswith("dataset_loss_"):
                log_dict[k] = v
            if k.startswith(("latent_control_", "adaptive_anchor_", "parent_kv_", "source_", "match_fm_", "hungarian_flow_", "cell_detr_", "anchor_bridge_", "parent_residual_")):
                log_dict[k] = v
            if k.startswith(("em/", "candidate/", "source/")):
                diagnostic_log[k] = v

        assert self.model.model_name in {"Cross_DiT", "Cell_DETR_Direct"}
        log_dict["training_Pert_loss_step"] = loss_dict["loss1"]
        log_dict["training_MMD_Pert_loss_step"] = loss_dict["mmd1"]

        self.log_data(log_dict, train=True)
        if diagnostic_log:
            self.log_dict(
                diagnostic_log,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=self.optimizer_cfg.micro_batch_size,
                logger=True,
                sync_dist=True,
            )

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Index value used for lookup or slicing.
        :param dataloader_idx: Index value used for lookup or slicing.
        :return: Computed output(s) for this function.
        """
        split = self.all_split_names[dataloader_idx]
        self._active_validation_split = split
        self._active_validation_batch_idx = batch_idx

        loss_dict = self._compute_loss(batch)
        loss = (loss_dict["loss"] * loss_dict["weights"]).mean()

        log_dict = {f"validation_{split}_loss": loss}

        for k,v in loss_dict.items():
            if k.startswith("dataset_loss_"):
                log_dict[k] = v
            if k.startswith(("latent_control_", "adaptive_anchor_", "parent_kv_", "source_", "match_fm_", "hungarian_flow_", "cell_detr_", "anchor_bridge_", "parent_residual_")):
                log_dict[k] = v

        assert self.model.model_name in {"Cross_DiT", "Cell_DETR_Direct"}
        log_dict[f"validation_{split}_Pert_loss"] = loss_dict["loss1"]

        log_dict[f"validation_{split}_MMD_Pert_loss"] = loss_dict["mmd1"]
        self.log_data(log_dict, train=False)

        self.validation_step_outputs.append({f"validation_{split}_loss": loss})
        getattr(self, f"validation_{split}_step_outputs").append({f"validation_{split}_loss": loss})
        return {f"validation_{split}_loss": loss}

    def on_validation_epoch_end(self):
        """Execute `on_validation_epoch_end` and return values used by downstream logic."""
        for split in self.all_split_names:
            arr = getattr(self, f"validation_{split}_step_outputs")
            if len(arr) > 0:
                vals = torch.stack([o[f"validation_{split}_loss"] for o in arr])
                mean_val = vals.mean()
                self.log(
                    f"validation_{split}_loss_epoch",
                    mean_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            getattr(self, f"validation_{split}_step_outputs").clear()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Execute `on_train_batch_end` and return values used by downstream logic."""
        if self._pending_online_anchor_em_update is not None:
            fit_loop = getattr(self.trainer, "fit_loop", None)
            should_accumulate = bool(
                fit_loop is not None
                and hasattr(fit_loop, "_should_accumulate")
                and fit_loop._should_accumulate()
            )
            if not should_accumulate:
                self.online_anchor_em.commit_update(
                    self._pending_online_anchor_em_update,
                    synchronize=True,
                )
                self._pending_online_anchor_em_update = None
        if batch_idx > 0 and batch_idx % self.trainer_cfg.log_every_n_steps == 0:
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            self._last_logged_batch_start_time = time.monotonic()
            time_per_step = elapsed_time / self.trainer_cfg.log_every_n_steps
            self.log("sec/step", time_per_step, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        if self.match_fm_ema_teacher_enabled:
            update_every = max(int(getattr(self.model_cfg, "match_fm_ema_update_every", 1)), 1)
            start_step = int(getattr(self.model_cfg, "match_fm_ema_start_step", 0) or 0)
            if int(self.global_step) >= start_step and int(self.global_step) % update_every == 0:
                decay = float(getattr(self.model_cfg, "match_fm_ema_decay", 0.999))
                with torch.no_grad():
                    first_update = not self._match_fm_teacher_initialized
                    for teacher_param, student_param in zip(
                        self.match_fm_teacher.parameters(), self.model.parameters()
                    ):
                        if first_update:
                            teacher_param.copy_(student_param.detach())
                        else:
                            teacher_param.mul_(decay).add_(
                                student_param.detach(), alpha=1.0 - decay
                            )
                    for teacher_buffer, student_buffer in zip(
                        self.match_fm_teacher.buffers(), self.model.buffers()
                    ):
                        teacher_buffer.copy_(student_buffer)
                    self._match_fm_teacher_initialized = True
        if getattr(self, "match_fm_lagged_prior_enabled", False):
            update_every = max(
                int(getattr(self.model_cfg, "match_fm_prior_ema_update_every", 1)), 1
            )
            start_step = int(
                getattr(self.model_cfg, "match_fm_prior_ema_start_step", 0) or 0
            )
            if int(self.global_step) >= start_step and int(self.global_step) % update_every == 0:
                decay = float(getattr(self.model_cfg, "match_fm_prior_ema_decay", 0.99))
                with torch.no_grad():
                    for teacher_param, student_param in zip(
                        self.match_fm_prior_teacher.parameters(),
                        self.match_fm_prior.parameters(),
                    ):
                        teacher_param.mul_(decay).add_(
                            student_param.detach(), alpha=1.0 - decay
                        )
                    for teacher_buffer, student_buffer in zip(
                        self.match_fm_prior_teacher.buffers(),
                        self.match_fm_prior.buffers(),
                    ):
                        teacher_buffer.copy_(student_buffer)

    def _encode_covariates(self, batch):
        """Execute `_encode_covariates` and return values used by downstream logic."""
        cov_reprs = self.cov_encoder(batch["cov_pert"], batch["cov_celltype"], batch["cov_batch"])

        return cov_reprs

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Avoid recursively walking BxG gene-name metadata for Cell-DETR."""
        if not (
            self.cell_detr_enabled
            or self.cell_detr_direct_enabled
            or self.parent_residual_enabled
        ) or not isinstance(batch, dict):
            return super().transfer_batch_to_device(
                batch, device, dataloader_idx
            )
        return {
            key: (
                value.to(device=device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in batch.items()
        }

    def _compute_loss(self, batch):
        """Compute a training loss value, optionally with latent control-cluster matching."""
        if self.cell_detr_direct_enabled:
            return self._compute_cell_detr_direct_loss(batch)
        if self.parent_residual_enabled:
            return self._compute_parent_residual_gene_dit_loss(batch)
        if self.cell_detr_enabled:
            return self._compute_cell_detr_loss(batch)
        if self.hungarian_flow_enabled:
            return self._compute_hungarian_flow_loss(batch)
        if str(self.model_cfg.model_type).lower() == "source_cross_dit":
            return self._compute_source_transport_loss(batch)
        if "latent_control_emb" in batch and getattr(self.model_cfg, "latent_control_mixture", False):
            return self._compute_latent_control_mixture_loss(batch)
        return self._compute_base_loss(batch)

    @staticmethod
    def _cell_detr_control_inputs(batch, device):
        """Resolve either the unique-bank path or the legacy per-cell path."""
        unique_keys = {
            "cell_detr_control_bank",
            "cell_detr_control_prior",
            "cell_detr_control_mask",
            "cell_detr_bank_inverse",
        }
        present_unique = unique_keys.intersection(batch)
        if present_unique:
            missing = sorted(unique_keys.difference(batch))
            if missing:
                raise ValueError(
                    "incomplete Cell-DETR unique control bank; missing "
                    + ", ".join(missing)
                )
            token_mask = batch.get("cell_detr_control_token_mask")
            if token_mask is not None:
                token_mask = token_mask.to(device=device, dtype=torch.bool)
            return (
                batch["cell_detr_control_bank"].to(device),
                batch["cell_detr_control_mask"].to(device).bool(),
                batch["cell_detr_control_prior"].to(device),
                batch["cell_detr_bank_inverse"].to(device).long(),
                token_mask,
            )

        legacy_keys = {
            "latent_control_emb",
            "latent_control_prior",
            "latent_control_mask",
        }
        missing = sorted(legacy_keys.difference(batch))
        if missing:
            raise ValueError(
                "cell_detr requires a unique or legacy control bank; missing "
                + ", ".join(missing)
            )
        token_mask = batch.get("latent_control_token_mask")
        if token_mask is not None:
            token_mask = token_mask.to(device=device, dtype=torch.bool)
        return (
            batch["latent_control_emb"].to(device),
            batch["latent_control_mask"].to(device).bool(),
            batch["latent_control_prior"].to(device),
            None,
            token_mask,
        )

    @staticmethod
    def _anchor_bridge_child_std(batch, context, source_mode):
        """Return a validated/padded Child standard-deviation bank if needed."""
        if not str(source_mode).endswith("mean_std"):
            return None

        if "cell_detr_control_bank" in batch:
            key = "cell_detr_control_stds"
        else:
            key = "latent_control_stds"
        if key not in batch:
            raise ValueError(
                f"Anchor-Bridge source mode {source_mode!r} requires batch[{key!r}]"
            )

        prototypes = context.control_prototypes
        child_std = batch[key].to(
            device=prototypes.device,
            dtype=prototypes.dtype,
        )
        if child_std.ndim != 3:
            raise ValueError(f"batch[{key!r}] must have shape [U,K,G]")
        unique_banks, capacity, genes = prototypes.shape
        if child_std.shape[0] != unique_banks:
            raise ValueError(
                f"batch[{key!r}] U={child_std.shape[0]} does not match "
                f"context U={unique_banks}"
            )
        if child_std.shape[2] != genes:
            raise ValueError(
                f"batch[{key!r}] G={child_std.shape[2]} does not match "
                f"context G={genes}"
            )
        if child_std.shape[1] > capacity:
            raise ValueError(
                f"batch[{key!r}] K={child_std.shape[1]} exceeds "
                f"context capacity={capacity}"
            )
        if not torch.isfinite(child_std).all():
            raise ValueError(f"batch[{key!r}] must be finite")
        if (child_std < 0).any():
            raise ValueError(f"batch[{key!r}] cannot contain negative values")
        if child_std.shape[1] < capacity:
            child_std = F.pad(
                child_std,
                (0, 0, 0, capacity - child_std.shape[1]),
            )
        return child_std

    def _compute_strict_parent_only_loss(self, batch):
        """Train the no-Child ablation without constructing routing state."""

        required = {"pert_emb", "cov_celltype", "cov_pert", "cov_batch"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "strict Parent-only training is missing " + ", ".join(missing)
            )
        if not getattr(self.diffusion, "is_rectified_flow", False):
            raise ValueError("strict Parent-only training requires RectifiedFlow")
        if not self.parent_residual_locked_flow_enabled:
            raise ValueError("strict Parent-only training requires locked flow")
        if batch["pert_emb"].ndim != 3 or batch["pert_emb"].shape[1] != 1:
            raise ValueError(
                "strict Parent-only training requires data.use_cell_set=1"
            )

        if (
            self.training
            and getattr(self.model, "fewshot_parent_adapter", None) is not None
        ):
            forbidden_condition = (
                self.model.fewshot_training_forbidden_condition_mask(
                    batch["cov_celltype"],
                    batch["cov_pert"],
                    cov_batch=batch.get("cov_batch"),
                )
            )
            if forbidden_condition.any():
                raise ValueError(
                    "hash-frozen audit or held-out deployment-line "
                    "non-support condition entered a training batch; exclude "
                    "it from the training dataloader"
                )

        device = batch["pert_emb"].device
        batch["batch_emb"] = self._encode_covariates(batch)
        raw_valid_mask = batch.get("valid_cell_mask")
        if raw_valid_mask is None:
            padded = batch.get("is_padded_list")
            if padded is None:
                raw_valid_mask = torch.ones(
                    batch["pert_emb"].shape[:2],
                    dtype=torch.bool,
                    device=device,
                )
            else:
                raw_valid_mask = ~padded.bool()
        supervision_mask = raw_valid_mask.to(device=device, dtype=torch.bool)

        context, prepared = self.model.prepare_strict_parent_only(
            cov_celltype=batch["cov_celltype"],
            cov_pert=batch["cov_pert"],
            cov_batch=batch.get("cov_batch"),
            semantic_condition=batch["batch_emb"],
            num_cells=1,
            stochastic=None,
        )
        batch["parent_residual_source"] = prepared.source.source
        batch.update(self.model.conditioning_tensors(context, prepared))
        if prepared.real_control_sample is not None:
            batch["parent_residual_real_control_member_indices"] = (
                prepared.real_control_sample.member_indices
            )

        from src.models.parent_locked_residual.lightning_integration import (
            compute_locked_flow_base_result,
        )

        result, locked_flow_output = compute_locked_flow_base_result(
            self,
            batch=batch,
            prepared=prepared,
            context=context,
            supervision_mask=supervision_mask,
            device=device,
            selection=None,
            parent_grouping=self.parent_residual_parent_grouping,
        )
        if self.parent_residual_de_aware_enabled:
            from src.models.de_aware_residual.lightning_integration import (
                augment_locked_result_with_de_aware,
            )

            augment_locked_result_with_de_aware(
                self,
                result=result,
                flow_output=locked_flow_output,
                batch=batch,
                prepared=prepared,
                supervision_mask=supervision_mask,
            )
        from src.models.sparse_hurdle_de.lightning_integration import (
            augment_locked_result_with_end_to_end_sparse_hurdle,
        )

        augment_locked_result_with_end_to_end_sparse_hurdle(
            self,
            result=result,
            flow_output=locked_flow_output,
            prepared=prepared,
            supervision_mask=supervision_mask,
        )

        row_valid = supervision_mask[:, 0].to(
            prepared.parent.parent_mean.dtype
        )
        target_available = prepared.parent.lookup.target_available.to(
            device=device
        )
        parent_weights = row_valid * target_available.to(row_valid)
        parent_denominator = parent_weights.sum().clamp_min(1.0)
        target_delta = prepared.parent.lookup.target_delta.to(
            prepared.parent.parent_mean
        )
        target_control = prepared.parent.lookup.control_mean.to(
            prepared.parent.parent_mean
        )
        target_endpoint = target_control + target_delta
        parent_mean = prepared.parent.parent_mean
        parent_mse_per_cell = (parent_mean - target_endpoint).square().mean(
            dim=-1
        )
        parent_mse = (
            parent_mse_per_cell * parent_weights
        ).sum() / parent_denominator

        predicted_delta = parent_mean - target_control
        predicted_centered = predicted_delta - predicted_delta.mean(
            dim=-1, keepdim=True
        )
        target_centered = target_delta - target_delta.mean(
            dim=-1, keepdim=True
        )
        delta_cosine = F.cosine_similarity(
            predicted_delta, target_delta, dim=-1, eps=1e-8
        )
        delta_pearson = F.cosine_similarity(
            predicted_centered, target_centered, dim=-1, eps=1e-8
        )
        direction_loss = (
            (1.0 - delta_cosine) * parent_weights
        ).sum() / parent_denominator
        pearson_loss = (
            (1.0 - delta_pearson) * parent_weights
        ).sum() / parent_denominator
        parent_pdcorr = (
            delta_pearson * parent_weights
        ).sum() / parent_denominator
        correction_l2 = prepared.parent.correction.square().mean()

        base_parent = target_control + prepared.parent.lookup.prior_delta.to(
            target_control
        )
        base_mse_per_cell = (base_parent - target_endpoint).square().mean(
            dim=-1
        )
        base_parent_mse = (
            base_mse_per_cell * parent_weights
        ).sum() / parent_denominator

        parent_mse_weight = float(
            getattr(self.model_cfg, "parent_residual_parent_mse_weight", 1.0)
        )
        direction_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_parent_direction_weight",
                2e-4,
            )
        )
        correction_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_correction_l2_weight",
                1e-2,
            )
        )
        pearson_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_parent_pearson_weight",
                2e-4,
            )
        )
        explicit_auxiliary = (
            parent_mse_weight * parent_mse
            + direction_weight * direction_loss
            + pearson_weight * pearson_loss
            + correction_weight * correction_l2
        )
        result["loss"] = result["loss"] + explicit_auxiliary

        from src.models.sparse_hurdle_de.lightning_integration import (
            augment_parent_signature_with_sparse_support,
        )

        augment_parent_signature_with_sparse_support(
            self,
            result=result,
            flow_output=locked_flow_output,
            prepared=prepared,
            parent_base_loss=explicit_auxiliary,
        )
        source_velocity = locked_flow_output["target_velocity"]
        result.update(
            {
                "parent_residual_strict_parent_only": parent_mse.new_tensor(1.0),
                "parent_residual_match_skipped": parent_mse.new_tensor(1.0),
                "parent_residual_parent_mse_raw": parent_mse.detach(),
                "parent_residual_parent_mse_weighted": (
                    parent_mse_weight * parent_mse
                ).detach(),
                "parent_residual_base_parent_mse": base_parent_mse.detach(),
                "parent_residual_parent_mse_gain": (
                    base_parent_mse - parent_mse
                ).detach(),
                "parent_residual_parent_direction": direction_loss.detach(),
                "parent_residual_parent_direction_weighted": (
                    direction_weight * direction_loss
                ).detach(),
                "parent_residual_parent_pearson_loss": pearson_loss.detach(),
                "parent_residual_parent_pdcorr": parent_pdcorr.detach(),
                "parent_residual_parent_target_coverage": (
                    parent_weights.mean()
                ).detach(),
                "parent_residual_prior_coverage": (
                    prepared.parent.lookup.prior_available.float().mean().detach()
                ),
                "parent_residual_correction_rms": (
                    prepared.parent.correction.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                ),
                "parent_residual_correction_l2": correction_l2.detach(),
                "parent_residual_explicit_auxiliary": explicit_auxiliary.detach(),
                "parent_residual_source_rms": (
                    prepared.source.source.detach().float().square().mean().sqrt()
                ),
                "parent_residual_source_mean_rms": (
                    prepared.source.mean.detach().float().square().mean().sqrt()
                ),
                "parent_residual_target_velocity_rms": (
                    source_velocity.detach().float().square().mean().sqrt()
                ),
            }
        )

        soft_gate_probability = prepared.parent.soft_gate_probability
        if soft_gate_probability is not None:
            soft_gate_slab = prepared.parent.soft_gate_slab
            legacy_delta = prepared.parent.legacy_delta
            effective_delta = prepared.parent.effective_delta
            shrinkage = legacy_delta - effective_delta
            result.update(
                {
                    "parent_residual_soft_gate_probability_mean": (
                        soft_gate_probability.detach().float().mean()
                    ),
                    "parent_residual_soft_gate_probability_min": (
                        soft_gate_probability.detach().float().min()
                    ),
                    "parent_residual_soft_gate_probability_max": (
                        soft_gate_probability.detach().float().max()
                    ),
                    "parent_residual_soft_gate_slab_rms": (
                        soft_gate_slab.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_soft_gate_legacy_delta_rms": (
                        legacy_delta.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_soft_gate_effective_delta_rms": (
                        effective_delta.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_soft_gate_shrinkage_rms": (
                        shrinkage.detach().float().square().mean().sqrt()
                    ),
                }
            )
        if prepared.parent.fewshot is not None:
            fewshot = prepared.parent.fewshot
            result.update(
                {
                    "parent_residual_fewshot_gate_mean": (
                        fewshot.gate.detach().float().mean()
                    ),
                    "parent_residual_fewshot_gate_max": (
                        fewshot.gate.detach().float().max()
                    ),
                    "parent_residual_fewshot_cv_eligible_fraction": (
                        fewshot.cv_eligible.detach().float().mean()
                    ),
                    "parent_residual_fewshot_query_fraction": (
                        fewshot.query_eligible.detach().float().mean()
                    ),
                    "parent_residual_fewshot_relative_cv_gain": (
                        fewshot.relative_cv_gain.detach().float().mean()
                    ),
                    "parent_residual_fewshot_correction_rms": (
                        fewshot.correction.detach().float().square().mean().sqrt()
                    ),
                }
            )
        if prepared.real_control_sample is not None:
            real_sample = prepared.real_control_sample
            result.update(
                {
                    "parent_residual_real_control_source_used": (
                        parent_mse.new_tensor(1.0)
                    ),
                    "parent_residual_real_control_support_mean": (
                        real_sample.member_counts.float().mean().detach()
                    ),
                    "parent_residual_real_control_support_min": (
                        real_sample.member_counts.min().to(parent_mse).detach()
                    ),
                    "parent_residual_real_control_residual_rms": (
                        real_sample.residual.float()
                        .square()
                        .mean()
                        .sqrt()
                        .detach()
                    ),
                    "parent_residual_real_control_residual_scale": (
                        parent_mse.new_tensor(
                            self.model.real_control_residual_scale
                        )
                    ),
                }
            )
        return result

    def _compute_parent_residual_gene_dit_loss(self, batch):
        """Train a prior-source FM plus explicit transferable Parent mean."""

        if self.parent_residual_strict_parent_only_enabled:
            return self._compute_strict_parent_only_loss(batch)

        set_guidance_step = getattr(
            self.model, "set_child_guidance_step", None
        )
        if callable(set_guidance_step):
            set_guidance_step(int(self.global_step))

        required = {"pert_emb", "cont_emb", "cov_celltype", "cov_pert"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "parent-residual training is missing " + ", ".join(missing)
            )
        if not getattr(self.diffusion, "is_rectified_flow", False):
            raise ValueError("Parent-Residual Gene-DiT requires RectifiedFlow")
        if batch["pert_emb"].ndim != 3 or batch["pert_emb"].shape[1] != 1:
            raise ValueError(
                "Parent-Residual training requires data.use_cell_set=1; the "
                "Gene-DiT treats each response cell independently"
            )

        parent_grouping = self.parent_residual_parent_grouping

        if (
            self.training
            and getattr(self.model, "fewshot_parent_adapter", None) is not None
        ):
            forbidden_condition = (
                self.model.fewshot_training_forbidden_condition_mask(
                    batch["cov_celltype"], batch["cov_pert"],
                    cov_batch=batch.get("cov_batch"),
                )
            )
            if forbidden_condition.any():
                raise ValueError(
                    "hash-frozen audit or held-out deployment-line "
                    "non-support condition entered a training batch; exclude "
                    "it from the training dataloader"
                )

        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        batch["batch_emb"] = self._encode_covariates(batch)
        supervision_mask = batch.get(
            "valid_cell_mask", ~batch["is_padded_list"].bool()
        ).to(device=device, dtype=torch.bool)
        epoch = int(self.current_epoch)
        match = None
        if self.parent_residual_matching_guidance_enabled:
            matching_group_ids = None
            if self.model.semi_balanced_ot_enabled:
                from src.models.parent_locked_residual.matched_integration import (
                    condition_group_ids,
                )

                matching_group_ids = condition_group_ids(
                    batch,
                    1,
                    parent_grouping=parent_grouping,
                )[:, 0]
            context, match = self.model.prepare_training_context(
                control_sets=control_sets,
                anchor_mask=anchor_mask,
                occupancy=occupancy,
                targets=batch["pert_emb"],
                semantic_condition=batch["batch_emb"],
                positive_multiplicity=self.model.positive_multiplicity(
                    epoch, step=int(self.global_step)
                ),
                supervision_mask=supervision_mask,
                token_mask=token_mask,
                bank_inverse=bank_inverse,
                matching_group_ids=matching_group_ids,
                epoch=epoch,
            )
        else:
            context = self.model.prepare_unmatched_training_context(
                control_sets=control_sets,
                anchor_mask=anchor_mask,
                occupancy=occupancy,
                semantic_condition=batch["batch_emb"],
                token_mask=token_mask,
                bank_inverse=bank_inverse,
            )
        if self.model.child_std_beta > 0.0:
            child_stds = self._anchor_bridge_child_std(
                batch, context, "prior_child_mean_std"
            )
        else:
            child_stds = torch.zeros_like(context.control_prototypes)
        matched_selection = None
        if self.parent_residual_matched_set_enabled:
            from src.models.parent_locked_residual.matched_integration import (
                prepare_matched_training_source,
            )

            prepared, matched_selection = prepare_matched_training_source(
                self,
                batch=batch,
                context=context,
                match=match,
                child_stds=child_stds,
            )
        else:
            prepared = self.model.build_prior_source(
                context=context,
                cov_celltype=batch["cov_celltype"],
                cov_pert=batch["cov_pert"],
                cov_batch=batch.get("cov_batch"),
                child_stds=child_stds,
                num_cells=1,
                stochastic=None,
        )

        batch["parent_residual_source"] = prepared.source.source
        batch.update(self.model.conditioning_tensors(context, prepared))
        batch["parent_residual_child_indices"] = prepared.child_indices
        if prepared.real_control_sample is not None:
            batch["parent_residual_real_control_member_indices"] = (
                prepared.real_control_sample.member_indices
            )
        if match is not None:
            batch["parent_residual_positive_child_indices"] = (
                match.positive_child_indices
            )
            batch["parent_residual_positive_mask"] = match.positive_mask

        locked_flow_output = None
        if self.parent_residual_locked_flow_enabled:
            from src.models.parent_locked_residual.lightning_integration import (
                compute_locked_flow_base_result,
            )

            result, locked_flow_output = compute_locked_flow_base_result(
                self,
                batch=batch,
                prepared=prepared,
                context=context,
                supervision_mask=supervision_mask,
                device=device,
                selection=matched_selection,
                parent_grouping=parent_grouping,
            )
            if self.parent_residual_de_aware_enabled:
                from src.models.de_aware_residual.lightning_integration import (
                    augment_locked_result_with_de_aware,
                )

                augment_locked_result_with_de_aware(
                    self,
                    result=result,
                    flow_output=locked_flow_output,
                    batch=batch,
                    prepared=prepared,
                    supervision_mask=supervision_mask,
                )
            from src.models.sparse_hurdle_de.lightning_integration import (
                augment_locked_result_with_end_to_end_sparse_hurdle,
            )

            augment_locked_result_with_end_to_end_sparse_hurdle(
                self,
                result=result,
                flow_output=locked_flow_output,
                prepared=prepared,
                supervision_mask=supervision_mask,
            )
        else:
            # Both Parent-derived FM entrances stay detached in the legacy path.
            result = self._compute_base_loss(
                batch,
                cont_emb_override=prepared.source.mean.detach(),
                noise_override=prepared.source.source.detach(),
            )

        row_valid = supervision_mask[:, 0].to(
            prepared.parent.parent_mean.dtype
        )
        target_available = prepared.parent.lookup.target_available.to(
            device=device
        )
        parent_weights = row_valid * target_available.to(row_valid)
        parent_denominator = parent_weights.sum().clamp_min(1.0)
        target_delta = prepared.parent.lookup.target_delta.to(
            prepared.parent.parent_mean
        )
        target_control = prepared.parent.lookup.control_mean.to(
            prepared.parent.parent_mean
        )
        target_endpoint = target_control + target_delta
        parent_mean = prepared.parent.parent_mean
        parent_mse_per_cell = (parent_mean - target_endpoint).square().mean(dim=-1)
        parent_mse = (
            parent_mse_per_cell * parent_weights
        ).sum() / parent_denominator

        # Optional safety budget for an enabled Child-guided Parent. P0 is a
        # detached moving teacher in the comparison: the auxiliary gradient
        # reaches only the final guided endpoint. The positive-parallel
        # wrapper has already detached all Child/Router evidence, so this term
        # can refine Parent without turning matching into a hidden Parent loss.
        unguided_parent_mse = parent_mse.new_zeros(())
        child_guidance_mse_no_regression = parent_mse.new_zeros(())
        unguided_parent_mean = prepared.parent.base_parent_mean
        if unguided_parent_mean is not None:
            unguided_mse_per_cell = (
                unguided_parent_mean - target_endpoint
            ).square().mean(dim=-1)
            unguided_parent_mse = (
                unguided_mse_per_cell * parent_weights
            ).sum() / parent_denominator
            child_guidance_mse_no_regression = (
                F.relu(parent_mse_per_cell - unguided_mse_per_cell.detach())
                * parent_weights
            ).sum() / parent_denominator

        predicted_delta = parent_mean - target_control
        predicted_centered = predicted_delta - predicted_delta.mean(
            dim=-1, keepdim=True
        )
        target_centered = target_delta - target_delta.mean(dim=-1, keepdim=True)
        delta_cosine = F.cosine_similarity(
            predicted_delta,
            target_delta,
            dim=-1,
            eps=1e-8,
        )
        delta_pearson = F.cosine_similarity(
            predicted_centered,
            target_centered,
            dim=-1,
            eps=1e-8,
        )
        direction_loss = (
            (1.0 - delta_cosine) * parent_weights
        ).sum() / parent_denominator
        pearson_loss = (
            (1.0 - delta_pearson) * parent_weights
        ).sum() / parent_denominator
        parent_pdcorr = (
            delta_pearson * parent_weights
        ).sum() / parent_denominator
        correction_l2 = prepared.parent.correction.square().mean()

        base_parent = target_control + prepared.parent.lookup.prior_delta.to(
            target_control
        )
        base_mse_per_cell = (base_parent - target_endpoint).square().mean(dim=-1)
        base_parent_mse = (
            base_mse_per_cell * parent_weights
        ).sum() / parent_denominator

        router_kl = parent_mse.new_zeros(())
        latent_match_loss = parent_mse.new_zeros(())
        match_metrics = {}
        if match is not None:
            posterior = match.child_distribution.detach()
            prior = prepared.prior.float().clamp_min(1e-8)
            posterior_safe = posterior.float().clamp_min(1e-8)
            router_kl_per_cell = (
                posterior.float()
                * (posterior_safe.log() - prior.log())
            ).sum(dim=-1)
            router_kl_per_cell_mean = (
                router_kl_per_cell * row_valid.float()
            ).sum() / row_valid.float().sum().clamp_min(1.0)
            condition_router_kl = router_kl_per_cell_mean.detach()
            condition_router_enabled = bool(
                getattr(
                    self.model_cfg,
                    "parent_residual_condition_router_kl_enabled",
                    False,
                )
            )
            if self.parent_residual_matched_set_enabled:
                from src.models.matched_set_transport import condition_level_posterior_prior_kl
                from src.models.parent_locked_residual.matched_integration import condition_group_ids

                condition_router_kl = condition_level_posterior_prior_kl(
                    posterior,
                    prepared.prior,
                    group_ids=condition_group_ids(
                        batch,
                        1,
                        parent_grouping=parent_grouping,
                    )[:, 0],
                    child_mask=context.anchor_mask,
                    row_mask=row_valid,
                )

            if condition_router_enabled and not self.parent_residual_matched_set_enabled:
                raise ValueError("condition Router KL requires matched-set training")
            router_kl = (
                condition_router_kl
                if condition_router_enabled
                else router_kl_per_cell_mean
            )
            latent_match_loss = match.latent_match_loss
            match_metrics = {
                "parent_residual_router_kl_per_cell": router_kl_per_cell_mean.detach(),
                "parent_residual_router_kl_condition": condition_router_kl.detach(),
                "parent_residual_router_kl_uses_condition": parent_mse.new_tensor(
                    float(condition_router_enabled)
                ),
                "parent_residual_latent_match": latent_match_loss.detach(),
                "parent_residual_positive_multiplicity_requested": (
                    match.mean_cost.new_tensor(float(match.positive_multiplicity))
                ),
                "parent_residual_positive_multiplicity_effective_min": (
                    match.effective_positive_multiplicity_min.detach()
                ),
                "parent_residual_positive_multiplicity_effective_mean": (
                    match.effective_positive_multiplicity_mean.detach()
                ),
                "parent_residual_semi_balanced_ot_enabled": match.ot_enabled.detach(),
                "parent_residual_balanced_ot_enabled": match.ot_balanced.detach(),
                "parent_residual_semi_balanced_ot_column_l1": (
                    match.ot_column_l1.detach()
                ),
                "parent_residual_semi_balanced_ot_column_kl": (
                    match.ot_column_kl.detach()
                ),
                "parent_residual_semi_balanced_ot_effective_children": (
                    match.ot_effective_children.detach()
                ),
                "parent_residual_semi_balanced_ot_cost_scale": (
                    match.ot_cost_scale.detach()
                ),
            }

        parent_mse_weight = float(
            getattr(self.model_cfg, "parent_residual_parent_mse_weight", 1.0)
        )
        direction_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_parent_direction_weight",
                2e-4,
            )
        )
        correction_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_correction_l2_weight",
                1e-2,
            )
        )
        pearson_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_parent_pearson_weight",
                2e-4,
            )
        )
        router_weight = float(
            getattr(self.model_cfg, "parent_residual_router_kl_weight", 5e-4)
        )
        latent_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_latent_match_weight",
                5e-5,
            )
        )
        child_guidance_mse_no_reg_weight = float(
            getattr(
                self.model_cfg,
                "parent_residual_child_guidance_mse_no_reg_weight",
                0.0,
            )
        )
        if child_guidance_mse_no_reg_weight < 0.0 or not math.isfinite(
            child_guidance_mse_no_reg_weight
        ):
            raise ValueError(
                "parent_residual_child_guidance_mse_no_reg_weight must be "
                "finite and nonnegative"
            )
        if (
            child_guidance_mse_no_reg_weight > 0.0
            and unguided_parent_mean is None
        ):
            raise ValueError(
                "Child-guidance MSE no-regression requires an enabled "
                "Child-guided Parent endpoint"
            )
        teacher_auxiliary = parent_mse.new_zeros(())
        teacher_logs = {}
        if self.parent_residual_matched_set_enabled:
            from src.models.parent_locked_residual.matched_integration import (
                matched_teacher_auxiliary,
            )

            teacher_auxiliary, teacher_logs = matched_teacher_auxiliary(
                self, match
            )
        explicit_auxiliary = (
            parent_mse_weight * parent_mse
            + direction_weight * direction_loss
            + pearson_weight * pearson_loss
            + correction_weight * correction_l2
            + router_weight * router_kl
            + latent_weight * latent_match_loss
            + child_guidance_mse_no_reg_weight
            * child_guidance_mse_no_regression
            + teacher_auxiliary
        )
        result["loss"] = result["loss"] + explicit_auxiliary
        if match is not None:
            _augment_joint_parent_child_result(
                self,
                result,
                prepared,
                match,
                batch,
                supervision_mask,
                parent_grouping,
                parent_mse_weight * parent_mse,
            )
        if locked_flow_output is not None:
            from src.models.sparse_hurdle_de.lightning_integration import (
                augment_parent_signature_with_sparse_support,
            )

            augment_parent_signature_with_sparse_support(
                self,
                result=result,
                flow_output=locked_flow_output,
                prepared=prepared,
                parent_base_loss=explicit_auxiliary,
            )

        _augment_parent_listwise_pds_result(
            self,
            result=result,
            prepared=prepared,
            batch=batch,
            supervision_mask=supervision_mask,
            parent_grouping=parent_grouping,
        )

        prior_entropy = -(
            prepared.prior.float()
            * prepared.prior.float().clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        effective_children = prior_entropy.exp()
        source_velocity = (
            locked_flow_output["target_velocity"]
            if locked_flow_output is not None
            else batch["pert_emb"] - prepared.source.source
        )
        result.update(
            {
                "parent_residual_parent_mse_raw": parent_mse.detach(),
                "parent_residual_parent_mse_weighted": (
                    parent_mse_weight * parent_mse
                ).detach(),
                "parent_residual_base_parent_mse": base_parent_mse.detach(),
                "parent_residual_parent_mse_gain": (
                    base_parent_mse - parent_mse
                ).detach(),
                "parent_residual_child_guidance_unguided_parent_mse": (
                    unguided_parent_mse.detach()
                ),
                "parent_residual_child_guidance_mse_gain": (
                    unguided_parent_mse - parent_mse
                ).detach(),
                "parent_residual_child_guidance_mse_no_regression": (
                    child_guidance_mse_no_regression.detach()
                ),
                "parent_residual_child_guidance_mse_no_regression_weighted": (
                    child_guidance_mse_no_reg_weight
                    * child_guidance_mse_no_regression
                ).detach(),
                "parent_residual_parent_direction": direction_loss.detach(),
                "parent_residual_parent_direction_weighted": (direction_weight * direction_loss).detach(),
                "parent_residual_parent_pearson_loss": pearson_loss.detach(),
                "parent_residual_parent_pdcorr": parent_pdcorr.detach(),
                "parent_residual_parent_target_coverage": (
                    parent_weights.mean()
                ).detach(),
                "parent_residual_prior_coverage": prepared.parent.lookup.prior_available.float().mean().detach(),
                "parent_residual_correction_rms": prepared.parent.correction.detach().float().square().mean().sqrt(),
                "parent_residual_correction_l2": correction_l2.detach(),
                "parent_residual_router_entropy": prior_entropy.detach(),
                "parent_residual_effective_children": effective_children.detach(),
                "parent_residual_matching_guidance_enabled": parent_mse.new_tensor(
                    float(self.parent_residual_matching_guidance_enabled)
                ),
                "parent_residual_match_skipped": parent_mse.new_tensor(
                    float(not self.parent_residual_matching_guidance_enabled)
                ),
                "parent_residual_explicit_auxiliary": explicit_auxiliary.detach(),
                "parent_residual_source_rms": prepared.source.source.detach().float().square().mean().sqrt(),
                "parent_residual_source_mean_rms": prepared.source.mean.detach().float().square().mean().sqrt(),
                "parent_residual_target_velocity_rms": source_velocity.detach().float().square().mean().sqrt(),
                "parent_residual_child_beta": parent_mse.new_tensor(self.model.child_beta),
                "parent_residual_child_std_beta": parent_mse.new_tensor(self.model.child_std_beta),
            }
        )
        if match is not None:
            match_metrics.update(
                {
                    "parent_residual_router_kl_weighted": (
                        router_weight * router_kl
                    ).detach(),
                    "parent_residual_router_kl": router_kl.detach(),
                }
            )
            result.update(match_metrics)
        bridge_shrink = prepared.parent.child_consensus_bridge_shrink
        if bridge_shrink is not None:
            bridge_metrics = {
                "legacy_rms": prepared.parent.child_consensus_bridge_legacy_rms,
                "full_raw_rms": prepared.parent.child_consensus_bridge_full_raw_rms,
                "collapsed_raw_rms": (
                    prepared.parent.child_consensus_bridge_collapsed_raw_rms
                ),
                "contrast_raw_rms": (
                    prepared.parent.child_consensus_bridge_contrast_raw_rms
                ),
                "realized_delta_rms": (
                    prepared.parent.child_consensus_bridge_realized_delta_rms
                ),
                "delta_to_legacy_ratio": (
                    prepared.parent.child_consensus_bridge_delta_to_legacy_ratio
                ),
                "shrink": bridge_shrink,
                "saturation": prepared.parent.child_consensus_bridge_saturation,
                "legacy_zero": prepared.parent.child_consensus_bridge_legacy_zero,
            }
            for metric_name, metric_value in bridge_metrics.items():
                if metric_value is None:
                    raise RuntimeError(
                        "enabled Child consensus bridge omitted diagnostic "
                        f"{metric_name}"
                    )
                result[
                    "parent_residual_child_consensus_bridge_" + metric_name
                ] = metric_value.detach().float().mean()
            result[
                "parent_residual_child_consensus_bridge_delta_to_legacy_ratio_max"
            ] = (
                prepared.parent.child_consensus_bridge_delta_to_legacy_ratio
                .detach()
                .float()
                .max()
            )

        tied_gene_raw_rms = prepared.parent.tied_gene_raw_rms
        if tied_gene_raw_rms is not None:
            tied_metrics = {
                "parent_residual_tied_gene_raw_rms": tied_gene_raw_rms,
                "parent_residual_tied_gene_legacy_raw_rms": (
                    prepared.parent.tied_gene_legacy_raw_rms
                ),
                "parent_residual_tied_gene_to_legacy_ratio": (
                    prepared.parent.tied_gene_to_legacy_ratio
                ),
            }
            for metric_name, metric_value in tied_metrics.items():
                if metric_value is None:
                    raise RuntimeError(
                        "enabled tied gene correction omitted diagnostic "
                        f"{metric_name}"
                    )
                result[metric_name] = metric_value.detach().float().mean()

        child_guidance_alpha = prepared.parent.child_guidance_alpha
        if child_guidance_alpha is not None:
            alpha = child_guidance_alpha.detach().float()
            contribution = prepared.parent.child_guidance_residual
            result.update(
                {
                    "parent_residual_child_guidance_projection_mean": alpha.mean(),
                    "parent_residual_child_guidance_projection_positive_fraction": (
                        (alpha > 0.0).float().mean()
                    ),
                    "parent_residual_child_guidance_projection_max": alpha.max(),
                    "parent_residual_child_guidance_topk_mass_mean": (
                        prepared.parent.child_guidance_topk_mass.detach()
                        .float()
                        .mean()
                    ),
                    "parent_residual_child_guidance_coupling": parent_mse.new_tensor(
                        float(prepared.parent.child_guidance_coupling)
                    ),
                }
            )
            if contribution is not None:
                result["parent_residual_child_guidance_contribution_rms"] = (
                    contribution.detach().float().square().mean().sqrt()
                )
        child_marginal_alpha = prepared.parent.child_marginal_alpha
        if child_marginal_alpha is not None:
            marginal = prepared.parent.child_marginal_mean
            marginal_control = prepared.parent.child_marginal_control_mean
            marginal_delta = prepared.parent.child_marginal_delta
            marginal_residual = prepared.parent.child_marginal_residual
            marginal_contribution = prepared.parent.child_marginal_contribution
            result.update(
                {
                    "parent_residual_child_marginal_alpha_mean": (
                        child_marginal_alpha.detach().float().mean()
                    ),
                    "parent_residual_child_marginal_mean_rms": (
                        marginal.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_child_marginal_control_rms": (
                        marginal_control.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_child_marginal_delta_rms": (
                        marginal_delta.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_child_marginal_residual_rms": (
                        marginal_residual.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_child_marginal_contribution_rms": (
                        marginal_contribution.detach().float().square().mean().sqrt()
                    ),
                    "parent_residual_child_marginal_router_entropy": (
                        prepared.parent.child_marginal_router_entropy.detach()
                        .float()
                        .mean()
                    ),
                    "parent_residual_child_marginal_effective_children": (
                        prepared.parent.child_marginal_router_effective_children
                        .detach()
                        .float()
                        .mean()
                    ),
                }
            )

        soft_gate_probability = prepared.parent.soft_gate_probability
        if soft_gate_probability is not None:
            soft_gate_slab = prepared.parent.soft_gate_slab
            legacy_delta = prepared.parent.legacy_delta
            effective_delta = prepared.parent.effective_delta
            shrinkage = legacy_delta - effective_delta
            result.update(
                {
                    "parent_residual_soft_gate_probability_mean": soft_gate_probability.detach().float().mean(),
                    "parent_residual_soft_gate_probability_min": soft_gate_probability.detach().float().min(),
                    "parent_residual_soft_gate_probability_max": soft_gate_probability.detach().float().max(),
                    "parent_residual_soft_gate_slab_rms": soft_gate_slab.detach().float().square().mean().sqrt(),
                    "parent_residual_soft_gate_legacy_delta_rms": legacy_delta.detach().float().square().mean().sqrt(),
                    "parent_residual_soft_gate_effective_delta_rms": effective_delta.detach().float().square().mean().sqrt(),
                    "parent_residual_soft_gate_shrinkage_rms": shrinkage.detach().float().square().mean().sqrt(),
                }
            )
        if prepared.parent.fewshot is not None:
            fewshot = prepared.parent.fewshot
            result.update(
                {
                    "parent_residual_fewshot_gate_mean": fewshot.gate.detach().float().mean(),
                    "parent_residual_fewshot_gate_max": fewshot.gate.detach().float().max(),
                    "parent_residual_fewshot_cv_eligible_fraction": fewshot.cv_eligible.detach().float().mean(),
                    "parent_residual_fewshot_query_fraction": fewshot.query_eligible.detach().float().mean(),
                    "parent_residual_fewshot_relative_cv_gain": fewshot.relative_cv_gain.detach().float().mean(),
                    "parent_residual_fewshot_correction_rms": fewshot.correction.detach().float().square().mean().sqrt(),
                }
            )
        if prepared.real_control_sample is not None:
            real_sample = prepared.real_control_sample
            result.update(
                {
                    "parent_residual_real_control_source_used": parent_mse.new_tensor(1.0),
                    "parent_residual_real_control_support_mean": real_sample.member_counts.float().mean().detach(),
                    "parent_residual_real_control_support_min": real_sample.member_counts.min().to(parent_mse).detach(),
                    "parent_residual_real_control_residual_rms": real_sample.residual.float().square().mean().sqrt().detach(),
                    "parent_residual_real_control_residual_scale": parent_mse.new_tensor(
                        self.model.real_control_residual_scale
                    ),
                }
            )
        result.update(teacher_logs)
        return result

    @torch.no_grad()
    def _apply_strict_parent_only_inference_context(
        self, batch, stochastic=None
    ):
        """Prepare inference state without constructing any Child state."""

        required = {"pert_emb", "cov_celltype", "cov_pert", "cov_batch"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "strict Parent-only inference is missing " + ", ".join(missing)
            )
        if batch["pert_emb"].ndim != 3:
            raise ValueError("strict Parent-only pert_emb must have shape [B,S,G]")
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        context, prepared = self.model.prepare_strict_parent_only(
            cov_celltype=batch["cov_celltype"],
            cov_pert=batch["cov_pert"],
            cov_batch=batch.get("cov_batch"),
            semantic_condition=batch["batch_emb"],
            num_cells=int(batch["pert_emb"].shape[1]),
            stochastic=stochastic,
        )
        batch["parent_residual_source"] = prepared.source.source
        batch["cont_emb"] = prepared.source.mean
        batch.update(self.model.conditioning_tensors(context, prepared))
        if prepared.real_control_sample is not None:
            batch["parent_residual_real_control_member_indices"] = (
                prepared.real_control_sample.member_indices
            )

        hurdle_fields = {}
        if self.parent_residual_locked_flow_enabled:
            from src.models.parent_locked_residual.lightning_integration import (
                attach_locked_inference_fields,
            )

            attach_locked_inference_fields(
                self,
                batch,
                prepared,
                parent_grouping=self.parent_residual_parent_grouping,
            )
            from src.models.sparse_hurdle_de.lightning_integration import (
                attach_end_to_end_hurdle_inference_fields,
            )

            hurdle_fields = attach_end_to_end_hurdle_inference_fields(
                self,
                batch=batch,
                prepared=prepared,
                context=context,
            )
        marker = prepared.parent.parent_mean.new_tensor(1.0)
        return {
            "parent_mean": prepared.parent.parent_mean,
            "parent_correction": prepared.parent.correction,
            "source": prepared.source.source,
            "source_mean": prepared.source.mean,
            "context": context,
            "parent_residual_strict_parent_only": marker,
            "parent_residual_match_skipped": marker,
            **hurdle_fields,
        }

    @torch.no_grad()
    def apply_parent_residual_inference_context(self, batch, stochastic=None):
        """Build the exact same target-free prior source used in training."""
        if self.parent_residual_strict_parent_only_enabled:
            return self._apply_strict_parent_only_inference_context(batch, stochastic)

        set_guidance_step = getattr(
            self.model, "set_child_guidance_step", None
        )
        if callable(set_guidance_step):
            set_guidance_step(PlModel._inference_child_guidance_step(self))


        if "pert_emb" not in batch:
            raise ValueError("parent-residual inference requires pert_emb")
        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        parent_grouping = self.parent_residual_parent_grouping
        context = self.model.build_context(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            semantic_condition=batch["batch_emb"],
            token_mask=token_mask,
            bank_inverse=bank_inverse,
        )
        if self.model.child_std_beta > 0.0:
            child_stds = self._anchor_bridge_child_std(
                batch, context, "prior_child_mean_std"
            )
        else:
            child_stds = torch.zeros_like(context.control_prototypes)
        sampling_mode = self.model.inference_child_sampling_mode
        routing_group_ids = None
        if sampling_mode != "legacy":
            from src.models.parent_locked_residual.matched_integration import condition_group_ids

            condition_ids = condition_group_ids(
                batch,
                set_size=int(batch["pert_emb"].shape[1]),
                parent_grouping=parent_grouping,
            )
            if not torch.equal(condition_ids, condition_ids[:, :1].expand_as(condition_ids)):
                raise ValueError("all cells in an inference row must share one condition")
            routing_group_ids = condition_ids[:, 0]

        prepared = self.model.build_prior_source(
            context=context,
            cov_celltype=batch["cov_celltype"],
            cov_pert=batch["cov_pert"],
            cov_batch=batch.get("cov_batch"),
            child_stds=child_stds,
            num_cells=batch["pert_emb"].shape[1],
            stochastic=stochastic,
            sampling_mode=sampling_mode,
            routing_group_ids=routing_group_ids,
        )
        batch["parent_residual_source"] = prepared.source.source
        batch["cont_emb"] = prepared.source.mean
        batch.update(self.model.conditioning_tensors(context, prepared))
        batch["parent_residual_child_indices"] = prepared.child_indices
        if prepared.real_control_sample is not None:
            batch["parent_residual_real_control_member_indices"] = (
                prepared.real_control_sample.member_indices
            )
        hurdle_fields = {}
        if self.parent_residual_locked_flow_enabled:
            from src.models.parent_locked_residual.lightning_integration import (
                attach_locked_inference_fields,
            )

            attach_locked_inference_fields(
                self,
                batch,
                prepared,
                parent_grouping=parent_grouping,
            )
            from src.models.sparse_hurdle_de.lightning_integration import (
                attach_end_to_end_hurdle_inference_fields,
            )

            hurdle_fields = attach_end_to_end_hurdle_inference_fields(
                self, batch=batch, prepared=prepared, context=context
            )
        return {
            "prior": prepared.prior,
            "child_indices": prepared.child_indices,
            "selected_child_tokens": prepared.selected_child_tokens,
            "parent_mean": prepared.parent.parent_mean,
            "parent_correction": prepared.parent.correction,
            "source": prepared.source.source,
            "source_mean": prepared.source.mean,
            "context": context,
            **hurdle_fields,
        }

    def _compute_cell_detr_loss(self, batch):
        """Train FM with target-free routing and target-only matching labels."""
        required = {
            "pert_emb",
            "cont_emb",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "cell_detr requires a fixed control bank; missing "
                + ", ".join(missing)
            )
        if not getattr(self.diffusion, "is_rectified_flow", False):
            raise ValueError("cell_detr must use PerturbDiff RectifiedFlow")
        if batch["pert_emb"].ndim != 3 or batch["pert_emb"].shape[1] != 1:
            raise ValueError(
                "cell_detr requires data.use_cell_set=1 so every batch row is independent"
            )

        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        batch["batch_emb"] = self._encode_covariates(batch)
        supervision_mask = batch.get(
            "valid_cell_mask", ~batch["is_padded_list"].bool()
        ).to(device=device, dtype=torch.bool)
        current_epoch = int(self.current_epoch)
        multiplicity = self.model.positive_multiplicity(
            current_epoch, step=int(self.global_step)
        )
        context, match = self.model.prepare_training_context(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            targets=batch["pert_emb"],
            semantic_condition=batch["batch_emb"],
            positive_multiplicity=multiplicity,
            supervision_mask=supervision_mask,
            token_mask=token_mask,
            bank_inverse=bank_inverse,
            epoch=current_epoch,
        )
        cfg = self.model.cell_detr_cfg
        routed = None
        if cfg.cold_start_gate_enabled:
            # This is the same target-free Router+Gate path used at test time.
            routed = self.model.route_context(
                context,
                num_cells=1,
                stochastic=bool(self.training),
                epoch=current_epoch,
            )
            selected_control = routed["selected_control"]
            selected_child_tokens = routed["selected_child_tokens"]
            selected_child_indices = routed["child_indices"]
        else:
            selected_control = match.selected_control
            selected_child_tokens = match.selected_child_tokens
            selected_child_indices = match.child_indices

        # Keep Router-vs-oracle diagnostics tied to the target-free routed
        # Child even when Anchor-Bridge uses the posterior match as its source.
        metric_child_indices = selected_child_indices
        bridge_source = None
        bridge_path = None
        flow_t = None
        flow_weights = None
        if self.anchor_bridge_enabled:
            bridge_cfg = self.model.anchor_bridge_cfg
            child_std = self._anchor_bridge_child_std(
                batch,
                context,
                bridge_cfg.training_source_mode,
            )
            bridge_source = self.model.build_anchor_bridge_training_source(
                context=context,
                match=match,
                child_std=child_std,
            )
            selected_control = bridge_source.mean
            selected_child_indices = bridge_source.child_indices
            if (selected_child_indices < 0).all():
                selected_child_tokens = context.parent_token[:, None].expand(
                    -1,
                    selected_child_indices.shape[1],
                    -1,
                )
            elif (selected_child_indices >= 0).all():
                cell_rows = torch.arange(
                    selected_child_indices.shape[0],
                    device=selected_child_indices.device,
                )[:, None]
                selected_child_tokens = context.child_tokens[
                    cell_rows,
                    selected_child_indices,
                ]
            else:
                raise ValueError(
                    "Anchor-Bridge source cannot mix Parent and Child indices"
                )
            batch["anchor_bridge_source"] = bridge_source.source

        batch["cell_detr_parent_token"] = context.parent_token
        batch["cell_detr_child_tokens"] = selected_child_tokens
        batch["cell_detr_routed_token"] = context.routed_token
        batch["cell_detr_child_indices"] = selected_child_indices
        batch["cell_detr_positive_child_indices"] = match.positive_child_indices
        batch["cell_detr_positive_mask"] = match.positive_mask

        if self.anchor_bridge_enabled:
            flow_t, flow_weights = self.schedule_sampler.sample(
                batch["pert_emb"].shape[0],
                device,
            )
            continuous = self.diffusion._continuous_time(
                flow_t,
                bridge_source.source,
            )
            bridge_path = self.model.build_anchor_bridge_path(
                bridge_source.source,
                batch["pert_emb"],
                continuous,
            )
            result = self._compute_base_loss(
                batch,
                cont_emb_override=bridge_source.mean,
                t_override=flow_t,
                weights_override=flow_weights,
                noise_override=bridge_source.source,
            )
            result.update(
                {
                    "anchor_bridge_source_rms": bridge_source.source.detach().float().square().mean().sqrt(),
                    "anchor_bridge_path_rms": bridge_path.current.detach().float().square().mean().sqrt(),
                    "anchor_bridge_velocity_rms": bridge_path.target_velocity.detach().float().square().mean().sqrt(),
                }
            )
        else:
            result = self._compute_base_loss(
                batch,
                cont_emb_override=selected_control,
            )
        weights = supervision_mask[:, 0].to(match.delta_loss.dtype)
        cell_denominator = weights.sum().clamp_min(1.0)
        bridge_router_loss = match.router_loss.new_zeros(())
        bridge_uses_router = bool(
            self.anchor_bridge_enabled
            and self.model.anchor_bridge_cfg.inference_source_mode != "parent"
        )
        if bridge_uses_router:
            posterior_target = match.child_distribution.detach()
            inference_router_distribution = self.model.inference_probabilities(
                context
            )
            bridge_router_per_cell = -(
                posterior_target
                * inference_router_distribution.clamp_min(1e-8).log()
            ).sum(dim=-1)
            bridge_router_loss = (
                bridge_router_per_cell * weights
            ).sum() / cell_denominator
            router_supervision_loss = bridge_router_loss
            router_weight = float(cfg.router_loss_weight)
        elif self.anchor_bridge_enabled:
            router_supervision_loss = bridge_router_loss
            router_weight = 0.0
        else:
            router_supervision_loss = match.router_loss
            router_weight = cfg.effective_router_loss_weight(current_epoch)
        auxiliary = (
            cfg.delta_loss_weight * match.delta_loss
            + cfg.direction_loss_weight * match.direction_loss
            + cfg.magnitude_loss_weight * match.magnitude_loss
            + router_weight * router_supervision_loss
            + cfg.compactness_loss_weight * context.compactness_loss
            + cfg.separation_loss_weight * context.separation_loss
        )

        zero = match.delta_loss.new_zeros(())
        parent_auxiliary = zero
        parent_delta = zero
        parent_direction = zero
        parent_magnitude = zero
        parent_cost = zero
        balance_weight = 0.0
        if cfg.cold_start_gate_enabled:
            parent_delta = (
                match.parent_delta_per_cell * weights
            ).sum() / cell_denominator
            parent_direction = (
                match.parent_direction_per_cell * weights
            ).sum() / cell_denominator
            parent_magnitude = (
                match.parent_magnitude_per_cell * weights
            ).sum() / cell_denominator
            parent_cost = (
                match.parent_cost_per_cell * weights
            ).sum() / cell_denominator
            parent_auxiliary = cfg.parent_auxiliary_loss_weight * (
                cfg.delta_loss_weight * parent_delta
                + cfg.direction_loss_weight * parent_direction
                + cfg.magnitude_loss_weight * parent_magnitude
            )
            balance_weight = cfg.balance_loss_weight_for_epoch(current_epoch)
            auxiliary = (
                auxiliary
                + parent_auxiliary
                + cfg.latent_match_loss_weight * match.latent_match_loss
                + cfg.gate_loss_weight * match.gate_loss
                + balance_weight * match.balance_loss
            )

        auxiliary_scale = (
            float(self.model.anchor_bridge_cfg.auxiliary_scale)
            if self.anchor_bridge_enabled
            else 1.0
        )
        scaled_auxiliary = auxiliary * auxiliary_scale
        result["loss"] = result["loss"] + scaled_auxiliary
        batch_usage = match.child_distribution.mean(dim=0)
        effective_children = 1.0 / batch_usage.square().sum().clamp_min(1e-8)
        router_entropy = -(
            match.router_distribution
            * match.router_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        result.update(
            {
                "cell_detr_auxiliary": auxiliary.detach(),
                "cell_detr_delta": match.delta_loss.detach(),
                "cell_detr_direction": match.direction_loss.detach(),
                "cell_detr_magnitude": match.magnitude_loss.detach(),
                "cell_detr_router": router_supervision_loss.detach(),
                "cell_detr_match_cost": match.mean_cost.detach(),
                "cell_detr_positive_multiplicity": match.mean_cost.new_tensor(
                    float(match.positive_multiplicity)
                ),
                "cell_detr_positive_multiplicity_effective_min": (
                    match.effective_positive_multiplicity_min.detach()
                ),
                "cell_detr_positive_multiplicity_effective_mean": (
                    match.effective_positive_multiplicity_mean.detach()
                ),
                "cell_detr_active_children": context.anchor_mask.sum(dim=-1).float().mean().detach(),
                "cell_detr_effective_children": effective_children.detach(),
                "cell_detr_router_entropy": router_entropy.detach(),
                "cell_detr_router_max_probability": match.router_distribution.max(dim=-1).values.mean().detach(),
                "cell_detr_compactness": context.compactness_loss.detach(),
                "cell_detr_separation": context.separation_loss.detach(),
                "cell_detr_supervised_fraction": supervision_mask.float().mean().detach(),
            }
        )
        if self.anchor_bridge_enabled:
            result["anchor_bridge_router_ce"] = (
                bridge_router_loss.detach()
            )
            result["anchor_bridge_router_loss_weight"] = (
                bridge_router_loss.new_tensor(float(router_weight))
            )
            result["anchor_bridge_auxiliary_raw"] = auxiliary.detach()
            result["anchor_bridge_auxiliary_scaled"] = (
                scaled_auxiliary.detach()
            )
            result["anchor_bridge_auxiliary_scale"] = auxiliary.new_tensor(
                float(auxiliary_scale)
            )

        if cfg.cold_start_gate_enabled:
            settings = cfg.cold_start_settings_for_epoch(current_epoch)
            positive_valid = weights[:, None].expand_as(match.relative_gain)
            positive_valid = positive_valid * match.positive_mask.to(
                positive_valid.dtype
            )
            positive_denominator = positive_valid.sum().clamp_min(1.0)
            relative_gain = (
                match.relative_gain * positive_valid
            ).sum() / positive_denominator
            gain_pass = (
                (match.relative_gain > settings["relative_margin"]).to(weights.dtype)
                * positive_valid
            ).sum() / positive_denominator
            gate_target_mean = (
                match.gate_target * positive_valid
            ).sum() / positive_denominator
            gate_predicted_mean = (
                match.selected_gate_probability * positive_valid
            ).sum() / positive_denominator
            fallback_target = (
                match.fallback_target_probability * weights
            ).sum() / cell_denominator
            router_confidence = (
                match.router_confidence * weights
            ).sum() / cell_denominator
            routed_gate = routed["selected_gate_probability"][..., 0]
            routed_gate_mean = (routed_gate[:, 0] * weights).sum() / cell_denominator
            routed_fallback = 1.0 - routed_gate_mean
            oracle_match = (
                metric_child_indices.unsqueeze(-1)
                == match.positive_child_indices[:, None, :]
            ) & match.positive_mask[:, None, :]
            oracle_hit = oracle_match.any(dim=-1).to(weights.dtype)[:, 0]
            oracle_hit = (oracle_hit * weights).sum() / cell_denominator

            all_gate = torch.sigmoid(context.child_gate_logits.float())
            committed = match.router_distribution * all_gate
            committed_mass = committed.sum(dim=-1, keepdim=True)
            conditional = committed / committed_mass.clamp_min(1e-8)
            committed_effective = 1.0 / conditional.square().sum(dim=-1).clamp_min(
                1e-8
            )
            committed_effective = torch.where(
                committed_mass[:, 0] > 1e-6,
                committed_effective,
                torch.zeros_like(committed_effective),
            )
            committed_effective = (
                committed_effective * weights
            ).sum() / cell_denominator

            result.update(
                {
                    "cell_detr_parent_auxiliary": parent_auxiliary.detach(),
                    "cell_detr_parent_delta": parent_delta.detach(),
                    "cell_detr_parent_direction": parent_direction.detach(),
                    "cell_detr_parent_cost": parent_cost.detach(),
                    "cell_detr_latent_match": match.latent_match_loss.detach(),
                    "cell_detr_gate": match.gate_loss.detach(),
                    "cell_detr_balance": match.balance_loss.detach(),
                    "cell_detr_child_relative_gain": relative_gain.detach(),
                    "cell_detr_gain_pass_fraction": gain_pass.detach(),
                    "cell_detr_gate_target_mean": gate_target_mean.detach(),
                    "cell_detr_gate_predicted_mean": gate_predicted_mean.detach(),
                    "cell_detr_fallback_target_fraction": fallback_target.detach(),
                    "cell_detr_fallback_actual_fraction": routed_fallback.detach(),
                    "cell_detr_router_confidence": router_confidence.detach(),
                    "cell_detr_router_oracle_topr_hit": oracle_hit.detach(),
                    "cell_detr_committed_effective_children": committed_effective.detach(),
                    "cell_detr_gain_margin": match.mean_cost.new_tensor(
                        float(settings["relative_margin"])
                    ),
                    "cell_detr_gate_exploration": match.mean_cost.new_tensor(
                        float(settings["exploration_weight"])
                    ),
                    "cell_detr_router_loss_weight": match.mean_cost.new_tensor(
                        float(router_weight)
                    ),
                    "cell_detr_balance_loss_weight": match.mean_cost.new_tensor(
                        float(balance_weight)
                    ),
                }
            )
        return result

    def _compute_cell_detr_direct_loss(self, batch):
        # Direct matching has no sampled t and never calls diffusion.training_losses.
        required = {"pert_emb", "cont_emb"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "cell_detr_direct requires a fixed control bank; missing "
                + ", ".join(missing)
            )
        if batch["pert_emb"].ndim != 3 or batch["pert_emb"].shape[1] != 1:
            raise ValueError(
                "cell_detr_direct training requires data.use_cell_set=1"
            )

        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        batch["batch_emb"] = self._encode_covariates(batch)
        supervision_mask = batch.get(
            "valid_cell_mask", ~batch["is_padded_list"].bool()
        ).to(device=device, dtype=torch.bool)
        multiplicity = self.model.positive_multiplicity(
            int(self.current_epoch), step=int(self.global_step)
        )
        context, match = self.model.prepare_training_context(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            targets=batch["pert_emb"],
            semantic_condition=batch["batch_emb"],
            positive_multiplicity=multiplicity,
            supervision_mask=supervision_mask,
            token_mask=token_mask,
            bank_inverse=bank_inverse,
        )
        batch["cell_detr_direct_positive_child_indices"] = match.positive_child_indices
        batch["cell_detr_direct_positive_mask"] = match.positive_mask

        cfg = self.model.cell_detr_cfg
        direct_mse = match.delta_per_cell
        per_cell_loss = (
            direct_mse
            + cfg.direction_loss_weight * match.direction_per_cell
            + cfg.magnitude_loss_weight * match.magnitude_per_cell
            + cfg.router_loss_weight * match.router_per_cell
        )
        structure_loss = (
            cfg.compactness_loss_weight * context.compactness_loss
            + cfg.separation_loss_weight * context.separation_loss
        )
        per_cell_loss = per_cell_loss + structure_loss
        weights = supervision_mask[:, 0].to(per_cell_loss.dtype)
        denominator = weights.sum().clamp_min(1.0)
        zero = per_cell_loss.new_zeros(())
        result = {
            "loss": per_cell_loss,
            "weights": weights,
            "loss1": (direct_mse * weights).sum() / denominator,
            "loss1_per_sample": direct_mse,
            "mmd1": zero,
            "mmd1_per_sample": torch.zeros_like(direct_mse),
        }

        dataset_names = np.array(
            [get_short_dsname(name) for name in batch["ds_name"]]
        )
        for dataset_name in {
            get_short_dsname(name) for name in self.model_cfg.dataset_dict
        }:
            selected = dataset_names == dataset_name
            if selected.any():
                result[f"dataset_loss_mse1_{dataset_name}"] = (
                    direct_mse[selected].mean().detach()
                )
            else:
                result[f"dataset_loss_mse1_{dataset_name}"] = zero
            result[f"dataset_loss_mmd1_{dataset_name}"] = zero

        batch_usage = match.child_distribution.mean(dim=0)
        effective_children = 1.0 / batch_usage.square().sum().clamp_min(1e-8)
        router_entropy = -(
            match.router_distribution
            * match.router_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        result.update(
            {
                "cell_detr_direct_mse": (direct_mse * weights).sum().detach() / denominator,
                "cell_detr_direct_direction": match.direction_loss.detach(),
                "cell_detr_direct_magnitude": match.magnitude_loss.detach(),
                "cell_detr_direct_router": match.router_loss.detach(),
                "cell_detr_direct_match_cost": match.mean_cost.detach(),
                "cell_detr_direct_structure": structure_loss.detach(),
                "cell_detr_direct_positive_multiplicity": zero.new_tensor(
                    float(match.positive_multiplicity)
                ),
                "cell_detr_direct_positive_multiplicity_effective_min": (
                    match.effective_positive_multiplicity_min.detach()
                ),
                "cell_detr_direct_positive_multiplicity_effective_mean": (
                    match.effective_positive_multiplicity_mean.detach()
                ),
                "cell_detr_direct_active_children": context.anchor_mask.sum(dim=-1).float().mean().detach(),
                "cell_detr_direct_effective_children": effective_children.detach(),
                "cell_detr_direct_router_entropy": router_entropy.detach(),
                "cell_detr_direct_router_max_probability": match.router_distribution.max(dim=-1).values.mean().detach(),
                "cell_detr_direct_supervised_fraction": weights.mean().detach(),
            }
        )
        return result

    @torch.no_grad()
    def apply_cell_detr_direct_prediction(self, batch, stochastic=True):
        # Route every output cell to one Child and decode it in one forward pass.
        if "pert_emb" not in batch:
            raise ValueError("cell_detr_direct inference requires pert_emb")
        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        prepared = self.model.prepare_inference_context(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            semantic_condition=batch["batch_emb"],
            num_cells=batch["pert_emb"].shape[1],
            token_mask=token_mask,
            stochastic=stochastic,
            bank_inverse=bank_inverse,
        )
        context = prepared["context"]
        batch["cont_emb"] = prepared["selected_control"]
        batch["cell_detr_parent_token"] = context.parent_token
        batch["cell_detr_child_tokens"] = prepared["selected_child_tokens"]
        batch["cell_detr_routed_token"] = context.routed_token
        batch["cell_detr_child_indices"] = prepared["child_indices"]
        return prepared

    @torch.no_grad()
    def apply_cell_detr_inference_context(self, batch, stochastic=True):
        """Choose each generated cell's Child using the learned router."""
        required = {
            "pert_emb",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "cell_detr inference is missing " + ", ".join(missing)
            )
        device = batch["pert_emb"].device
        control_sets, anchor_mask, occupancy, bank_inverse, token_mask = (
            self._cell_detr_control_inputs(batch, device)
        )
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        prepared = self.model.prepare_inference_context(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            semantic_condition=batch["batch_emb"],
            num_cells=batch["pert_emb"].shape[1],
            token_mask=token_mask,
            stochastic=stochastic,
            bank_inverse=bank_inverse,
        )
        context = prepared["context"]
        selected_control = prepared["selected_control"]
        selected_child_tokens = prepared["selected_child_tokens"]
        selected_child_indices = prepared["child_indices"]
        if self.anchor_bridge_enabled:
            bridge_cfg = self.model.anchor_bridge_cfg
            child_std = self._anchor_bridge_child_std(
                batch,
                context,
                bridge_cfg.inference_source_mode,
            )
            bridge_source = self.model.build_anchor_bridge_routed_source(
                context=context,
                routed=prepared,
                child_std=child_std,
            )
            selected_control = bridge_source.mean
            selected_child_indices = bridge_source.child_indices
            if (selected_child_indices < 0).all():
                selected_child_tokens = context.parent_token[:, None].expand(
                    -1,
                    selected_child_indices.shape[1],
                    -1,
                )
            elif (selected_child_indices >= 0).all():
                cell_rows = torch.arange(
                    selected_child_indices.shape[0],
                    device=selected_child_indices.device,
                )[:, None]
                selected_child_tokens = context.child_tokens[
                    cell_rows,
                    selected_child_indices,
                ]
            else:
                raise ValueError(
                    "Anchor-Bridge source cannot mix Parent and Child indices"
                )
            batch["anchor_bridge_source"] = bridge_source.source
            prepared["anchor_bridge_source"] = bridge_source.source
            prepared["anchor_bridge_mean"] = bridge_source.mean
            prepared["selected_control"] = selected_control
            prepared["selected_child_tokens"] = selected_child_tokens
            prepared["child_indices"] = selected_child_indices

        batch["cont_emb"] = selected_control
        batch["cell_detr_parent_token"] = context.parent_token
        batch["cell_detr_child_tokens"] = selected_child_tokens
        batch["cell_detr_routed_token"] = context.routed_token
        batch["cell_detr_child_indices"] = selected_child_indices
        return prepared

    def _compute_hungarian_flow_loss(self, batch):
        """Select t-free control conditions, then run the unchanged FM loss."""
        required = {
            "pert_emb",
            "cont_emb",
            "latent_control_emb",
            "latent_control_prior",
            "latent_control_mask",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "Hungarian-flow requires a fixed Anchor context bank; missing "
                + ", ".join(missing)
            )
        if not getattr(self.diffusion, "is_rectified_flow", False):
            raise ValueError("Hungarian-flow must use PerturbDiff RectifiedFlow")

        device = batch["pert_emb"].device
        batch["batch_emb"] = self._encode_covariates(batch)
        controls = batch["latent_control_emb"].to(device)
        occupancy = batch["latent_control_prior"].to(device)
        anchor_mask = batch["latent_control_mask"].to(device).bool()
        token_mask = batch.get("latent_control_token_mask")
        if token_mask is not None:
            token_mask = token_mask.to(device).bool()
        supervision_mask = batch.get(
            "valid_cell_mask", ~batch["is_padded_list"].bool()
        ).to(device=device, dtype=torch.bool)
        # Cell-set padding is sampled with replacement from the same biological
        # group. Those are real treated cells and the base FM loss trains them,
        # so every FM position needs a Hungarian-selected control. The original
        # (non-replicated) mask only de-duplicates auxiliary supervision.
        target_mask = torch.ones_like(supervision_mask)

        context, match = self.model.prepare_training_context(
            control_sets=controls,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            targets=batch["pert_emb"],
            semantic_condition=batch["batch_emb"],
            target_mask=target_mask,
            supervision_mask=supervision_mask,
            token_mask=token_mask,
        )
        selected_control = match.selected_control
        batch["hungarian_parent_token"] = context.parent_token
        batch["hungarian_child_tokens"] = match.selected_child_tokens
        batch["hungarian_routed_token"] = context.routed_token
        batch["hungarian_child_indices"] = match.child_indices
        batch["hungarian_slot_indices"] = match.slot_indices

        result = self._compute_base_loss(
            batch,
            cont_emb_override=selected_control,
        )
        cfg = self.model.hungarian_cfg
        episode_consistency = match.delta_loss.new_zeros(())
        if "meta_episode_id" in batch:
            episode_consistency = episode_distribution_consistency(
                match.soft_child_distribution,
                batch["meta_episode_id"],
            )
        auxiliary = (
            cfg.episode_consistency_weight * episode_consistency
            + cfg.delta_loss_weight * match.delta_loss
            + cfg.direction_loss_weight * match.direction_loss
            + cfg.magnitude_loss_weight * match.magnitude_loss
            + cfg.compactness_loss_weight * context.compactness_loss
            + cfg.separation_loss_weight * context.separation_loss
        )
        result["loss"] = result["loss"] + auxiliary
        active = context.anchor_mask.sum(dim=-1).float()
        effective = 1.0 / match.child_distribution.square().sum(dim=-1).clamp_min(1e-8)
        result.update(
            {
                "hungarian_flow_auxiliary": auxiliary.detach(),
                "hungarian_flow_episode_consistency": episode_consistency.detach(),
                "hungarian_flow_delta": match.delta_loss.detach(),
                "hungarian_flow_direction": match.direction_loss.detach(),
                "hungarian_flow_magnitude": match.magnitude_loss.detach(),
                "hungarian_flow_compactness": context.compactness_loss.detach(),
                "hungarian_flow_separation": context.separation_loss.detach(),
                "hungarian_flow_match_cost": match.mean_cost.detach(),
                "hungarian_flow_active_children": active.mean().detach(),
                "hungarian_flow_effective_children": effective.mean().detach(),
                "hungarian_flow_matched_fraction": match.matched_mask.float().mean().detach(),
                "hungarian_flow_supervised_fraction": supervision_mask.float().mean().detach(),
            }
        )
        return result

    @torch.no_grad()
    def apply_hungarian_flow_inference_context(self, batch):
        """Build Parent/Child context without treated-dependent matching."""
        required = {
            "pert_emb",
            "latent_control_emb",
            "latent_control_prior",
            "latent_control_mask",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "Hungarian-flow inference is missing " + ", ".join(missing)
            )
        device = batch["pert_emb"].device
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        token_mask = batch.get("latent_control_token_mask")
        if token_mask is not None:
            token_mask = token_mask.to(device).bool()
        prepared = self.model.prepare_inference_context(
            control_sets=batch["latent_control_emb"].to(device),
            anchor_mask=batch["latent_control_mask"].to(device).bool(),
            occupancy=batch["latent_control_prior"].to(device),
            semantic_condition=batch["batch_emb"],
            num_cells=batch["pert_emb"].shape[1],
            token_mask=token_mask,
        )
        context = prepared["context"]
        batch["cont_emb"] = prepared["selected_control"]
        batch["hungarian_parent_token"] = context.parent_token
        batch["hungarian_child_tokens"] = prepared["selected_child_tokens"]
        batch["hungarian_routed_token"] = context.routed_token
        batch["hungarian_child_indices"] = prepared["child_indices"]
        batch["hungarian_slot_indices"] = prepared["slot_indices"]
        return prepared


    def _repeat_source_batch(self, value, num_sources):
        """Repeat a batch-major value for every source candidate."""
        if torch.is_tensor(value):
            return value.unsqueeze(1).expand(
                -1,
                num_sources,
                *value.shape[1:],
            ).reshape(value.shape[0] * num_sources, *value.shape[1:])
        if isinstance(value, list):
            return [item for item in value for _ in range(num_sources)]
        return value

    def _source_prior_scale(self, em_cfg):
        """Return the configured linear prior warmup scale at the current step."""
        start = float(getattr(em_cfg, "prior_scale_start", 1.0))
        end = float(getattr(em_cfg, "prior_scale_end", 1.0))
        warmup_steps = int(getattr(em_cfg, "prior_warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return end
        progress = min(max(float(self.global_step) / warmup_steps, 0.0), 1.0)
        return start + (end - start) * progress

    def _source_delta_score_weight(self, em_cfg):
        """Return the E-step-only delta score weight at the current step."""
        configured = float(getattr(em_cfg, "delta_score_weight", 0.0))
        start = float(getattr(em_cfg, "delta_score_weight_start", configured))
        end = float(getattr(em_cfg, "delta_score_weight_end", configured))
        warmup_steps = int(getattr(em_cfg, "delta_score_warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return end
        progress = min(max(float(self.global_step) / warmup_steps, 0.0), 1.0)
        return start + (end - start) * progress

    def _source_support_policy(self, em_cfg):
        """Resolve the configured support stage without affecting legacy EM runs."""
        schedule = getattr(em_cfg, "support_schedule", None)
        if not schedule:
            return None, {"mode": "legacy_fixed", "stage_index": -1}
        from src.models.source_em import resolve_support_schedule

        resolved = resolve_support_schedule(
            int(self.global_step),
            schedule,
            transition_steps=int(
                getattr(em_cfg, "support_transition_steps", 0) or 0
            ),
        )
        policy = {
            key: resolved[key]
            for key in (
                "mode",
                "topk",
                "previous_mode",
                "previous_topk",
                "transition_lambda",
            )
        }
        return policy, resolved

    @staticmethod
    def _metric_float(value):
        if torch.is_tensor(value):
            return float(value.detach().float().cpu())
        return float(value)

    def _append_curriculum_csv(self, filename, row):
        em_cfg = getattr(self.model_cfg, "em", None)
        metrics_dir = str(getattr(em_cfg, "metrics_dir", "") or "")
        if not metrics_dir or not self.trainer.is_global_zero:
            return
        path = Path(metrics_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def _source_validation_diagnostics(
        self,
        prediction,
        x_t,
        x_0,
        control_input,
        expanded_t,
        condition,
        batch_size,
        num_sources,
        set_size,
        gene_dim,
    ):
        """Measure source sensitivity while holding every non-source input fixed."""
        source_tokens = self.model.encode_source_bank()

        def run_with_condition(current_condition):
            output = self.diffusion.get_model_output(
                model=self.model,
                x_t=torch.cat([x_t, x_0], dim=-1),
                control_input_t=torch.cat(
                    [control_input, torch.zeros_like(control_input)],
                    dim=-1,
                ),
                t=expanded_t,
                self_condition=current_condition,
                model_kwargs={},
            )["x"]
            return output.reshape(
                batch_size,
                num_sources,
                set_size,
                gene_dim,
            )

        permuted_condition = dict(condition)
        permuted_condition["source_indices"] = (
            condition["source_indices"] + 1
        ) % num_sources
        permuted = run_with_condition(permuted_condition)

        zero_condition = dict(condition)
        zero_condition["source_token_override"] = torch.zeros(
            batch_size * num_sources,
            source_tokens.shape[-1],
            device=prediction.device,
            dtype=source_tokens.dtype,
        )
        zero_output = run_with_condition(zero_condition)

        mean_condition = dict(condition)
        mean_condition["source_token_override"] = source_tokens.mean(
            dim=0,
            keepdim=True,
        ).expand(batch_size * num_sources, -1)
        mean_output = run_with_condition(mean_condition)

        pred_flat = prediction.float().flatten(start_dim=2)
        perm_flat = permuted.float().flatten(start_dim=2)
        permutation_mse = (pred_flat - perm_flat).square().mean()
        permutation_cosine = F.cosine_similarity(
            pred_flat,
            perm_flat,
            dim=-1,
        ).mean()
        zero_delta = (prediction.float() - zero_output.float()).square().mean()
        mean_delta = (prediction.float() - mean_output.float()).square().mean()

        pseudobulk = prediction.float().mean(dim=2)
        pairwise_mmd = prediction.new_zeros((), dtype=torch.float32)
        pairwise_distance = prediction.new_zeros((), dtype=torch.float32)
        pair_count = 0
        pairwise_rows = []
        split = str(getattr(self, "_active_validation_split", "validation"))
        batch_index = int(getattr(self, "_active_validation_batch_idx", -1))
        for source_i in range(num_sources):
            for source_j in range(source_i + 1, num_sources):
                delta = pseudobulk[:, source_i] - pseudobulk[:, source_j]
                linear_mmd = delta.square().mean(dim=-1).mean()
                pb_distance = delta.square().sum(dim=-1).sqrt().mean()
                pairwise_mmd = pairwise_mmd + linear_mmd
                pairwise_distance = pairwise_distance + pb_distance
                pair_count += 1
                pairwise_rows.append({
                    "global_step": int(self.global_step),
                    "split": split,
                    "batch_index": batch_index,
                    "source_i": source_i,
                    "source_j": source_j,
                    "linear_mmd": self._metric_float(linear_mmd),
                    "pseudobulk_l2": self._metric_float(pb_distance),
                })
        if pair_count:
            pairwise_mmd = pairwise_mmd / pair_count
            pairwise_distance = pairwise_distance / pair_count
        for row in pairwise_rows:
            self._append_curriculum_csv("source_pairwise_distance.csv", row)

        record = {
            "global_step": int(self.global_step),
            "split": split,
            "batch_index": batch_index,
            "source_permutation_output_mse": self._metric_float(permutation_mse),
            "source_permutation_output_cosine": self._metric_float(permutation_cosine),
            "source_zero_token_delta": self._metric_float(zero_delta),
            "source_mean_token_delta": self._metric_float(mean_delta),
            "source_pairwise_output_mmd": self._metric_float(pairwise_mmd),
            "source_pairwise_pseudobulk_distance": self._metric_float(pairwise_distance),
        }
        self._validation_source_records.append(record)
        em_cfg = getattr(self.model_cfg, "em", None)
        metrics_dir = str(getattr(em_cfg, "metrics_dir", "") or "")
        if metrics_dir and self.trainer.is_global_zero:
            output_path = Path(metrics_dir) / "source_permutation_test.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as handle:
                json.dump(self._validation_source_records, handle, indent=2)
        return {
            "source_permutation_output_mse": permutation_mse,
            "source_permutation_output_cosine": permutation_cosine,
            "source_zero_token_delta": zero_delta,
            "source_mean_token_delta": mean_delta,
            "source_pairwise_output_mmd": pairwise_mmd,
            "source_pairwise_pseudobulk_distance": pairwise_distance,
        }

    def _compute_source_transport_loss(self, batch):
        """Compute source-origin candidates with configured EM-v0 or EM-v1."""
        batch["batch_emb"] = self._encode_covariates(batch)
        target = batch["pert_emb"]
        device = target.device
        batch_size, set_size, gene_dim = target.shape
        num_sources = self.model.num_sources
        within_cell_line = self.model.source_scope == "within_cell_line"
        source_cell_line_ids = None
        source_mask = None
        source_centroid_candidates = None
        source_prior = None
        model_mask = None
        if within_cell_line:
            if "source_cell_line_id" not in batch or "source_mask" not in batch:
                raise ValueError("G3 batch is missing source cell-line metadata")
            source_cell_line_ids = batch["source_cell_line_id"].to(
                device=device, dtype=torch.long
            )
            source_mask = batch["source_mask"].to(device=device, dtype=torch.bool)
            if source_cell_line_ids.shape != (batch_size,):
                raise ValueError("source_cell_line_id must have shape [B]")
            if source_mask.shape != (batch_size, num_sources):
                raise ValueError("source_mask must have shape [B,K]")
            if not source_mask.any(dim=-1).all():
                raise ValueError("every target must have a valid local source")
            (
                source_centroid_candidates,
                _,
                source_prior,
                model_mask,
            ) = self.model.source_bank_for_cell_lines(source_cell_line_ids)
            if not torch.equal(source_mask, model_mask):
                raise ValueError("dataset/model source masks disagree")
        em_cfg = getattr(self.model_cfg, "em", None)
        em_version = str(getattr(em_cfg, "version", "v0_set_level"))

        if em_version == "v1_cell_level_online_gem":
            if not bool(getattr(em_cfg, "detach_candidate_loss", True)):
                raise ValueError("EM-v1 requires detach_candidate_loss=true")
            if bool(getattr(em_cfg, "include_mmd_in_estep", False)):
                raise ValueError("EM-v1 forbids MMD in the E-step")
            if bool(getattr(em_cfg, "use_learned_matcher", False)):
                raise ValueError("EM-v1 does not use a learned matcher")
            if bool(getattr(em_cfg, "use_hard_assignment", False)):
                raise ValueError("EM-v1 forbids hard assignments")

        t, weights = self.schedule_sampler.sample(batch_size, device)
        source_indices = torch.arange(
            num_sources,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1).reshape(-1)
        expanded_cell_line_ids = None
        if within_cell_line:
            expanded_cell_line_ids = source_cell_line_ids.unsqueeze(1).expand(
                -1, num_sources
            ).reshape(-1)
        shared_noise = torch.randn_like(target, dtype=torch.float64)
        from src.models.source_em import expand_shared_source_randomness

        expanded_t, expanded_noise = expand_shared_source_randomness(
            t,
            shared_noise,
            num_sources,
        )

        source_start = None
        if bool(getattr(self.model_cfg, "source_gaussian_initialization", False)):
            x_t = expanded_noise
        else:
            source_start = self.model.sample_source(
                source_indices,
                set_size,
                cell_line_ids=expanded_cell_line_ids,
            )
            x_t = self.diffusion.q_sample(
                source_start,
                expanded_t,
                noise=expanded_noise,
            )

        condition = {
            "batch_emb": self._repeat_source_batch(batch["batch_emb"], num_sources),
            "gene_emb": None,
            "ds_name": self._repeat_source_batch(batch["ds_name"], num_sources),
            "source_indices": source_indices,
            "source_cell_line_ids": expanded_cell_line_ids,
        }
        x_0 = torch.zeros_like(x_t)
        control_input = torch.zeros_like(x_t)
        if bool((torch.rand(1, device=device) > 0.5).item()):
            with torch.no_grad():
                initial = self.diffusion.get_model_output(
                    model=self.model,
                    x_t=torch.cat([x_t, x_0], dim=-1),
                    control_input_t=torch.cat(
                        [control_input, torch.zeros_like(control_input)],
                        dim=-1,
                    ),
                    t=expanded_t,
                    self_condition=condition,
                    model_kwargs={},
                )
            x_0 = initial["x"]

        model_output = self.diffusion.get_model_output(
            model=self.model,
            x_t=torch.cat([x_t, x_0], dim=-1),
            control_input_t=torch.cat(
                [control_input, torch.zeros_like(control_input)],
                dim=-1,
            ),
            t=expanded_t,
            self_condition=condition,
            model_kwargs={},
        )
        prediction = model_output["x"].reshape(
            batch_size,
            num_sources,
            set_size,
            gene_dim,
        )
        source_start_candidates = None
        if source_start is not None:
            source_start_candidates = source_start.reshape(
                batch_size,
                num_sources,
                set_size,
                gene_dim,
            )

        prediction_type = "x0" if bool(self.model_cfg.predict_xstart) else "epsilon"
        objective_target = target if prediction_type == "x0" else shared_noise
        temperature = float(
            getattr(
                em_cfg,
                "temperature",
                getattr(self.model_cfg, "source_em_temperature", 0.1),
            )
        )
        prior_scale = self._source_prior_scale(em_cfg) if em_cfg is not None else 1.0
        responsibility_topk = getattr(em_cfg, "responsibility_topk", None)
        support_policy, support_state = self._source_support_policy(em_cfg)
        if within_cell_line:
            source_mask = source_mask & model_mask
        else:
            source_prior = self.model.source_prior
            source_centroid_candidates = self.model.source_centroid.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        if str(getattr(em_cfg, "source_prior", "empirical")) in (
            "uniform", "uniform_within_cell_line"
        ):
            if source_mask is None:
                source_prior = torch.full_like(source_prior, 1.0 / num_sources)
            else:
                source_prior = source_mask.to(source_prior.dtype)
                source_prior = source_prior / source_prior.sum(dim=-1, keepdim=True)
        source_tokens = model_output["source_tokens"]
        if within_cell_line:
            source_tokens = source_tokens.reshape(
                batch_size, num_sources, self.model.hidden_size
            )

        # ---- Router forward pass ----
        router_logits = None
        router_cfg = getattr(self.model_cfg, "router", None)
        router_enabled = bool(getattr(self.model_cfg, "router_enabled", False))
        if router_enabled and self.model.router is not None:
            # Compute router logits from perturbation cells + class embedding
            class_emb = condition.get("batch_emb", batch["batch_emb"])
            if class_emb.ndim == 3:
                class_emb = class_emb.mean(dim=1)  # pool set dimension
            router_logits = self.model.router(
                pert_cells=target,
                class_emb=class_emb.to(device=target.device, dtype=target.dtype),
            )
        # Determine router alpha (scheduled)
        router_alpha = 1.0
        if router_cfg is not None:
            alpha_start = float(getattr(router_cfg, "alpha_start", 0.5))
            alpha_end = float(getattr(router_cfg, "alpha_end", 0.2))
            alpha_steps = int(getattr(router_cfg, "alpha_steps", 5000))
            if alpha_steps > 0:
                progress = min(1.0, self.global_step / max(alpha_steps, 1))
                router_alpha = alpha_start + (alpha_end - alpha_start) * progress

        em_topk = None
        if router_cfg is not None:
            em_topk = getattr(router_cfg, "topk", None)
            if em_topk is not None:
                em_topk = int(em_topk)

        delta_score_weight = self._source_delta_score_weight(em_cfg)
        objective = source_transport_loss(
            prediction=prediction,
            target=objective_target,
            source_prior=source_prior,
            source_tokens=source_tokens,
            temperature=temperature,
            lambda_balance=float(getattr(self.model_cfg, "source_lambda_balance", 0.1)),
            lambda_mmd=float(getattr(self.model_cfg, "source_lambda_mmd", 0.1)),
            lambda_diversity=float(getattr(self.model_cfg, "source_lambda_diversity", 0.01)),
            use_em=bool(getattr(self.model_cfg, "source_use_em", True)),
            em_version=em_version,
            prior_scale=prior_scale,
            responsibility_topk=responsibility_topk,
            lambda_path_sep=float(getattr(self.model_cfg, "source_lambda_path_sep", 0.0)),
            path_sep_margin=float(getattr(self.model_cfg, "source_path_sep_margin", 0.0)),
            source_start=source_start_candidates,
            source_centroid=source_centroid_candidates,
            prediction_type=prediction_type,
            support_policy=support_policy,
            source_mask=source_mask,
            router_logits=router_logits,
            router_alpha=router_alpha,
            lambda_router=float(getattr(self.model_cfg, "source_lambda_router", 0.5)),
            lambda_usage=float(getattr(self.model_cfg, "source_lambda_usage", 0.05)),
            em_topk=em_topk,
            em_candidate_score=str(
                getattr(em_cfg, "candidate_score", "diffusion_mse")
            ),
            delta_score_weight=delta_score_weight,
            delta_center_genes=bool(
                getattr(em_cfg, "delta_center_genes", True)
            ),
        )

        per_sample = (
            objective["responsibilities"]
            * objective["diffusion_loss_per_source"]
        ).sum(dim=-1)
        if per_sample.ndim == 2:
            per_sample = per_sample.mean(dim=1)
        if hasattr(self.schedule_sampler, "update_with_local_losses"):
            self.schedule_sampler.update_with_local_losses(t, per_sample.detach())

        diagnostics = objective["diagnostics"]
        top1 = diagnostics["top1_assignments"].reshape(-1)
        top1_switch_rate = top1.new_zeros((), dtype=torch.float32)
        if (
            self.training
            and self._previous_top1_assignments is not None
            and self._previous_top1_assignments.shape == top1.shape
        ):
            top1_switch_rate = (
                top1 != self._previous_top1_assignments.to(top1.device)
            ).float().mean()
        if self.training:
            self._previous_top1_assignments = top1.detach().cpu()

        delta_signature = objective["delta_signature_loss"].float()
        estep_score = objective["estep_candidate_score"].float()
        result = {
            "loss": objective["loss"],
            "weights": weights,
            "loss1": objective["em_loss"],
            "mmd1": objective["mmd_loss"],
            "source_em_loss": objective["em_loss"],
            "source_raw_diffusion_loss": objective["diffusion_loss_per_source"].mean(),
            "source_total_loss": objective["loss"],
            "source_balance_loss": objective["balance_loss"],
            "source_mmd_loss": objective["mmd_loss"],
            "source_diversity_loss": objective["diversity_loss"],
            "source_path_separation_loss": objective["path_separation_loss"],
            "source_router_loss": objective.get("router_loss", prediction.new_zeros(())),
            "source_usage_loss": objective.get("usage_loss", prediction.new_zeros(())),
            "source_delta_signature_loss": delta_signature.mean(),
            "source_delta_score_std": delta_signature.std(dim=-1, unbiased=False).mean(),
            "source_delta_score_range": (
                delta_signature.max(dim=-1).values
                - delta_signature.min(dim=-1).values
            ).mean(),
            "source_estep_score_std": estep_score.std(dim=-1, unbiased=False).mean(),
            "source_delta_score_weight": prediction.new_tensor(delta_score_weight),
            "source_entropy": diagnostics["entropy_mean"],
            "source_max_responsibility": diagnostics["max_responsibility_mean"],
            "source_gate_mean": model_output.get("source_gate_mean", prediction.new_zeros(())),
            "source_gate_std": model_output.get("source_gate_std", prediction.new_zeros(())),
        }
        if (
            not self.training
            and bool(getattr(em_cfg, "source_validation_diagnostics", False))
        ):
            result.update(
                self._source_validation_diagnostics(
                    prediction=prediction,
                    x_t=x_t,
                    x_0=x_0,
                    control_input=control_input,
                    expanded_t=expanded_t,
                    condition=condition,
                    batch_size=batch_size,
                    num_sources=num_sources,
                    set_size=set_size,
                    gene_dim=gene_dim,
                )
            )

        log_interval = int(getattr(em_cfg, "log_every_n_steps", 100) or 100)
        if self.training and int(self.global_step) % log_interval == 0:
            is_v1 = float(em_version == "v1_cell_level_online_gem")
            q = objective["responsibilities"]
            pre = diagnostics["pre_support"]
            post = diagnostics["post_support"]
            result.update({
                "em/version": q.new_tensor(is_v1),
                "em/responsibility_level": q.new_tensor(float(q.ndim == 3)),
                "em/pre_support_entropy_mean": pre["entropy_mean"],
                "em/pre_support_entropy_std": pre["entropy_std"],
                "em/pre_support_effective_num_sources": pre["effective_num_sources"],
                "em/pre_support_max_responsibility": pre["max_responsibility_mean"],
                "em/pre_support_top1_top2_gap": pre["top1_top2_gap"],
                "em/pre_support_kl_to_uniform": pre["kl_to_uniform"],
                "em/post_support_entropy_mean": post["entropy_mean"],
                "em/post_support_entropy_std": post["entropy_std"],
                "em/post_support_effective_num_sources": post["effective_num_sources"],
                "em/post_support_max_responsibility": post["max_responsibility_mean"],
                "em/post_support_top1_top2_gap": post["top1_top2_gap"],
                "em/current_support_k": diagnostics["current_support_k"],
                "em/topk_set_jaccard_across_cells": diagnostics["topk_set_jaccard_across_cells"],
                "em/top1_switch_rate": top1_switch_rate,
                "em/support_transition_lambda": diagnostics["support_transition_lambda"],
                "em/prior_scale": q.new_tensor(prior_scale),
                "em/temperature": q.new_tensor(temperature),
                "em/delta_score_weight": q.new_tensor(delta_score_weight),
                "em/q_ndim": q.new_tensor(float(q.ndim)),
                "em/q_shape_k": q.new_tensor(float(q.shape[-1])),
                "candidate/loss_mean": diagnostics["candidate_loss_mean"],
                "candidate/loss_std_across_sources": diagnostics["candidate_loss_std_across_sources"],
                "candidate/loss_range_across_sources": diagnostics["candidate_loss_range_across_sources"],
                "candidate/top1_top2_loss_gap": diagnostics["candidate_top1_top2_loss_gap"],
                "candidate/source_rank_stability": diagnostics["candidate_source_rank_stability"],
                "candidate/delta_signature_mean": delta_signature.mean(),
                "candidate/delta_signature_std_across_sources": delta_signature.std(
                    dim=-1, unbiased=False
                ).mean(),
                "candidate/estep_score_std_across_sources": estep_score.std(
                    dim=-1, unbiased=False
                ).mean(),
            })
            if q.ndim == 3:
                result["em/q_shape_b"] = q.new_tensor(float(q.shape[0]))
                result["em/q_shape_s"] = q.new_tensor(float(q.shape[1]))
            membership = diagnostics["topk_membership_frequency_per_source"]
            top1_frequency = diagnostics["top1_source_frequency"]
            for source_idx in range(q.shape[-1]):
                result[f"em/topk_membership_frequency_source_{source_idx}"] = membership[source_idx]
                result[f"em/top1_source_frequency_{source_idx}"] = top1_frequency[source_idx]

            if within_cell_line:
                for line_id, line_name in enumerate(self.model.source_cell_line_names):
                    line_rows = source_cell_line_ids == line_id
                    if line_rows.any():
                        line_q = q[line_rows]
                        result[f"em/{line_name}/entropy_mean"] = -(
                            line_q * line_q.clamp_min(1e-8).log()
                        ).sum(dim=-1).mean()
                        for source_idx in range(num_sources):
                            result[
                                f"em/{line_name}/responsibility_mass_source_{source_idx}"
                            ] = line_q[..., source_idx].mean()

            metric_payload = {
                "global_step": int(self.global_step),
                "em/version": em_version,
                "em/support_mode": support_state["mode"],
                "em/stage_index": support_state["stage_index"],
                "em/q_shape": list(q.shape),
                "em/pre_support_entropy_mean": self._metric_float(pre["entropy_mean"]),
                "em/pre_support_entropy_std": self._metric_float(pre["entropy_std"]),
                "em/pre_support_effective_num_sources": self._metric_float(pre["effective_num_sources"]),
                "em/pre_support_max_responsibility": self._metric_float(pre["max_responsibility_mean"]),
                "em/pre_support_top1_top2_gap": self._metric_float(pre["top1_top2_gap"]),
                "em/pre_support_kl_to_uniform": self._metric_float(pre["kl_to_uniform"]),
                "em/post_support_entropy_mean": self._metric_float(post["entropy_mean"]),
                "em/post_support_entropy_std": self._metric_float(post["entropy_std"]),
                "em/post_support_effective_num_sources": self._metric_float(post["effective_num_sources"]),
                "em/post_support_max_responsibility": self._metric_float(post["max_responsibility_mean"]),
                "em/post_support_top1_top2_gap": self._metric_float(post["top1_top2_gap"]),
                "em/current_support_k": self._metric_float(diagnostics["current_support_k"]),
                "em/topk_membership_frequency_per_source": membership.detach().float().cpu().tolist(),
                "em/topk_set_jaccard_across_cells": self._metric_float(diagnostics["topk_set_jaccard_across_cells"]),
                "em/top1_source_frequency": top1_frequency.detach().float().cpu().tolist(),
                "em/top1_switch_rate": self._metric_float(top1_switch_rate),
                "em/support_transition_lambda": self._metric_float(diagnostics["support_transition_lambda"]),
                "candidate/loss_mean": self._metric_float(diagnostics["candidate_loss_mean"]),
                "candidate/loss_std_across_sources": self._metric_float(diagnostics["candidate_loss_std_across_sources"]),
                "candidate/loss_range_across_sources": self._metric_float(diagnostics["candidate_loss_range_across_sources"]),
                "candidate/top1_top2_loss_gap": self._metric_float(diagnostics["candidate_top1_top2_loss_gap"]),
                "candidate/source_rank_stability": self._metric_float(diagnostics["candidate_source_rank_stability"]),
                "candidate/delta_signature_mean": self._metric_float(delta_signature.mean()),
                "candidate/delta_signature_std_across_sources": self._metric_float(
                    delta_signature.std(dim=-1, unbiased=False).mean()
                ),
                "candidate/estep_score_std_across_sources": self._metric_float(
                    estep_score.std(dim=-1, unbiased=False).mean()
                ),
                "em/delta_score_weight": float(delta_score_weight),
            }
            self.py_logger.info("soft_em_curriculum_metrics=%s", metric_payload)

            common = {
                "global_step": int(self.global_step),
                "stage_index": support_state["stage_index"],
                "support_mode": support_state["mode"],
            }
            self._append_curriculum_csv("metrics.csv", {
                **common,
                "total_loss": self._metric_float(objective["loss"]),
                "em_loss": self._metric_float(objective["em_loss"]),
                "mmd_loss": self._metric_float(objective["mmd_loss"]),
                "path_separation_loss": self._metric_float(objective["path_separation_loss"]),
            })
            self._append_curriculum_csv("em_pre_post_support_stats.csv", {
                **common,
                "pre_entropy_mean": metric_payload["em/pre_support_entropy_mean"],
                "pre_entropy_std": metric_payload["em/pre_support_entropy_std"],
                "pre_effective_num_sources": metric_payload["em/pre_support_effective_num_sources"],
                "pre_max_responsibility": metric_payload["em/pre_support_max_responsibility"],
                "pre_top1_top2_gap": metric_payload["em/pre_support_top1_top2_gap"],
                "pre_kl_to_uniform": metric_payload["em/pre_support_kl_to_uniform"],
                "post_entropy_mean": metric_payload["em/post_support_entropy_mean"],
                "post_entropy_std": metric_payload["em/post_support_entropy_std"],
                "post_effective_num_sources": metric_payload["em/post_support_effective_num_sources"],
                "post_max_responsibility": metric_payload["em/post_support_max_responsibility"],
                "post_top1_top2_gap": metric_payload["em/post_support_top1_top2_gap"],
                "current_support_k": metric_payload["em/current_support_k"],
                "transition_lambda": metric_payload["em/support_transition_lambda"],
                "membership_frequency": json.dumps(metric_payload["em/topk_membership_frequency_per_source"]),
                "topk_set_jaccard": metric_payload["em/topk_set_jaccard_across_cells"],
                "top1_frequency": json.dumps(metric_payload["em/top1_source_frequency"]),
                "top1_switch_rate": metric_payload["em/top1_switch_rate"],
            })
            self._append_curriculum_csv("candidate_loss_stats.csv", {
                **common,
                "loss_mean": metric_payload["candidate/loss_mean"],
                "loss_std_across_sources": metric_payload["candidate/loss_std_across_sources"],
                "loss_range_across_sources": metric_payload["candidate/loss_range_across_sources"],
                "top1_top2_loss_gap": metric_payload["candidate/top1_top2_loss_gap"],
                "source_rank_stability": metric_payload["candidate/source_rank_stability"],
                "delta_signature_mean": metric_payload["candidate/delta_signature_mean"],
                "delta_signature_std_across_sources": metric_payload[
                    "candidate/delta_signature_std_across_sources"
                ],
                "estep_score_std_across_sources": metric_payload[
                    "candidate/estep_score_std_across_sources"
                ],
                "delta_score_weight": metric_payload["em/delta_score_weight"],
            })
            elapsed = max(time.monotonic() - self._curriculum_resource_time, 1e-8)
            step_delta = max(int(self.global_step) - self._curriculum_resource_step, 0)
            resource_row = {
                **common,
                "seconds_per_step": elapsed / max(step_delta, 1),
                "cuda_memory_allocated_mb": 0.0,
                "cuda_memory_reserved_mb": 0.0,
                "cuda_max_memory_allocated_mb": 0.0,
            }
            if torch.cuda.is_available():
                resource_row.update({
                    "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                    "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                    "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                })
            self._append_curriculum_csv("resource_usage.csv", resource_row)
            self._curriculum_resource_time = time.monotonic()
            self._curriculum_resource_step = int(self.global_step)
        return result

    def _expand_batch_for_latent_controls(self, batch, k_val):
        """Repeat batch fields along a latent-control candidate axis."""
        expanded = {}
        for key, value in batch.items():
            if key.startswith("latent_control_"):
                continue
            if torch.is_tensor(value) and value.shape[0] == batch["pert_emb"].shape[0]:
                expanded[key] = value.unsqueeze(1).expand(-1, k_val, *value.shape[1:]).reshape(
                    value.shape[0] * k_val, *value.shape[1:]
                )
            elif isinstance(value, list) and len(value) == batch["pert_emb"].shape[0]:
                expanded[key] = [item for item in value for _ in range(k_val)]
            else:
                expanded[key] = value
        return expanded

    def _latent_control_temperature(self):
        start = getattr(self.model_cfg, "latent_control_temperature_start", None)
        end = getattr(self.model_cfg, "latent_control_temperature_end", None)
        anneal_steps = int(getattr(self.model_cfg, "latent_control_temperature_anneal_steps", 0) or 0)
        if start is None or end is None or anneal_steps <= 0:
            return float(getattr(self.model_cfg, "latent_control_temperature", 1.0) or 1.0)
        progress = min(float(self.global_step) / float(anneal_steps), 1.0)
        return float(start) + (float(end) - float(start)) * progress

    def _latent_control_delta_score_weight(self):
        configured = float(getattr(self.model_cfg, "latent_control_delta_score_weight", 0.0) or 0.0)
        start = getattr(self.model_cfg, "latent_control_delta_score_weight_start", None)
        end = getattr(self.model_cfg, "latent_control_delta_score_weight_end", None)
        start = configured if start is None else float(start)
        end = configured if end is None else float(end)
        warmup_steps = int(getattr(self.model_cfg, "latent_control_delta_score_warmup_steps", 0) or 0)
        start_step = int(getattr(self.model_cfg, "latent_control_delta_score_start_step", 0) or 0)
        if int(self.global_step) < start_step:

            return start
        if warmup_steps <= 0:
            return end
        progress = min(
            max(float(self.global_step - start_step) / float(warmup_steps), 0.0),
            1.0,
        )
        return start + (end - start) * progress

    def _adaptive_anchor_residual_alpha(self):
        start = float(
            getattr(self.model_cfg, "adaptive_anchor_residual_alpha_start", 0.0)
        )
        end = float(
            getattr(self.model_cfg, "adaptive_anchor_residual_alpha_end", 0.5)
        )
        start_step = int(
            getattr(self.model_cfg, "adaptive_anchor_residual_start_step", 2000)
        )
        warmup_steps = int(
            getattr(self.model_cfg, "adaptive_anchor_residual_warmup_steps", 8000)
        )
        if int(self.global_step) < start_step:
            return start
        if warmup_steps <= 0:
            return end
        progress = min(
            max(float(int(self.global_step) - start_step) / warmup_steps, 0.0),
            1.0,
        )
        return start + (end - start) * progress

    def _adaptive_anchor_scheduled_weight(self, stem, default, default_start, default_warmup):
        target = float(getattr(self.model_cfg, f"adaptive_anchor_{stem}_weight", default))
        start_step = int(getattr(self.model_cfg, f"adaptive_anchor_{stem}_start_step", default_start))
        warmup_steps = int(getattr(self.model_cfg, f"adaptive_anchor_{stem}_warmup_steps", default_warmup))
        step = int(self.global_step)
        if step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return target
        progress = min(max(float(step - start_step) / float(warmup_steps), 0.0), 1.0)
        return target * progress

    def _prepare_adaptive_anchors(self, batch, candidates, priors, masks):
        if not self.adaptive_anchor_enabled:
            return None
        output = self.adaptive_state_anchor(
            candidates=candidates,
            empirical_prior=priors,
            source_mask=masks > 0,
            perturbation_ids=batch["cov_pert"],
            cell_type_ids=batch["cov_celltype"],
            cluster_ids=batch.get("latent_control_cluster_id"),
            parent_ids=batch.get("latent_control_parent_id"),
        )
        # Child IDs are stable routing identities; parent context is a
        # control-derived description of the current cell line.
        batch["latent_control_cluster_id"] = output.cluster_ids
        batch["adaptive_anchor_parent_id"] = output.parent_ids
        batch["adaptive_anchor_child_id"] = output.child_ids
        batch["anchor_parent_context"] = output.parent_context
        if output.parent_memory_tokens is None:
            batch["anchor_parent_tokens"] = output.parent_tokens
            batch["anchor_parent_mask"] = masks > 0
        else:
            batch["anchor_parent_tokens"] = output.parent_memory_tokens
            batch["anchor_parent_mask"] = output.parent_memory_mask
        batch["anchor_perturbation_query"] = output.perturbation_query
        return output

    def _match_fm_prior_blend(self):
        start = float(getattr(self.model_cfg, "match_fm_prior_blend_start", 1.0))
        end = float(getattr(self.model_cfg, "match_fm_prior_blend_end", 1.0))
        steps = int(getattr(self.model_cfg, "match_fm_prior_blend_steps", 0) or 0)
        if steps <= 0:
            return end
        progress = min(max(float(self.global_step) / float(steps), 0.0), 1.0)
        return start + (end - start) * progress

    def _match_fm_prior_distribution(
        self, batch, candidates, empirical_prior, source_mask, lagged=False
    ):
        """Predict r(z | perturbation, cell line, control-state set)."""
        if not self.match_fm_enabled or self.match_fm_prior is None:
            raise RuntimeError("MATCH-FM prior requested while match_fm_enabled=false")
        device = candidates.device
        cluster_ids = batch.get("latent_control_cluster_id")
        if cluster_ids is not None:
            cluster_ids = cluster_ids.to(device)
        prior_module = (
            self.match_fm_prior_teacher
            if lagged and self.match_fm_prior_teacher is not None
            else self.match_fm_prior
        )
        return prior_module(
            perturbation_ids=batch["cov_pert"].to(device),
            cell_type_ids=batch["cov_celltype"].to(device),
            control_sets=candidates,
            empirical_prior=empirical_prior.to(device),
            source_mask=source_mask.to(device).bool(),
            cluster_ids=cluster_ids,
            semantic_context=(
                batch.get("batch_emb").to(device)
                if batch.get("batch_emb") is not None else None
            ),
            parent_context=(
                batch.get("anchor_parent_context").to(device)
                if batch.get("anchor_parent_context") is not None else None
            ),
        )

    @torch.no_grad()
    def apply_match_fm_inference_prior(self, batch):
        """Allocate control contexts from the learned or empirical inference prior."""
        required = {"latent_control_emb", "latent_control_prior", "latent_control_mask"}
        if not required.issubset(batch):
            raise ValueError("MATCH-FM inference requires latent control candidates")
        device = batch["pert_emb"].device
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        candidates = batch["latent_control_emb"].to(device)
        seed_prior = batch["latent_control_prior"].to(device)
        seed_mask = batch["latent_control_mask"].to(device)
        anchor_output = None
        if bool(getattr(self, "adaptive_anchor_enabled", False)):
            anchor_output = self._prepare_adaptive_anchors(
                batch, candidates, seed_prior, seed_mask
            )
        if anchor_output is not None:
            candidates = anchor_output.candidates
            seed_prior = anchor_output.prior
            source_mask = anchor_output.active_mask
        else:
            source_mask = seed_mask > 0
        if self.match_fm_enabled:
            prior = self._match_fm_prior_distribution(
                batch,
                candidates,
                seed_prior,
                source_mask,
            )
        else:
            prior = seed_prior
            prior = prior.masked_fill(~source_mask, 0.0)
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        perturbation_ids = batch["cov_pert"][:, 0].to(device).long()
        cell_type_ids = batch["cov_celltype"][:, 0].to(device).long()
        offset = (
            perturbation_ids * 1103515245 + cell_type_ids * 12345
        ).remainder(1000003).to(prior.dtype) / 1000003.0
        from src.models.match_fm import deterministic_prior_assignments

        assignment = deterministic_prior_assignments(
            prior,
            num_cells=candidates.shape[2],
            source_mask=source_mask,
            offset=offset,
        )
        gather_index = assignment[:, None, :, None]
        selected = candidates.gather(
            1,
            gather_index.expand(-1, 1, -1, candidates.shape[-1]),
        ).squeeze(1)
        batch["anchor_baseline"] = selected
        batch["cont_emb"] = selected
        return {"prior": prior, "assignment": assignment}

    def _compute_latent_control_mixture_loss(self, batch):
        """Compute soft latent-cluster mixture loss without changing the denoiser architecture."""
        pert_emb = batch["pert_emb"]
        device = pert_emb.device
        if batch.get("batch_emb") is None:
            batch["batch_emb"] = self._encode_covariates(batch)
        candidates = batch["latent_control_emb"].to(device)
        priors = batch["latent_control_prior"].to(device)
        masks = batch["latent_control_mask"].to(device)
        anchor_output = self._prepare_adaptive_anchors(
            batch, candidates, priors, masks
        )
        if anchor_output is not None:
            candidates = anchor_output.candidates
            priors = anchor_output.prior
            masks = anchor_output.mask.to(dtype=priors.dtype)
            source_centroid = anchor_output.centroids
        else:
            source_centroid = candidates.mean(dim=2)
        valid_sources = masks > 0
        valid_cell_mask = batch.get(
            "valid_cell_mask", ~batch["is_padded_list"].bool()
        ).to(device=device, dtype=torch.bool)
        empirical_prior = torch.where(
            valid_sources, priors.clamp_min(1e-8), torch.zeros_like(priors)
        )
        empirical_prior = empirical_prior / empirical_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        if self.match_fm_enabled:
            inference_prior = self._match_fm_prior_distribution(
                batch, candidates, priors, valid_sources, lagged=False
            )
            constraint_prior = self._match_fm_prior_distribution(
                batch, candidates, priors, valid_sources, lagged=True
            ).detach()
            prior_blend = min(max(self._match_fm_prior_blend(), 0.0), 1.0)
            constraint_prior = (
                (1.0 - prior_blend) * empirical_prior
                + prior_blend * constraint_prior
            )
            constraint_prior = constraint_prior / constraint_prior.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            inference_prior = empirical_prior
            constraint_prior = empirical_prior
            prior_blend = 0.0

        bsz, k_val = candidates.shape[:2]
        t, weights = self.schedule_sampler.sample(bsz, device)
        noise = torch.randn_like(pert_emb, dtype=torch.float64)

        expanded_batch = self._expand_batch_for_latent_controls(batch, k_val)
        expanded_candidates = candidates.reshape(bsz * k_val, *candidates.shape[2:])
        if self.adaptive_anchor_enabled:
            expanded_batch["anchor_baseline"] = expanded_candidates
        expanded_t = t.unsqueeze(1).expand(-1, k_val).reshape(-1)
        expanded_weights = weights.unsqueeze(1).expand(-1, k_val).reshape(-1)
        expanded_noise = noise.unsqueeze(1).expand(-1, k_val, *noise.shape[1:]).reshape(bsz * k_val, *noise.shape[1:])

        responsibility_level = str(
            getattr(self.model_cfg, "latent_control_responsibility_level", "set")
        ).lower()
        if responsibility_level not in ("set", "cell"):
            raise ValueError("latent_control_responsibility_level must be set or cell")

        expanded_return = self._compute_base_loss(
            expanded_batch,
            cont_emb_override=expanded_candidates,
            t_override=expanded_t,
            weights_override=expanded_weights,
            noise_override=expanded_noise,
            return_model_output=responsibility_level == "cell",
        )
        teacher_return = None
        teacher_time = None
        ema_start_step = int(
            getattr(self.model_cfg, "match_fm_ema_start_step", 0) or 0
        )
        teacher_ready = (
            self.match_fm_ema_teacher_enabled
            and self._match_fm_teacher_initialized
            and int(self.global_step) >= ema_start_step
        )
        if teacher_ready and responsibility_level == "cell":
            if getattr(self.diffusion, "is_rectified_flow", False):
                time_min = float(
                    getattr(self.model_cfg, "match_fm_teacher_time_min", 0.1)
                )
                time_max = float(
                    getattr(self.model_cfg, "match_fm_teacher_time_max", 0.5)
                )
                if not 0.0 <= time_min < time_max <= 1.0:
                    raise ValueError(
                        "MATCH-FM teacher time range must satisfy 0 <= min < max <= 1"
                    )
                teacher_time = time_min + (time_max - time_min) * torch.rand(
                    bsz, device=device
                )
            else:
                teacher_time = t
            expanded_teacher_time = teacher_time.unsqueeze(1).expand(
                -1, k_val
            ).reshape(-1)
            with torch.no_grad():
                teacher_return = self._compute_base_loss(
                    expanded_batch,
                    cont_emb_override=expanded_candidates,
                    t_override=expanded_teacher_time,
                    weights_override=expanded_weights,
                    noise_override=expanded_noise,
                    return_model_output=True,
                    model_override=self.match_fm_teacher,
                    compute_mmd=False,
                    update_schedule_sampler=False,
                )

        if responsibility_level == "cell":
            from src.models.source_em import compute_cell_cluster_mixture

            prediction = expanded_return["model_output"]["x"].reshape(
                bsz, k_val, *pert_emb.shape[1:]
            )
            delta_weight = self._latent_control_delta_score_weight()
            magnitude_weight = float(
                getattr(self.model_cfg, "latent_control_delta_magnitude_score_weight", 0.05)
            )
            cell_objective = compute_cell_cluster_mixture(
                prediction=prediction,
                target=pert_emb,
                source_centroid=source_centroid,
                source_prior=constraint_prior.detach(),
                temperature=max(self._latent_control_temperature(), 1e-6),
                delta_score_weight=delta_weight,
                delta_magnitude_score_weight=magnitude_weight,
                source_mask=masks > 0,
                center_genes=bool(
                    getattr(self.model_cfg, "latent_control_delta_center_genes", True)
                ),
                source_sets=candidates,
                reliable_gene_weighting=bool(
                    getattr(self.model_cfg, "match_fm_reliable_gene_energy", False)
                ),
                reliability_max_weight=float(
                    getattr(self.model_cfg, "match_fm_reliability_max_weight", 20.0)
                ),
                pca_projection_basis=(
                    self.match_fm_pca_basis
                    if self.match_fm_pca_basis.numel()
                    else None
                ),
                pca_score_weight=float(
                    getattr(self.model_cfg, "match_fm_pca_score_weight", 0.0)
                ),
                normalize_matching_energy=bool(
                    getattr(self.model_cfg, "match_fm_energy_normalization", False)
                ),
                matching_energy_clip=float(
                    getattr(self.model_cfg, "match_fm_energy_clip", 5.0)
                ),
            )
            matching_objective = cell_objective
            if teacher_return is not None:
                teacher_prediction = teacher_return["model_output"]["x"].reshape(
                    bsz, k_val, *pert_emb.shape[1:]
                )
                matching_objective = compute_cell_cluster_mixture(
                    prediction=teacher_prediction,
                    target=pert_emb,
                    source_centroid=source_centroid,
                    source_prior=constraint_prior.detach(),
                    temperature=max(self._latent_control_temperature(), 1e-6),
                    delta_score_weight=delta_weight,
                    delta_magnitude_score_weight=magnitude_weight,
                    source_mask=valid_sources,
                    center_genes=bool(
                        getattr(self.model_cfg, "latent_control_delta_center_genes", True)
                    ),
                    source_sets=candidates,
                    reliable_gene_weighting=bool(
                        getattr(self.model_cfg, "match_fm_reliable_gene_energy", False)
                    ),
                    reliability_max_weight=float(
                        getattr(self.model_cfg, "match_fm_reliability_max_weight", 20.0)
                    ),
                    pca_projection_basis=(
                        self.match_fm_pca_basis
                        if self.match_fm_pca_basis.numel()
                        else None
                    ),
                    pca_score_weight=float(
                        getattr(self.model_cfg, "match_fm_pca_score_weight", 0.0)
                    ),
                    normalize_matching_energy=bool(
                        getattr(self.model_cfg, "match_fm_energy_normalization", False)
                    ),
                    matching_energy_clip=float(
                        getattr(self.model_cfg, "match_fm_energy_clip", 5.0)
                    ),
                )
            assignment = matching_objective["responsibilities"]
            online_em_prior = None
            if self.online_anchor_em_enabled:
                if k_val != self.online_anchor_em.num_slots:
                    raise ValueError(
                        f"online EM expects {self.online_anchor_em.num_slots} child slots, "
                        f"but this batch has {k_val}; enable data.latent_control_fixed_max_k"
                    )
                if "cell_index" not in batch:
                    raise ValueError("online anchor EM requires batch['cell_index']")
                cell_ids = batch["cell_index"].to(device=device, dtype=torch.long)
                num_perturbations = int(self.cov_encoding_cfg.num_pert)
                key_ids = (
                    batch["cov_celltype"].to(device=device, dtype=torch.long)
                    * num_perturbations
                    + batch["cov_pert"].to(device=device, dtype=torch.long)
                )
                base_cell_prior = constraint_prior.detach().unsqueeze(1).expand(
                    -1, pert_emb.shape[1], -1
                )
                cell_source_mask = valid_sources.unsqueeze(1).expand_as(
                    base_cell_prior
                )
                probe_interval = int(
                    getattr(self.model_cfg, "online_anchor_em_probe_interval", 200)
                )
                probe_inactive = (
                    self.training
                    and probe_interval > 0
                    and int(self.global_step) % probe_interval == 0
                )
                online_em_prior = self.online_anchor_em.get_soft_prior(
                    cell_ids,
                    key_ids,
                    step=int(self.global_step),
                    base_prior=base_cell_prior,
                    source_mask=cell_source_mask,
                    probe_inactive=probe_inactive,
                )
                from src.models.source_em import compute_soft_em_responsibility
                assignment = compute_soft_em_responsibility(
                    matching_objective["matching_energy"],
                    online_em_prior.prior,
                    temperature=max(self._latent_control_temperature(), 1e-6),
                    source_mask=online_em_prior.support_mask,
                )
            assignment_confidence = torch.ones(
                assignment.shape[:2], device=device, dtype=assignment.dtype
            )
            prior_kl = None
            posterior_marginal = None
            prior_empirical_kl = None
            if self.match_fm_enabled:
                from src.models.match_fm import (
                    confidence_shrink_posterior,
                    distribution_kl,
                    posterior_prior_kl,
                    set_constrained_posterior,
                )

                if bool(getattr(self.model_cfg, "match_fm_set_constraint", True)):
                    assignment = set_constrained_posterior(
                        matching_objective["matching_energy"],
                        constraint_prior.detach(),
                        source_mask=valid_sources,
                        temperature=max(self._latent_control_temperature(), 1e-6),
                        marginal_strength=float(
                            getattr(self.model_cfg, "match_fm_marginal_strength", 1.0)
                        ),
                        iterations=int(getattr(self.model_cfg, "match_fm_set_iterations", 12)),
                        damping=float(getattr(self.model_cfg, "match_fm_set_damping", 0.5)),
                        row_prior_scale=float(
                            getattr(self.model_cfg, "match_fm_row_prior_scale", 0.0)
                        ),
                    )
                if bool(getattr(self.model_cfg, "match_fm_confidence_shrinkage", False)):
                    assignment, assignment_confidence = confidence_shrink_posterior(
                        assignment,
                        constraint_prior,
                        source_mask=valid_sources,
                        power=float(
                            getattr(self.model_cfg, "match_fm_confidence_power", 1.0)
                        ),
                        max_confidence=float(
                            getattr(self.model_cfg, "match_fm_confidence_max", 1.0)
                        ),
                    )
                prior_kl, posterior_marginal = posterior_prior_kl(
                    assignment,
                    inference_prior,
                    source_mask=valid_sources,
                )
                prior_empirical_kl = distribution_kl(
                    inference_prior, empirical_prior, source_mask=valid_sources
                )

            low_count = None
            min_group_cells = int(getattr(self.model_cfg, "latent_control_min_perturb_cells", 0) or 0)
            if min_group_cells > 0 and "latent_control_perturb_group_size" in batch:
                group_size = batch["latent_control_perturb_group_size"].to(device).float()
                low_count = group_size < float(min_group_cells)
                valid_prior = torch.where(
                    valid_sources,
                    constraint_prior.clamp_min(1e-8),
                    torch.zeros_like(constraint_prior),
                )
                valid_prior = valid_prior / valid_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                prior_q = valid_prior.unsqueeze(1).expand_as(assignment)
                assignment = torch.where(low_count[:, None, None], prior_q, assignment)
            if self.training and self.online_anchor_em_enabled:
                if self._pending_online_anchor_em_update is not None:
                    raise RuntimeError("uncommitted online-EM update from previous batch")
                observed_delta = (
                    pert_emb.detach().float().unsqueeze(1)
                    - source_centroid.detach().float().unsqueeze(2)
                )
                if bool(
                    getattr(
                        self.model_cfg,
                        "latent_control_delta_center_genes",
                        True,
                    )
                ):
                    observed_delta = observed_delta - observed_delta.mean(
                        dim=-1, keepdim=True
                    )
                delta_features = torch.einsum(
                    "bksg,dg->bskd",
                    observed_delta,
                    self.match_fm_pca_basis.float(),
                )
                update_source_mask = valid_sources.unsqueeze(1).expand(
                    -1, pert_emb.shape[1], -1
                )
                self._pending_online_anchor_em_update = (
                    self.online_anchor_em.build_update(
                        cell_ids,
                        key_ids,
                        assignment.detach(),
                        delta_features,
                        step=int(self.global_step),
                        valid_mask=valid_cell_mask,
                        source_mask=update_source_mask,
                    )
                )
            diffusion_loss = cell_objective["diffusion_loss"]
            optimization_loss = expanded_return["model_output"].get(
                "optimization_loss_per_cell"
            )
            if optimization_loss is not None:
                diffusion_loss = optimization_loss.reshape(
                    bsz, k_val, pert_emb.shape[1]
                ).permute(0, 2, 1).contiguous()
            mixture_mse_per_cell = (assignment * diffusion_loss).sum(dim=-1)
            valid_float = valid_cell_mask.to(dtype=mixture_mse_per_cell.dtype)
            mixture_mse = (
                (mixture_mse_per_cell * valid_float).sum(dim=-1)
                / valid_float.sum(dim=-1).clamp_min(1.0)
            )
            generator_warmup_steps = int(
                getattr(self.model_cfg, "match_fm_generator_warmup_steps", 0) or 0
            )
            generator_active = int(self.global_step) >= generator_warmup_steps
            optimization_mse = mixture_mse if generator_active else mixture_mse.detach()

            source_mmd = expanded_return["mmd1_per_sample"].reshape(bsz, k_val)
            mmd = _responsibility_weighted_mmd(source_mmd, assignment)
            return_dict = {k: v for k, v in expanded_return.items() if k.startswith("dataset_loss_")}
            return_dict["mmd1"] = mmd
            return_dict["loss1"] = (mixture_mse * weights).mean()
            return_dict["loss1_per_sample"] = mixture_mse
            return_dict["loss"] = optimization_mse
            if anchor_output is not None:
                from src.models.adaptive_pairing import state_conditioned_pair_consistency
                from src.models.adaptive_state_anchor import (
                    anchor_parent_prior_kl,
                    anchor_sibling_diversity,
                    posterior_prior_distillation,
                )

                model_output = expanded_return["model_output"]
                if "anchor_delta" not in model_output:
                    raise RuntimeError(
                        "adaptive anchor is enabled but Cross-DiT returned no delta head"
                    )
                anchor_delta = model_output["anchor_delta"].reshape(
                    bsz, k_val, *pert_emb.shape[1:]
                )
                delta_target = (
                    pert_emb.unsqueeze(1)
                    - source_centroid.detach().unsqueeze(2).to(dtype=pert_emb.dtype)
                )
                delta_regression = F.smooth_l1_loss(
                    anchor_delta.float(),
                    delta_target.float(),
                    reduction="none",
                ).mean(dim=-1).permute(0, 2, 1)
                predicted_direction = anchor_delta.float()
                target_direction = delta_target.float()
                if bool(
                    getattr(
                        self.model_cfg,
                        "latent_control_delta_center_genes",
                        True,
                    )
                ):
                    predicted_direction = predicted_direction - predicted_direction.mean(
                        dim=-1, keepdim=True
                    )
                    target_direction = target_direction - target_direction.mean(
                        dim=-1, keepdim=True
                    )
                direction_per_candidate = (
                    1.0
                    - F.cosine_similarity(
                        predicted_direction,
                        target_direction,
                        dim=-1,
                        eps=1e-6,
                    )
                ).permute(0, 2, 1)
                detached_q = assignment.detach()
                delta_per_cell = (detached_q * delta_regression).sum(dim=-1)
                direction_per_cell = (
                    detached_q * direction_per_candidate
                ).sum(dim=-1)
                valid_denominator = valid_float.sum().clamp_min(1.0)
                anchor_delta_aux = (
                    delta_per_cell * valid_float
                ).sum() / valid_denominator
                anchor_direction_aux = (
                    direction_per_cell * valid_float
                ).sum() / valid_denominator

                pair_consistency = anchor_delta.new_zeros(())
                pair_active_states = anchor_delta.new_zeros(())
                if "meta_view_id" in batch:
                    num_views = int(
                        getattr(self.model_cfg, "adaptive_anchor_num_views", 3)
                    )
                    pair_consistency, pair_active_states = (
                        state_conditioned_pair_consistency(
                            anchor_delta=anchor_delta,
                            responsibilities=detached_q,
                            valid_cell_mask=valid_cell_mask,
                            source_mask=valid_sources,
                            num_views=num_views,
                            episode_ids=batch.get("meta_episode_id"),
                            view_ids=batch.get("meta_view_id"),
                            min_occupancy=float(
                                getattr(
                                    self.model_cfg,
                                    "adaptive_anchor_pair_min_occupancy",
                                    0.05,
                                )
                            )
                        )
                    )

                occupancy_distill, posterior_occupancy = (
                    posterior_prior_distillation(
                        assignment,
                        priors,
                        valid_cell_mask,
                    )
                )
                parent_prior_kl = anchor_parent_prior_kl(anchor_output)
                sibling_diversity = anchor_sibling_diversity(
                    anchor_output,
                    split_factor=int(
                        getattr(
                            self.model_cfg,
                            "adaptive_anchor_split_factor",
                            2,
                        )
                    ),
                    margin=float(
                        getattr(
                            self.model_cfg,
                            "adaptive_anchor_diversity_margin",
                            0.005,
                        )
                    ),
                )
                valid_slot = anchor_output.mask.to(
                    dtype=anchor_output.offsets.dtype
                )
                offset_l2 = (
                    anchor_output.offsets.float().square().mean(dim=-1)
                    * valid_slot.float()
                ).sum() / valid_slot.float().sum().clamp_min(1.0)
                objectness = anchor_output.objectness
                split_factor = int(
                    getattr(self.model_cfg, "adaptive_anchor_split_factor", 2)
                )
                slot_index = torch.arange(objectness.shape[-1], device=device) % split_factor
                optional_slot = anchor_output.mask & slot_index.unsqueeze(0).ne(0)
                objectness_l1 = (
                    torch.where(optional_slot, objectness, torch.zeros_like(objectness)).sum()
                    / optional_slot.sum().clamp_min(1)
                )
                hard_winner = F.one_hot(
                    assignment.detach().argmax(dim=-1),
                    num_classes=k_val,
                ).to(dtype=objectness.dtype)
                hard_count = (
                    hard_winner * valid_cell_mask.unsqueeze(-1).to(hard_winner)
                ).sum(dim=1)
                objectness_target = 1.0 - torch.exp(-hard_count)
                num_leaves = k_val // split_factor
                leaf_occupancy = posterior_occupancy.reshape(
                    bsz, num_leaves, split_factor
                )
                mandatory_child = F.one_hot(
                    leaf_occupancy.argmax(dim=-1),
                    num_classes=split_factor,
                ).reshape(bsz, k_val).to(dtype=objectness.dtype)
                objectness_target = torch.maximum(
                    objectness_target, mandatory_child
                )
                structural_mask = anchor_output.mask
                objectness_bce = F.binary_cross_entropy_with_logits(
                    anchor_output.objectness_logits.float(),
                    objectness_target.float(),
                    reduction="none",
                )
                objectness_bce = (
                    objectness_bce * structural_mask.float()
                ).sum() / structural_mask.float().sum().clamp_min(1.0)

                delta_aux_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_delta_loss_weight",
                        0.1,
                    )
                )
                direction_aux_weight = self._adaptive_anchor_scheduled_weight(
                    "direction_loss", 0.05, 500, 2000
                )
                pair_weight = self._adaptive_anchor_scheduled_weight(
                    "pair_consistency", 0.05, 2000, 4000
                )
                objectness_supervision_weight = self._adaptive_anchor_scheduled_weight(
                    "objectness_loss", 0.05, 2000, 4000
                )
                occupancy_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_occupancy_loss_weight",
                        0.05,
                    )
                )
                parent_prior_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_parent_prior_weight",
                        0.02,
                    )
                )
                offset_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_offset_l2_weight",
                        0.01,
                    )
                )
                diversity_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_diversity_weight",
                        0.01,
                    )
                )
                sparsity_weight = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_objectness_l1_weight",
                        0.001,
                    )
                )
                sparsity_start_step = int(getattr(
                    self.model_cfg, "adaptive_anchor_objectness_start_step", 10000
                ))
                if int(self.global_step) < sparsity_start_step:
                    sparsity_weight = 0.0
                anchor_regularizer = (
                    delta_aux_weight * anchor_delta_aux
                    + direction_aux_weight * anchor_direction_aux
                    + pair_weight * pair_consistency
                    + occupancy_weight * occupancy_distill
                    + parent_prior_weight * parent_prior_kl
                    + offset_weight * offset_l2
                    + diversity_weight * sibling_diversity
                    + objectness_supervision_weight * objectness_bce
                    + sparsity_weight * objectness_l1
                )
                if self.training:
                    return_dict["loss"] = return_dict["loss"] + anchor_regularizer
                active_threshold = float(
                    getattr(
                        self.model_cfg,
                        "adaptive_anchor_objectness_threshold",
                        0.5,
                    )
                )
                active_count = (
                    (objectness > active_threshold) & anchor_output.mask
                ).sum(dim=-1).float().mean()
                return_dict.update(
                    {
                        "adaptive_anchor_delta_aux": anchor_delta_aux,
                        "adaptive_anchor_direction_aux": anchor_direction_aux,
                        "adaptive_anchor_pair_consistency": pair_consistency,
                        "adaptive_anchor_pair_active_states": pair_active_states,
                        "adaptive_anchor_occupancy_kl": occupancy_distill,
                        "adaptive_anchor_parent_prior_kl": parent_prior_kl,
                        "adaptive_anchor_offset_l2": offset_l2,
                        "adaptive_anchor_sibling_diversity": sibling_diversity,
                        "adaptive_anchor_objectness_bce": objectness_bce,
                        "adaptive_anchor_objectness_target_count": (
                            objectness_target * structural_mask
                        ).sum(dim=-1).mean(),
                        "adaptive_anchor_gate_mean": model_output["anchor_gate_mean"],
                        "adaptive_anchor_gate_std": model_output["anchor_gate_std"],
                        "adaptive_anchor_objectness_l1": objectness_l1,
                        "adaptive_anchor_objectness_weight": torch.as_tensor(
                            sparsity_weight, device=device
                        ),
                        "adaptive_anchor_active_count": active_count,
                        "adaptive_anchor_posterior_effective_k": (
                            1.0
                            / posterior_occupancy.square().sum(
                                dim=-1
                            ).clamp_min(1e-8)
                        ).mean(),
                        "adaptive_anchor_residual_alpha": torch.as_tensor(
                            self._adaptive_anchor_residual_alpha(),
                            device=device,
                        ),
                    }
                )
            if bool(getattr(self.model, "parent_kv_enabled", False)):
                return_dict["parent_kv_residual_rms"] = model_output[
                    "parent_kv_residual_rms"
                ]
                return_dict["parent_kv_attention_entropy"] = model_output[
                    "parent_kv_attention_entropy"
                ]
                return_dict[
                    "parent_kv_attention_normalized_entropy"
                ] = model_output["parent_kv_attention_normalized_entropy"]
                return_dict[
                    "parent_kv_attention_max_probability"
                ] = model_output["parent_kv_attention_max_probability"]
            mmd_factor = float(self.optimizer_cfg.MMD_loss_factor)
            if mmd_factor != 0.0:
                optimization_mmd = mmd if generator_active else mmd.detach()
                return_dict["loss"] = return_dict["loss"] + optimization_mmd * mmd_factor
            if prior_kl is not None:
                prior_weight = float(getattr(self.model_cfg, "match_fm_prior_loss_weight", 0.1))
                return_dict["loss"] = return_dict["loss"] + prior_weight * prior_kl
                return_dict["latent_control_prior_kl"] = prior_kl.mean()
                return_dict["latent_control_prior_weight"] = torch.as_tensor(
                    prior_weight, device=device
                )
                empirical_weight = float(
                    getattr(self.model_cfg, "match_fm_prior_empirical_kl_weight", 0.0)
                )
                if empirical_weight != 0.0:
                    return_dict["loss"] = (
                        return_dict["loss"] + empirical_weight * prior_empirical_kl
                    )
                return_dict["match_fm_prior_empirical_kl"] = prior_empirical_kl.mean()
                return_dict["match_fm_prior_empirical_weight"] = torch.as_tensor(
                    empirical_weight, device=device
                )
                prior_entropy = -(
                    inference_prior
                    * inference_prior.clamp_min(1e-8).log()
                ).sum(dim=-1)
                marginal_entropy = -(
                    posterior_marginal
                    * posterior_marginal.clamp_min(1e-8).log()
                ).sum(dim=-1)
                return_dict["latent_control_inference_prior_entropy"] = prior_entropy.mean()
                return_dict["latent_control_posterior_marginal_entropy"] = marginal_entropy.mean()
                empirical = torch.where(
                    valid_sources,
                    priors.clamp_min(1e-8),
                    torch.zeros_like(priors),
                )
                empirical = empirical / empirical.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                return_dict["latent_control_prior_shift_l1"] = (
                    inference_prior - empirical
                ).abs().sum(dim=-1).mean()
            return_dict["weights"] = weights
            entropy = -(assignment * torch.log(assignment.clamp_min(1e-8))).sum(dim=-1)
            return_dict["latent_control_entropy"] = (
                entropy * valid_float
            ).sum() / valid_float.sum().clamp_min(1.0)
            max_probability = assignment.max(dim=-1).values
            return_dict["latent_control_max_prob"] = (
                max_probability * valid_float
            ).sum() / valid_float.sum().clamp_min(1.0)
            return_dict["latent_control_temperature"] = torch.as_tensor(
                self._latent_control_temperature(), device=device
            )
            return_dict["latent_control_delta_score_weight"] = torch.as_tensor(
                delta_weight, device=device
            )
            return_dict["latent_control_delta_magnitude_score_weight"] = torch.as_tensor(
                magnitude_weight, device=device
            )
            return_dict["latent_control_delta_loss"] = matching_objective["delta_loss"].mean()
            return_dict["latent_control_delta_magnitude_loss"] = matching_objective[
                "delta_magnitude_loss"
            ].mean()
            return_dict["match_fm_pca_delta_loss"] = matching_objective[
                "pca_delta_loss"
            ].mean()
            return_dict["match_fm_pca_score_weight"] = torch.as_tensor(
                float(getattr(self.model_cfg, "match_fm_pca_score_weight", 0.0)),
                device=device,
            )
            return_dict["match_fm_assignment_confidence"] = assignment_confidence.mean()
            return_dict["match_fm_prior_blend"] = torch.as_tensor(
                prior_blend, device=device
            )
            return_dict["match_fm_generator_active"] = torch.as_tensor(
                float(generator_active), device=device
            )
            return_dict["match_fm_assignment_cell_std"] = assignment.float().std(
                dim=1, unbiased=False
            ).mean()
            return_dict["match_fm_energy_source_std"] = matching_objective[
                "matching_energy"
            ].float().std(dim=-1, unbiased=False).mean()
            return_dict["match_fm_flow_energy_source_std"] = matching_objective[
                "diffusion_loss"
            ].detach().float().std(dim=-1, unbiased=False).mean()
            return_dict["match_fm_delta_energy_source_std"] = matching_objective[
                "delta_loss"
            ].float().std(dim=-1, unbiased=False).mean()
            return_dict["match_fm_delta_magnitude_energy_source_std"] = matching_objective[
                "delta_magnitude_loss"
            ].float().std(dim=-1, unbiased=False).mean()
            if online_em_prior is not None:
                return_dict["adaptive_anchor_online_history_weight"] = (
                    online_em_prior.history_weight.float().mean()
                )
                return_dict["adaptive_anchor_online_cache_hit"] = (
                    online_em_prior.cache_hit.float().mean()
                )
                return_dict["adaptive_anchor_online_history_confidence"] = (
                    online_em_prior.confidence.float().mean()
                )
                touched_keys = torch.unique(key_ids)
                return_dict["adaptive_anchor_online_active_count"] = (
                    self.online_anchor_em.active_mask.index_select(
                        0, touched_keys
                    ).float().sum(dim=-1).mean()
                )
            return_dict["match_fm_pca_energy_source_std"] = matching_objective[
                "pca_delta_loss"
            ].float().std(dim=-1, unbiased=False).mean()
            if teacher_time is not None:
                return_dict["match_fm_teacher_time_mean"] = teacher_time.float().mean()
                return_dict["match_fm_teacher_noise_level_mean"] = (
                    1.0 - teacher_time.float()
                ).mean()
            if low_count is not None:
                return_dict["latent_control_low_count_frac"] = low_count.float().mean()
            if hasattr(self.schedule_sampler, "update_with_local_losses"):
                self.schedule_sampler.update_with_local_losses(t, mixture_mse.detach())
            return return_dict

        candidate_loss = expanded_return["loss"].reshape(bsz, k_val)
        candidate_loss1 = expanded_return["loss1_per_sample"].reshape(bsz, k_val)
        valid = masks > 0
        temperature = max(self._latent_control_temperature(), 1e-6)
        log_prior = torch.log(torch.clamp(priors, min=1e-8))
        scores = log_prior - candidate_loss.detach() / temperature
        learned_score = None
        if self.latent_control_learned_scoring and "latent_control_cluster_id" in batch:
            cluster_ids = batch["latent_control_cluster_id"].to(device).long()
            max_id = self.latent_control_cluster_embedding.num_embeddings - 1
            cluster_ids = torch.clamp(cluster_ids, min=0, max=max_id)
            cluster_emb = self.latent_control_cluster_embedding(cluster_ids)
            score_input = torch.cat([cluster_emb, log_prior.unsqueeze(-1)], dim=-1)
            learned_score = self.latent_control_score(score_input).squeeze(-1)
            scores = scores + learned_score

        min_group_cells = int(getattr(self.model_cfg, "latent_control_min_perturb_cells", 0) or 0)
        low_count = None
        if min_group_cells > 0 and "latent_control_perturb_group_size" in batch:
            group_size = batch["latent_control_perturb_group_size"].to(device).float()
            low_count = group_size < float(min_group_cells)
            prior_scores = log_prior
            scores = torch.where(low_count[:, None], prior_scores, scores)

        scores = scores.masked_fill(~valid, -torch.inf)
        assignment = torch.softmax(scores, dim=1)
        assignment = torch.where(valid, assignment, torch.zeros_like(assignment))
        assignment = assignment / torch.clamp(assignment.sum(dim=1, keepdim=True), min=1e-8)

        mixture_loss = (assignment * candidate_loss).sum(dim=1)
        mixture_loss1 = (assignment * candidate_loss1).sum(dim=1)

        return_dict = {k: v for k, v in expanded_return.items() if k.startswith("dataset_loss_")}
        return_dict["mmd1"] = expanded_return["mmd1"]
        return_dict["loss1"] = (mixture_loss1 * weights).mean()
        return_dict["loss1_per_sample"] = mixture_loss1
        return_dict["loss"] = mixture_loss
        return_dict["weights"] = weights
        entropy = -(assignment * torch.log(torch.clamp(assignment, min=1e-8))).sum(dim=1)
        return_dict["latent_control_entropy"] = entropy.mean()
        return_dict["latent_control_max_prob"] = assignment.max(dim=1).values.mean()
        return_dict["latent_control_temperature"] = torch.as_tensor(temperature, device=device)
        if low_count is not None:
            return_dict["latent_control_low_count_frac"] = low_count.float().mean()
        if learned_score is not None:
            return_dict["latent_control_learned_score"] = (assignment * learned_score).sum(dim=1).mean()

        if hasattr(self.schedule_sampler, "update_with_local_losses"):
            self.schedule_sampler.update_with_local_losses(t, mixture_loss.detach())
        return return_dict

    def _compute_base_loss(
        self,
        batch,
        cont_emb_override=None,
        t_override=None,
        weights_override=None,
        noise_override=None,
        return_model_output=False,
        model_override=None,
        update_schedule_sampler=True,
        compute_mmd=None,
    ):
        """
        Compute a training loss value.

        :param batch: Current batch.
        :return: Loss tensor(s) and/or scalar metrics.
        """
        active_model = self.model if model_override is None else model_override
        batch["batch_emb"] = self._encode_covariates(batch)
        pert_emb = batch["pert_emb"]
        device = pert_emb.device

        if self.gene_embedding is None:
            gene_emb = None
        else:
            gene_emb = []
            for i, dataset_name in enumerate(batch["ds_name"]):
                dataset_name = dataset_name[0]
                if dataset_name not in self.gene_name_embedding_cache:
                    self.gene_name_embedding_cache[dataset_name] = torch.stack([self.gene_embedding.get(id, torch.zeros(5120)) for id in batch["col_genes"][i]])
                gene_emb.append(self.gene_name_embedding_cache[dataset_name])

            gene_emb = torch.stack(gene_emb).to(device)
        #batch["gene_emb"] = torch.stack([self.gene_embedding.get(id, torch.zeros(5120)) for id in batch["col_genes"]])
        #batch["gene_emb"] = batch["gene_emb"].expand(pert_emb.shape[0], -1, -1).to(device)

        cond = {"batch_emb": batch["batch_emb"],
                "cont_emb": batch["cont_emb"] if cont_emb_override is None else cont_emb_override,
                "gene_emb": gene_emb,
                "cov_celltype": batch["cov_celltype"],
                "cov_pert": batch["cov_pert"],
                "ds_name": batch["ds_name"],
                }
        if self.hungarian_flow_enabled:
            for key in (
                "hungarian_parent_token",
                "hungarian_child_tokens",
                "hungarian_routed_token",
            ):
                if key not in batch:
                    raise ValueError(f"Hungarian-flow loss requires batch[{key!r}]")
                cond[key] = batch[key]
        if self.cell_detr_enabled:
            for key in (
                "cell_detr_parent_token",
                "cell_detr_child_tokens",
                "cell_detr_routed_token",
            ):
                if key not in batch:
                    raise ValueError(f"cell_detr loss requires batch[{key!r}]")
                cond[key] = batch[key]
        if self.parent_residual_enabled:
            strict_parent_only = bool(
                getattr(active_model, "strict_parent_only_enabled", False)
            )
            required_keys = (
                ("parent_residual_parent_token",)
                if strict_parent_only
                else (
                    "parent_residual_parent_token",
                    "parent_residual_child_tokens",
                    "parent_residual_routed_token",
                )
            )
            for key in required_keys:
                if key not in batch:
                    raise ValueError(
                        f"parent-residual loss requires batch[{key!r}]"
                    )
                cond[key] = batch[key]
            if (
                not strict_parent_only
                and bool(
                    getattr(
                        active_model,
                        "strong_condition_memory_enabled",
                        False,
                    )
                )
            ):
                for key in (
                    "parent_residual_all_child_tokens",
                    "parent_residual_child_mask",
                    "parent_residual_perturbation_query",
                ):
                    if key not in batch:
                        raise ValueError(
                            "strong parent-residual memory requires "
                            f"batch[{key!r}]"
                        )
                    cond[key] = batch[key]
        if self.adaptive_anchor_enabled:
            if "anchor_baseline" not in batch:
                raise ValueError(
                    "adaptive-anchor loss requires batch['anchor_baseline']"
                )
            if "anchor_parent_context" not in batch:
                raise ValueError(
                    "adaptive-anchor loss requires batch['anchor_parent_context']"
                )
            cond["anchor_baseline"] = batch["anchor_baseline"]
            cond["anchor_parent_context"] = batch["anchor_parent_context"]
            if bool(getattr(active_model, "parent_kv_enabled", False)):
                if (
                    "anchor_parent_tokens" not in batch
                    or "anchor_parent_mask" not in batch
                ):
                    raise ValueError(
                        "Parent-KV loss requires parent tokens and their validity mask"
                    )
                cond["anchor_parent_tokens"] = batch["anchor_parent_tokens"]
                cond["anchor_parent_mask"] = batch["anchor_parent_mask"]
                if "anchor_perturbation_query" in batch:
                    cond["anchor_perturbation_query"] = batch[
                        "anchor_perturbation_query"
                    ]
            cond["anchor_residual_alpha"] = self._adaptive_anchor_residual_alpha()


        if t_override is None or weights_override is None:
            t, weights = self.schedule_sampler.sample(pert_emb.shape[0], device)
        else:
            t, weights = t_override, weights_override

        if compute_mmd is None:
            compute_mmd = float(self.optimizer_cfg.MMD_loss_factor) != 0.0
        mmd_loss_fn = self.loss_fn
        if not compute_mmd:
            mmd_loss_fn = lambda target, prediction: target.new_zeros(target.shape[0])

        diffusion_result = self.diffusion.training_losses(
            active_model,
            pert_emb, 
            t, 
            self_condition=cond, 
            model_kwargs=None, 
            noise=noise_override,
            p_drop_cond=self.model_cfg.p_drop_cond,
            MMD_loss_fn=mmd_loss_fn,
            return_model_output=return_model_output,
        )
        if return_model_output:
            losses, model_output = diffusion_result
        else:
            losses = diffusion_result
        return_dict = {}
        
        keys = list(set([(get_short_dsname(x)) for x in active_model.model_cfg.dataset_dict]))
        name_arr = np.array([get_short_dsname(x) for x in cond["ds_name"]])
        for ds_name in keys:
            if (name_arr == ds_name).any():
                return_dict[f"dataset_loss_mse1_{ds_name}"] = losses["loss1"][name_arr == ds_name].nanmean().item()
                return_dict[f"dataset_loss_mmd1_{ds_name}"] = losses["mmd1_list"][name_arr == ds_name].nanmean().item()
            else:
                return_dict[f"dataset_loss_mse1_{ds_name}"] =  0
                return_dict[f"dataset_loss_mmd1_{ds_name}"] = 0
        
        assert active_model.model_name == "Cross_DiT"
        losses["loss"] = losses["loss1"]
        
        if not self.optimizer_cfg.use_mse_loss:
            losses["loss"] = 0

        mmd_factor = float(self.optimizer_cfg.MMD_loss_factor)
        if mmd_factor != 0.0:
            losses["loss"] += losses["mmd1"] * mmd_factor
        return_dict["mmd1"] = losses["mmd1"]
        return_dict["mmd1_per_sample"] = losses["mmd1_per_sample"]

        return_dict["loss1"] = (losses["loss1"] * weights).mean()
        return_dict["loss1_per_sample"] = losses["loss1"]

        if update_schedule_sampler and hasattr(self.schedule_sampler, "update_with_local_losses"):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        return_dict["loss"] = losses["loss"]
        return_dict["weights"] = weights
        if return_model_output:
            return_dict["model_output"] = model_output

        return return_dict

    def configure_optimizers(self):
        """
        return optimizers and schedulers
        """
        decay_names = set(get_parameter_names(self.model, [torch.nn.LayerNorm]))
        decay_names_cov = set(get_parameter_names(self.cov_encoder, [torch.nn.LayerNorm]))
        decay_names.update(decay_names_cov)
        decay_names = {n for n in decay_names if "bias" not in n and "layer_norm" not in n and "layernorm" not in n}

        adaptive_params = []
        if self.adaptive_anchor_enabled:
            adaptive_params.extend(list(self.model.anchor_delta_head.parameters()))
            adaptive_params.extend(list(self.model.anchor_gate_head.parameters()))
            adaptive_params.extend(list(self.model.anchor_parent_projector.parameters()))
            adaptive_params.extend(list(self.model.parent_kv_adapters.parameters()))
            if self.model.anchor_perturbation_projector is not None:
                adaptive_params.extend(
                    list(self.model.anchor_perturbation_projector.parameters())
                )
            adaptive_params.extend(list(self.adaptive_state_anchor.parameters()))
        adaptive_params = [
            parameter for parameter in adaptive_params if parameter.requires_grad
        ]
        adaptive_param_ids = {id(parameter) for parameter in adaptive_params}

        params_decay, params_nodecay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in adaptive_param_ids:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        for n, p in self.cov_encoder.named_parameters():
            if not p.requires_grad:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        extra_params = []
        if self.match_fm_enabled:
            extra_params.extend(list(self.match_fm_prior.parameters()))
        if self.latent_control_learned_scoring:
            extra_params.extend(list(self.latent_control_cluster_embedding.parameters()))
            extra_params.extend(list(self.latent_control_score.parameters()))
        extra_params = [
            parameter for parameter in extra_params if parameter.requires_grad
        ]

        main_groups = [
            {"params": params_decay, "weight_decay": self.optimizer_cfg.optimizer.weight_decay},
            {"params": params_nodecay + extra_params, "weight_decay": 0.0},
        ]
        if adaptive_params:
            multiplier = float(getattr(self.model_cfg, "adaptive_anchor_lr_multiplier", 5.0))
            adaptive_lr = float(self.optimizer_cfg.optimizer.lr) * multiplier
            main_groups.append({"params": adaptive_params, "weight_decay": 0.0, "lr": adaptive_lr})
        grouped_params = [parameter for group in main_groups for parameter in group["params"]]
        assert len(grouped_params) == len({id(parameter) for parameter in grouped_params})
        expected_ids = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        assert {id(parameter) for parameter in grouped_params} == expected_ids
        opt_main = get_optimizer(main_groups, self.optimizer_cfg.optimizer)
        
        sched_main = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=opt_main)
        return [opt_main], [{"scheduler": sched_main, "interval": "step", "frequency": 1, "monitor": "validation_loss", }]

    def on_after_backward(self):
        """Fail fast on non-finite gradients in newly introduced modules."""
        if not self.adaptive_anchor_enabled:
            return
        step = int(self.global_step) + 1
        warmup_steps = int(getattr(self.model_cfg, "adaptive_anchor_gradient_finite_warmup_steps", 20))
        interval = int(getattr(self.model_cfg, "adaptive_anchor_gradient_finite_interval", 1000))
        if step > warmup_steps and (interval <= 0 or step % interval != 0):
            return
        modules = (
            ("adaptive_state_anchor", self.adaptive_state_anchor),
            ("anchor_delta_head", self.model.anchor_delta_head),
            ("parent_kv_adapters", self.model.parent_kv_adapters),
            ("anchor_gate_head", self.model.anchor_gate_head),
            ("anchor_parent_projector", self.model.anchor_parent_projector),
        )
        for module_name, module in modules:
            for parameter_name, parameter in module.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient: {module_name}.{parameter_name} at step {step}")

    def on_before_optimizer_step(self, optimizer):
        """Log full and grouped gradient norms at the configured interval."""
        if (self.cell_detr_enabled or self.cell_detr_direct_enabled) and torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated() / float(1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved() / float(1024 ** 3)
            self.log(
                "cell_detr_peak_allocated_gib",
                peak_allocated,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "cell_detr_peak_reserved_gib",
                peak_reserved,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if int(self.global_step) == 0 and int(self.global_rank) == 0:
                self.py_logger.info(
                    "cell_detr CUDA peak after backward: allocated=%.3f GiB reserved=%.3f GiB",
                    peak_allocated,
                    peak_reserved,
                )
        is_source_model = str(self.model_cfg.model_type).lower() == "source_cross_dit"
        interval_key = "source_grad_norm_every" if is_source_model else "grad_norm_every"
        interval = int(getattr(self.model_cfg, interval_key, 1000))
        if interval <= 0 or (int(self.global_step) + 1) % interval != 0:
            return
        norms = grad_norm(self.model, norm_type=2)
        device = next(self.parameters()).device
        norms = {key: value.to(device) for key, value in norms.items()}

        if is_source_model:
            grouped_squares = {
                "encoder": torch.zeros((), device=device),
                "adapter": torch.zeros((), device=device),
                "backbone": torch.zeros((), device=device),
            }
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                squared = parameter.grad.detach().float().square().sum()
                if "source_encoder" in name:
                    grouped_squares["encoder"] += squared
                elif "source_adapter" in name:
                    grouped_squares["adapter"] += squared
                else:
                    grouped_squares["backbone"] += squared
            encoder_norm = grouped_squares["encoder"].sqrt()
            adapter_norm = grouped_squares["adapter"].sqrt()
            backbone_norm = grouped_squares["backbone"].sqrt()
            compact = {
                "source/encoder_grad_norm": encoder_norm,
                "source/adapter_grad_norm": adapter_norm,
                "source/backbone_grad_norm_ratio": (
                    (encoder_norm + adapter_norm) / backbone_norm.clamp_min(1e-12)
                ),
            }
            norms.update(compact)
            self.py_logger.info(
                "source_gradient_metrics=%s",
                {
                    "global_step": int(self.global_step) + 1,
                    **{
                        key: self._metric_float(value)
                        for key, value in compact.items()
                    },
                },
            )

        self.log_dict(
            norms,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
        )

"""Factory and builder utilities split from lightning_module (logic-preserving)."""

import os
import torch
from torch.optim import AdamW, Adam

from src.models.rectified_flow import RectifiedFlow
from src.models.resampling import LogitNormalSampler, UniformSampler
from src.models.weight_averaging_callback import WeightAveraging


class EMAWeightAveraging(WeightAveraging):
    """Emaweightaveraging implementation used by the PerturbDiff pipeline."""
    def __init__(self, decay=0.999, update_steps=100):
        """Special method `__init__`."""
        self._update_steps = update_steps
        # Never average buffers. AveragedModel(use_buffers=False) still copies
        # current buffers into the averaged module on every update, while
        # avoiding floating-point EMA followed by truncating copy-back for
        # integer runtime-ID tables and boolean masks.
        super().__init__(
            use_buffers=False,
            avg_fn=torch.optim.swa_utils.get_ema_avg_fn(decay=decay),
        )

    def should_update(self, step_idx=None, epoch_idx=None):
        """Execute `should_update` and return values used by downstream logic."""
        if epoch_idx is not None:
            return True
        elif step_idx is not None:
            return step_idx % self._update_steps == 0
        else:
            raise ValueError("step_idx and epoch_idx cannot be both None.")


def model_init_fn(model_cfg, cov_cfg=None):
    """Execute `model_init_fn` and return values used by downstream logic."""
    model_type = model_cfg.model_type.lower()
    if model_type != "parent_residual_gene_dit":
        raise NotImplementedError(
            "This release contains only model_type=parent_residual_gene_dit"
        )
    if bool(
        getattr(
            model_cfg,
            "parent_residual_donor_aware_delta_enabled",
            False,
        )
    ):
        from src.models.donor_aware_parent_delta.artifact_model import (
            ArtifactDonorAwareParentResidualGeneDiTModel as ParentResidualGeneDiTModel,
        )
    else:
        from src.models.parent_residual_gene_dit.wrapper import (
            ParentResidualGeneDiTModel,
        )

    model = ParentResidualGeneDiTModel(
        model_cfg=model_cfg,
        covariate_config=cov_cfg,
    )
    return model


def create_diffusion(model_cfg):
    """Create the rectified-flow process used by every released model."""
    process = str(getattr(model_cfg, "generative_process", "diffusion")).lower()
    if process != "rectified_flow":
        raise ValueError(
            "This release contains only generative_process=rectified_flow"
        )
    return RectifiedFlow(
        num_timesteps=int(model_cfg.steps),
        sampling_steps=int(getattr(model_cfg, "flow_sampling_steps", 50)),
        time_scale=float(getattr(model_cfg, "flow_time_scale", 1000.0)),
        final_nonnegative=bool(getattr(model_cfg, "flow_final_nonnegative", True)),
        final_max=getattr(model_cfg, "flow_final_max", 0.85),
        time_mean=float(getattr(model_cfg, "flow_time_mean", -0.8)),
        time_std=float(getattr(model_cfg, "flow_time_std", 0.8)),
    )


def get_optimizer(optim_groups, optimizer_cfg):
    """Execute `get_optimizer` and return values used by downstream logic."""
    optim_cls = AdamW if optimizer_cfg.adam_w_mode else Adam
    if hasattr(optimizer_cfg, "lr"):
        return optim_cls(
            optim_groups,
            lr=optimizer_cfg.lr,
            eps=optimizer_cfg.eps,
            betas=(optimizer_cfg.betas[0], optimizer_cfg.betas[1]),
        )
    else:
        return optim_cls(
            optim_groups,
            eps=optimizer_cfg.eps,
            betas=(optimizer_cfg.betas[0], optimizer_cfg.betas[1]),
        )


def create_named_schedule_sampler(name, diffusion):
    """Execute `create_named_schedule_sampler` and return values used by downstream logic."""
    if getattr(diffusion, "is_rectified_flow", False):
        return LogitNormalSampler(diffusion)
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

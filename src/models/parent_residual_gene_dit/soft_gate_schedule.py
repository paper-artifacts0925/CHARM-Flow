"""Training-only blend schedule for soft-gated Parent regression."""

from __future__ import annotations

import math

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class SoftGateBlendSchedule(pl.Callback):
    """Warm the Parent soft gate from the legacy path to its target blend.

    The schedule is a pure function of ``trainer.global_step``.  It therefore
    resumes at the correct blend without storing mutable progress in the
    checkpoint.  Validation, test, prediction, and completed training always
    use the target blend.
    """

    def __init__(self, target_blend: float = 0.25, warmup_steps: int = 500):
        super().__init__()
        self.target_blend = float(target_blend)
        try:
            parsed_warmup_steps = int(warmup_steps)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("warmup_steps must be a non-negative integer") from error
        if not math.isfinite(self.target_blend):
            raise ValueError("target_blend must be finite")
        if not 0.0 <= self.target_blend <= 1.0:
            raise ValueError("target_blend must lie in [0, 1]")
        if (
            isinstance(warmup_steps, bool)
            or parsed_warmup_steps != warmup_steps
            or parsed_warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a non-negative integer")
        self.warmup_steps = parsed_warmup_steps
        self._validated_model_id = None
        self._first_train_batch_seen = False
        self._boundary_markers_logged = set()

    @property
    def state_key(self):
        return (
            f"{self.__class__.__qualname__}:"
            f"target={self.target_blend}:warmup={self.warmup_steps}"
        )

    def _parent_model(self, pl_module):
        model = getattr(pl_module, "model", None)
        if model is None:
            raise RuntimeError(
                "SoftGateBlendSchedule requires pl_module.model"
            )
        if not bool(getattr(model, "soft_gated_regression_enabled", False)):
            raise RuntimeError(
                "SoftGateBlendSchedule requires soft-gated Parent regression"
            )
        if not hasattr(model, "soft_gate_blend"):
            raise RuntimeError(
                "soft-gated Parent model is missing soft_gate_blend"
            )
        model_id = id(model)
        if self._validated_model_id != model_id:
            configured_warmup = int(
                getattr(model, "soft_gate_warmup_steps", self.warmup_steps)
            )
            if configured_warmup != self.warmup_steps:
                raise RuntimeError(
                    "model.parent_residual_soft_gate_warmup_steps must match "
                    "SoftGateBlendSchedule.warmup_steps"
                )
            configured_target = float(model.soft_gate_blend)
            if not math.isclose(
                configured_target,
                self.target_blend,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    "model.parent_residual_soft_gate_blend must match "
                    "SoftGateBlendSchedule.target_blend"
                )
            self._validated_model_id = model_id
        return model

    def blend_at_step(self, step: int) -> float:
        """Return zero at step 0 and the target at/after warmup_steps."""

        parsed_step = int(step)
        if isinstance(step, bool) or parsed_step != step or parsed_step < 0:
            raise ValueError("step must be a non-negative integer")
        if self.warmup_steps == 0:
            return self.target_blend
        progress = min(float(parsed_step) / float(self.warmup_steps), 1.0)
        return self.target_blend * progress

    def _set_blend(self, pl_module, blend: float) -> None:
        self._parent_model(pl_module).soft_gate_blend = float(blend)

    def _restore_target(self, pl_module) -> None:
        self._set_blend(pl_module, self.target_blend)

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
    ) -> None:
        del batch, batch_idx
        step = int(trainer.global_step)
        blend = self.blend_at_step(step)
        self._set_blend(pl_module, blend)
        log = getattr(pl_module, "log", None)
        if callable(log):
            log(
                "parent_residual_soft_gate_effective_blend",
                blend,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        boundary = step in {0, self.warmup_steps}
        first_batch = not self._first_train_batch_seen
        if first_batch or (
            boundary and step not in self._boundary_markers_logged
        ):
            rank_zero_info(
                "soft_gate_blend_schedule step=%d blend=%.8f "
                "target=%.8f warmup_steps=%d",
                step,
                blend,
                self.target_blend,
                self.warmup_steps,
            )
        self._first_train_batch_seen = True
        if boundary:
            self._boundary_markers_logged.add(step)

    def on_validation_start(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)

    def on_predict_start(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:
        del trainer
        self._restore_target(pl_module)


__all__ = ["SoftGateBlendSchedule"]


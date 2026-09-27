"""Parent-locked rectified flow operating only in centred residual space."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .centering import project_group_zero_mean
from .coordinates import (
    NonnegativeFixedMeanProjection,
    ParentLockedResidualPair,
    compose_parent_residual,
    prepare_mean_carrying_target,
    prepare_source_residual,
    prepare_training_residual_pair,
)


@dataclass(frozen=True)
class ParentLockedSample:
    expression: torch.Tensor
    residual: torch.Tensor
    source_residual: torch.Tensor
    endpoint_mean: Optional[torch.Tensor] = None
    integrated_mean_shift: Optional[torch.Tensor] = None


class ParentLockedResidualFlow:
    """A small process adapter compatible with ParentResidualGeneDiTModel.

    The neural network sees only residual coordinates.  The protected Parent
    expression is added once, after integration, through
    :func:`compose_parent_residual`.  Both training and inference call the same
    source-coordinate function, removing the former train/inference ambiguity.
    """

    is_rectified_flow = True
    is_parent_locked_residual_flow = True

    def __init__(
        self,
        *,
        sampling_steps: int = 50,
        time_scale: float = 1000.0,
        child_mean_flow_enabled: bool = False,
        child_mean_loss_weight: float = 1.0,
        child_mean_endpoint_blend: float = 1.0,
        common_velocity_loss_weight: float = 0.0,
    ):
        self.sampling_steps = int(sampling_steps)
        self.time_scale = float(time_scale)
        self.child_mean_flow_enabled = bool(child_mean_flow_enabled)
        self.child_mean_loss_weight = float(child_mean_loss_weight)
        self.child_mean_endpoint_blend = float(child_mean_endpoint_blend)
        self.common_velocity_loss_weight = float(common_velocity_loss_weight)
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if not torch.isfinite(torch.tensor(self.time_scale)) or self.time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if (
            not torch.isfinite(torch.tensor(self.child_mean_loss_weight))
            or self.child_mean_loss_weight < 0
        ):
            raise ValueError("child_mean_loss_weight must be finite and nonnegative")
        if (
            not torch.isfinite(torch.tensor(self.child_mean_endpoint_blend))
            or not 0.0 <= self.child_mean_endpoint_blend <= 1.0
        ):
            raise ValueError("child_mean_endpoint_blend must lie in [0,1]")
        if (
            not torch.isfinite(torch.tensor(self.common_velocity_loss_weight))
            or self.common_velocity_loss_weight < 0
        ):
            raise ValueError(
                "common_velocity_loss_weight must be finite and nonnegative"
            )
        if self.child_mean_flow_enabled and self.common_velocity_loss_weight != 0.0:
            raise ValueError(
                "common_velocity_loss_weight must be zero when Child-mean flow "
                "uses a nonzero common-velocity target"
            )

    @staticmethod
    def _time_for_set(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(time) or not time.is_floating_point():
            raise TypeError("time must be a floating-point tensor")
        batch_size = reference.shape[0]
        if time.ndim == 1 and time.shape[0] == batch_size:
            expanded = time[:, None, None]
        elif time.ndim == 2 and tuple(time.shape) == (batch_size, 1):
            expanded = time[:, :, None]
        else:
            raise ValueError("time must have shape [B] or [B,1]")
        if not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
            raise ValueError("continuous flow time must lie in [0,1]")
        return expanded.to(device=reference.device, dtype=reference.dtype)

    def _predict_raw_velocity(
        self,
        model,
        current: torch.Tensor,
        time: torch.Tensor,
        source_residual: torch.Tensor,
        self_condition: Dict,
        model_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        if model_kwargs is None:
            model_kwargs = {}
        condition = copy.copy(self_condition)
        condition["cont_emb"] = source_residual
        condition["parent_locked_residual_space"] = True
        zeros = torch.zeros_like(current)
        model_time = time * self.time_scale
        output = model(
            torch.cat([current, zeros], dim=-1),
            torch.cat([source_residual, zeros], dim=-1),
            model_time[:, None],
            self_condition=condition,
            **model_kwargs,
        )
        velocity = output["x"] if isinstance(output, dict) else output
        if not torch.is_tensor(velocity) or velocity.shape != current.shape:
            raise ValueError("velocity model must return x with shape [B,S,G]")
        if not torch.isfinite(velocity).all():
            raise ValueError("velocity model returned non-finite values")
        return velocity

    def _project_velocity(
        self,
        velocity: torch.Tensor,
        group_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return project_group_zero_mean(
            velocity,
            group_ids=group_ids,
            valid_mask=valid_mask,
            singleton_policy="keep",
        ).value

    @staticmethod
    def _masked_loss(
        squared_error: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-batch and scalar cell-masked losses."""

        numeric_mask = mask.to(squared_error)
        raw_denominator = numeric_mask.sum(dim=1)
        per_batch = (squared_error * numeric_mask).sum(dim=1) / (
            raw_denominator.clamp_min(1.0)
        )
        loss = per_batch.sum() / (raw_denominator > 0).sum().clamp_min(1)
        return per_batch, loss

    @staticmethod
    def _split_child_velocity(
        velocity: torch.Tensor,
        group_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split into group-common and centred modes.

        Child mode deliberately assigns all singleton velocity to the common
        mode.  A singleton can supervise/integrate its endpoint mean, but it
        contains no evidence for within-condition heterogeneity.
        """

        projection = project_group_zero_mean(
            velocity,
            group_ids=group_ids,
            valid_mask=valid_mask,
            singleton_policy="zero",
        )
        cell_mask = projection.valid_mask[..., None]
        centred = projection.value
        common = torch.where(cell_mask, velocity - centred, torch.zeros_like(velocity))
        return centred, common

    def _child_training_losses(
        self,
        model,
        *,
        pair: ParentLockedResidualPair,
        target_condition_mean: torch.Tensor,
        target_condition_mean_mask: Optional[torch.Tensor],
        time: torch.Tensor,
        self_condition: Dict,
        group_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        model_kwargs: Optional[Dict],
    ) -> Dict[str, torch.Tensor | ParentLockedResidualPair]:
        expanded_time = self._time_for_set(time, pair.target_residual)
        mean_target = prepare_mean_carrying_target(
            target_condition_mean=target_condition_mean,
            effective_parent=pair.source_projection.effective_parent,
            reference=pair.target_residual,
            group_ids=group_ids,
            valid_mask=valid_mask,
            supervision_mask=target_condition_mean_mask,
        )

        current_centred = (
            (1.0 - expanded_time) * pair.source_residual
            + expanded_time * pair.target_residual
        )
        current_mean_shift = expanded_time * mean_target.mean_shift
        current = current_centred + current_mean_shift
        target_centred_velocity = pair.target_residual - pair.source_residual
        target_common_velocity = mean_target.mean_shift
        raw_velocity = self._predict_raw_velocity(
            model,
            current,
            time.to(current),
            pair.source_residual,
            self_condition,
            model_kwargs,
        )
        centred_velocity, common_velocity = self._split_child_velocity(
            raw_velocity, group_ids, valid_mask
        )

        centred_squared_error = (
            centred_velocity - target_centred_velocity
        ).square().mean(dim=-1)
        residual_loss_per_batch, residual_loss = self._masked_loss(
            centred_squared_error, pair.target_projection.valid_mask
        )
        common_squared_error = (
            common_velocity - target_common_velocity
        ).square().mean(dim=-1)
        child_mean_loss_per_batch, child_mean_loss = self._masked_loss(
            common_squared_error, mean_target.supervision_mask
        )
        loss_per_batch = residual_loss_per_batch + (
            self.child_mean_loss_weight * child_mean_loss_per_batch
        )
        loss = residual_loss + self.child_mean_loss_weight * child_mean_loss

        endpoint_centred = current_centred + (
            1.0 - expanded_time
        ) * centred_velocity
        endpoint_centred = project_group_zero_mean(
            endpoint_centred,
            group_ids=group_ids,
            valid_mask=valid_mask,
            singleton_policy="zero",
        ).value
        integrated_mean_shift = current_mean_shift + (
            1.0 - expanded_time
        ) * common_velocity
        endpoint_mean = pair.source_projection.effective_parent + (
            self.child_mean_endpoint_blend * integrated_mean_shift
        )
        composed = compose_parent_residual(
            parent_mean=endpoint_mean,
            residual=endpoint_centred,
            group_ids=group_ids,
            valid_mask=valid_mask,
            enforce_zero_mean=True,
            detach_parent_mean=False,
        )
        common_velocity_rms = common_velocity.float().square().mean().sqrt()
        return {
            "loss": loss,
            "loss_per_batch": loss_per_batch,
            "centered_flow_loss": residual_loss,
            "centered_flow_loss_per_batch": residual_loss_per_batch,
            "velocity_mse_per_cell": centred_squared_error,
            "raw_velocity": raw_velocity,
            # Keep these two compatibility fields in centred-residual space.
            "velocity": centred_velocity,
            "target_velocity": target_centred_velocity,
            "child_target_velocity": (
                target_centred_velocity + target_common_velocity
            ),
            "common_velocity": common_velocity,
            "target_common_velocity": target_common_velocity,
            "current_residual": current_centred,
            "current_mean_shift": current_mean_shift,
            "endpoint_residual": composed.residual,
            "prediction": composed.expression,
            "child_mean_loss": child_mean_loss,
            "child_mean_loss_per_batch": child_mean_loss_per_batch,
            "child_mean_mse_per_cell": common_squared_error,
            "child_mean_supervision_mask": mean_target.supervision_mask,
            "child_endpoint_mean": composed.parent_mean,
            "integrated_mean_shift": integrated_mean_shift[:, 0, :],
            "common_velocity_rms": common_velocity_rms,
            # Retain the old diagnostic key for downstream log compatibility.
            "removed_common_velocity_rms": common_velocity_rms,
            "pair": pair,
        }

    def training_losses(
        self,
        model,
        *,
        target_expression: torch.Tensor,
        source_expression: torch.Tensor,
        parent_mean: torch.Tensor,
        target_condition_mean: Optional[torch.Tensor],
        time: torch.Tensor,
        self_condition: Dict,
        group_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        target_condition_mean_mask: Optional[torch.Tensor] = None,
        target_projection_override: Optional[object] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor | ParentLockedResidualPair]:
        """Compute residual velocity matching with a protected Parent skip."""

        pair = prepare_training_residual_pair(
            target_expression=target_expression,
            source_expression=source_expression,
            parent_mean=parent_mean,
            target_condition_mean=target_condition_mean,
            group_ids=group_ids,
            valid_mask=valid_mask,
            target_projection_override=target_projection_override,
        )
        if self.child_mean_flow_enabled:
            if target_condition_mean is None:
                raise ValueError(
                    "child mean-carrying flow requires target_condition_mean"
                )
            return self._child_training_losses(
                model,
                pair=pair,
                target_condition_mean=target_condition_mean,
                target_condition_mean_mask=target_condition_mean_mask,
                time=time,
                self_condition=self_condition,
                group_ids=group_ids,
                valid_mask=valid_mask,
                model_kwargs=model_kwargs,
            )
        expanded_time = self._time_for_set(time, pair.target_residual)
        current = (
            (1.0 - expanded_time) * pair.source_residual
            + expanded_time * pair.target_residual
        )
        target_velocity = pair.target_residual - pair.source_residual
        raw_velocity = self._predict_raw_velocity(
            model,
            current,
            time.to(current),
            pair.source_residual,
            self_condition,
            model_kwargs,
        )
        velocity = self._project_velocity(raw_velocity, group_ids, valid_mask)
        squared_error = (velocity - target_velocity).square().mean(dim=-1)
        mask = pair.target_projection.valid_mask.to(squared_error)
        per_batch_denominator = mask.sum(dim=1).clamp_min(1.0)
        loss_per_batch = (squared_error * mask).sum(dim=1) / per_batch_denominator
        centered_flow_loss_per_batch = loss_per_batch
        centered_flow_loss = loss_per_batch.sum() / (
            per_batch_denominator > 0
        ).sum().clamp_min(1)

        # ``velocity`` is the only component used by the parent-locked flow,
        # so the group-common part of ``raw_velocity`` otherwise lies in an
        # exact null space of the objective. Give that nuisance mode its
        # mathematically correct zero target without changing the represented
        # centred vector field. The reduction deliberately reuses the flow
        # mask so padding and per-batch weighting retain their semantics.
        removed_common_velocity = raw_velocity.to(velocity) - velocity
        common_squared_error = removed_common_velocity.float().square().mean(
            dim=-1
        )
        common_velocity_loss_per_batch, common_velocity_loss = self._masked_loss(
            common_squared_error,
            pair.target_projection.valid_mask,
        )
        if self.common_velocity_loss_weight > 0.0:
            loss_per_batch = centered_flow_loss_per_batch + (
                self.common_velocity_loss_weight
                * common_velocity_loss_per_batch.to(centered_flow_loss_per_batch)
            )
            loss = centered_flow_loss + (
                self.common_velocity_loss_weight
                * common_velocity_loss.to(centered_flow_loss)
            )
        else:
            # Preserve the literal historical objective when the new
            # stabilizer is disabled.
            loss = centered_flow_loss

        endpoint_residual = current + (1.0 - expanded_time) * velocity
        composed = compose_parent_residual(
            parent_mean=pair.parent_mean,
            residual=endpoint_residual,
            group_ids=group_ids,
            valid_mask=valid_mask,
            enforce_zero_mean=True,
        )
        return {
            "loss": loss,
            "loss_per_batch": loss_per_batch,
            "centered_flow_loss": centered_flow_loss,
            "centered_flow_loss_per_batch": centered_flow_loss_per_batch,
            "common_velocity_loss": common_velocity_loss,
            "common_velocity_loss_per_batch": common_velocity_loss_per_batch,
            "weighted_common_velocity_loss": (
                self.common_velocity_loss_weight * common_velocity_loss
            ),
            "velocity_mse_per_cell": squared_error,
            "raw_velocity": raw_velocity,
            "velocity": velocity,
            "target_velocity": target_velocity,
            "current_residual": current,
            "endpoint_residual": composed.residual,
            "prediction": composed.expression,
            "removed_common_velocity_rms": removed_common_velocity.float()
            .square()
            .mean()
            .sqrt(),
            "pair": pair,
        }

    def _guided_velocity(
        self,
        model,
        current: torch.Tensor,
        time: torch.Tensor,
        source_residual: torch.Tensor,
        self_condition: Dict,
        model_kwargs: Optional[Dict],
        guidance_strength: float,
        group_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        conditional = self._project_velocity(
            self._predict_raw_velocity(
                model,
                current,
                time,
                source_residual,
                self_condition,
                model_kwargs,
            ),
            group_ids,
            valid_mask,
        )
        strength = float(guidance_strength)
        if strength == 0.0:
            return conditional
        unconditional_condition = copy.copy(self_condition)
        unconditional_condition["batch_emb"] = None
        unconditional = self._project_velocity(
            self._predict_raw_velocity(
                model,
                current,
                time,
                source_residual,
                unconditional_condition,
                model_kwargs,
            ),
            group_ids,
            valid_mask,
        )
        # A linear combination of zero-mean residual velocities remains
        # zero-mean, including under classifier-free guidance.
        return conditional + strength * (conditional - unconditional)

    def _guided_raw_velocity(
        self,
        model,
        current: torch.Tensor,
        time: torch.Tensor,
        source_residual: torch.Tensor,
        self_condition: Dict,
        model_kwargs: Optional[Dict],
        guidance_strength: float,
    ) -> torch.Tensor:
        """Apply CFG in raw Child space before common/centred decomposition."""

        conditional = self._predict_raw_velocity(
            model,
            current,
            time,
            source_residual,
            self_condition,
            model_kwargs,
        )
        strength = float(guidance_strength)
        if strength == 0.0:
            return conditional
        unconditional_condition = copy.copy(self_condition)
        unconditional_condition["batch_emb"] = None
        unconditional = self._predict_raw_velocity(
            model,
            current,
            time,
            source_residual,
            unconditional_condition,
            model_kwargs,
        )
        return conditional + strength * (conditional - unconditional)

    def _sample_child_mean(
        self,
        model,
        *,
        source_projection,
        self_condition: Dict,
        group_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        model_kwargs: Optional[Dict],
        guidance_strength: float,
        sampling_steps: Optional[int],
    ) -> ParentLockedSample:
        source_residual = source_projection.residual
        current_centred = source_residual.clone()
        integrated_mean_shift = torch.zeros_like(source_residual)
        steps = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        if steps < 1:
            raise ValueError("sampling_steps must be positive")
        times = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=current_centred.device,
            dtype=current_centred.dtype,
        )
        for index in range(steps):
            start = times[index]
            end = times[index + 1]
            dt = end - start
            start_time = start.expand(current_centred.shape[0])
            current = current_centred + integrated_mean_shift
            raw_velocity = self._guided_raw_velocity(
                model,
                current,
                start_time,
                source_residual,
                self_condition,
                model_kwargs,
                guidance_strength,
            )
            centred_velocity, common_velocity = self._split_child_velocity(
                raw_velocity, group_ids, valid_mask
            )
            proposed_centred = current_centred + dt * centred_velocity
            proposed_centred = project_group_zero_mean(
                proposed_centred,
                group_ids=group_ids,
                valid_mask=valid_mask,
                singleton_policy="zero",
            ).value
            proposed_mean_shift = integrated_mean_shift + dt * common_velocity
            if index == steps - 1:
                current_centred = proposed_centred
                integrated_mean_shift = proposed_mean_shift
            else:
                proposal = proposed_centred + proposed_mean_shift
                end_raw_velocity = self._guided_raw_velocity(
                    model,
                    proposal,
                    end.expand(current_centred.shape[0]),
                    source_residual,
                    self_condition,
                    model_kwargs,
                    guidance_strength,
                )
                end_centred_velocity, end_common_velocity = (
                    self._split_child_velocity(
                        end_raw_velocity, group_ids, valid_mask
                    )
                )
                current_centred = current_centred + 0.5 * dt * (
                    centred_velocity + end_centred_velocity
                )
                current_centred = project_group_zero_mean(
                    current_centred,
                    group_ids=group_ids,
                    valid_mask=valid_mask,
                    singleton_policy="zero",
                ).value
                integrated_mean_shift = integrated_mean_shift + 0.5 * dt * (
                    common_velocity + end_common_velocity
                )

        endpoint_mean = source_projection.effective_parent + (
            self.child_mean_endpoint_blend * integrated_mean_shift
        )
        composed = compose_parent_residual(
            parent_mean=endpoint_mean,
            residual=current_centred,
            group_ids=group_ids,
            valid_mask=valid_mask,
            enforce_zero_mean=True,
            detach_parent_mean=False,
        )
        return ParentLockedSample(
            expression=composed.expression,
            residual=composed.residual,
            source_residual=source_residual,
            endpoint_mean=composed.parent_mean,
            integrated_mean_shift=integrated_mean_shift[:, 0, :],
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        *,
        source_expression: torch.Tensor,
        parent_mean: torch.Tensor,
        self_condition: Dict,
        group_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        guidance_strength: float = 0.0,
        sampling_steps: Optional[int] = None,
    ) -> ParentLockedSample:
        """Integrate a residual trajectory and add the Parent exactly once."""

        source_projection = prepare_source_residual(
            source_expression,
            parent_mean,
            group_ids=group_ids,
            valid_mask=valid_mask,
        )
        return self.sample_prepared(
            model,
            source_projection=source_projection,
            parent_mean=parent_mean,
            self_condition=self_condition,
            group_ids=group_ids,
            valid_mask=valid_mask,
            model_kwargs=model_kwargs,
            guidance_strength=guidance_strength,
            sampling_steps=sampling_steps,
        )

    @torch.no_grad()
    def sample_prepared(
        self,
        model,
        *,
        source_projection: NonnegativeFixedMeanProjection,
        parent_mean: torch.Tensor,
        self_condition: Dict,
        group_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        guidance_strength: float = 0.0,
        sampling_steps: Optional[int] = None,
    ) -> ParentLockedSample:
        """Integrate from an already prepared feasible source projection.

        This is the exact-source counterpart of :meth:`sample`. It exists for
        paired analyses that must share one numerical source residual between
        a no-flow arm and the integration arm. No source projection is
        repeated here. Callers must obtain ``source_projection`` from the
        production :func:`prepare_source_residual` using the supplied Parent
        and grouping. The ordinary :meth:`sample` path retains its historical
        behaviour by preparing once and delegating to this method.
        """

        if not isinstance(source_projection, NonnegativeFixedMeanProjection):
            raise TypeError(
                "source_projection must be a NonnegativeFixedMeanProjection"
            )
        source_residual = source_projection.residual
        if source_residual.ndim != 3 or not source_residual.is_floating_point():
            raise ValueError("prepared source residual must have shape [B,S,G]")
        if not torch.isfinite(source_residual).all():
            raise ValueError("prepared source residual must be finite")
        if source_projection.value.shape != source_residual.shape:
            raise ValueError("prepared source value and residual shapes differ")
        if source_projection.value.device != source_residual.device:
            raise ValueError("prepared source value and residual devices differ")
        if self.child_mean_flow_enabled:
            return self._sample_child_mean(
                model,
                source_projection=source_projection,
                self_condition=self_condition,
                group_ids=group_ids,
                valid_mask=valid_mask,
                model_kwargs=model_kwargs,
                guidance_strength=guidance_strength,
                sampling_steps=sampling_steps,
            )
        current = source_residual.clone()
        steps = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        if steps < 1:
            raise ValueError("sampling_steps must be positive")
        times = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=current.device,
            dtype=current.dtype,
        )
        for index in range(steps):
            start = times[index]
            end = times[index + 1]
            dt = end - start
            start_time = start.expand(current.shape[0])
            velocity = self._guided_velocity(
                model,
                current,
                start_time,
                source_residual,
                self_condition,
                model_kwargs,
                guidance_strength,
                group_ids,
                valid_mask,
            )
            proposal = current + dt * velocity
            proposal = project_group_zero_mean(
                proposal,
                group_ids=group_ids,
                valid_mask=valid_mask,
                singleton_policy="keep",
            ).value
            if index == steps - 1:
                current = proposal
            else:
                end_velocity = self._guided_velocity(
                    model,
                    proposal,
                    end.expand(current.shape[0]),
                    source_residual,
                    self_condition,
                    model_kwargs,
                    guidance_strength,
                    group_ids,
                    valid_mask,
                )
                current = current + 0.5 * dt * (velocity + end_velocity)
                current = project_group_zero_mean(
                    current,
                    group_ids=group_ids,
                    valid_mask=valid_mask,
                    singleton_policy="keep",
                ).value

        composed = compose_parent_residual(
            parent_mean=parent_mean,
            residual=current,
            group_ids=group_ids,
            valid_mask=valid_mask,
            enforce_zero_mean=True,
        )
        return ParentLockedSample(
            expression=composed.expression,
            residual=composed.residual,
            source_residual=source_residual,
        )


__all__ = ["ParentLockedResidualFlow", "ParentLockedSample"]

"""OCOO-T-style rectified flow process for PerturbDiff-compatible models."""

from __future__ import annotations

import copy

import torch


def _mean_flat(value):
    return value.mean(dim=tuple(range(1, value.ndim)))


class RectifiedFlow:
    """Noise-to-expression rectified flow with Heun ODE sampling."""

    is_rectified_flow = True

    def __init__(
        self,
        num_timesteps=1000,
        sampling_steps=50,
        time_scale=1000.0,
        final_nonnegative=True,
        final_max=None,
        time_mean=-0.8,
        time_std=0.8,
    ):
        self.num_timesteps = int(num_timesteps)
        self.sampling_steps = int(sampling_steps)
        self.time_scale = float(time_scale)
        self.final_nonnegative = bool(final_nonnegative)
        self.final_max = None if final_max is None else float(final_max)
        self.time_mean = float(time_mean)
        self.time_std = float(time_std)
        if self.num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if self.time_std <= 0:
            raise ValueError("time_std must be positive")
        if self.final_max is not None and self.final_max <= 0:
            raise ValueError("final_max must be positive when provided")

    @staticmethod
    def _expand_time(time, value):
        return time.reshape(-1, *([1] * (value.ndim - 1)))

    def _continuous_time(self, time, reference):
        is_discrete = not time.is_floating_point()
        time = time.to(device=reference.device, dtype=reference.dtype)
        if is_discrete or (
            time.numel() and float(time.detach().max()) > 1.0
        ):
            time = (time.float() + 0.5) / float(self.num_timesteps)
            time = time.to(reference)
        return time.clamp(0.0, 1.0)

    def _model_time(self, time):
        return time * self.time_scale

    def q_sample(self, x_start, time, noise=None, x_control=None):
        del x_control
        if noise is None:
            noise = torch.randn_like(x_start)
        continuous = self._continuous_time(time, x_start)
        expanded = self._expand_time(continuous, x_start)
        return (1.0 - expanded) * noise + expanded * x_start

    def _predict_velocity(self, model, current, time, self_condition, model_kwargs):
        control = self_condition["cont_emb"].to(
            device=current.device, dtype=current.dtype
        )
        zeros = torch.zeros_like(current)
        control_zeros = torch.zeros_like(control)
        output = model(
            torch.cat([current, zeros], dim=-1),
            torch.cat([control, control_zeros], dim=-1),
            self._model_time(time).unsqueeze(1),
            self_condition=self_condition,
            **model_kwargs,
        )
        return output

    def training_losses(
        self,
        model,
        x_start,
        t,
        self_condition=None,
        model_kwargs=None,
        noise=None,
        p_drop_cond=0.0,
        MMD_loss_fn=None,
        return_model_output=False,
    ):
        if self_condition is None:
            raise ValueError("rectified flow requires control/covariate conditioning")
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            noise = noise.to(device=x_start.device, dtype=x_start.dtype)

        continuous = self._continuous_time(t, x_start)
        expanded = self._expand_time(continuous, x_start)
        current = (1.0 - expanded) * noise + expanded * x_start
        target_velocity = x_start - noise
        condition = copy.copy(self_condition)
        control = condition["cont_emb"]

        if model.model_cfg.p_drop_control > 0.0 and (
            torch.rand((), device=x_start.device) < model.model_cfg.p_drop_control
        ):
            condition["cont_emb"] = torch.zeros_like(control)
        if p_drop_cond > 0.0 and (
            torch.rand((), device=x_start.device) < p_drop_cond
        ):
            condition["batch_emb"] = None

        output = self._predict_velocity(
            model, current, continuous, condition, model_kwargs
        )
        velocity = output["x"]
        endpoint = current + (1.0 - expanded) * velocity
        velocity_mse_per_cell = (velocity - target_velocity.to(velocity)).square().mean(dim=-1)
        mse = velocity_mse_per_cell.mean(dim=-1)

        if MMD_loss_fn is None:
            mmd_list = x_start.new_zeros(x_start.shape[0])
        else:
            mmd_list = MMD_loss_fn(x_start.type_as(endpoint), endpoint)
        mmd = mmd_list.nanmean()
        terms = {
            "mse1": mse,
            "loss1": mse,
            "mmd1": mmd,
            "mmd1_per_sample": mmd_list,
            "mmd1_list": mmd_list,
        }
        if not return_model_output:
            return terms
        matching_output = dict(output)
        matching_output["velocity"] = velocity
        matching_output["x"] = endpoint
        matching_output["flow_time"] = continuous
        matching_output["optimization_loss_per_cell"] = velocity_mse_per_cell
        return terms, matching_output

    @torch.no_grad()
    def _guided_velocity(
        self,
        model,
        current,
        time,
        self_condition,
        model_kwargs,
        guidance_strength,
    ):
        conditional = self._predict_velocity(
            model, current, time, self_condition, model_kwargs
        )["x"]
        strength = float(guidance_strength)
        if strength == 0.0:
            return conditional
        unconditional_condition = copy.copy(self_condition)
        unconditional_condition["batch_emb"] = None
        unconditional = self._predict_velocity(
            model, current, time, unconditional_condition, model_kwargs
        )["x"]
        return conditional + strength * (conditional - unconditional)

    @torch.no_grad()
    def _sample_loop(
        self,
        model,
        shape,
        self_condition,
        noise=None,
        clip_denoised=False,
        model_kwargs=None,
        device=None,
        progress=False,
        guidance_strength=0.0,
    ):
        if device is None:
            device = next(model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}
        current = (
            torch.randn(*shape, device=device)
            if noise is None
            else noise.to(device=device)
        )
        times = torch.linspace(
            0.0, 1.0, self.sampling_steps + 1, device=device, dtype=current.dtype
        )
        iterator = range(self.sampling_steps)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator)
        trajectory = []
        for index in iterator:
            start = times[index]
            end = times[index + 1]
            dt = end - start
            time = start.expand(shape[0])
            velocity = self._guided_velocity(
                model,
                current,
                time,
                self_condition,
                model_kwargs,
                guidance_strength,
            )
            proposal = current + dt * velocity
            if index == self.sampling_steps - 1:
                current = proposal
            else:
                end_velocity = self._guided_velocity(
                    model,
                    proposal,
                    end.expand(shape[0]),
                    self_condition,
                    model_kwargs,
                    guidance_strength,
                )
                current = current + 0.5 * dt * (velocity + end_velocity)
            if clip_denoised:
                current = current.clamp(-1.0, 1.0)
        if self.final_nonnegative:
            current = current.clamp_min(0.0)
        if self.final_max is not None:
            current = current.clamp_max(self.final_max)
        return current, trajectory

    def ddim_sample_loop(
        self,
        model,
        shape,
        self_condition,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_time=None,
        eta=0.0,
        guidance_strength=0.0,
        sample_kwargs=None,
    ):
        del denoised_fn, cond_fn, start_time, eta, sample_kwargs
        return self._sample_loop(
            model,
            shape,
            self_condition,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            guidance_strength=guidance_strength,
        )

    def p_sample_loop(
        self,
        model,
        shape,
        self_condition,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_time=None,
        nw=None,
        start_guide_steps=None,
        guidance_strength=0.0,
        sample_kwargs=None,
    ):
        del denoised_fn, cond_fn, start_time, nw, start_guide_steps, sample_kwargs
        return self._sample_loop(
            model,
            shape,
            self_condition,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            guidance_strength=guidance_strength,
        )


__all__ = ["RectifiedFlow"]

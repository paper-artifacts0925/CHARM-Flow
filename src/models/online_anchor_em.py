"""Persistent online-EM state for hierarchical control-state anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch import nn


_SCHEMA_VERSION = 1


@dataclass
class OnlineEMPrior:
    """Historical E-step prior and diagnostics for one collection of cells."""

    prior: torch.Tensor
    history: torch.Tensor
    history_weight: torch.Tensor
    confidence: torch.Tensor
    staleness: torch.Tensor
    cache_hit: torch.Tensor
    support_mask: torch.Tensor


@dataclass
class AssignmentUpdates:
    """Detached per-cell responsibilities to write into the temporal cache."""

    cell_ids: torch.Tensor
    key_ids: torch.Tensor
    responsibilities: torch.Tensor
    steps: torch.Tensor
    layout_versions: torch.Tensor

    @property
    def num_rows(self) -> int:
        return int(self.cell_ids.numel())


@dataclass
class SufficientStatistics:
    """Sparse per-key sufficient statistics.

    ``packed`` has shape ``[num_touched_keys, K, 2 + 3D]`` with the layout
    ``mass, log_magnitude_sum, direction_sum, delta_sum, delta_square_sum``.
    Keeping this packed allows one DDP collective for the expensive payload.
    """

    key_ids: torch.Tensor
    packed: torch.Tensor
    cell_count: torch.Tensor
    step: int

    @property
    def num_rows(self) -> int:
        return int(self.key_ids.numel())


@dataclass
class OnlineEMBatchUpdate:
    """One optimizer-step update, before or after distributed synchronization."""

    assignments: AssignmentUpdates
    statistics: SufficientStatistics


def _as_long_tensor(values: Iterable[int] | torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.long)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _normalize_masked(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values = torch.where(mask, values.clamp_min(0.0), torch.zeros_like(values))
    denominator = values.sum(dim=-1, keepdim=True)
    fallback = mask.to(values.dtype)
    fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(
        denominator > float(eps),
        values / denominator.clamp_min(float(eps)),
        fallback,
    )


def _all_gather_variable_rows(
    values: torch.Tensor,
    process_group=None,
) -> torch.Tensor:
    """All-gather a tensor with a variable first dimension on every rank."""
    if not dist.is_available() or not dist.is_initialized():
        return values
    world_size = dist.get_world_size(group=process_group)
    if world_size == 1:
        return values

    local_size = torch.tensor(
        [values.shape[0]], device=values.device, dtype=torch.long
    )
    gathered_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size, group=process_group)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if max_size == 0:
        return values.new_empty((0, *values.shape[1:]))

    padded = values.new_zeros((max_size, *values.shape[1:]))
    if values.shape[0]:
        padded[: values.shape[0]].copy_(values)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=process_group)
    pieces = [tensor[:size] for tensor, size in zip(gathered, sizes) if size]
    if not pieces:
        return values.new_empty((0, *values.shape[1:]))
    return torch.cat(pieces, dim=0)


class OnlineAnchorEM(nn.Module):
    """Stateful mini-batch generalized EM for fixed-capacity anchor slots.

    Parameters are intentionally absent: every registered tensor is a buffer.
    This prevents the optimizer or model-weight EMA from treating assignment
    memory as trainable weights.  For repositories that swap model state_dicts
    for parameter EMA, callers may also save this module under a dedicated
    checkpoint key.
    """

    def __init__(
        self,
        num_cells: int,
        num_keys: int,
        parent_ids: Iterable[int] | torch.Tensor,
        child_ids: Iterable[int] | torch.Tensor,
        feature_dim: int,
        *,
        initial_active_mask: Optional[torch.Tensor] = None,
        burn_in_steps: int = 4000,
        history_ramp_steps: int = 20000,
        history_max_weight: float = 0.35,
        history_stale_half_life_steps: float = 50000.0,
        history_max_age_steps: Optional[int] = None,
        occupancy_half_life_cells: float = 256.0,
        prototype_half_life_cells: float = 128.0,
        minimum_prototype_mass: float = 1e-3,
        activate_threshold: float = 0.03,
        deactivate_threshold: float = 0.005,
        activate_patience: int = 3,
        deactivate_patience: int = 20,
        min_active_per_parent: int = 1,
        inactive_probe_prior: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        parent_ids = _as_long_tensor(parent_ids, "parent_ids")
        child_ids = _as_long_tensor(child_ids, "child_ids")
        if parent_ids.shape != child_ids.shape or parent_ids.numel() < 1:
            raise ValueError("parent_ids and child_ids must have the same non-empty shape")
        if int(num_cells) < 1 or int(num_keys) < 1 or int(feature_dim) < 1:
            raise ValueError("num_cells, num_keys and feature_dim must be positive")
        identities = torch.stack([parent_ids, child_ids], dim=-1)
        if torch.unique(identities, dim=0).shape[0] != parent_ids.numel():
            raise ValueError("every (parent_id, child_id) slot must be unique")
        if not 0.0 <= float(history_max_weight) < 1.0:
            raise ValueError("history_max_weight must be in [0, 1)")
        if float(activate_threshold) <= float(deactivate_threshold):
            raise ValueError("activate_threshold must exceed deactivate_threshold")
        if int(activate_patience) < 1 or int(deactivate_patience) < 1:
            raise ValueError("activity patience values must be positive")
        if int(min_active_per_parent) < 1:
            raise ValueError("min_active_per_parent must be positive")
        if float(occupancy_half_life_cells) <= 0.0:
            raise ValueError("occupancy_half_life_cells must be positive")
        if float(prototype_half_life_cells) <= 0.0:
            raise ValueError("prototype_half_life_cells must be positive")

        self.num_cells = int(num_cells)
        self.num_keys = int(num_keys)
        self.num_slots = int(parent_ids.numel())
        self.feature_dim = int(feature_dim)
        self.burn_in_steps = int(burn_in_steps)
        self.history_ramp_steps = int(history_ramp_steps)
        self.history_max_weight = float(history_max_weight)
        self.history_stale_half_life_steps = float(history_stale_half_life_steps)
        self.history_max_age_steps = (
            None if history_max_age_steps is None else int(history_max_age_steps)
        )
        self.occupancy_half_life_cells = float(occupancy_half_life_cells)
        self.prototype_half_life_cells = float(prototype_half_life_cells)
        self.minimum_prototype_mass = float(minimum_prototype_mass)
        self.activate_threshold = float(activate_threshold)
        self.deactivate_threshold = float(deactivate_threshold)
        self.activate_patience = int(activate_patience)
        self.deactivate_patience = int(deactivate_patience)
        self.min_active_per_parent = int(min_active_per_parent)
        self.inactive_probe_prior = float(inactive_probe_prior)
        self.eps = float(eps)

        if initial_active_mask is None:
            active = torch.ones(self.num_keys, self.num_slots, dtype=torch.bool)
        else:
            active = torch.as_tensor(initial_active_mask, dtype=torch.bool)
            if active.shape == (self.num_slots,):
                active = active.unsqueeze(0).expand(self.num_keys, -1).clone()
            if active.shape != (self.num_keys, self.num_slots):
                raise ValueError("initial_active_mask must have shape [K] or [num_keys,K]")
        for parent in torch.unique(parent_ids):
            parent_mask = parent_ids.eq(parent)
            if (active[:, parent_mask].sum(dim=-1) < self.min_active_per_parent).any():
                raise ValueError("initial_active_mask violates min_active_per_parent")

        initial_occupancy = active.float()
        initial_occupancy /= initial_occupancy.sum(dim=-1, keepdim=True)

        self.register_buffer("parent_ids", parent_ids, persistent=True)
        self.register_buffer("child_ids", child_ids, persistent=True)
        self.register_buffer("active_mask", active, persistent=True)
        self.register_buffer(
            "slot_generation",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "key_layout_version",
            torch.zeros(self.num_keys, dtype=torch.long),
            persistent=True,
        )

        # The only large buffer is deliberately FP16.  It is an EM warm start,
        # not a numerical accumulator or gradient-bearing tensor.
        self.register_buffer(
            "q_cache",
            torch.zeros(self.num_cells, self.num_slots, dtype=torch.float16),
            persistent=True,
        )
        self.register_buffer(
            "q_cache_seen", torch.zeros(self.num_cells, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "q_cache_key", torch.full((self.num_cells,), -1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "q_cache_step", torch.full((self.num_cells,), -1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "q_cache_layout_version",
            torch.full((self.num_cells,), -1, dtype=torch.long),
            persistent=True,
        )

        self.register_buffer("occupancy", initial_occupancy, persistent=True)
        self.register_buffer(
            "ema_mass",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "direction",
            torch.zeros(
                self.num_keys, self.num_slots, self.feature_dim, dtype=torch.float32
            ),
            persistent=True,
        )
        self.register_buffer(
            "log_magnitude",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "delta_mean",
            torch.zeros(
                self.num_keys, self.num_slots, self.feature_dim, dtype=torch.float32
            ),
            persistent=True,
        )
        self.register_buffer(
            "variance",
            torch.zeros(
                self.num_keys, self.num_slots, self.feature_dim, dtype=torch.float32
            ),
            persistent=True,
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "last_slot_update_step",
            torch.full((self.num_keys, self.num_slots), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "key_observation_count",
            torch.zeros(self.num_keys, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "activation_streak",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "deactivation_streak",
            torch.zeros(self.num_keys, self.num_slots, dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "last_committed_step", torch.tensor(-1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "num_commits", torch.tensor(0, dtype=torch.long), persistent=True
        )

    @property
    def packed_stat_dim(self) -> int:
        return 2 + 3 * self.feature_dim

    def get_extra_state(self):
        """Store non-tensor schedule and hysteresis configuration in state_dict."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "num_cells": self.num_cells,
            "num_keys": self.num_keys,
            "num_slots": self.num_slots,
            "feature_dim": self.feature_dim,
            "burn_in_steps": self.burn_in_steps,
            "history_ramp_steps": self.history_ramp_steps,
            "history_max_weight": self.history_max_weight,
            "history_stale_half_life_steps": self.history_stale_half_life_steps,
            "history_max_age_steps": self.history_max_age_steps,
            "occupancy_half_life_cells": self.occupancy_half_life_cells,
            "prototype_half_life_cells": self.prototype_half_life_cells,
            "minimum_prototype_mass": self.minimum_prototype_mass,
            "activate_threshold": self.activate_threshold,
            "deactivate_threshold": self.deactivate_threshold,
            "activate_patience": self.activate_patience,
            "deactivate_patience": self.deactivate_patience,
            "min_active_per_parent": self.min_active_per_parent,
            "inactive_probe_prior": self.inactive_probe_prior,
            "eps": self.eps,
        }

    def set_extra_state(self, state) -> None:
        if int(state.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise RuntimeError("unsupported OnlineAnchorEM state schema")
        for name in ("num_cells", "num_keys", "num_slots", "feature_dim"):
            if int(state[name]) != int(getattr(self, name)):
                raise RuntimeError(
                    f"OnlineAnchorEM checkpoint {name}={state[name]} does not match "
                    f"constructed {name}={getattr(self, name)}"
                )
        for name in (
            "burn_in_steps",
            "history_ramp_steps",
            "history_max_weight",
            "history_stale_half_life_steps",
            "history_max_age_steps",
            "occupancy_half_life_cells",
            "prototype_half_life_cells",
            "minimum_prototype_mass",
            "activate_threshold",
            "deactivate_threshold",
            "activate_patience",
            "deactivate_patience",
            "min_active_per_parent",
            "inactive_probe_prior",
            "eps",
        ):
            setattr(self, name, state[name])

    def _validate_ids(self, cell_ids: torch.Tensor, key_ids: torch.Tensor) -> None:
        if cell_ids.shape != key_ids.shape:
            raise ValueError("cell_ids and key_ids must have identical shapes")
        if cell_ids.numel() and (
            int(cell_ids.min()) < 0 or int(cell_ids.max()) >= self.num_cells
        ):
            raise IndexError("cell_id is outside the configured stable-cell range")
        if key_ids.numel() and (
            int(key_ids.min()) < 0 or int(key_ids.max()) >= self.num_keys
        ):
            raise IndexError("key_id is outside the configured key range")

    def history_schedule(self, step: int) -> float:
        """Return the maximum historical responsibility weight at ``step``."""
        step = int(step)
        if step < self.burn_in_steps:
            return 0.0
        if self.history_ramp_steps <= 0:
            return self.history_max_weight
        progress = min(
            max(float(step - self.burn_in_steps) / self.history_ramp_steps, 0.0),
            1.0,
        )
        return self.history_max_weight * progress

    @torch.no_grad()
    def get_soft_prior(
        self,
        cell_ids: torch.Tensor,
        key_ids: torch.Tensor,
        *,
        step: int,
        base_prior: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        probe_inactive: bool = False,
    ) -> OnlineEMPrior:
        """Blend a structural prior with cached q without ever making q a label.

        The blend is a confidence- and staleness-weighted geometric mixture.
        A uniform historical assignment has zero confidence and therefore
        returns the structural prior exactly.  ``probe_inactive`` gives dormant
        slots a small prior so they can accumulate reactivation evidence.
        """
        original_shape = cell_ids.shape
        cells = cell_ids.detach().to(device=self.q_cache.device, dtype=torch.long).reshape(-1)
        keys = key_ids.detach().to(device=self.q_cache.device, dtype=torch.long).reshape(-1)
        self._validate_ids(cells, keys)
        count = cells.numel()

        active = self.active_mask.index_select(0, keys)
        if source_mask is None:
            external_mask = torch.ones_like(active)
        else:
            external_mask = source_mask.detach().to(device=active.device, dtype=torch.bool)
            external_mask = external_mask.reshape(count, self.num_slots)
        support = external_mask & (torch.ones_like(active) if probe_inactive else active)
        if not support.any(dim=-1).all():
            raise ValueError("every cell must have at least one supported anchor slot")

        if base_prior is None:
            base = self.occupancy.index_select(0, keys).float()
        else:
            base = base_prior.detach().to(device=active.device, dtype=torch.float32)
            base = base.reshape(count, self.num_slots)
        if probe_inactive and self.inactive_probe_prior > 0.0:
            probe_floor = torch.full_like(base, self.inactive_probe_prior)
            base = torch.where(~active & support, torch.maximum(base, probe_floor), base)
        base = _normalize_masked(base, support, self.eps)

        cached = self.q_cache.index_select(0, cells).float()
        cached = _normalize_masked(cached, support, self.eps)
        cache_hit = self.q_cache_seen.index_select(0, cells)
        cache_hit &= self.q_cache_key.index_select(0, cells).eq(keys)
        cache_hit &= self.q_cache_layout_version.index_select(0, cells).eq(
            self.key_layout_version.index_select(0, keys)
        )

        valid_count = support.sum(dim=-1).float()
        entropy = -(cached * cached.clamp_min(self.eps).log()).sum(dim=-1)
        max_entropy = valid_count.clamp_min(2.0).log()
        confidence = 1.0 - entropy / max_entropy
        confidence = torch.where(valid_count <= 1.0, torch.ones_like(confidence), confidence)
        confidence = confidence.clamp(0.0, 1.0)

        age = (
            int(step) - self.q_cache_step.index_select(0, cells)
        ).clamp_min(0).float()
        if self.history_stale_half_life_steps > 0.0:
            staleness = torch.exp2(-age / self.history_stale_half_life_steps)
        else:
            staleness = torch.ones_like(age)
        if self.history_max_age_steps is not None:
            staleness = torch.where(
                age <= float(self.history_max_age_steps),
                staleness,
                torch.zeros_like(staleness),
            )
        staleness = torch.where(cache_hit, staleness, torch.zeros_like(staleness))
        confidence = torch.where(cache_hit, confidence, torch.zeros_like(confidence))

        scheduled = self.history_schedule(step)
        history_weight = (scheduled * confidence * staleness).clamp(
            0.0, self.history_max_weight
        )
        mixed_log = (
            (1.0 - history_weight.unsqueeze(-1)) * base.clamp_min(self.eps).log()
            + history_weight.unsqueeze(-1) * cached.clamp_min(self.eps).log()
        )
        mixed_log = mixed_log.masked_fill(~support, -torch.inf)
        prior = torch.softmax(mixed_log, dim=-1)
        prior = torch.where(support, prior, torch.zeros_like(prior))
        prior = _normalize_masked(prior, support, self.eps)

        out_shape = (*original_shape, self.num_slots)
        return OnlineEMPrior(
            prior=prior.reshape(out_shape),
            history=cached.reshape(out_shape),
            history_weight=history_weight.reshape(original_shape),
            confidence=confidence.reshape(original_shape),
            staleness=staleness.reshape(original_shape),
            cache_hit=cache_hit.reshape(original_shape),
            support_mask=support.reshape(out_shape),
        )

    def _reshape_delta_features(
        self,
        delta_features: torch.Tensor,
        cell_shape: torch.Size,
    ) -> torch.Tensor:
        if delta_features.shape == (*cell_shape, self.feature_dim):
            flat = delta_features.reshape(-1, self.feature_dim)
            return flat.unsqueeze(1).expand(-1, self.num_slots, -1)
        expected = (*cell_shape, self.num_slots, self.feature_dim)
        if delta_features.shape != expected:
            raise ValueError(
                "delta_features must have shape [...,D] or [...,K,D], "
                f"got {tuple(delta_features.shape)} expected suffixes D={self.feature_dim}, "
                f"K={self.num_slots}"
            )
        return delta_features.reshape(-1, self.num_slots, self.feature_dim)

    @torch.no_grad()
    def build_update(
        self,
        cell_ids: torch.Tensor,
        key_ids: torch.Tensor,
        responsibilities: torch.Tensor,
        delta_features: torch.Tensor,
        *,
        step: int,
        valid_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> OnlineEMBatchUpdate:
        """Build a detached cache/statistics update from one local rank.

        ``delta_features`` may be shared across candidates (``[...,D]``) or
        candidate-specific (``[...,K,D]``), which is required when observed
        deltas are measured from different parent anchors.
        """
        cell_shape = cell_ids.shape
        cells = cell_ids.detach().to(device=self.q_cache.device, dtype=torch.long).reshape(-1)
        keys = key_ids.detach().to(device=self.q_cache.device, dtype=torch.long).reshape(-1)
        self._validate_ids(cells, keys)
        q = responsibilities.detach().to(device=self.q_cache.device, dtype=torch.float32)
        if q.shape != (*cell_shape, self.num_slots):
            raise ValueError("responsibilities must have shape cell_ids.shape + [K]")
        q = q.reshape(-1, self.num_slots)
        delta = self._reshape_delta_features(
            delta_features.detach().to(device=self.q_cache.device, dtype=torch.float32),
            cell_shape,
        )
        if not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("responsibilities must be finite and non-negative")
        if not torch.isfinite(delta).all():
            raise ValueError("delta_features must be finite")

        if source_mask is None:
            mask = torch.ones_like(q, dtype=torch.bool)
        else:
            mask = source_mask.detach().to(device=q.device, dtype=torch.bool)
            mask = mask.reshape(-1, self.num_slots)
        q = _normalize_masked(q, mask, self.eps)

        if valid_mask is None:
            valid = torch.ones(cells.shape[0], device=cells.device, dtype=torch.bool)
        else:
            valid = valid_mask.detach().to(device=cells.device, dtype=torch.bool).reshape(-1)
            if valid.shape != cells.shape:
                raise ValueError("valid_mask must have the same shape as cell_ids")
        cells = cells[valid]
        keys = keys[valid]
        q = q[valid]
        delta = delta[valid]

        steps = torch.full_like(cells, int(step))
        layout_versions = self.key_layout_version.index_select(0, keys)
        assignments = AssignmentUpdates(
            cell_ids=cells,
            key_ids=keys,
            responsibilities=q,
            steps=steps,
            layout_versions=layout_versions,
        )

        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        packed = torch.zeros(
            unique_keys.numel(),
            self.num_slots,
            self.packed_stat_dim,
            device=q.device,
            dtype=torch.float32,
        )
        cell_count = torch.zeros(unique_keys.numel(), device=q.device, dtype=torch.float32)
        if keys.numel():
            norms = delta.square().sum(dim=-1).clamp_min(0.0).sqrt()
            unit = delta / norms.clamp_min(self.eps).unsqueeze(-1)
            unit = torch.where(norms.unsqueeze(-1) > self.eps, unit, torch.zeros_like(unit))
            log_magnitude = torch.log(norms + self.eps)
            local = torch.cat(
                [
                    q.unsqueeze(-1),
                    (q * log_magnitude).unsqueeze(-1),
                    q.unsqueeze(-1) * unit,
                    q.unsqueeze(-1) * delta,
                    q.unsqueeze(-1) * delta.square(),
                ],
                dim=-1,
            )
            packed.index_add_(0, inverse, local)
            cell_count.index_add_(
                0, inverse, torch.ones_like(inverse, dtype=torch.float32)
            )
        statistics = SufficientStatistics(
            key_ids=unique_keys,
            packed=packed,
            cell_count=cell_count,
            step=int(step),
        )
        return OnlineEMBatchUpdate(assignments=assignments, statistics=statistics)

    @staticmethod
    def merge_assignment_updates(updates: Iterable[AssignmentUpdates]) -> AssignmentUpdates:
        """Merge rank-local cache writes and deterministically average duplicates."""
        updates = list(updates)
        if not updates:
            raise ValueError("at least one assignment update is required")
        cell_ids = torch.cat([update.cell_ids for update in updates], dim=0)
        key_ids = torch.cat([update.key_ids for update in updates], dim=0)
        q = torch.cat([update.responsibilities for update in updates], dim=0)
        steps = torch.cat([update.steps for update in updates], dim=0)
        versions = torch.cat([update.layout_versions for update in updates], dim=0)
        if not cell_ids.numel():
            return AssignmentUpdates(cell_ids, key_ids, q, steps, versions)

        order = torch.argsort(cell_ids, stable=True)
        cell_ids, key_ids, q = cell_ids[order], key_ids[order], q[order]
        steps, versions = steps[order], versions[order]
        unique_cells, counts = torch.unique_consecutive(cell_ids, return_counts=True)
        if (counts == 1).all():
            return AssignmentUpdates(cell_ids, key_ids, q, steps, versions)

        merged_keys, merged_q, merged_steps, merged_versions = [], [], [], []
        cursor = 0
        for count in counts.tolist():
            stop = cursor + count
            group_keys = key_ids[cursor:stop]
            group_versions = versions[cursor:stop]
            if not group_keys.eq(group_keys[0]).all():
                raise ValueError("one stable cell_id was associated with multiple key_ids")
            if not group_versions.eq(group_versions[0]).all():
                raise ValueError("one stable cell_id was updated under multiple layouts")
            merged_keys.append(group_keys[0])
            merged_q.append(q[cursor:stop].mean(dim=0))
            merged_steps.append(steps[cursor:stop].max())
            merged_versions.append(group_versions[0])
            cursor = stop
        merged_q_tensor = torch.stack(merged_q)
        merged_q_tensor /= merged_q_tensor.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return AssignmentUpdates(
            cell_ids=unique_cells,
            key_ids=torch.stack(merged_keys),
            responsibilities=merged_q_tensor,
            steps=torch.stack(merged_steps),
            layout_versions=torch.stack(merged_versions),
        )

    @staticmethod
    def merge_statistics(updates: Iterable[SufficientStatistics]) -> SufficientStatistics:
        """Merge sparse sufficient statistics from multiple ranks."""
        updates = list(updates)
        if not updates:
            raise ValueError("at least one statistics update is required")
        key_ids = torch.cat([update.key_ids for update in updates], dim=0)
        packed = torch.cat([update.packed for update in updates], dim=0)
        cell_count = torch.cat([update.cell_count for update in updates], dim=0)
        step = max(update.step for update in updates)
        if not key_ids.numel():
            return SufficientStatistics(key_ids, packed, cell_count, step)
        unique_keys, inverse = torch.unique(key_ids, sorted=True, return_inverse=True)
        merged_packed = packed.new_zeros((unique_keys.numel(), *packed.shape[1:]))
        merged_count = cell_count.new_zeros(unique_keys.numel())
        merged_packed.index_add_(0, inverse, packed)
        merged_count.index_add_(0, inverse, cell_count)
        return SufficientStatistics(unique_keys, merged_packed, merged_count, step)

    @classmethod
    def merge_batch_updates(
        cls, updates: Iterable[OnlineEMBatchUpdate]
    ) -> OnlineEMBatchUpdate:
        updates = list(updates)
        if not updates:
            raise ValueError("at least one batch update is required")
        return OnlineEMBatchUpdate(
            assignments=cls.merge_assignment_updates(
                [update.assignments for update in updates]
            ),
            statistics=cls.merge_statistics([update.statistics for update in updates]),
        )

    @classmethod
    def synchronize_batch_update(
        cls,
        update: OnlineEMBatchUpdate,
        process_group=None,
    ) -> OnlineEMBatchUpdate:
        """All-gather and merge a sparse update on every initialized DDP rank."""
        if not dist.is_available() or not dist.is_initialized():
            return update

        assignments = update.assignments
        gathered_assignments = AssignmentUpdates(
            cell_ids=_all_gather_variable_rows(
                assignments.cell_ids.unsqueeze(-1), process_group
            ).squeeze(-1),
            key_ids=_all_gather_variable_rows(
                assignments.key_ids.unsqueeze(-1), process_group
            ).squeeze(-1),
            responsibilities=_all_gather_variable_rows(
                assignments.responsibilities, process_group
            ),
            steps=_all_gather_variable_rows(
                assignments.steps.unsqueeze(-1), process_group
            ).squeeze(-1),
            layout_versions=_all_gather_variable_rows(
                assignments.layout_versions.unsqueeze(-1), process_group
            ).squeeze(-1),
        )
        statistics = update.statistics
        gathered_statistics = SufficientStatistics(
            key_ids=_all_gather_variable_rows(
                statistics.key_ids.unsqueeze(-1), process_group
            ).squeeze(-1),
            packed=_all_gather_variable_rows(statistics.packed, process_group),
            cell_count=_all_gather_variable_rows(
                statistics.cell_count.unsqueeze(-1), process_group
            ).squeeze(-1),
            step=statistics.step,
        )
        # The gather already concatenated every rank, so a single-item merge
        # performs duplicate-key/cell reduction without another collective.
        return OnlineEMBatchUpdate(
            assignments=cls.merge_assignment_updates([gathered_assignments]),
            statistics=cls.merge_statistics([gathered_statistics]),
        )

    @torch.no_grad()
    def commit_assignments(self, update: AssignmentUpdates) -> None:
        """Write synchronized responsibilities into the FP16 temporal cache."""
        if not update.cell_ids.numel():
            return
        cells = update.cell_ids.to(device=self.q_cache.device, dtype=torch.long)
        keys = update.key_ids.to(device=self.q_cache.device, dtype=torch.long)
        self._validate_ids(cells, keys)
        q = update.responsibilities.to(device=self.q_cache.device, dtype=torch.float32)
        if q.shape != (cells.numel(), self.num_slots):
            raise ValueError("assignment update has an invalid responsibility shape")
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        self.q_cache.index_copy_(0, cells, q.to(torch.float16))
        self.q_cache_seen.index_fill_(0, cells, True)
        self.q_cache_key.index_copy_(0, cells, keys)
        self.q_cache_step.index_copy_(
            0, cells, update.steps.to(device=self.q_cache.device, dtype=torch.long)
        )
        self.q_cache_layout_version.index_copy_(
            0,
            cells,
            update.layout_versions.to(device=self.q_cache.device, dtype=torch.long),
        )

    def _activity_update(
        self,
        key: int,
        batch_occupancy: torch.Tensor,
    ) -> bool:
        """Apply activation/deactivation hysteresis for one key."""
        active = self.active_mask[key].clone()
        updated_occupancy = self.occupancy[key]
        high = batch_occupancy >= self.activate_threshold
        low = updated_occupancy <= self.deactivate_threshold

        self.activation_streak[key] = torch.where(
            ~active & high,
            self.activation_streak[key] + 1,
            torch.zeros_like(self.activation_streak[key]),
        )
        self.deactivation_streak[key] = torch.where(
            active & low,
            self.deactivation_streak[key] + 1,
            torch.zeros_like(self.deactivation_streak[key]),
        )
        active |= self.activation_streak[key] >= self.activate_patience

        for parent in torch.unique(self.parent_ids).tolist():
            parent_mask = self.parent_ids.eq(int(parent))
            active_indices = torch.nonzero(active & parent_mask, as_tuple=False).flatten()
            if active_indices.numel() <= self.min_active_per_parent:
                continue
            candidates = active_indices[
                self.deactivation_streak[key, active_indices]
                >= self.deactivate_patience
            ]
            can_remove = int(active_indices.numel()) - self.min_active_per_parent
            if candidates.numel() > can_remove:
                order = torch.argsort(updated_occupancy[candidates])
                candidates = candidates[order[:can_remove]]
            active[candidates] = False

        changed = not torch.equal(active, self.active_mask[key])
        if changed:
            self.active_mask[key].copy_(active)
            self.key_layout_version[key].add_(1)
            self.activation_streak[key].zero_()
            self.deactivation_streak[key].zero_()
            self.occupancy[key] = _normalize_masked(
                self.occupancy[key], active, self.eps
            )
        return changed

    @torch.no_grad()
    def commit_statistics(self, update: SufficientStatistics) -> None:
        """Apply synchronized online sufficient statistics and slot hysteresis."""
        if update.packed.shape != (
            update.key_ids.numel(),
            self.num_slots,
            self.packed_stat_dim,
        ):
            raise ValueError("statistics update has an invalid packed shape")
        if update.cell_count.shape != update.key_ids.shape:
            raise ValueError("statistics cell_count must match key_ids")
        keys = update.key_ids.to(device=self.occupancy.device, dtype=torch.long)
        if keys.numel() and (int(keys.min()) < 0 or int(keys.max()) >= self.num_keys):
            raise IndexError("statistics key_id is out of range")
        packed = update.packed.to(device=self.occupancy.device, dtype=torch.float32)
        cell_count = update.cell_count.to(device=self.occupancy.device, dtype=torch.float32)
        direction_start = 2
        delta_start = direction_start + self.feature_dim
        square_start = delta_start + self.feature_dim

        for row, key_tensor in enumerate(keys):
            key = int(key_tensor.item())
            count = float(cell_count[row].item())
            if count <= 0.0:
                continue
            mass = packed[row, :, 0].clamp_min(0.0)
            if float(mass.sum()) <= self.eps:
                continue
            batch_occupancy = mass / mass.sum().clamp_min(self.eps)
            occupancy_eta = 1.0 - math.pow(
                2.0, -count / self.occupancy_half_life_cells
            )
            self.occupancy[key].mul_(1.0 - occupancy_eta).add_(
                batch_occupancy, alpha=occupancy_eta
            )
            occupancy_support = self.active_mask[key] | batch_occupancy.gt(0)
            self.occupancy[key] = _normalize_masked(
                self.occupancy[key], occupancy_support, self.eps
            )
            self.ema_mass[key].mul_(1.0 - occupancy_eta).add_(
                mass, alpha=occupancy_eta
            )
            self.key_observation_count[key].add_(count)

            for slot in range(self.num_slots):
                slot_mass = float(mass[slot].item())
                if slot_mass < self.minimum_prototype_mass:
                    continue
                prototype_eta = 1.0 - math.pow(
                    2.0, -slot_mass / self.prototype_half_life_cells
                )
                batch_direction = (
                    packed[row, slot, direction_start:delta_start] / slot_mass
                )
                direction_norm = batch_direction.norm()
                if float(direction_norm) > self.eps:
                    batch_direction = batch_direction / direction_norm
                batch_logmag = packed[row, slot, 1] / slot_mass
                batch_mean = packed[row, slot, delta_start:square_start] / slot_mass
                batch_second = packed[row, slot, square_start:] / slot_mass
                batch_variance = (batch_second - batch_mean.square()).clamp_min(0.0)

                if not bool(self.prototype_initialized[key, slot]):
                    self.direction[key, slot].copy_(batch_direction)
                    self.log_magnitude[key, slot].copy_(batch_logmag)
                    self.delta_mean[key, slot].copy_(batch_mean)
                    self.variance[key, slot].copy_(batch_variance)
                    self.prototype_initialized[key, slot] = True
                else:
                    old_mean = self.delta_mean[key, slot].clone()
                    old_variance = self.variance[key, slot].clone()
                    mixed_direction = (
                        (1.0 - prototype_eta) * self.direction[key, slot]
                        + prototype_eta * batch_direction
                    )
                    mixed_norm = mixed_direction.norm()
                    if float(mixed_norm) > self.eps:
                        mixed_direction /= mixed_norm
                    self.direction[key, slot].copy_(mixed_direction)
                    self.log_magnitude[key, slot].mul_(1.0 - prototype_eta).add_(
                        batch_logmag, alpha=prototype_eta
                    )
                    new_mean = (
                        (1.0 - prototype_eta) * old_mean
                        + prototype_eta * batch_mean
                    )
                    # Variance of a mixture, retaining between-batch mean shift.
                    new_variance = (
                        (1.0 - prototype_eta)
                        * (old_variance + (old_mean - new_mean).square())
                        + prototype_eta
                        * (batch_variance + (batch_mean - new_mean).square())
                    )
                    self.delta_mean[key, slot].copy_(new_mean)
                    self.variance[key, slot].copy_(new_variance.clamp_min(0.0))
                self.last_slot_update_step[key, slot] = int(update.step)

            self._activity_update(key, batch_occupancy)

    @torch.no_grad()
    def commit_update(
        self,
        update: OnlineEMBatchUpdate,
        *,
        synchronize: bool = False,
        process_group=None,
    ) -> OnlineEMBatchUpdate:
        """Optionally synchronize and atomically commit one optimizer-step update."""
        if synchronize:
            update = self.synchronize_batch_update(update, process_group=process_group)
        # Cache the E-step under the layout it used.  If hysteresis changes the
        # layout below, the version bump intentionally invalidates this history.
        self.commit_assignments(update.assignments)
        self.commit_statistics(update.statistics)
        self.last_committed_step.fill_(int(update.statistics.step))
        self.num_commits.add_(1)
        return update

    @torch.no_grad()
    def reset_slots(
        self,
        key_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        *,
        activate: bool = True,
    ) -> None:
        """Reset/reseed child-slot state and invalidate historical q for its key."""
        keys = key_ids.to(device=self.occupancy.device, dtype=torch.long).reshape(-1)
        slots = slot_ids.to(device=self.occupancy.device, dtype=torch.long).reshape(-1)
        if keys.shape != slots.shape:
            raise ValueError("key_ids and slot_ids must have identical shapes")
        touched = set()
        for key_tensor, slot_tensor in zip(keys, slots):
            key, slot = int(key_tensor), int(slot_tensor)
            if not 0 <= key < self.num_keys or not 0 <= slot < self.num_slots:
                raise IndexError("reset slot index is out of range")
            self.slot_generation[key, slot].add_(1)
            self.ema_mass[key, slot].zero_()
            self.direction[key, slot].zero_()
            self.log_magnitude[key, slot].zero_()
            self.delta_mean[key, slot].zero_()
            self.variance[key, slot].zero_()
            self.prototype_initialized[key, slot] = False
            self.last_slot_update_step[key, slot] = -1
            self.activation_streak[key, slot] = 0
            self.deactivation_streak[key, slot] = 0
            if activate:
                self.active_mask[key, slot] = True
            touched.add(key)
        for key in touched:
            self.key_layout_version[key].add_(1)
            self.occupancy[key] = _normalize_masked(
                self.occupancy[key], self.active_mask[key], self.eps
            )


__all__ = [
    "AssignmentUpdates",
    "OnlineAnchorEM",
    "OnlineEMBatchUpdate",
    "OnlineEMPrior",
    "SufficientStatistics",
]

"""Validated configuration for the isolated per-cell DETR model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellDETRConfig:
    """Parent/Child context, matching curriculum, and auxiliary losses."""

    anchor_capacity: int = 128
    hidden_dim: int = 128
    num_heads: int = 8
    parent_depth: int = 2
    dropout: float = 0.0
    max_child_similarity: float = 0.85
    child_encoding_mode: str = "joint"

    endpoint_cost_weight: float = 1.0
    direction_cost_weight: float = 0.25
    magnitude_cost_weight: float = 0.05
    router_cost_weight: float = 0.05
    matching_temperature: float = 0.25

    positive_multiplicities: tuple[int, ...] = (8, 4, 2, 1)
    multiplicity_transition_epochs: tuple[int, ...] = (5, 12, 20)

    delta_loss_weight: float = 0.10
    direction_loss_weight: float = 0.05
    # Optional optimizer-step curriculum. Epoch scheduling stays the default
    # because old runs may have very different numbers of steps per epoch.
    positive_multiplicity_step_enabled: bool = False
    # Opt-in support for ragged Child banks. The matcher keeps its requested
    # output width R while capping each row's real positives to its active K.
    positive_multiplicity_cap_to_active_children: bool = False
    positive_multiplicity_step_values: tuple[int, ...] = (32, 16, 8, 4, 2, 1)
    positive_multiplicity_step_transitions: tuple[int, ...] = (
        500,
        1200,
        2200,
        3500,
        5000,
    )
    magnitude_loss_weight: float = 0.01
    router_loss_weight: float = 0.05
    latent_match_loss_weight: float = 0.0
    gate_loss_weight: float = 0.0
    parent_auxiliary_loss_weight: float = 0.0
    compactness_loss_weight: float = 0.0
    separation_loss_weight: float = 0.001
    context_scale: float = 0.1
    inference_temperature: float = 1.0
    inference_occupancy_logit_weight: float = 0.1

    # Optional cold-start-safe hierarchy. Disabled by default so historical
    # Cell-DETR experiments retain their exact behavior.
    cold_start_gate_enabled: bool = False
    cold_start_parent_epochs: int = 3
    cold_start_transition_end_epoch: int = 20
    gate_initial_probability: float = 0.02
    gate_relative_margin_start: float = 0.30
    gate_relative_margin_end: float = 0.02
    gate_temperature_start: float = 0.08
    gate_temperature_end: float = 0.04
    gate_exploration_start: float = 0.10
    gate_exploration_end: float = 0.02
    cold_start_router_loss_weight_start: float = 0.005
    cold_start_router_loss_weight_end: float = 0.05
    router_temperature_start: float = 2.0
    router_temperature_end: float = 1.0
    router_occupancy_logit_weight_start: float = 1.0
    router_occupancy_logit_weight_end: float = 0.1
    balance_target_power_start: float = 0.5
    balance_target_power_end: float = 1.0
    balance_loss_weight_start: float = 0.005
    balance_loss_weight_end: float = 0.0

    @classmethod
    def from_model_cfg(cls, cfg):
        prefix = "cell_detr_"
        kwargs = {}
        for field in cls.__dataclass_fields__.values():
            value = getattr(cfg, prefix + field.name, field.default)
            if field.name in {
                "positive_multiplicities",
                "multiplicity_transition_epochs",
                "positive_multiplicity_step_values",
                "positive_multiplicity_step_transitions",
            }:
                value = tuple(int(item) for item in value)
            kwargs[field.name] = value
        parsed = cls(**kwargs)
        parsed.validate()
        return parsed

    def validate(self):
        if self.child_encoding_mode not in {"joint", "independent"}:
            raise ValueError("cell_detr child_encoding_mode must be joint or independent")
        if self.anchor_capacity not in (64, 128, 256, 512, 1024):
            raise ValueError(
                "cell_detr_anchor_capacity must be one of 64, 128, 256, 512, 1024"
            )
        if self.hidden_dim < 8 or self.hidden_dim % self.num_heads:
            raise ValueError("cell_detr hidden_dim must be divisible by num_heads")
        if self.parent_depth < 1:
            raise ValueError("cell_detr_parent_depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("cell_detr_dropout must be in [0, 1)")
        if not -1.0 < self.max_child_similarity < 1.0:
            raise ValueError("cell_detr_max_child_similarity must be in (-1, 1)")
        if self.matching_temperature <= 0.0 or self.inference_temperature <= 0.0:
            raise ValueError("cell_detr temperatures must be positive")
        if not self.positive_multiplicities:
            raise ValueError("positive_multiplicities cannot be empty")
        if len(self.multiplicity_transition_epochs) != len(self.positive_multiplicities) - 1:
            raise ValueError("multiplicity schedule needs one fewer transition than values")
        if any(value < 1 for value in self.positive_multiplicities):
            raise ValueError("all positive multiplicities must be positive")
        if any(
            left >= right
            for left, right in zip(
                self.multiplicity_transition_epochs,
                self.multiplicity_transition_epochs[1:],
            )
        ):
            raise ValueError("multiplicity transition epochs must be increasing")
        if self.positive_multiplicities[-1] != 1:
            raise ValueError("the final matching phase must be one-to-one")
        if not self.positive_multiplicity_step_values:
            raise ValueError("positive_multiplicity_step_values cannot be empty")
        if len(self.positive_multiplicity_step_transitions) != len(
            self.positive_multiplicity_step_values
        ) - 1:
            raise ValueError(
                "step multiplicity schedule needs one fewer transition than values"
            )
        if any(value < 1 for value in self.positive_multiplicity_step_values):
            raise ValueError("all step positive multiplicities must be positive")
        if any(
            left >= right
            for left, right in zip(
                self.positive_multiplicity_step_transitions,
                self.positive_multiplicity_step_transitions[1:],
            )
        ):
            raise ValueError("step multiplicity transitions must be increasing")
        if any(value < 0 for value in self.positive_multiplicity_step_transitions):
            raise ValueError("step multiplicity transitions must be non-negative")
        if self.positive_multiplicity_step_values[-1] != 1:
            raise ValueError("the final step matching phase must be one-to-one")
        nonnegative = (
            self.endpoint_cost_weight,
            self.direction_cost_weight,
            self.magnitude_cost_weight,
            self.router_cost_weight,
            self.delta_loss_weight,
            self.direction_loss_weight,
            self.magnitude_loss_weight,
            self.router_loss_weight,
            self.latent_match_loss_weight,
            self.gate_loss_weight,
            self.parent_auxiliary_loss_weight,
            self.compactness_loss_weight,
            self.separation_loss_weight,
            self.context_scale,
            self.inference_occupancy_logit_weight,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("cell_detr weights must be non-negative")
        if self.cold_start_parent_epochs < 0:
            raise ValueError("cold_start_parent_epochs must be non-negative")
        if self.cold_start_transition_end_epoch <= self.cold_start_parent_epochs:
            raise ValueError(
                "cold_start_transition_end_epoch must follow the Parent warmup"
            )
        if not 0.0 < self.gate_initial_probability < 1.0:
            raise ValueError("gate_initial_probability must be in (0, 1)")
        unit_interval = (
            self.gate_exploration_start,
            self.gate_exploration_end,
        )
        if any(value < 0.0 or value > 1.0 for value in unit_interval):
            raise ValueError("gate exploration values must be in [0, 1]")
        positive = (
            self.gate_temperature_start,
            self.gate_temperature_end,
            self.router_temperature_start,
            self.router_temperature_end,
            self.balance_target_power_start,
            self.balance_target_power_end,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("cold-start temperatures and powers must be positive")
        cold_nonnegative = (
            self.gate_relative_margin_start,
            self.gate_relative_margin_end,
            self.cold_start_router_loss_weight_start,
            self.cold_start_router_loss_weight_end,
            self.router_occupancy_logit_weight_start,
            self.router_occupancy_logit_weight_end,
            self.balance_loss_weight_start,
            self.balance_loss_weight_end,
        )
        if any(value < 0.0 for value in cold_nonnegative):
            raise ValueError("cold-start schedule values must be non-negative")

    def multiplicity_for_epoch(self, epoch: int) -> int:
        phase = 0
        for transition in self.multiplicity_transition_epochs:
            if int(epoch) < transition:
                break
            phase += 1
        return int(self.positive_multiplicities[phase])

    def multiplicity_for_step(self, step: int) -> int:
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        phase = 0
        for transition in self.positive_multiplicity_step_transitions:
            if step < transition:
                break
            phase += 1
        return int(self.positive_multiplicity_step_values[phase])

    def positive_multiplicity(self, *, epoch: int, step: int | None = None) -> int:
        if not self.positive_multiplicity_step_enabled:
            return self.multiplicity_for_epoch(epoch)
        if step is None:
            raise ValueError(
                "step is required when positive_multiplicity_step_enabled=true"
            )
        return self.multiplicity_for_step(step)

    @staticmethod
    def _linear(epoch: int, start_epoch: int, end_epoch: int) -> float:
        value = (int(epoch) - int(start_epoch)) / float(end_epoch - start_epoch)
        return min(1.0, max(0.0, value))

    @staticmethod
    def _lerp(start: float, end: float, fraction: float) -> float:
        return float(start) + (float(end) - float(start)) * float(fraction)

    def cold_start_fraction(self, epoch: int) -> float:
        return self._linear(
            epoch,
            self.cold_start_parent_epochs,
            self.cold_start_transition_end_epoch,
        )

    def cold_start_settings_for_epoch(self, epoch: int) -> dict[str, float | bool]:
        fraction = self.cold_start_fraction(epoch)
        return {
            "force_parent": int(epoch) < self.cold_start_parent_epochs,
            "relative_margin": self._lerp(
                self.gate_relative_margin_start,
                self.gate_relative_margin_end,
                fraction,
            ),
            "gate_temperature": self._lerp(
                self.gate_temperature_start,
                self.gate_temperature_end,
                fraction,
            ),
            "exploration_weight": self._lerp(
                self.gate_exploration_start,
                self.gate_exploration_end,
                fraction,
            ),
            "router_temperature": self._lerp(
                self.router_temperature_start,
                self.router_temperature_end,
                fraction,
            ),
            "router_occupancy_logit_weight": self._lerp(
                self.router_occupancy_logit_weight_start,
                self.router_occupancy_logit_weight_end,
                fraction,
            ),
            "balance_target_power": self._lerp(
                self.balance_target_power_start,
                self.balance_target_power_end,
                fraction,
            ),
        }

    def effective_router_loss_weight(self, epoch: int) -> float:
        if not self.cold_start_gate_enabled:
            return float(self.router_loss_weight)
        return self._lerp(
            self.cold_start_router_loss_weight_start,
            self.cold_start_router_loss_weight_end,
            self.cold_start_fraction(epoch),
        )

    def balance_loss_weight_for_epoch(self, epoch: int) -> float:
        if not self.cold_start_gate_enabled:
            return 0.0
        return self._lerp(
            self.balance_loss_weight_start,
            self.balance_loss_weight_end,
            self.cold_start_fraction(epoch),
        )

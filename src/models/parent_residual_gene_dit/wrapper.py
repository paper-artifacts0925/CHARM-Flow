"""Standalone Parent-Residual Gene-DiT model and target-free source API."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.cell_detr.config import CellDETRConfig
from src.models.cell_detr.context import CellDETRContextEncoder
from src.models.cell_detr.matching import PerCellHungarianMatcher

from .config import ParentResidualGeneDiTConfig
from .child_guidance import (
    ConditionReliabilityGate,
    coupling_at_step,
    parent_fill_residual_proposal,
    positive_parallel_child_guidance,
    rms_trust_clip,
    scale_gradient,
)
from .model import ParentResidualGeneDiT
from .parent_module_decoder import (
    ChildAwareParentModuleDecoder,
    ChildAwareParentModuleDecoderConfig,
)
from .prior_bank import ParentPriorLookup, ParentResidualPriorBank
from .source import (
    ParentResidualSource,
    _banked_center,
    build_parent_residual_prior_source,
)
from .tied_gene_correction import TiedGeneParentCorrection


@dataclass(frozen=True)
class ParentMeanPrediction:
    lookup: ParentPriorLookup
    prior_feature: torch.Tensor
    raw_correction: torch.Tensor
    correction: torch.Tensor
    parent_mean: torch.Tensor
    fewshot: object = None
    legacy_delta: Optional[torch.Tensor] = None
    soft_gate_logits: Optional[torch.Tensor] = None
    soft_gate_probability: Optional[torch.Tensor] = None
    soft_gate_slab: Optional[torch.Tensor] = None
    effective_delta: Optional[torch.Tensor] = None
    soft_gate_effective_blend: Optional[float] = None
    unified_sparse_program: object = None
    base_parent_mean: Optional[torch.Tensor] = None
    child_guidance_proposal: Optional[torch.Tensor] = None
    child_guidance_residual: Optional[torch.Tensor] = None
    child_guidance_alpha: Optional[torch.Tensor] = None
    child_guidance_coupling: Optional[float] = None
    child_guidance_topk_mass: Optional[torch.Tensor] = None
    child_marginal_mean: Optional[torch.Tensor] = None
    child_marginal_control_mean: Optional[torch.Tensor] = None
    child_marginal_delta: Optional[torch.Tensor] = None
    child_marginal_residual: Optional[torch.Tensor] = None
    child_marginal_contribution: Optional[torch.Tensor] = None
    child_marginal_alpha: Optional[torch.Tensor] = None
    child_marginal_router_entropy: Optional[torch.Tensor] = None
    child_marginal_router_effective_children: Optional[torch.Tensor] = None
    child_consensus_bridge_legacy_rms: Optional[torch.Tensor] = None
    child_consensus_bridge_full_raw_rms: Optional[torch.Tensor] = None
    child_consensus_bridge_collapsed_raw_rms: Optional[torch.Tensor] = None
    child_consensus_bridge_contrast_raw_rms: Optional[torch.Tensor] = None
    child_consensus_bridge_realized_delta_rms: Optional[torch.Tensor] = None
    child_consensus_bridge_delta_to_legacy_ratio: Optional[torch.Tensor] = None
    child_consensus_bridge_shrink: Optional[torch.Tensor] = None
    child_consensus_bridge_saturation: Optional[torch.Tensor] = None
    child_consensus_bridge_legacy_zero: Optional[torch.Tensor] = None
    tied_gene_raw_rms: Optional[torch.Tensor] = None
    tied_gene_legacy_raw_rms: Optional[torch.Tensor] = None
    tied_gene_to_legacy_ratio: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class PreparedPriorSource:
    source: ParentResidualSource
    parent: ParentMeanPrediction
    prior: torch.Tensor
    child_indices: torch.Tensor
    selected_child_tokens: torch.Tensor
    real_control_sample: object = None


@dataclass(frozen=True)
class StrictParentOnlyContext:
    """Target-free context containing only Parent and perturbation semantics."""

    parent_token: torch.Tensor
    perturbation_query: torch.Tensor
    parent_control: torch.Tensor
    parent_control_feature: torch.Tensor


@dataclass(frozen=True)
class ParentOnlyResidualSource:
    """Parent-only FM source with no routing or Child identity fields."""

    source: torch.Tensor
    mean: torch.Tensor
    residual: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class PreparedParentOnlySource:
    """Strict Parent-only source and Parent prediction."""

    source: ParentOnlyResidualSource
    parent: ParentMeanPrediction
    real_control_sample: object = None



class ParentResidualGeneDiTModel(nn.Module):
    """Gene-module FM backbone with Parent mean and centred Child states.

    Target expression is accepted only by ``prepare_training_context`` to form
    detached matcher labels.  The sole source constructor is ``build_prior_source``;
    its signature intentionally has no target or match argument and is shared by
    training and inference.
    """

    model_type = "parent_residual_gene_dit"
    model_name = "Cross_DiT"

    @staticmethod
    def _context_config(model_cfg) -> CellDETRConfig:
        """Parse the independent Parent-Residual context namespace."""
        aliases = {
            "anchor_capacity": ("parent_residual_anchor_capacity",),
            "hidden_dim": ("parent_residual_context_hidden_dim",),
            "num_heads": ("parent_residual_context_heads",),
            "parent_depth": (
                "parent_residual_context_parent_depth",
                "parent_residual_parent_depth",
            ),
            "dropout": ("parent_residual_context_dropout",),
            "max_child_similarity": (
                "parent_residual_context_max_child_similarity",
            ),
            "endpoint_cost_weight": (
                "parent_residual_match_endpoint_cost_weight",
            ),
            "direction_cost_weight": (
                "parent_residual_match_direction_cost_weight",
            ),
            "magnitude_cost_weight": (
                "parent_residual_match_magnitude_cost_weight",
            ),
            "matching_temperature": (
                "parent_residual_match_temperature",
            ),
        }
        values = {}
        for field in CellDETRConfig.__dataclass_fields__.values():
            names = aliases.get(
                field.name,
                ("parent_residual_" + field.name,),
            )
            value = getattr(model_cfg, "cell_detr_" + field.name, field.default)
            for name in reversed(names):
                value = getattr(model_cfg, name, value)
            if field.name in {
                "positive_multiplicities",
                "multiplicity_transition_epochs",
                "positive_multiplicity_step_values",
                "positive_multiplicity_step_transitions",
            }:
                value = tuple(int(item) for item in value)
            values[field.name] = value
        parsed = CellDETRConfig(**values)
        parsed.validate()
        return parsed

    @staticmethod
    def _child_context_mode_from_config(model_cfg) -> str:
        mode = str(
            getattr(model_cfg, "parent_residual_child_context_mode", "native")
        ).lower()
        if mode not in {"native", "global_replicated"}:
            raise ValueError(
                "parent_residual_child_context_mode must be native or "
                "global_replicated"
            )
        return mode

    @staticmethod
    def _source_selection_mode_from_config(model_cfg) -> str:
        mode = str(
            getattr(model_cfg, "parent_residual_source_selection_mode", "router")
        ).lower()
        if mode not in {"router", "occupancy"}:
            raise ValueError(
                "parent_residual_source_selection_mode must be router or occupancy"
            )
        return mode

    @staticmethod
    def _field_context_mode_from_config(model_cfg) -> str:
        mode = str(
            getattr(model_cfg, "parent_residual_field_context_mode", "hierarchical")
        ).lower()
        if mode not in {"hierarchical", "population_only"}:
            raise ValueError(
                "parent_residual_field_context_mode must be hierarchical or "
                "population_only"
            )
        return mode

    @staticmethod
    def _gene_module_ids_from_config(model_cfg, artifact_module_ids):
        mode = str(
            getattr(model_cfg, "parent_residual_gene_partition_mode", "artifact")
        ).lower()
        if mode not in {"artifact", "random_size_preserving"}:
            raise ValueError(
                "parent_residual_gene_partition_mode must be artifact or "
                "random_size_preserving"
            )
        if mode == "artifact":
            return artifact_module_ids
        seed = getattr(model_cfg, "parent_residual_gene_partition_seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "parent_residual_gene_partition_seed must be a non-negative integer"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(
            int(artifact_module_ids.numel()), generator=generator
        )
        randomized = artifact_module_ids.detach().cpu()[permutation]
        return randomized.to(device=artifact_module_ids.device)

    @staticmethod
    def _parent_module_decoder_config(model_cfg) -> dict:
        """Parse the fixed-capacity Child-aware Parent decoder switches."""

        enabled_name = "parent_residual_parent_module_decoder_enabled"
        enabled = getattr(model_cfg, enabled_name, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{enabled_name} must be boolean")
        detach_name = (
            "parent_residual_parent_module_decoder_detach_child_context"
        )
        detach_child_context = getattr(model_cfg, detach_name, True)
        if not isinstance(detach_child_context, bool):
            raise ValueError(f"{detach_name} must be boolean")
        beta_name = "parent_residual_parent_module_decoder_router_beta"
        raw_beta = getattr(model_cfg, beta_name, 0.5)
        if isinstance(raw_beta, bool):
            raise ValueError(f"{beta_name} must be a finite non-negative number")
        try:
            router_beta = float(raw_beta)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{beta_name} must be a finite non-negative number"
            ) from error
        if not math.isfinite(router_beta) or router_beta < 0.0:
            raise ValueError(
                f"{beta_name} must be a finite non-negative number"
            )
        incompatible = (
            "parent_residual_child_consensus_bridge_enabled",
            "parent_residual_child_mean_flow_enabled",
            "parent_residual_child_guidance_enabled",
            "parent_residual_child_marginal_enabled",
        )
        active_incompatible = [
            name
            for name in incompatible
            if enabled and bool(getattr(model_cfg, name, False))
        ]
        if active_incompatible:
            raise ValueError(
                "Parent Module Decoder first arm requires these switches=false: "
                + ", ".join(active_incompatible)
            )
        return {
            "enabled": enabled,
            "detach_child_context": detach_child_context,
            "router_beta": router_beta,
        }

    @staticmethod
    def _tied_gene_correction_config(model_cfg) -> dict:
        """Parse the isolated factorized Parent gene-output exit."""

        enabled_name = "parent_residual_tied_gene_correction_enabled"
        enabled = getattr(model_cfg, enabled_name, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{enabled_name} must be boolean")

        rank_name = "parent_residual_tied_gene_correction_rank"
        rank = getattr(model_cfg, rank_name, 64)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"{rank_name} must be a positive integer")

        mode_name = "parent_residual_tied_gene_correction_mode"
        mode = getattr(model_cfg, mode_name, "replace")
        valid_modes = {"replace", "residual", "bounded_residual"}
        if not isinstance(mode, str) or mode not in valid_modes:
            raise ValueError(
                f"{mode_name} must be replace, residual, or bounded_residual"
            )

        scale_name = "parent_residual_tied_gene_correction_residual_scale"
        raw_scale = getattr(model_cfg, scale_name, 1.0)
        if isinstance(raw_scale, bool):
            raise ValueError(f"{scale_name} must lie in [0, 1]")
        try:
            residual_scale = float(raw_scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{scale_name} must lie in [0, 1]") from error
        if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
            raise ValueError(f"{scale_name} must lie in [0, 1]")
        if enabled and mode == "replace" and residual_scale != 1.0:
            raise ValueError(f"{scale_name} must equal 1 in replace mode")

        bounded_name = "parent_residual_tied_gene_correction_bounded_enabled"
        bounded_enabled = getattr(model_cfg, bounded_name, False)
        if not isinstance(bounded_enabled, bool):
            raise ValueError(f"{bounded_name} must be boolean")

        detach_name = "parent_residual_tied_gene_correction_detach_gene_identity"
        detach_gene_identity = getattr(model_cfg, detach_name, True)
        if not isinstance(detach_gene_identity, bool):
            raise ValueError(f"{detach_name} must be boolean")

        incompatible = (
            "parent_residual_parent_module_decoder_enabled",
            "parent_residual_child_consensus_bridge_enabled",
        )
        active_incompatible = [
            name
            for name in incompatible
            if enabled and bool(getattr(model_cfg, name, False))
        ]
        if active_incompatible:
            raise ValueError(
                "tied gene correction requires these switches=false: "
                + ", ".join(active_incompatible)
            )
        return {
            "enabled": enabled,
            "rank": rank,
            "mode": mode,
            "residual_scale": residual_scale,
            "bounded_enabled": bounded_enabled,
            "detach_gene_identity": detach_gene_identity,
        }

    @staticmethod
    def _child_consensus_bridge_config(model_cfg) -> dict:
        """Parse the isolated legacy-MLP plus Child-contrast bridge arm."""

        enabled_name = "parent_residual_child_consensus_bridge_enabled"
        enabled = getattr(model_cfg, enabled_name, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{enabled_name} must be boolean")

        detach_name = (
            "parent_residual_child_consensus_bridge_detach_child_context"
        )
        detach_child_context = getattr(model_cfg, detach_name, True)
        if not isinstance(detach_child_context, bool):
            raise ValueError(f"{detach_name} must be boolean")

        beta_name = "parent_residual_child_consensus_bridge_router_beta"
        raw_beta = getattr(model_cfg, beta_name, 0.5)
        if isinstance(raw_beta, bool):
            raise ValueError(f"{beta_name} must be a finite non-negative number")
        try:
            router_beta = float(raw_beta)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{beta_name} must be a finite non-negative number"
            ) from error
        if not math.isfinite(router_beta) or router_beta < 0.0:
            raise ValueError(
                f"{beta_name} must be a finite non-negative number"
            )

        ratio_name = (
            "parent_residual_child_consensus_bridge_max_delta_to_legacy_ratio"
        )
        raw_ratio = getattr(model_cfg, ratio_name, 0.25)
        if isinstance(raw_ratio, bool):
            raise ValueError(f"{ratio_name} must lie in [0, 0.25]")
        try:
            max_delta_to_legacy_ratio = float(raw_ratio)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{ratio_name} must lie in [0, 0.25]") from error
        if (
            not math.isfinite(max_delta_to_legacy_ratio)
            or not 0.0 <= max_delta_to_legacy_ratio <= 0.25
        ):
            raise ValueError(f"{ratio_name} must lie in [0, 0.25]")
        bootstrap_name = (
            "parent_residual_child_consensus_bridge_bootstrap_reference_rms"
        )
        raw_bootstrap = getattr(model_cfg, bootstrap_name, 1.0e-4)
        if isinstance(raw_bootstrap, bool):
            raise ValueError(f"{bootstrap_name} must be finite and positive")
        try:
            bootstrap_reference_rms = float(raw_bootstrap)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{bootstrap_name} must be finite and positive"
            ) from error
        if (
            not math.isfinite(bootstrap_reference_rms)
            or bootstrap_reference_rms <= 0.0
        ):
            raise ValueError(f"{bootstrap_name} must be finite and positive")
        scale_name = "parent_residual_correction_max_scale"
        raw_scale = getattr(model_cfg, scale_name, 0.05)
        if isinstance(raw_scale, bool):
            raise ValueError(f"{scale_name} must equal 0.05 for the bridge")
        try:
            correction_max_scale = float(raw_scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{scale_name} must equal 0.05 for the bridge"
            ) from error
        if enabled and correction_max_scale != 0.05:
            raise ValueError(f"{scale_name} must equal 0.05 for the bridge")

        incompatible = (
            "parent_residual_parent_module_decoder_enabled",
            "parent_residual_child_mean_flow_enabled",
            "parent_residual_child_guidance_enabled",
            "parent_residual_child_marginal_enabled",
            "parent_residual_strict_parent_only_enabled",
        )
        active_incompatible = [
            name
            for name in incompatible
            if enabled and bool(getattr(model_cfg, name, False))
        ]
        if active_incompatible:
            raise ValueError(
                "Child consensus bridge requires these switches=false: "
                + ", ".join(active_incompatible)
            )
        return {
            "enabled": enabled,
            "detach_child_context": detach_child_context,
            "router_beta": router_beta,
            "max_delta_to_legacy_ratio": max_delta_to_legacy_ratio,
            "bootstrap_reference_rms": bootstrap_reference_rms,
        }

    @staticmethod
    def _real_control_sampling_mode_from_config(model_cfg) -> str:
        mode = str(
            getattr(
                model_cfg,
                "parent_residual_real_control_sampling_mode",
                "child",
            )
        ).strip().lower()
        if mode not in {"child", "global_marginal"}:
            raise ValueError(
                "parent_residual_real_control_sampling_mode must be child or "
                "global_marginal"
            )
        return mode

    @staticmethod
    def _validate_matching_guidance_disabled_config(model_cfg) -> None:
        """Fail closed for the target-free, native-Child ablation contract."""
        forbidden_flags = (
            "parent_residual_matched_set_enabled",
            "parent_residual_condition_router_kl_enabled",
            "parent_residual_semi_balanced_ot_enabled",
            "parent_residual_balanced_ot_enabled",
            "parent_residual_joint_parent_child_enabled",
        )
        enabled = [
            name for name in forbidden_flags if bool(getattr(model_cfg, name, False))
        ]
        if enabled:
            raise ValueError(
                "matching-guidance-disabled mode forbids matching switches: "
                + ", ".join(enabled)
            )

        required_zero = (
            "parent_residual_posterior_start_fraction",
            "parent_residual_posterior_end_fraction",
            "parent_residual_router_kl_weight",
            "parent_residual_latent_match_weight",
            "parent_residual_teacher_delta_weight",
            "parent_residual_teacher_direction_weight",
            "parent_residual_teacher_magnitude_weight",
            "parent_residual_teacher_latent_weight",
            "parent_residual_child_endpoint_pds_weight",
            "parent_residual_parent_listwise_pds_weight",
        )
        nonzero = []
        for name in required_zero:
            try:
                value = float(getattr(model_cfg, name, 0.0))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value != 0.0:
                nonzero.append(name)
        if nonzero:
            raise ValueError(
                "matching-guidance-disabled mode requires zero matching weights: "
                + ", ".join(nonzero)
            )

        required_modes = {
            "parent_residual_child_context_mode": "native",
            "parent_residual_source_selection_mode": "occupancy",
            "parent_residual_real_control_sampling_mode": "child",
        }
        wrong_modes = []
        for name, expected in required_modes.items():
            actual = str(getattr(model_cfg, name, "")).strip().lower()
            if actual != expected:
                wrong_modes.append(f"{name}={actual!r} (expected {expected!r})")
        if wrong_modes:
            raise ValueError(
                "matching-guidance-disabled mode requires native occupancy Child "
                "sampling: " + ", ".join(wrong_modes)
            )

        required_flags = (
            "parent_residual_real_control_source_enabled",
            "parent_residual_locked_flow_enabled",
            "parent_residual_grouped_set_loss_enabled",
        )
        disabled = [
            name for name in required_flags if not bool(getattr(model_cfg, name, False))
        ]
        if disabled:
            raise ValueError(
                "matching-guidance-disabled mode requires these switches=true: "
                + ", ".join(disabled)
            )
        if bool(
            getattr(model_cfg, "parent_residual_strict_parent_only_enabled", False)
        ):
            raise ValueError(
                "parent_residual_strict_parent_only_enabled must be false when "
                "matching guidance is disabled; this mode retains native Child and "
                "cannot be combined with strict Parent-only"
            )

    @staticmethod
    def _validate_strict_parent_only_config(model_cfg) -> None:
        """Reject every routing/Child objective on the strict ablation path."""
        locked_flow = bool(
            getattr(model_cfg, "parent_residual_locked_flow_enabled", False)
        )
        direct_flow = bool(
            getattr(
                model_cfg,
                "parent_residual_strict_direct_flow_enabled",
                False,
            )
        )
        if locked_flow == direct_flow:
            raise ValueError(
                "strict Parent-only mode requires exactly one of locked flow "
                "or strict direct flow"
            )

        forbidden_flags = (
            "parent_residual_matched_set_enabled",
            "parent_residual_condition_router_kl_enabled",
            "parent_residual_semi_balanced_ot_enabled",
            "parent_residual_balanced_ot_enabled",
            "parent_residual_child_mean_flow_enabled",
            "parent_residual_child_guidance_enabled",
            "parent_residual_child_marginal_enabled",
            "parent_residual_global_local_memory_split_enabled",
            "parent_residual_parent_module_decoder_enabled",
            "parent_residual_child_consensus_bridge_enabled",
        )
        enabled = [
            name for name in forbidden_flags if bool(getattr(model_cfg, name, False))
        ]
        if enabled:
            raise ValueError(
                "strict Parent-only mode forbids routing/Child switches: "
                + ", ".join(enabled)
            )

        if direct_flow:
            direct_forbidden_flags = (
                "parent_residual_grouped_set_loss_enabled",
                "parent_residual_de_aware_enabled",
                "parent_residual_hurdle_geometry_enabled",
                "parent_residual_sparse_support_enabled",
                "parent_residual_soft_gated_regression_enabled",
                "parent_residual_strong_condition_memory_enabled",
            )
            direct_enabled = [
                name
                for name in direct_forbidden_flags
                if bool(getattr(model_cfg, name, False))
            ]
            if direct_enabled:
                raise ValueError(
                    "strict direct flow forbids locked/Child auxiliary switches: "
                    + ", ".join(direct_enabled)
                )
            if not bool(
                getattr(
                    model_cfg,
                    "parent_residual_real_control_source_enabled",
                    False,
                )
            ):
                raise ValueError(
                    "strict direct flow requires a real-control residual source"
                )
            sampling_mode = str(
                getattr(
                    model_cfg,
                    "parent_residual_real_control_sampling_mode",
                    "child",
                )
            ).strip().lower()
            if sampling_mode != "global_marginal":
                raise ValueError(
                    "strict direct flow requires global_marginal control sampling"
                )
            residual_scale = float(
                getattr(
                    model_cfg,
                    "parent_residual_real_control_residual_scale",
                    1.0,
                )
            )
            if not math.isfinite(residual_scale) or residual_scale != 1.0:
                raise ValueError(
                    "strict direct flow requires real-control residual scale 1"
                )
            if not bool(
                getattr(
                    model_cfg,
                    "parent_residual_real_control_stochastic",
                    True,
                )
            ):
                raise ValueError(
                    "strict direct flow requires stochastic real-control sampling"
                )

        forbidden_nonzero = {
            "parent_residual_posterior_start_fraction": 1.0,
            "parent_residual_posterior_end_fraction": 0.0,
            "parent_residual_router_kl_weight": 5e-4,
            "parent_residual_latent_match_weight": 5e-5,
            "parent_residual_teacher_delta_weight": 0.0,
            "parent_residual_teacher_direction_weight": 0.0,
            "parent_residual_teacher_magnitude_weight": 0.0,
            "parent_residual_teacher_latent_weight": 0.0,
            "parent_residual_beta": 0.0,
            "parent_residual_child_beta": 0.0,
            "parent_residual_beta_std": 0.0,
            "parent_residual_child_std_beta": 0.0,
            "parent_residual_child_endpoint_pds_weight": 0.0,
            "parent_residual_parent_listwise_pds_weight": 0.0,
        }
        nonzero = []
        for name, default in forbidden_nonzero.items():
            try:
                value = float(getattr(model_cfg, name, default))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value != 0.0:
                nonzero.append(name)
        if nonzero:
            raise ValueError(
                "strict Parent-only mode requires zero routing/Child weights: "
                + ", ".join(nonzero)
            )

    @staticmethod
    def _child_guidance_config(model_cfg) -> dict:
        """Parse the optional, conservative Child-to-Parent coupling contract."""

        def positive_integer(name, default):
            raw = getattr(model_cfg, name, default)
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if isinstance(raw, bool) or value != raw or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            return value

        def nonnegative_integer(name, default):
            raw = getattr(model_cfg, name, default)
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} must be a non-negative integer"
                ) from error
            if isinstance(raw, bool) or value != raw or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            return value

        def bounded_float(name, default, *, lower, upper, lower_open=False):
            raw = getattr(model_cfg, name, default)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be a finite number")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            lower_ok = value > lower if lower_open else value >= lower
            if not lower_ok or value > upper:
                left = "(" if lower_open else "["
                raise ValueError(
                    f"{name} must lie in {left}{lower}, {upper}]"
                )
            return value

        detach_name = "parent_residual_child_guidance_detach_router_probabilities"
        detach_router = getattr(model_cfg, detach_name, True)
        if not isinstance(detach_router, bool):
            raise ValueError(f"{detach_name} must be boolean")
        mode_name = "parent_residual_child_guidance_mode"
        mode = str(getattr(model_cfg, mode_name, "residual_gate")).strip().lower()
        if mode not in {
            "residual_gate",
            "positive_parallel",
            "occupancy_contrast",
        }:
            raise ValueError(
                f"{mode_name} must be residual_gate, positive_parallel, or "
                "occupancy_contrast"
            )
        return {
            "mode": mode,
            "top_k": positive_integer(
                "parent_residual_child_guidance_top_k", 32
            ),
            "max_rms_ratio": bounded_float(
                "parent_residual_child_guidance_max_rms_ratio",
                0.1,
                lower=0.0,
                upper=1.0,
                # Zero is the exact paired-control boundary for the
                # occupancy-contrast experiment and is a safe no-op for the
                # historical guidance modes as well.
                lower_open=False,
            ),
            "gate_hidden_dim": positive_integer(
                "parent_residual_child_guidance_gate_hidden_dim", 32
            ),
            "max_alpha": bounded_float(
                "parent_residual_child_guidance_max_alpha",
                0.25,
                lower=0.0,
                upper=1.0,
                lower_open=True,
            ),
            "detach_router_probabilities": detach_router,
            "gradient_scale": bounded_float(
                "parent_residual_child_guidance_gradient_scale",
                0.1,
                lower=0.0,
                upper=1.0,
            ),
            "start_step": nonnegative_integer(
                "parent_residual_child_guidance_start_step", 500
            ),
            "warmup_steps": nonnegative_integer(
                "parent_residual_child_guidance_warmup_steps", 1500
            ),
            "max_coupling": bounded_float(
                "parent_residual_child_guidance_max_coupling",
                1.0,
                lower=0.0,
                upper=1.0,
            ),
            "parallel_scale": bounded_float(
                "parent_residual_child_guidance_parallel_scale",
                0.1,
                lower=0.0,
                upper=1.0,
            ),
            "parallel_max_projection": bounded_float(
                "parent_residual_child_guidance_parallel_max_projection",
                1.0,
                lower=0.0,
                upper=1.0,
                lower_open=True,
            ),
        }

    @staticmethod
    def _child_marginal_config(model_cfg) -> dict:
        """Parse the parameter-free Child-barycentric Parent coupling."""

        enabled_name = "parent_residual_child_marginal_enabled"
        enabled = getattr(model_cfg, enabled_name, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{enabled_name} must be boolean")

        detach_name = "parent_residual_child_marginal_detach_router_probabilities"
        detach_router = getattr(model_cfg, detach_name, True)
        if not isinstance(detach_router, bool):
            raise ValueError(f"{detach_name} must be boolean")

        alpha_name = "parent_residual_child_marginal_alpha"
        raw_alpha = getattr(model_cfg, alpha_name, 0.25)
        if isinstance(raw_alpha, bool):
            raise ValueError(f"{alpha_name} must be a finite number")
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{alpha_name} must be a finite number") from error
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"{alpha_name} must lie in [0, 1]")

        def nonnegative_integer(name, default):
            raw = getattr(model_cfg, name, default)
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} must be a non-negative integer"
                ) from error
            if isinstance(raw, bool) or value != raw or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            return value

        return {
            "enabled": enabled,
            "alpha": alpha,
            "start_step": nonnegative_integer(
                "parent_residual_child_marginal_start_step", 500
            ),
            "warmup_steps": nonnegative_integer(
                "parent_residual_child_marginal_warmup_steps", 1500
            ),
            "detach_router_probabilities": detach_router,
        }

    def __init__(self, model_cfg, covariate_config):
        super().__init__()
        configured = str(getattr(model_cfg, "model_type", "")).lower()
        if configured != self.model_type:
            raise ValueError(
                f"ParentResidualGeneDiTModel requires model_type={self.model_type!r}"
            )
        if str(getattr(model_cfg, "generative_process", "")).lower() != "rectified_flow":
            raise ValueError("Parent-Residual Gene-DiT requires rectified_flow")
        self.model_cfg = model_cfg
        population_response_prior_enabled = getattr(
            model_cfg,
            "parent_residual_population_response_prior_enabled",
            True,
        )
        if not isinstance(population_response_prior_enabled, bool):
            raise ValueError(
                "parent_residual_population_response_prior_enabled must be a boolean"
            )
        self.population_response_prior_enabled = (
            population_response_prior_enabled
        )
        parent_module_decoder = self._parent_module_decoder_config(model_cfg)
        self.parent_module_decoder_enabled = parent_module_decoder["enabled"]
        self.parent_module_decoder_detach_child_context = (
            parent_module_decoder["detach_child_context"]
        )
        self.parent_module_decoder_router_beta = parent_module_decoder[
            "router_beta"
        ]
        tied_gene_correction = self._tied_gene_correction_config(model_cfg)
        self.tied_gene_correction_enabled = tied_gene_correction["enabled"]
        self.tied_gene_correction_rank = tied_gene_correction["rank"]
        self.tied_gene_correction_mode = tied_gene_correction["mode"]
        self.tied_gene_correction_residual_scale = tied_gene_correction[
            "residual_scale"
        ]
        self.tied_gene_correction_bounded_enabled = tied_gene_correction[
            "bounded_enabled"
        ]
        self.tied_gene_correction_detach_gene_identity = (
            tied_gene_correction["detach_gene_identity"]
        )
        child_consensus_bridge = self._child_consensus_bridge_config(model_cfg)
        self.child_consensus_bridge_enabled = child_consensus_bridge["enabled"]
        self.child_consensus_bridge_detach_child_context = (
            child_consensus_bridge["detach_child_context"]
        )
        self.child_consensus_bridge_router_beta = child_consensus_bridge[
            "router_beta"
        ]
        self.child_consensus_bridge_max_delta_to_legacy_ratio = (
            child_consensus_bridge["max_delta_to_legacy_ratio"]
        )
        self.child_consensus_bridge_bootstrap_reference_rms = (
            child_consensus_bridge["bootstrap_reference_rms"]
        )
        self.strict_parent_only_enabled = bool(
            getattr(
                model_cfg, "parent_residual_strict_parent_only_enabled", False
            )
        )
        matching_guidance = getattr(
            model_cfg, "parent_residual_matching_guidance_enabled", True
        )
        if not isinstance(matching_guidance, bool):
            raise ValueError(
                "parent_residual_matching_guidance_enabled must be a boolean"
            )
        self.matching_guidance_enabled = matching_guidance
        if not self.matching_guidance_enabled:
            self._validate_matching_guidance_disabled_config(model_cfg)
        self.strict_direct_flow_enabled = bool(
            self.strict_parent_only_enabled
            and getattr(
                model_cfg,
                "parent_residual_strict_direct_flow_enabled",
                False,
            )
        )
        if self.strict_parent_only_enabled:
            self._validate_strict_parent_only_config(model_cfg)
        self.field_context_mode = self._field_context_mode_from_config(model_cfg)
        if self.strict_parent_only_enabled and self.field_context_mode != "hierarchical":
            raise ValueError(
                "strict Parent-only and population-only field ablations cannot be "
                "enabled together"
            )
        self.gene_dim = int(getattr(model_cfg, "input_dim", 2000))
        self.flow_time_scale = float(getattr(model_cfg, "flow_time_scale", 1000.0))
        if not math.isfinite(self.flow_time_scale) or self.flow_time_scale <= 0.0:
            raise ValueError("flow_time_scale must be finite and positive")

        artifact_path = getattr(model_cfg, "parent_residual_artifact_path", None)
        if artifact_path in (None, "", "null"):
            raise ValueError("parent_residual_artifact_path is required")
        self.prior_bank = ParentResidualPriorBank(
            artifact_path,
            covariate_config,
            gene_dim=self.gene_dim,
            expected_sha256=getattr(
                model_cfg, "parent_residual_artifact_sha256", None
            ),
            prior_mode=str(
                getattr(model_cfg, "parent_residual_prior_mode", "equal_line")
            ),
            calibration=str(
                getattr(
                    model_cfg,
                    "parent_residual_calibration",
                    "no_intercept",
                )
            ),
            parent_grouping=str(
                getattr(model_cfg, "parent_residual_parent_grouping", "celltype")
            ),
            expected_normalization_divisor=float(
                getattr(
                    model_cfg,
                    "parent_residual_normalization_divisor",
                    10.0,
                )
            ),
        )
        self.fewshot_parent_adapter_enabled = bool(
            getattr(
                model_cfg,
                "parent_residual_fewshot_adapter_enabled",
                False,
            )
        )
        if (
            self.fewshot_parent_adapter_enabled
            and not self.population_response_prior_enabled
        ):
            raise ValueError(
                "few-shot Parent adapter requires response prior to be enabled"
            )
        self.fewshot_parent_adapter = None
        if self.fewshot_parent_adapter_enabled:
            if self.prior_bank.prior_mode != "equal_line":
                raise ValueError(
                    "few-shot Parent adapter requires prior_mode='equal_line'"
                )
            if self.prior_bank.calibration != "no_intercept":
                raise ValueError(
                    "few-shot Parent adapter requires calibration='no_intercept'"
                )
            if float(
                getattr(model_cfg, "parent_residual_correction_max_scale", 0.0)
            ) != 0.0:
                raise ValueError(
                    "few-shot Parent adapter requires the legacy Parent correction "
                    "head to be disabled (correction_max_scale=0)"
                )
            artifact_path = getattr(
                model_cfg,
                "parent_residual_fewshot_adapter_artifact_path",
                None,
            )
            if artifact_path in (None, "", "null"):
                raise ValueError(
                    "parent_residual_fewshot_adapter_artifact_path is required"
                )
            from src.models.fewshot_parent_adapter import (
                FewShotParentAdapter,
                FewShotParentAdapterConfig,
                FewShotParentArtifact,
            )

            fewshot_artifact = FewShotParentArtifact(
                artifact_path,
                expected_sha256=getattr(
                    model_cfg,
                    "parent_residual_fewshot_adapter_artifact_sha256",
                    None,
                ),
                expected_parent_sha256=self.prior_bank.artifact_sha256,
                expected_normalization_divisor=float(
                    getattr(
                        model_cfg,
                        "parent_residual_normalization_divisor",
                        10.0,
                    )
                ),
            )
            heldout_deployment_line_id = getattr(
                model_cfg,
                "parent_residual_fewshot_adapter_heldout_deployment_line_id",
                None,
            )
            if heldout_deployment_line_id in (None, "", "null"):
                raise ValueError(
                    "few-shot Parent adapter requires an explicit "
                    "heldout_deployment_line_id"
                )
            fewshot_config = FewShotParentAdapterConfig(
                rank=int(
                    getattr(model_cfg, "parent_residual_fewshot_adapter_rank", 32)
                ),
                support_size=int(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_support_size",
                        32,
                    )
                ),
                ridge_lambdas=tuple(
                    float(value)
                    for value in getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_ridge_lambdas",
                        (1e-3, 1e-2, 1e-1, 1.0, 10.0),
                    )
                ),
                parent_calibration_lambdas=tuple(
                    float(value)
                    for value in getattr(
                        model_cfg,
                        "parent_residual_fewshot_parent_calibration_lambdas",
                        (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                    )
                ),
                parent_calibration_default_lambda=float(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_parent_calibration_default_lambda",
                        1e-2,
                    )
                ),
                donor_ridge=float(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_donor_ridge",
                        1e-2,
                    )
                ),
                cv_folds=int(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_cv_folds",
                        4,
                    )
                ),
                minimum_relative_cv_gain=float(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_minimum_cv_gain",
                        0.0,
                    )
                ),
                seed=int(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_seed",
                        42,
                    )
                ),
                gate_maximum=float(
                    getattr(
                        model_cfg,
                        "parent_residual_fewshot_adapter_gate_maximum",
                        1.0,
                    )
                ),
                heldout_deployment_line_id=int(
                    heldout_deployment_line_id
                ),
            )
            self.fewshot_parent_adapter = FewShotParentAdapter(
                fewshot_artifact,
                self.prior_bank,
                fewshot_config,
            )

        self.cell_detr_cfg = self._context_config(model_cfg)
        self.child_context_mode = self._child_context_mode_from_config(model_cfg)
        self.source_selection_mode = self._source_selection_mode_from_config(
            model_cfg
        )
        context_cfg = self.cell_detr_cfg
        self.context_encoder = CellDETRContextEncoder(
            gene_dim=self.gene_dim,
            hidden_dim=context_cfg.hidden_dim,
            anchor_capacity=context_cfg.anchor_capacity,
            num_heads=context_cfg.num_heads,
            parent_depth=context_cfg.parent_depth,
            dropout=context_cfg.dropout,
            max_child_similarity=context_cfg.max_child_similarity,
            child_encoding_mode=context_cfg.child_encoding_mode,
            cold_start_gate_enabled=context_cfg.cold_start_gate_enabled,
            gate_initial_probability=context_cfg.gate_initial_probability,
        )
        self.matcher = PerCellHungarianMatcher(
            endpoint_cost_weight=context_cfg.endpoint_cost_weight,
            direction_cost_weight=context_cfg.direction_cost_weight,
            magnitude_cost_weight=context_cfg.magnitude_cost_weight,
            router_cost_weight=0.0,
            temperature=context_cfg.matching_temperature,
        )
        self.semi_balanced_ot_enabled = bool(
            getattr(model_cfg, "parent_residual_semi_balanced_ot_enabled", False)
        )
        self.semi_balanced_ot_epsilon = float(
            getattr(model_cfg, "parent_residual_semi_balanced_ot_epsilon", 0.1)
        )
        self.semi_balanced_ot_rho = float(
            getattr(model_cfg, "parent_residual_semi_balanced_ot_rho", 1.0)
        )
        self.semi_balanced_ot_iterations = int(
            getattr(model_cfg, "parent_residual_semi_balanced_ot_iterations", 50)
        )
        self.semi_balanced_ot_normalize_cost_by_median = bool(
            getattr(
                model_cfg,
                "parent_residual_semi_balanced_ot_cost_median_normalization",
                True,
            )
        )
        self.balanced_ot_enabled = bool(
            getattr(model_cfg, "parent_residual_balanced_ot_enabled", False)
        )
        self.balanced_ot_column_tolerance = float(
            getattr(
                model_cfg,
                "parent_residual_balanced_ot_column_tolerance",
                1.0e-3,
            )
        )
        if self.balanced_ot_enabled and not self.semi_balanced_ot_enabled:
            raise ValueError("balanced OT requires parent residual OT matching")
        if (
            not math.isfinite(self.balanced_ot_column_tolerance)
            or self.balanced_ot_column_tolerance <= 0.0
        ):
            raise ValueError(
                "balanced OT column tolerance must be finite and positive"
            )
        if (
            not math.isfinite(self.semi_balanced_ot_epsilon)
            or self.semi_balanced_ot_epsilon <= 0.0
        ):
            raise ValueError(
                "parent residual semi-balanced OT epsilon must be finite and positive"
            )
        if (
            not math.isfinite(self.semi_balanced_ot_rho)
            or self.semi_balanced_ot_rho < 0.0
        ):
            raise ValueError(
                "parent residual semi-balanced OT rho must be finite and non-negative"
            )
        if self.semi_balanced_ot_iterations < 1:
            raise ValueError(
                "parent residual semi-balanced OT iterations must be positive"
            )

        num_modules = int(
            getattr(
                model_cfg,
                "parent_residual_num_modules",
                128,
            )
        )
        if num_modules != self.prior_bank.num_modules:
            raise ValueError(
                "parent_residual_num_modules must match artifact module count "
                f"{self.prior_bank.num_modules}"
            )
        if (
            self.parent_module_decoder_enabled
            or self.child_consensus_bridge_enabled
        ):
            required = {
                "input_dim": (self.gene_dim, 2000),
                "parent_residual_num_modules": (num_modules, 128),
                "parent_residual_context_hidden_dim": (
                    context_cfg.hidden_dim,
                    128,
                ),
                "parent_residual_anchor_capacity": (
                    context_cfg.anchor_capacity,
                    128,
                ),
            }
            mismatches = [
                f"{name}={actual} (required {expected})"
                for name, (actual, expected) in required.items()
                if int(actual) != int(expected)
            ]
            if mismatches:
                raise ValueError(
                    "Child-aware Parent decoder fixed contract mismatch: "
                    + ", ".join(mismatches)
                )
        core_cfg = ParentResidualGeneDiTConfig(
            gene_dim=self.gene_dim,
            num_modules=num_modules,
            hidden_dim=int(
                getattr(model_cfg, "parent_residual_hidden_dim", 256)
            ),
            depth=int(getattr(model_cfg, "parent_residual_depth", 8)),
            num_heads=int(
                getattr(model_cfg, "parent_residual_num_heads", 8)
            ),
            context_dim=context_cfg.hidden_dim,
            condition_dim=int(
                getattr(
                    model_cfg,
                    "parent_residual_condition_dim",
                    getattr(covariate_config, "output_dim", self.gene_dim),
                )
            ),
            time_embedding_dim=int(
                getattr(model_cfg, "parent_residual_time_embedding_dim", 256)
            ),
            mlp_ratio=float(
                getattr(model_cfg, "parent_residual_mlp_ratio", 4.0)
            ),
            dropout=float(
                getattr(model_cfg, "parent_residual_transformer_dropout", 0.0)
            ),
            decoder_chunk_size=int(
                getattr(model_cfg, "parent_residual_decoder_chunk_size", 256)
            ),
            norm_eps=float(
                getattr(model_cfg, "parent_residual_norm_eps", 1e-6)
            ),
            strong_condition_memory_enabled=bool(
                getattr(
                    model_cfg,
                    "parent_residual_strong_condition_memory_enabled",
                    getattr(
                        model_cfg,
                        "parent_residual_strong_condition_memory",
                        False,
                    ),
                )
            ),
            strong_condition_num_heads=int(
                getattr(
                    model_cfg,
                    "parent_residual_strong_condition_num_heads",
                    getattr(model_cfg, "parent_residual_num_heads", 8),
                )
            ),
            strong_condition_child_chunk_size=int(
                getattr(
                    model_cfg,
                    "parent_residual_strong_condition_child_chunk_size",
                    32,
                )
            ),
            strong_condition_adaln_hidden_ratio=float(
                getattr(
                    model_cfg,
                    "parent_residual_strong_condition_adaln_hidden_ratio",
                    2.0,
                )
            ),
            source_reference_attention_enabled=bool(
                getattr(
                    model_cfg,
                    "parent_residual_source_reference_attention_enabled",
                    True,
                )
            ),
            source_reference_gate_limit=(
                None
                if getattr(
                    model_cfg,
                    "parent_residual_source_reference_gate_limit",
                    None,
                )
                is None
                else float(
                    getattr(
                        model_cfg,
                        "parent_residual_source_reference_gate_limit",
                    )
                )
            ),
            velocity_output_fp32=bool(
                getattr(
                    model_cfg,
                    "parent_residual_velocity_output_fp32",
                    False,
                )
            ),
            global_local_memory_split_enabled=bool(
                getattr(
                    model_cfg,
                    "parent_residual_global_local_memory_split_enabled",
                    False,
                )
            ),
            local_child_memory_last_n_layers=int(
                getattr(
                    model_cfg,
                    "parent_residual_local_child_memory_last_n_layers",
                    0,
                )
            ),
            response_only_last_n_layers=getattr(
                model_cfg,
                "parent_residual_response_only_last_n_layers",
                0,
            ),
        )
        gene_module_ids = self._gene_module_ids_from_config(
            model_cfg, self.prior_bank.module_ids
        )
        self.gene_partition_mode = str(
            getattr(model_cfg, "parent_residual_gene_partition_mode", "artifact")
        ).lower()
        self.gene_partition_seed = int(
            getattr(model_cfg, "parent_residual_gene_partition_seed", 42)
        )
        self.gene_dit = ParentResidualGeneDiT(gene_module_ids, core_cfg)
        self.strong_condition_memory_enabled = (
            self.gene_dit.strong_condition_memory_enabled
        )
        hidden = context_cfg.hidden_dim
        self.child_consensus_bridge = None
        self.tied_gene_parent_correction = None
        if self.parent_module_decoder_enabled:
            self.parent_correction_head = ChildAwareParentModuleDecoder(
                gene_module_ids,
                ChildAwareParentModuleDecoderConfig(
                    gene_dim=self.gene_dim,
                    num_modules=num_modules,
                    context_dim=context_cfg.hidden_dim,
                    hidden_dim=256,
                    num_latent_queries=4,
                    depth=2,
                    num_heads=8,
                    mlp_ratio=4.0,
                    dropout=float(
                        getattr(
                            model_cfg,
                            "parent_residual_transformer_dropout",
                            0.0,
                        )
                    ),
                    decoder_chunk_size=int(
                        getattr(
                            model_cfg,
                            "parent_residual_decoder_chunk_size",
                            256,
                        )
                    ),
                    norm_eps=float(
                        getattr(model_cfg, "parent_residual_norm_eps", 1.0e-6)
                    ),
                    router_log_probability_beta=(
                        self.parent_module_decoder_router_beta
                    ),
                ),
                detach_child_context=(
                    self.parent_module_decoder_detach_child_context
                ),
            )
        else:
            self.parent_correction_head = nn.Sequential(
                nn.Linear(4 * hidden, 2 * hidden),
                nn.SiLU(),
                nn.LayerNorm(2 * hidden),
                nn.Linear(2 * hidden, self.gene_dim),
            )
            nn.init.zeros_(self.parent_correction_head[-1].weight)
            nn.init.zeros_(self.parent_correction_head[-1].bias)
            if self.tied_gene_correction_enabled:
                # Preserve every historical RNG draw and the caller's data RNG
                # while giving the opt-in factorized exit deterministic weights.
                with torch.random.fork_rng(devices=[]):
                    self.tied_gene_parent_correction = (
                        TiedGeneParentCorrection(
                            hidden_dim=2 * hidden,
                            gene_identity_dim=core_cfg.hidden_dim,
                            rank=self.tied_gene_correction_rank,
                            detach_gene_identity=(
                                self.tied_gene_correction_detach_gene_identity
                            ),
                            norm_eps=core_cfg.norm_eps,
                        )
                    )
            if self.child_consensus_bridge_enabled:
                # The bridge is an additive opt-in arm.  Forking the CPU RNG
                # makes every historical module initialized after this point,
                # as well as the caller's data-order RNG stream, identical to
                # a same-seed B0 construction.
                with torch.random.fork_rng(devices=[]):
                    self.child_consensus_bridge = (
                        ChildAwareParentModuleDecoder(
                            gene_module_ids,
                            ChildAwareParentModuleDecoderConfig(
                                gene_dim=self.gene_dim,
                                num_modules=num_modules,
                                context_dim=context_cfg.hidden_dim,
                                hidden_dim=256,
                                num_latent_queries=4,
                                depth=2,
                                num_heads=8,
                                mlp_ratio=4.0,
                                # Full and collapsed counterfactuals share one
                                # decoder invocation contract.  Dropout would
                                # create a spurious contrast even for identical
                                # Child evidence, so this bridge is deterministic.
                                dropout=0.0,
                                decoder_chunk_size=int(
                                    getattr(
                                        model_cfg,
                                        "parent_residual_decoder_chunk_size",
                                        256,
                                    )
                                ),
                                norm_eps=float(
                                    getattr(
                                        model_cfg,
                                        "parent_residual_norm_eps",
                                        1.0e-6,
                                    )
                                ),
                                router_log_probability_beta=(
                                    self.child_consensus_bridge_router_beta
                                ),
                            ),
                            detach_child_context=(
                                self.child_consensus_bridge_detach_child_context
                            ),
                        )
                    )
                    # A shared additive output bias cancels identically in
                    # full-minus-collapsed contrast and can never learn.
                    self.child_consensus_bridge.gene_decoder.output_bias.requires_grad_(
                        False
                    )

        raw_guidance_enabled = getattr(
            model_cfg, "parent_residual_child_guidance_enabled", False
        )
        if not isinstance(raw_guidance_enabled, bool):
            raise ValueError(
                "parent_residual_child_guidance_enabled must be boolean"
            )
        self.child_guidance_enabled = raw_guidance_enabled
        child_guidance = self._child_guidance_config(model_cfg)
        self.child_guidance_mode = child_guidance["mode"]
        self.child_guidance_top_k = child_guidance["top_k"]
        if self.child_guidance_top_k > context_cfg.anchor_capacity:
            raise ValueError(
                "parent_residual_child_guidance_top_k cannot exceed "
                "parent_residual_anchor_capacity"
            )
        self.child_guidance_max_rms_ratio = child_guidance["max_rms_ratio"]
        self.child_guidance_detach_router_probabilities = child_guidance[
            "detach_router_probabilities"
        ]
        self.child_guidance_gradient_scale = child_guidance["gradient_scale"]
        self.child_guidance_start_step = child_guidance["start_step"]
        self.child_guidance_warmup_steps = child_guidance["warmup_steps"]
        self.child_guidance_max_coupling = child_guidance["max_coupling"]
        self.child_guidance_parallel_scale = child_guidance["parallel_scale"]
        self.child_guidance_parallel_max_projection = child_guidance[
            "parallel_max_projection"
        ]
        child_marginal = self._child_marginal_config(model_cfg)
        self.child_marginal_enabled = child_marginal["enabled"]
        self.child_marginal_alpha = child_marginal["alpha"]
        self.child_marginal_start_step = child_marginal["start_step"]
        self.child_marginal_warmup_steps = child_marginal["warmup_steps"]
        self.child_marginal_detach_router_probabilities = child_marginal[
            "detach_router_probabilities"
        ]
        if self.child_guidance_enabled and self.child_marginal_enabled:
            raise ValueError(
                "Child guidance and Child-marginal Parent coupling are "
                "mutually exclusive"
            )
        # Plain runtime state: checkpoint/state_dict compatibility is unaffected.
        self._child_guidance_step = 0
        self.child_guidance_gate = None
        if self.child_guidance_enabled and self.child_guidance_mode == "residual_gate":
            # Semantic Parent/perturbation features distinguish conditions;
            # eight bounded router statistics regularize the low-capacity gate.
            self.child_guidance_gate = ConditionReliabilityGate(
                2 * hidden + 8,
                hidden_dim=child_guidance["gate_hidden_dim"],
                max_alpha=child_guidance["max_alpha"],
            )

        # Absent on the default path: old state_dict and RNG stay unchanged.
        self.sparse_support_enabled = bool(
            getattr(model_cfg, "parent_residual_sparse_support_enabled", False)
        )
        self.gene_relation_enabled = bool(
            getattr(
                model_cfg,
                "parent_residual_gene_relation_enabled",
                False,
            )
        )
        raw_gene_relation_rank = getattr(
            model_cfg, "parent_residual_gene_relation_rank", 64
        )
        try:
            self.gene_relation_rank = int(raw_gene_relation_rank)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "parent_residual_gene_relation_rank must be a positive integer"
            ) from error
        if (
            isinstance(raw_gene_relation_rank, bool)
            or self.gene_relation_rank != raw_gene_relation_rank
            or self.gene_relation_rank < 1
        ):
            raise ValueError(
                "parent_residual_gene_relation_rank must be a positive integer"
            )
        if self.gene_relation_enabled and not self.sparse_support_enabled:
            raise ValueError(
                "parent_residual_gene_relation_enabled=true requires "
                "parent_residual_sparse_support_enabled=true"
            )
        self.sparse_joint_context_enabled = bool(
            getattr(
                model_cfg,
                "parent_residual_sparse_joint_context_enabled",
                False,
            )
        )
        self.sparse_hurdle_adapter = None
        if self.sparse_support_enabled:
            from src.models.sparse_hurdle_de import (
                SparseHurdleDEAdapter,
                SparseHurdleDEConfig,
            )

            self.sparse_hurdle_adapter = SparseHurdleDEAdapter(
                SparseHurdleDEConfig(
                    condition_dim=4 * hidden,
                    gene_dim=self.gene_dim,
                    hidden_dim=256,
                    dropout=0.0,
                    projection_alpha=0.0,
                    # Explicit gradient routing: head labels cannot rewrite
                    # Parent statistics.  The opt-in joint path additionally
                    # trains the target-free condition context.
                    detach_base_features=not self.sparse_joint_context_enabled,
                    ranking_false_positive_weight=4.0,
                    # R1 replaces the condition-only dense support exit with
                    # q(condition)^T k(response_gene). Keys consume the full
                    # learned Gene-DiT identity plus a low-rank residual.
                    gene_relation_enabled=self.gene_relation_enabled,
                    gene_relation_rank=self.gene_relation_rank,
                    gene_relation_identity_dim=core_cfg.hidden_dim,
                )
            )

        # Optional joint classification/regression path.  It deliberately
        # reuses the online S head: DE supervision and Parent regression then
        # update the same per-gene response probability.  No module or RNG is
        # added to the historical default-false path.
        self.soft_gated_regression_enabled = bool(
            getattr(
                model_cfg,
                "parent_residual_soft_gated_regression_enabled",
                False,
            )
        )
        self.soft_gate_blend = float(
            getattr(model_cfg, "parent_residual_soft_gate_blend", 0.0)
        )
        self.soft_gate_detach_regression = bool(
            getattr(
                model_cfg,
                "parent_residual_soft_gate_detach_regression",
                False,
            )
        )
        raw_soft_gate_warmup_steps = getattr(
            model_cfg,
            "parent_residual_soft_gate_warmup_steps",
            0,
        )
        try:
            self.soft_gate_warmup_steps = int(raw_soft_gate_warmup_steps)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "parent_residual_soft_gate_warmup_steps must be a "
                "non-negative integer"
            ) from error
        if (
            isinstance(raw_soft_gate_warmup_steps, bool)
            or self.soft_gate_warmup_steps != raw_soft_gate_warmup_steps
            or self.soft_gate_warmup_steps < 0
        ):
            raise ValueError(
                "parent_residual_soft_gate_warmup_steps must be a "
                "non-negative integer"
            )
        self.soft_gate_temperature = float(
            getattr(model_cfg, "parent_residual_soft_gate_temperature", 1.0)
        )
        self.soft_gate_slab_scale = float(
            getattr(model_cfg, "parent_residual_soft_gate_slab_scale", 1.0)
        )
        initial_probability = float(
            getattr(
                model_cfg,
                "parent_residual_soft_gate_initial_probability",
                0.95,
            )
        )
        for name, value in (
            ("parent_residual_soft_gate_blend", self.soft_gate_blend),
            (
                "parent_residual_soft_gate_temperature",
                self.soft_gate_temperature,
            ),
            (
                "parent_residual_soft_gate_slab_scale",
                self.soft_gate_slab_scale,
            ),
            (
                "parent_residual_soft_gate_initial_probability",
                initial_probability,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.soft_gate_blend <= 1.0:
            raise ValueError(
                "parent_residual_soft_gate_blend must lie in [0, 1]"
            )
        if self.soft_gate_temperature <= 0.0:
            raise ValueError(
                "parent_residual_soft_gate_temperature must be positive"
            )
        if self.soft_gate_slab_scale <= 0.0:
            raise ValueError(
                "parent_residual_soft_gate_slab_scale must be positive"
            )
        if not 0.0 < initial_probability < 1.0:
            raise ValueError(
                "parent_residual_soft_gate_initial_probability must lie in (0, 1)"
            )
        self.soft_gate_initial_log_odds = math.log(
            initial_probability
        ) - math.log1p(-initial_probability)
        if (
            self.soft_gated_regression_enabled
            and self.sparse_hurdle_adapter is None
        ):
            raise ValueError(
                "soft-gated Parent regression requires "
                "parent_residual_sparse_support_enabled=true so classification "
                "and regression share one S head"
            )

        self.correction_max_scale = float(
            getattr(model_cfg, "parent_residual_correction_max_scale", 0.05)
        )
        if (
            (
                self.parent_module_decoder_enabled
                or self.child_consensus_bridge_enabled
            )
            and self.correction_max_scale != 0.05
        ):
            raise ValueError(
                "Child-aware Parent decoder requires "
                "parent_residual_correction_max_scale=0.05"
            )
        self.child_beta = float(
            getattr(
                model_cfg,
                "parent_residual_beta",
                getattr(model_cfg, "parent_residual_child_beta", 0.0),
            )
        )
        self.child_std_beta = float(
            getattr(
                model_cfg,
                "parent_residual_beta_std",
                getattr(model_cfg, "parent_residual_child_std_beta", 0.0),
            )
        )
        self.prior_stochastic = bool(
            getattr(model_cfg, "parent_residual_prior_stochastic", True)
        )
        self.inference_child_sampling_mode = str(
            getattr(
                model_cfg,
                "parent_residual_inference_child_sampling_mode",
                "legacy",
            )
        ).lower()
        if self.inference_child_sampling_mode not in {"legacy", "quota", "stratified"}:
            raise ValueError("inference Child sampling mode must be legacy, quota, or stratified")
        named_scales = {
            "parent_residual_correction_max_scale": self.correction_max_scale,
            "parent_residual_beta": self.child_beta,
            "parent_residual_beta_std": self.child_std_beta,
        }
        for name, value in named_scales.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.real_control_source_enabled = bool(
            getattr(
                model_cfg,
                "parent_residual_real_control_source_enabled",
                False,
            )
        )
        self.real_control_sampling_mode = (
            self._real_control_sampling_mode_from_config(model_cfg)
        )
        if (
            self.strict_parent_only_enabled
            and self.real_control_source_enabled
            and self.real_control_sampling_mode != "global_marginal"
        ):
            raise ValueError(
                "strict Parent-only empirical source requires explicitly configured "
                "parent_residual_real_control_sampling_mode=global_marginal"
            )
        if (
            self.real_control_sampling_mode != "child"
            and not self.real_control_source_enabled
        ):
            raise ValueError(
                "global_marginal real-control sampling requires "
                "parent_residual_real_control_source_enabled=true"
            )
        if (
            self.child_context_mode == "global_replicated"
            and self.real_control_source_enabled
            and self.real_control_sampling_mode != "global_marginal"
        ):
            raise ValueError(
                "global_replicated no-Child context with an empirical "
                "real-control source requires global_marginal sampling; "
                "per-Child sampling would leak Child identity"
            )
        self.real_control_residual_scale = float(
            getattr(
                model_cfg,
                "parent_residual_real_control_residual_scale",
                1.0,
            )
        )
        if (
            not math.isfinite(self.real_control_residual_scale)
            or self.real_control_residual_scale < 0.0
        ):
            raise ValueError(
                "parent_residual_real_control_residual_scale must be finite "
                "and non-negative"
            )
        self.real_control_stochastic = bool(
            getattr(
                model_cfg,
                "parent_residual_real_control_stochastic",
                True,
            )
        )
        self.real_control_reservoir = None
        if self.real_control_source_enabled:
            if self.child_beta != 0.0 or self.child_std_beta != 0.0:
                raise ValueError(
                    "real-control residual source replaces centroid/Gaussian "
                    "noise; set parent_residual_beta and "
                    "parent_residual_beta_std to zero"
                )
            if not (
                bool(
                    getattr(
                        model_cfg,
                        "parent_residual_locked_flow_enabled",
                        False,
                    )
                )
                or self.strict_direct_flow_enabled
            ):
                raise ValueError(
                    "real-control residual source requires "
                    "locked flow or strict direct flow"
                )
            if (
                not self.strict_parent_only_enabled
                and self.matching_guidance_enabled
                and not bool(
                    getattr(model_cfg, "parent_residual_matched_set_enabled", False)
                )
            ):
                raise ValueError(
                    "real-control training requires Hungarian matched-set routing"
                )
            posterior_max = max(
                float(getattr(model_cfg, "parent_residual_posterior_start_fraction", 1.0)),
                float(getattr(model_cfg, "parent_residual_posterior_end_fraction", 0.0)),
            )
            child_mean_target_free = bool(
                getattr(
                    model_cfg,
                    "parent_residual_child_mean_flow_enabled",
                    False,
                )
            )
            positive_parallel_target_free = bool(
                getattr(
                    model_cfg,
                    "parent_residual_child_guidance_enabled",
                    False,
                )
            ) and str(
                getattr(
                    model_cfg,
                    "parent_residual_child_guidance_mode",
                    "residual_gate",
                )
            ).strip().lower() == "positive_parallel"
            child_marginal_target_free = bool(
                getattr(model_cfg, "parent_residual_child_marginal_enabled", False)
            )
            if (
                not self.strict_parent_only_enabled
                and self.matching_guidance_enabled
                and posterior_max <= 0.0
                and not child_mean_target_free
                and not positive_parallel_target_free
                and not child_marginal_target_free
            ):
                raise ValueError("real-control training needs a nonzero Hungarian phase")
            reservoir_path = getattr(
                model_cfg,
                "parent_residual_real_control_reservoir_path",
                None,
            )
            if reservoir_path in (None, "", "null"):
                raise ValueError(
                    "parent_residual_real_control_reservoir_path is required "
                    "when the real-control source is enabled"
                )
            from src.models.real_control_residual_source import (
                RealControlResidualReservoir,
            )

            self.real_control_reservoir = RealControlResidualReservoir(
                reservoir_path,
                covariate_config,
                gene_dim=self.gene_dim,
                child_capacity=context_cfg.anchor_capacity,
                sampling_mode=self.real_control_sampling_mode,
                expected_sha256=getattr(
                    model_cfg,
                    "parent_residual_real_control_reservoir_sha256",
                    None,
                ),
                expected_normalization_divisor=float(
                    getattr(
                        model_cfg,
                        "parent_residual_normalization_divisor",
                        10.0,
                    )
                ),
                validate_runtime_prototypes=bool(
                    getattr(
                        model_cfg,
                        "parent_residual_real_control_validate_prototypes",
                        True,
                    )
                ),
                control_grouping=str(
                    getattr(
                        model_cfg,
                        "parent_residual_real_control_grouping",
                        "celltype",
                    )
                ),
                recenter_to_runtime_control_mean=bool(
                    getattr(model_cfg, "parent_residual_real_control_recenter", False)
                ),
                prototype_atol=float(
                    getattr(
                        model_cfg,
                        "parent_residual_real_control_prototype_atol",
                        2e-5,
                    )
                ),
                prototype_rtol=float(
                    getattr(
                        model_cfg,
                        "parent_residual_real_control_prototype_rtol",
                        2e-4,
                    )
                ),
            )

    @staticmethod
    def _padded_child_stds(context, child_stds: torch.Tensor) -> torch.Tensor:
        """Align raw ``[U,K,G]`` stds with the padded context bank."""
        if not torch.is_tensor(child_stds):
            raise TypeError("child_stds must be a torch.Tensor")
        if not child_stds.is_floating_point():
            raise TypeError("child_stds must be floating point")
        if child_stds.ndim != 3:
            raise ValueError("child_stds must have shape [U,K,G]")
        prototypes = context.control_prototypes
        unique_banks, capacity, genes = prototypes.shape
        if child_stds.shape[0] != unique_banks or child_stds.shape[2] != genes:
            raise ValueError("child_stds must match context U and G dimensions")
        if child_stds.shape[1] > capacity:
            raise ValueError("child_stds K exceeds the padded anchor capacity")
        if not torch.isfinite(child_stds).all():
            raise ValueError("child_stds must be finite")
        if (child_stds < 0).any():
            raise ValueError("child_stds cannot be negative")
        child_stds = child_stds.to(prototypes)
        if child_stds.shape[1] == capacity:
            return child_stds
        padded = prototypes.new_zeros(prototypes.shape)
        padded[:, : child_stds.shape[1]] = child_stds
        return padded

    @staticmethod
    def _validate_selected_indices(context, selected_indices: torch.Tensor) -> None:
        if not torch.is_tensor(selected_indices):
            raise TypeError("selected_indices must be a torch.Tensor")
        if selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must have dtype torch.long")
        if selected_indices.ndim != 2:
            raise ValueError("selected_indices must have shape [B,S]")
        if selected_indices.shape[0] != context.parent_token.shape[0]:
            raise ValueError("selected_indices B must match the context batch")
        if selected_indices.shape[1] < 1:
            raise ValueError("selected_indices must contain at least one cell")
        if selected_indices.device != context.parent_token.device:
            raise ValueError("selected_indices must share the context device")
        capacity = context.child_tokens.shape[1]
        if ((selected_indices < 0) | (selected_indices >= capacity)).any():
            raise ValueError("selected_indices contains an out-of-range Child")
        if not context.anchor_mask.gather(1, selected_indices).all():
            raise ValueError("selected_indices selects a padded Child")

    def initialize_weights(self):
        """Restore only the explicit safe cold-start boundaries."""

        self.gene_dit.reset_output_projection()
        if self.parent_module_decoder_enabled:
            self.parent_correction_head.reset_output_projection()
        else:
            nn.init.zeros_(self.parent_correction_head[-1].weight)
            nn.init.zeros_(self.parent_correction_head[-1].bias)
        if self.tied_gene_parent_correction is not None:
            self.tied_gene_parent_correction.reset_output_projection()
        if self.child_consensus_bridge is not None:
            self.child_consensus_bridge.reset_output_projection()
        self.context_encoder.reset_cold_start_parameters()
        self.gene_dit.reset_strong_condition_residual()
        if self.child_guidance_gate is not None:
            nn.init.zeros_(
                self.child_guidance_gate.reliability_projection.weight
            )
            nn.init.zeros_(
                self.child_guidance_gate.reliability_projection.bias
            )

    def set_child_guidance_step(self, step: int) -> None:
        """Set the optimizer step used by the deterministic coupling schedule."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Child-guidance step must be a non-negative integer")
        self._child_guidance_step = int(step)

    def positive_multiplicity(self, epoch: int, step: int | None = None) -> int:
        return self.cell_detr_cfg.positive_multiplicity(epoch=epoch, step=step)

    def build_context(
        self,
        control_sets,
        anchor_mask,
        occupancy,
        semantic_condition,
        token_mask=None,
        bank_inverse=None,
    ):
        if getattr(self, "strict_parent_only_enabled", False):
            raise RuntimeError(
                "strict Parent-only mode forbids CellDETRContextEncoder.forward; "
                "use prepare_strict_parent_only"
            )
        if self.child_context_mode == "global_replicated":
            control_sets, occupancy = self._global_replicated_child_context(
                control_sets=control_sets,
                anchor_mask=anchor_mask,
                occupancy=occupancy,
                token_mask=token_mask,
            )
        return self.context_encoder(
            control_sets=control_sets,
            anchor_mask=anchor_mask,
            occupancy=occupancy,
            semantic_condition=semantic_condition,
            token_mask=token_mask,
            bank_inverse=bank_inverse,
        )

    @staticmethod
    def _global_replicated_child_context(
        *, control_sets, anchor_mask, occupancy, token_mask=None
    ):
        """Remove Child content while preserving the context topology.

        Each active Child receives the occupancy-weighted global control mean.
        K, S, anchor/token masks, and padded entries stay unchanged; occupancy
        becomes uniform across active Children. This creates a clean no-Child
        training ablation without introducing a one-token topology shift.
        """
        if control_sets.ndim != 4:
            raise ValueError("control_sets must have shape [U,K,S,G]")
        bank_count, child_count, set_size, _ = control_sets.shape
        anchor_mask_local = anchor_mask.to(
            device=control_sets.device, dtype=torch.bool
        )
        if anchor_mask_local.shape != (bank_count, child_count):
            raise ValueError("anchor_mask must have shape [U,K]")
        if not anchor_mask_local.any(dim=-1).all():
            raise ValueError("every cell must have at least one active Child")

        if token_mask is None:
            token_mask_local = anchor_mask_local.unsqueeze(-1).expand(
                -1, -1, set_size
            )
        else:
            token_mask_local = token_mask.to(
                device=control_sets.device, dtype=torch.bool
            )
            if token_mask_local.shape != (bank_count, child_count, set_size):
                raise ValueError("token_mask must have shape [U,K,S]")
            token_mask_local = token_mask_local & anchor_mask_local.unsqueeze(-1)

        occupancy_local = occupancy.to(
            device=control_sets.device, dtype=control_sets.dtype
        )
        if occupancy_local.shape != (bank_count, child_count):
            raise ValueError("occupancy must have shape [U,K]")
        occupancy_local = torch.where(
            anchor_mask_local, occupancy_local.clamp_min(0.0), 0.0
        )
        occupancy_local = occupancy_local / occupancy_local.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        token_weights = token_mask_local.to(control_sets.dtype)
        child_means = (
            control_sets * token_weights.unsqueeze(-1)
        ).sum(dim=2) / token_weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        global_mean = (
            child_means * occupancy_local.unsqueeze(-1)
        ).sum(dim=1)
        replicated = torch.where(
            token_mask_local.unsqueeze(-1),
            global_mean[:, None, None, :],
            control_sets,
        )

        active_count = anchor_mask_local.sum(dim=-1, keepdim=True).clamp_min(1)
        uniform_occupancy = anchor_mask_local.to(control_sets.dtype) / active_count
        return replicated, uniform_occupancy

    def conditioning_tensors(self, context, prepared):
        """Return the single conditioning contract used by train and sample.

        Legacy selected/routed tensors remain present for old checkpoints and
        the default four-token core.  The target-free full bank is added only
        when StrongConditionMemory is enabled.
        """

        if self.strict_parent_only_enabled:
            if not isinstance(context, StrictParentOnlyContext):
                raise TypeError("strict mode requires StrictParentOnlyContext")
            return {
                "parent_residual_parent_token": context.parent_control_feature,
            }
        tensors = {
            "parent_residual_parent_token": context.parent_token,
            "parent_residual_child_tokens": prepared.selected_child_tokens,
            "parent_residual_routed_token": context.routed_token,
        }
        if self.strong_condition_memory_enabled:
            tensors.update(
                {
                    "parent_residual_all_child_tokens": context.child_tokens,
                    "parent_residual_child_mask": context.anchor_mask,
                    "parent_residual_perturbation_query": (
                        context.perturbation_query
                    ),
                }
            )
        return tensors

    def prepare_training_context(
        self,
        control_sets,
        anchor_mask,
        occupancy,
        targets,
        semantic_condition,
        positive_multiplicity=1,
        supervision_mask=None,
        token_mask=None,
        bank_inverse=None,
        matching_group_ids=None,
        epoch=0,
    ):
        if not self.matching_guidance_enabled:
            raise RuntimeError(
                "prepare_training_context cannot be used when matching guidance is "
                "disabled; use prepare_unmatched_training_context"
            )
        context = self.build_context(
            control_sets,
            anchor_mask,
            occupancy,
            semantic_condition,
            token_mask=token_mask,
            bank_inverse=bank_inverse,
        )
        target_features = self.context_encoder.encode_expression(targets)
        settings = (
            self.cell_detr_cfg.cold_start_settings_for_epoch(epoch)
            if self.cell_detr_cfg.cold_start_gate_enabled
            else {}
        )
        match = self.matcher(
            context=context,
            targets=targets,
            target_features=target_features,
            decode_selected_delta=self.context_encoder.decode_selected_delta,
            positive_multiplicity=int(positive_multiplicity),
            supervision_mask=supervision_mask,
            decode_parent_delta=(
                self.context_encoder.decode_parent_delta
                if self.cell_detr_cfg.cold_start_gate_enabled
                else None
            ),
            gate_enabled=self.cell_detr_cfg.cold_start_gate_enabled,
            semi_balanced_ot_enabled=self.semi_balanced_ot_enabled,
            semi_balanced_ot_group_ids=matching_group_ids,
            semi_balanced_ot_epsilon=self.semi_balanced_ot_epsilon,
            semi_balanced_ot_rho=self.semi_balanced_ot_rho,
            semi_balanced_ot_iterations=self.semi_balanced_ot_iterations,
            semi_balanced_ot_normalize_cost_by_median=(
                self.semi_balanced_ot_normalize_cost_by_median
            ),
            balanced_ot_enabled=self.balanced_ot_enabled,
            balanced_ot_column_tolerance=self.balanced_ot_column_tolerance,
            positive_multiplicity_cap_to_active_children=(
                self.cell_detr_cfg.positive_multiplicity_cap_to_active_children
            ),
            **settings,
        )
        return context, match

    def prepare_unmatched_training_context(
        self,
        control_sets,
        anchor_mask,
        occupancy,
        semantic_condition,
        token_mask=None,
        bank_inverse=None,
    ):
        """Build target-free context without encoding targets or invoking matcher."""
        if self.matching_guidance_enabled:
            raise RuntimeError(
                "prepare_unmatched_training_context is only valid when matching "
                "guidance is disabled"
            )
        return self.build_context(
            control_sets,
            anchor_mask,
            occupancy,
            semantic_condition,
            token_mask=token_mask,
            bank_inverse=bank_inverse,
        )

    @staticmethod
    def _routing_probabilities(context, temperature, occupancy_weight):
        logits = context.router_logits.float() / float(temperature)
        if float(occupancy_weight):
            logits = logits + float(occupancy_weight) * context.occupancy.float().clamp_min(
                1e-8
            ).log()
        logits = logits.masked_fill(~context.anchor_mask, -torch.inf)
        return torch.softmax(logits, dim=-1)

    def inference_probabilities(self, context):
        return self._routing_probabilities(
            context,
            self.cell_detr_cfg.inference_temperature,
            self.cell_detr_cfg.inference_occupancy_logit_weight,
        )

    def source_selection_probabilities(self, context):
        """Return the target-free Child distribution used only for x0."""

        if self.source_selection_mode == "router":
            return self.inference_probabilities(context)
        occupancy = context.occupancy.float().masked_fill(
            ~context.anchor_mask, 0.0
        )
        return occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _apply_child_marginal(self, context, parent_mean):
        """Refine P0 with the full target-free Child endpoint marginal."""

        probabilities = self.inference_probabilities(context)
        routing_weights = (
            probabilities.detach()
            if self.child_marginal_detach_router_probabilities
            else probabilities
        )
        batch_size = parent_mean.shape[0]
        control_prototypes = context.control_prototypes
        bank_inverse = context.bank_inverse
        if bank_inverse is None:
            if control_prototypes.shape[0] != batch_size:
                raise ValueError(
                    "Child marginal requires bank_inverse when U differs from B"
                )
            bank_inverse = torch.arange(
                batch_size,
                device=control_prototypes.device,
                dtype=torch.long,
            )
        else:
            bank_inverse = bank_inverse.to(
                device=control_prototypes.device,
                dtype=torch.long,
            )
            if bank_inverse.shape != (batch_size,):
                raise ValueError("context.bank_inverse must have shape [B]")
            if (
                (bank_inverse < 0)
                | (bank_inverse >= control_prototypes.shape[0])
            ).any():
                raise ValueError("context.bank_inverse contains an invalid bank")

        # Reuse the grouped bank matmul: never materialize [B,K,G] controls.
        child_control_mean = _banked_center(
            control_prototypes,
            routing_weights.to(control_prototypes),
            bank_inverse,
        )
        child_delta = self.context_encoder.decode_weighted_delta(
            context,
            routing_weights,
        ).to(parent_mean)
        child_marginal_mean = child_control_mean.to(parent_mean) + child_delta

        coupling = coupling_at_step(
            self._child_guidance_step,
            start_step=self.child_marginal_start_step,
            warmup_steps=self.child_marginal_warmup_steps,
            max_coupling=1.0,
        )
        alpha = parent_mean.new_full(
            (batch_size, 1),
            self.child_marginal_alpha * coupling,
        )
        # Forward value is the convex blend (1-a)*P0 + a*M. Detaching only P0
        # inside the residual keeps the established Parent branch's gradient at
        # exactly one while Child/Router learn from the endpoint objectives.
        residual = child_marginal_mean - parent_mean.detach()
        contribution = alpha * residual
        final_parent = parent_mean + contribution

        probability = probabilities.float().masked_fill(
            ~context.anchor_mask, 0.0
        )
        router_entropy = -(
            probability * probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return (
            final_parent,
            child_marginal_mean,
            child_control_mean.to(parent_mean),
            child_delta,
            residual,
            contribution,
            alpha,
            router_entropy,
            router_entropy.exp(),
        )

    @staticmethod
    def _child_guidance_router_statistics(
        probabilities, occupancy, anchor_mask, router_logits
    ):
        """Return the eight audited train-cache Router state features."""

        if probabilities.shape != anchor_mask.shape or anchor_mask.dtype != torch.bool:
            raise ValueError(
                "probabilities/anchor_mask must share [B,K], with bool mask"
            )
        mask = anchor_mask.to(device=probabilities.device)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every guidance row must have an active Child")
        if not bool(torch.isfinite(probabilities).all()) or bool(
            (probabilities < 0).any()
        ):
            raise ValueError(
                "routing probabilities must be finite and non-negative"
            )
        inactive = probabilities.masked_select(~mask)
        if inactive.numel() and bool((inactive > 1.0e-7).any()):
            raise ValueError(
                "routing probabilities place mass on padded Children"
            )
        probability = probabilities.float()
        probability = probability.masked_fill(~mask, 0.0)
        entropy = -(
            probability * probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        occupancy = occupancy.float().to(device=probability.device)
        occupancy = occupancy.masked_fill(~mask, 0.0)
        occupancy = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        occupancy_entropy = -(
            occupancy * occupancy.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        logits = router_logits.float().to(device=probability.device)
        if logits.shape != mask.shape:
            raise ValueError("router_logits and anchor_mask must share shape [B,K]")
        if not bool(torch.isfinite(logits.masked_select(mask)).all()):
            raise ValueError("active router logits must be finite")
        safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
        active_count = mask.sum(dim=-1).clamp_min(1).float()
        logit_mean = safe_logits.sum(dim=-1) / active_count
        logit_variance = torch.where(
            mask,
            (logits - logit_mean[:, None]).square(),
            torch.zeros_like(safe_logits),
        ).sum(dim=-1) / active_count
        top5_mass = probability.topk(
            k=min(5, probability.shape[1]), dim=-1
        ).values.sum(dim=-1)
        return torch.stack(
            (
                entropy,
                entropy.exp(),
                probability.max(dim=-1).values,
                top5_mass,
                0.5 * (probability - occupancy).abs().sum(dim=-1),
                occupancy_entropy.exp(),
                logit_mean,
                logit_variance.sqrt(),
            ),
            dim=-1,
        )

    def _apply_child_guidance(self, context, parent_mean, control_mean):
        """Compose the enabled Child proposal after the historical Parent P0."""

        probabilities = self.inference_probabilities(context)
        routing_for_parent = (
            probabilities.detach()
            if self.child_guidance_detach_router_probabilities
            else probabilities
        )
        selected_probabilities, selected_indices = torch.topk(
            routing_for_parent,
            k=self.child_guidance_top_k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        topk_mass = selected_probabilities.sum(dim=-1, keepdim=True)

        control_prototypes = context.control_prototypes
        batch_size = parent_mean.shape[0]
        bank_inverse = context.bank_inverse
        if bank_inverse is None:
            if control_prototypes.shape[0] != batch_size:
                raise ValueError(
                    "Child guidance requires bank_inverse when U differs from B"
                )
            bank_inverse = torch.arange(
                batch_size,
                device=control_prototypes.device,
                dtype=torch.long,
            )
        else:
            bank_inverse = bank_inverse.to(
                device=control_prototypes.device, dtype=torch.long
            )
            if bank_inverse.shape != (batch_size,):
                raise ValueError("context.bank_inverse must have shape [B]")
            if (
                (bank_inverse < 0)
                | (bank_inverse >= control_prototypes.shape[0])
            ).any():
                raise ValueError("context.bank_inverse contains an invalid bank")

        if self.child_guidance_mode == "occupancy_contrast":
            # Parent owns the common perturbation displacement. Child may only
            # contribute the signed change induced by moving mixture mass away
            # from the original control occupancy. Consequently p == a is an
            # exact zero correction and any Child effect shared by all states
            # cancels instead of being counted a second time by Parent.
            occupancy = context.occupancy.to(
                device=routing_for_parent.device,
                dtype=routing_for_parent.dtype,
            )
            occupancy = occupancy.masked_fill(~context.anchor_mask, 0.0)
            occupancy = occupancy / occupancy.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            routed_control = _banked_center(
                control_prototypes,
                routing_for_parent.to(control_prototypes),
                bank_inverse,
            )
            occupancy_control = _banked_center(
                control_prototypes,
                occupancy.to(control_prototypes),
                bank_inverse,
            )
            routed_delta = self.context_encoder.decode_weighted_delta(
                context, routing_for_parent
            )
            occupancy_delta = self.context_encoder.decode_weighted_delta(
                context, occupancy
            )
            proposal = (
                routed_control.to(parent_mean)
                + routed_delta.to(parent_mean)
                - occupancy_control.to(parent_mean)
                - occupancy_delta.to(parent_mean)
            )
            residual = rms_trust_clip(
                proposal,
                parent_mean - control_mean,
                self.child_guidance_max_rms_ratio,
            )
            residual = scale_gradient(
                residual, self.child_guidance_gradient_scale
            )
            coupling = coupling_at_step(
                self._child_guidance_step,
                start_step=self.child_guidance_start_step,
                warmup_steps=self.child_guidance_warmup_steps,
                max_coupling=self.child_guidance_max_coupling,
            )
            return (
                parent_mean + coupling * residual,
                proposal,
                residual,
                parent_mean.new_ones((batch_size, 1)),
                coupling,
                parent_mean.new_ones((batch_size, 1)),
            )

        row_controls = control_prototypes.index_select(0, bank_inverse)
        selected_controls = row_controls.gather(
            1,
            selected_indices.unsqueeze(-1).expand(
                -1, -1, row_controls.shape[-1]
            ),
        )
        selected_deltas = self.context_encoder.decode_selected_delta(
            context, selected_indices
        )
        child_endpoints = (
            selected_controls.to(parent_mean)
            + selected_deltas.to(parent_mean)
        )
        proposal = parent_fill_residual_proposal(
            parent_mean,
            child_endpoints,
            selected_probabilities.to(parent_mean),
            top_k=None,
            # Probabilities were detached above when configured; avoid a
            # redundant detach so the opt-in coupled path remains trainable.
            detach_router_probabilities=False,
        )
        coupling = coupling_at_step(
            self._child_guidance_step,
            start_step=self.child_guidance_start_step,
            warmup_steps=self.child_guidance_warmup_steps,
            max_coupling=self.child_guidance_max_coupling,
        )
        if self.child_guidance_mode == "positive_parallel":
            # Child matching supplies only detached, target-free evidence about
            # Parent amplitude. Parent objectives retain a gradient through
            # P0/d, but cannot rewrite Router, Child decoder, or shared context
            # through this coupling. Negative and orthogonal Child components
            # remain exclusively in the original heterogeneity/matching path.
            # Match the audited proposal construction used by the fixed F05
            # validation: Top-K mass is not renormalized, then q is bounded by
            # a Parent-effect-relative RMS trust region before projection.
            trusted_proposal = rms_trust_clip(
                proposal,
                parent_mean - control_mean,
                self.child_guidance_max_rms_ratio,
            )
            parallel = positive_parallel_child_guidance(
                parent_mean,
                control_mean,
                trusted_proposal,
                gamma=self.child_guidance_parallel_scale,
                projection_cap=self.child_guidance_parallel_max_projection,
                coupling=coupling,
            )
            return (
                parallel.endpoint,
                trusted_proposal,
                parallel.contribution,
                parallel.positive_projection,
                coupling,
                topk_mass,
            )

        residual = rms_trust_clip(
            proposal,
            parent_mean - control_mean,
            self.child_guidance_max_rms_ratio,
        )
        residual = scale_gradient(
            residual, self.child_guidance_gradient_scale
        )

        gate_probabilities = routing_for_parent
        statistics = self._child_guidance_router_statistics(
            gate_probabilities,
            context.occupancy,
            context.anchor_mask,
            context.router_logits,
        )
        gate_features = torch.cat(
            [
                context.perturbation_query.float(),
                context.parent_token.float(),
                statistics,
            ],
            dim=-1,
        ).detach()
        alpha = self.child_guidance_gate(gate_features).to(parent_mean)
        final_parent = parent_mean + coupling * alpha * residual
        return final_parent, proposal, residual, alpha, coupling, topk_mass

    def route_prior(
        self,
        context,
        num_cells: int,
        stochastic: Optional[bool] = None,
        sampling_mode: str = "legacy",
        routing_group_ids: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        num_cells = int(num_cells)
        if num_cells < 1:
            raise ValueError("num_cells must be positive")
        probabilities = self.source_selection_probabilities(context)
        stochastic = self.prior_stochastic if stochastic is None else bool(stochastic)
        from src.models.matched_set_transport import sample_grouped_child_indices

        sampling_mode = str(sampling_mode).lower()
        indices = sample_grouped_child_indices(
            probabilities,
            num_samples=num_cells,
            mode=sampling_mode,
            group_ids=routing_group_ids,
            stochastic=stochastic,
            generator=generator,
        )
        rows = torch.arange(indices.shape[0], device=indices.device)[:, None]
        return {
            "prior": probabilities,
            "child_indices": indices,
            "selected_child_tokens": context.child_tokens[rows, indices],
        }

    def fewshot_training_forbidden_condition_mask(
        self, cov_celltype, cov_pert, cov_batch=None
    ):
        """Map runtime IDs to audit/deployment rows forbidden in training."""

        runtime_line = self.prior_bank.runtime_parent_ids(
            cov_celltype, cov_batch
        )
        runtime_perturbation = self.prior_bank._condition_ids(
            cov_pert, "cov_pert"
        )
        if runtime_line.shape != runtime_perturbation.shape:
            raise ValueError("cov_celltype and cov_pert must share batch shape")
        if self.fewshot_parent_adapter is None:
            return torch.zeros_like(runtime_line, dtype=torch.bool)
        if (
            (runtime_line < 0)
            | (
                runtime_line
                >= self.prior_bank.runtime_cellline_to_artifact.numel()
            )
        ).any():
            raise ValueError("cov_celltype contains an out-of-range runtime ID")
        if (
            (runtime_perturbation < 0)
            | (
                runtime_perturbation
                >= self.prior_bank.runtime_perturbation_to_artifact.numel()
            )
        ).any():
            raise ValueError("cov_pert contains an out-of-range runtime ID")
        line = self.prior_bank.runtime_cellline_to_artifact.index_select(
            0, runtime_line
        )
        perturbation = (
            self.prior_bank.runtime_perturbation_to_artifact.index_select(
                0, runtime_perturbation
            )
        )
        return self.fewshot_parent_adapter.training_forbidden(
            line, perturbation
        )

    @staticmethod
    def _collapsed_child_consensus_inputs(
        child_tokens: torch.Tensor,
        child_mask: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse valid Children to their occupancy-weighted barycenter.

        The valid mask is intentionally not changed.  Both occupancy and the
        Router teacher are replaced by uniform mass over the same valid slots,
        leaving a counterfactual that contains Parent-level Child consensus but
        no within-condition Child heterogeneity.
        """

        if (
            not torch.is_tensor(child_tokens)
            or not child_tokens.is_floating_point()
            or child_tokens.ndim != 3
        ):
            raise TypeError("child_tokens must be floating point with shape [B,K,H]")
        batch_size, child_count, _ = child_tokens.shape
        if not torch.is_tensor(child_mask):
            raise TypeError("child_mask must be a torch.Tensor")
        if child_mask.dtype != torch.bool or tuple(child_mask.shape) != (
            batch_size,
            child_count,
        ):
            raise ValueError("child_mask must be boolean with shape [B,K]")
        if (
            not torch.is_tensor(occupancy)
            or not occupancy.is_floating_point()
            or tuple(occupancy.shape) != (batch_size, child_count)
        ):
            raise TypeError("occupancy must be floating point with shape [B,K]")
        if (
            child_mask.device != child_tokens.device
            or occupancy.device != child_tokens.device
        ):
            raise ValueError("collapsed Child inputs must share one device")
        if not child_mask.any(dim=1).all():
            raise ValueError("every row must contain at least one valid Child")
        if not torch.isfinite(child_tokens).all():
            raise ValueError("child_tokens must be finite")
        if not torch.isfinite(occupancy).all() or (occupancy < 0.0).any():
            raise ValueError("occupancy must be finite and non-negative")
        if (occupancy.masked_select(~child_mask) > 1.0e-7).any():
            raise ValueError("occupancy cannot place mass on a padded Child")

        accumulation_dtype = (
            torch.float64
            if child_tokens.dtype == torch.float64
            else torch.float32
        )
        occupancy_work = occupancy.to(accumulation_dtype).masked_fill(
            ~child_mask, 0.0
        )
        occupancy_mass = occupancy_work.sum(dim=1, keepdim=True)
        if (occupancy_mass <= 0.0).any():
            raise ValueError("each row needs positive valid occupancy mass")
        barycenter = (
            child_tokens.to(accumulation_dtype)
            * (occupancy_work / occupancy_mass).unsqueeze(-1)
        ).sum(dim=1)
        collapsed_tokens = torch.where(
            child_mask.unsqueeze(-1),
            barycenter.unsqueeze(1),
            torch.zeros_like(child_tokens, dtype=accumulation_dtype),
        )
        valid_count = child_mask.sum(dim=1, keepdim=True).float()
        uniform = (child_mask.float() / valid_count).to(occupancy)
        # Occupancy and inference-Router mass are distinct inputs but are both
        # uniform in the collapsed counterfactual by construction.
        return collapsed_tokens, uniform, uniform.clone()

    @staticmethod
    def _convex_child_consensus_correction(
        legacy_correction: torch.Tensor,
        candidate_child_delta: torch.Tensor,
        *,
        final_raw: torch.Tensor,
        max_delta_to_legacy_ratio: float,
        bootstrap_reference_rms: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply tanh then cap the realized Child delta per condition.

        The cap is a convex interpolation between the legacy correction and
        the full-contrast candidate.  Consequently the final correction stays
        inside the historical tanh bound without introducing a learned or
        separately zero-initialized gate.
        """

        if (
            legacy_correction.shape != candidate_child_delta.shape
            or legacy_correction.shape != final_raw.shape
            or legacy_correction.ndim != 2
        ):
            raise ValueError(
                "legacy_correction, candidate_child_delta and final_raw must "
                "share shape [B,G]"
            )
        if (
            not legacy_correction.is_floating_point()
            or not candidate_child_delta.is_floating_point()
            or not final_raw.is_floating_point()
        ):
            raise TypeError("bridge correction inputs must be floating point")
        if (
            legacy_correction.device != candidate_child_delta.device
            or legacy_correction.dtype != candidate_child_delta.dtype
        ):
            raise ValueError("legacy and candidate corrections must share device/dtype")
        if final_raw.device != legacy_correction.device:
            raise ValueError("final_raw must share the correction device")
        if not all(
            torch.isfinite(value).all()
            for value in (
                legacy_correction,
                candidate_child_delta,
                final_raw,
            )
        ):
            raise ValueError("bridge correction inputs must be finite")
        if (
            not math.isfinite(max_delta_to_legacy_ratio)
            or not 0.0 <= max_delta_to_legacy_ratio <= 0.25
        ):
            raise ValueError("max_delta_to_legacy_ratio must lie in [0, 0.25]")
        if (
            not math.isfinite(bootstrap_reference_rms)
            or bootstrap_reference_rms <= 0.0
        ):
            raise ValueError("bootstrap_reference_rms must be finite and positive")

        accumulation_dtype = (
            torch.float32
            if legacy_correction.dtype in (torch.float16, torch.bfloat16)
            else legacy_correction.dtype
        )
        legacy_rms = (
            legacy_correction.detach().to(accumulation_dtype)
            .square()
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )
        candidate_delta_rms = (
            candidate_child_delta.detach().to(accumulation_dtype)
            .square()
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )
        budget_reference_rms = legacy_rms.clamp_min(
            float(bootstrap_reference_rms)
        )
        allowed_delta_rms = (
            float(max_delta_to_legacy_ratio) * budget_reference_rms
        )
        shrink = torch.where(
            candidate_delta_rms <= allowed_delta_rms,
            torch.ones_like(candidate_delta_rms),
            allowed_delta_rms
            / candidate_delta_rms.clamp_min(torch.finfo(accumulation_dtype).tiny),
        ).clamp(min=0.0, max=1.0).detach()
        realized_delta = candidate_child_delta * shrink.to(candidate_child_delta)
        candidate_correction = legacy_correction + realized_delta
        zero_contrast = candidate_child_delta.detach().abs().amax(
            dim=-1, keepdim=True
        ) == 0.0
        # Preserve the historical bf16/fp32 value at the exact structural-null
        # boundary while using the candidate branch as a straight-through
        # gradient path.  This lets the zero output projection learn on step 1.
        zero_forward_candidate_gradient = legacy_correction.detach() + (
            candidate_correction - candidate_correction.detach()
        )
        correction = torch.where(
            zero_contrast,
            zero_forward_candidate_gradient,
            candidate_correction,
        )
        realized_delta_rms = (
            realized_delta.detach().to(accumulation_dtype)
            .square()
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )
        legacy_zero = legacy_rms == 0.0
        delta_to_legacy_ratio = torch.where(
            ~legacy_zero,
            realized_delta_rms
            / legacy_rms.clamp_min(torch.finfo(accumulation_dtype).tiny),
            torch.zeros_like(realized_delta_rms),
        )
        saturation = (
            torch.tanh(final_raw.detach().to(accumulation_dtype)).abs() >= 0.95
        ).float().mean(dim=-1, keepdim=True)
        diagnostics = {
            "legacy_rms": legacy_rms,
            "realized_delta_rms": realized_delta_rms,
            "delta_to_legacy_ratio": delta_to_legacy_ratio,
            "shrink": shrink,
            "saturation": saturation,
            "legacy_zero": legacy_zero.float(),
        }
        return correction, diagnostics

    def predict_parent_mean(self, context, cov_celltype, cov_pert, cov_batch=None):
        lookup = self.prior_bank.lookup(
            cov_celltype, cov_pert, cov_batch=cov_batch
        )
        if not self.population_response_prior_enabled:
            lookup = replace(
                lookup,
                prior_delta=torch.zeros_like(lookup.prior_delta),
                prior_available=torch.zeros_like(lookup.prior_available),
            )
        prior_delta = lookup.prior_delta.to(context.parent_control)
        fewshot = None
        if self.fewshot_parent_adapter is not None:
            fewshot = self.fewshot_parent_adapter(
                prior_delta,
                lookup.artifact_cellline_ids,
                lookup.artifact_perturbation_ids,
            )
            prior_delta = fewshot.delta
        prior_feature = self.context_encoder.encode_expression(prior_delta)
        if self.strict_parent_only_enabled:
            if not isinstance(context, StrictParentOnlyContext):
                raise TypeError("strict mode requires StrictParentOnlyContext")
            parent_condition = context.parent_control_feature
            routed_condition = torch.zeros_like(parent_condition)
            correction_reference = context.parent_control
        else:
            parent_condition = context.parent_token
            routed_condition = context.routed_token
            correction_reference = context.control_prototypes

        condition = torch.cat(
            [
                context.parent_token,
                context.perturbation_query,
                routed_condition,
                prior_feature,
            ],
            dim=-1,
        )
        bridge_diagnostics = None
        tied_gene_raw_rms = None
        tied_gene_legacy_raw_rms = None
        tied_gene_to_legacy_ratio = None
        if self.parent_module_decoder_enabled:
            if self.strict_parent_only_enabled:
                raise RuntimeError(
                    "strict Parent-only mode forbids Parent Module Decoder"
                )
            raw = self.parent_correction_head(
                parent_token=context.parent_token,
                perturbation_query=context.perturbation_query,
                routed_token=context.routed_token,
                prior_feature=prior_feature,
                child_tokens=context.child_tokens,
                child_mask=context.anchor_mask,
                occupancy=context.occupancy,
                router_probabilities=(
                    self.inference_probabilities(context).detach()
                ),
            )
        else:
            if self.tied_gene_parent_correction is None:
                legacy_raw = self.parent_correction_head(condition)
                raw = legacy_raw
            else:
                hidden = self.parent_correction_head[0](condition)
                hidden = self.parent_correction_head[1](hidden)
                hidden = self.parent_correction_head[2](hidden)
                legacy_raw = self.parent_correction_head[3](hidden)
                tied_raw = self.tied_gene_parent_correction(
                    hidden,
                    self.gene_dit.main_tokenizer.gene_identity,
                )
                tied_raw = tied_raw.to(legacy_raw)
                tied_contribution = tied_raw
                if self.tied_gene_correction_mode != "replace":
                    tied_contribution = (
                        tied_raw * self.tied_gene_correction_residual_scale
                    )
                if (
                    self.tied_gene_correction_mode == "bounded_residual"
                    or self.tied_gene_correction_bounded_enabled
                ):
                    # This is a genuine Parent-relative trust budget, not
                    # merely a multiplier that query weights can undo. At the
                    # exact zero/zero initialization boundary the identity
                    # branch preserves a useful first-step query gradient.
                    contribution_rms = (
                        tied_contribution.detach()
                        .float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt()
                    )
                    legacy_rms = (
                        legacy_raw.detach()
                        .float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt()
                    )
                    maximum_rms = (
                        legacy_rms
                        * self.tied_gene_correction_residual_scale
                    )
                    shrink = torch.where(
                        contribution_rms > maximum_rms,
                        maximum_rms
                        / contribution_rms.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ),
                        torch.ones_like(contribution_rms),
                    )
                    tied_contribution = tied_contribution * shrink.to(
                        tied_contribution
                    )
                raw = (
                    tied_contribution
                    if self.tied_gene_correction_mode == "replace"
                    else legacy_raw + tied_contribution
                )
                tied_gene_raw_rms = (
                    tied_contribution.detach()
                    .float()
                    .square()
                    .mean(dim=-1, keepdim=True)
                    .sqrt()
                )
                tied_gene_legacy_raw_rms = (
                    legacy_raw.detach()
                    .float()
                    .square()
                    .mean(dim=-1, keepdim=True)
                    .sqrt()
                )
                tied_gene_to_legacy_ratio = torch.where(
                    tied_gene_legacy_raw_rms > 0.0,
                    tied_gene_raw_rms
                    / tied_gene_legacy_raw_rms.clamp_min(
                        torch.finfo(torch.float32).tiny
                    ),
                    torch.zeros_like(tied_gene_raw_rms),
                )
            if self.child_consensus_bridge is not None:
                router_probabilities = self.inference_probabilities(
                    context
                ).detach()
                (
                    collapsed_child_tokens,
                    collapsed_occupancy,
                    collapsed_router_probabilities,
                ) = self._collapsed_child_consensus_inputs(
                    context.child_tokens,
                    context.anchor_mask,
                    context.occupancy,
                )
                batch_size = context.child_tokens.shape[0]
                # One concatenated FP32 call gives full/collapsed exactly the
                # same deterministic decoder execution contract.  Disabling
                # outer autocast also prevents bf16 from erasing a small Child
                # contrast before the correction-space trust budget.
                with torch.autocast(
                    device_type=context.child_tokens.device.type,
                    enabled=False,
                ):
                    bridge_output = self.child_consensus_bridge(
                        parent_token=torch.cat(
                            [
                                context.parent_token.detach().float(),
                                context.parent_token.detach().float(),
                            ],
                            dim=0,
                        ),
                        perturbation_query=torch.cat(
                            [
                                context.perturbation_query.detach().float(),
                                context.perturbation_query.detach().float(),
                            ],
                            dim=0,
                        ),
                        routed_token=torch.cat(
                            [
                                context.routed_token.detach().float(),
                                context.routed_token.detach().float(),
                            ],
                            dim=0,
                        ),
                        prior_feature=torch.cat(
                            [
                                prior_feature.detach().float(),
                                prior_feature.detach().float(),
                            ],
                            dim=0,
                        ),
                        child_tokens=torch.cat(
                            [
                                context.child_tokens.float(),
                                collapsed_child_tokens.float(),
                            ],
                            dim=0,
                        ),
                        child_mask=torch.cat(
                            [context.anchor_mask, context.anchor_mask], dim=0
                        ),
                        occupancy=torch.cat(
                            [
                                context.occupancy.float(),
                                collapsed_occupancy.float(),
                            ],
                            dim=0,
                        ),
                        router_probabilities=torch.cat(
                            [
                                router_probabilities.float(),
                                collapsed_router_probabilities.float(),
                            ],
                            dim=0,
                        ),
                    )
                bridge_full_raw, bridge_collapsed_raw = bridge_output.split(
                    batch_size, dim=0
                )
                contrast_raw = bridge_full_raw - bridge_collapsed_raw
                # This is the sole raw bridge composition.  There is no
                # learned gate and no second zero-initialized parameter.
                raw = legacy_raw.float() + contrast_raw
                legacy_bridge_correction_native = (
                    self.correction_max_scale * torch.tanh(legacy_raw)
                )
                anchor_correction = (
                    self.correction_max_scale
                    * torch.tanh(legacy_raw.detach().float())
                )
                candidate_bridge_correction = self.correction_max_scale * torch.tanh(
                    legacy_raw.detach().float() + contrast_raw
                )
                candidate_child_delta = (
                    candidate_bridge_correction - anchor_correction
                )
                correction, bridge_diagnostics = (
                    self._convex_child_consensus_correction(
                        legacy_bridge_correction_native.float(),
                        candidate_child_delta,
                        final_raw=raw,
                        max_delta_to_legacy_ratio=(
                            self.child_consensus_bridge_max_delta_to_legacy_ratio
                        ),
                        bootstrap_reference_rms=(
                            self.child_consensus_bridge_bootstrap_reference_rms
                        ),
                    )
                )
                bridge_diagnostics.update(
                    {
                        "full_raw_rms": bridge_full_raw.float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt(),
                        "collapsed_raw_rms": bridge_collapsed_raw.float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt(),
                        "contrast_raw_rms": contrast_raw.float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt(),
                    }
                )
        if bridge_diagnostics is None:
            correction = self.correction_max_scale * torch.tanh(raw)
        correction = correction.to(correction_reference)
        control_mean = lookup.control_mean.to(correction)
        prior_delta = prior_delta.to(correction)
        legacy_delta = prior_delta + correction
        legacy_parent_mean = control_mean + prior_delta + correction
        gate_logits = None
        gate_probability = None
        gate_slab = None
        effective_delta = legacy_delta
        effective_blend = None
        unified_sparse_program = None
        if self.soft_gated_regression_enabled:
            # Evaluate one program from the ungated legacy delta. Parent,
            # DE-BCE/count, zero-rate and terminal H reuse these exact tensors.
            response_gene_identity = (
                self.gene_dit.main_tokenizer.gene_identity
                if self.gene_relation_enabled
                else None
            )
            unified_sparse_program = (
                self.sparse_hurdle_adapter.predict_gene_program(
                    condition=condition,
                    base_parent_delta=legacy_delta,
                    control_mean=control_mean,
                    response_gene_identity=response_gene_identity,
                    detach_base_features=(
                        self.soft_gate_detach_regression
                        and not self.sparse_joint_context_enabled
                    ),
                    support_logit_offset=self.soft_gate_initial_log_odds,
                    support_temperature=self.soft_gate_temperature,
                    slab_scale=self.soft_gate_slab_scale,
                )
            )
            gate_logits = unified_sparse_program.shared_support_logits
            gate_probability = (
                unified_sparse_program.shared_support_probability
            )
            gate_slab = unified_sparse_program.signed_slab
            work_dtype = (
                torch.float64
                if legacy_delta.dtype == torch.float64
                else torch.float32
            )
            legacy_work = legacy_delta.to(work_dtype)
            effective_blend = self.soft_gate_blend
            if effective_blend == 0.0:
                # The schedule starts at the exact historical Parent path.
                effective_delta = legacy_delta
            else:
                regression_probability = (
                    gate_probability.detach()
                    if self.soft_gate_detach_regression
                    else gate_probability
                )
                effective_delta = (
                    (1.0 - effective_blend) * legacy_work
                    + effective_blend
                    * regression_probability
                    * gate_slab
                ).to(legacy_delta)
        # Retain the historical addition order at both disabled endpoints.
        # This matters for bitwise resume/audit comparisons in reduced dtypes.
        parent_mean = (
            legacy_parent_mean
            if not self.soft_gated_regression_enabled
            or effective_blend == 0.0
            else control_mean + effective_delta
        )
        if (
            not self.child_guidance_enabled
            and not self.child_marginal_enabled
        ):
            # Exact historical return: do not inspect Router/Child tensors.
            return ParentMeanPrediction(
                lookup=lookup,
                prior_feature=prior_feature,
                raw_correction=raw,
                correction=correction,
                parent_mean=parent_mean,
                fewshot=fewshot,
                legacy_delta=legacy_delta,
                soft_gate_logits=gate_logits,
                soft_gate_probability=gate_probability,
                soft_gate_slab=gate_slab,
                effective_delta=effective_delta,
                soft_gate_effective_blend=effective_blend,
                unified_sparse_program=unified_sparse_program,
                child_consensus_bridge_legacy_rms=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["legacy_rms"]
                ),
                child_consensus_bridge_full_raw_rms=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["full_raw_rms"]
                ),
                child_consensus_bridge_collapsed_raw_rms=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["collapsed_raw_rms"]
                ),
                child_consensus_bridge_contrast_raw_rms=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["contrast_raw_rms"]
                ),
                child_consensus_bridge_realized_delta_rms=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["realized_delta_rms"]
                ),
                child_consensus_bridge_delta_to_legacy_ratio=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["delta_to_legacy_ratio"]
                ),
                child_consensus_bridge_shrink=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["shrink"]
                ),
                child_consensus_bridge_saturation=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["saturation"]
                ),
                child_consensus_bridge_legacy_zero=(
                    None
                    if bridge_diagnostics is None
                    else bridge_diagnostics["legacy_zero"]
                ),
                tied_gene_raw_rms=tied_gene_raw_rms,
                tied_gene_legacy_raw_rms=tied_gene_legacy_raw_rms,
                tied_gene_to_legacy_ratio=tied_gene_to_legacy_ratio,
            )
        if self.child_marginal_enabled:
            (
                marginal_parent_mean,
                child_marginal_mean,
                child_marginal_control_mean,
                child_marginal_delta,
                child_marginal_residual,
                child_marginal_contribution,
                child_marginal_alpha,
                child_marginal_router_entropy,
                child_marginal_router_effective_children,
            ) = self._apply_child_marginal(context, parent_mean)
            return ParentMeanPrediction(
                lookup=lookup,
                prior_feature=prior_feature,
                raw_correction=raw,
                correction=correction,
                parent_mean=marginal_parent_mean,
                fewshot=fewshot,
                legacy_delta=legacy_delta,
                soft_gate_logits=gate_logits,
                soft_gate_probability=gate_probability,
                soft_gate_slab=gate_slab,
                effective_delta=effective_delta,
                soft_gate_effective_blend=effective_blend,
                unified_sparse_program=unified_sparse_program,
                base_parent_mean=parent_mean,
                child_marginal_mean=child_marginal_mean,
                child_marginal_control_mean=child_marginal_control_mean,
                child_marginal_delta=child_marginal_delta,
                child_marginal_residual=child_marginal_residual,
                child_marginal_contribution=child_marginal_contribution,
                child_marginal_alpha=child_marginal_alpha,
                child_marginal_router_entropy=child_marginal_router_entropy,
                child_marginal_router_effective_children=(
                    child_marginal_router_effective_children
                ),
                tied_gene_raw_rms=tied_gene_raw_rms,
                tied_gene_legacy_raw_rms=tied_gene_legacy_raw_rms,
                tied_gene_to_legacy_ratio=tied_gene_to_legacy_ratio,
            )

        (
            guided_parent_mean,
            guidance_proposal,
            guidance_residual,
            guidance_alpha,
            guidance_coupling,
            guidance_topk_mass,
        ) = self._apply_child_guidance(
            context,
            parent_mean,
            control_mean,
        )
        return ParentMeanPrediction(
            lookup=lookup,
            prior_feature=prior_feature,
            raw_correction=raw,
            correction=correction,
            parent_mean=guided_parent_mean,
            fewshot=fewshot,
            legacy_delta=legacy_delta,
            soft_gate_logits=gate_logits,
            soft_gate_probability=gate_probability,
            soft_gate_slab=gate_slab,
            effective_delta=effective_delta,
            soft_gate_effective_blend=effective_blend,
            unified_sparse_program=unified_sparse_program,
            base_parent_mean=parent_mean,
            child_guidance_proposal=guidance_proposal,
            child_guidance_residual=guidance_residual,
            child_guidance_alpha=guidance_alpha,
            child_guidance_coupling=guidance_coupling,
            child_guidance_topk_mass=guidance_topk_mass,
            tied_gene_raw_rms=tied_gene_raw_rms,
            tied_gene_legacy_raw_rms=tied_gene_legacy_raw_rms,
            tied_gene_to_legacy_ratio=tied_gene_to_legacy_ratio,
        )

    def prepare_strict_parent_only(
        self,
        *,
        cov_celltype: torch.Tensor,
        cov_pert: torch.Tensor,
        semantic_condition: torch.Tensor,
        num_cells: int,
        cov_batch: Optional[torch.Tensor] = None,
        stochastic: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        epsilon: Optional[torch.Tensor] = None,
    ) -> tuple[StrictParentOnlyContext, PreparedParentOnlySource]:
        """Build the shared target-free train/inference Parent-only source."""

        if not self.strict_parent_only_enabled:
            raise RuntimeError(
                "prepare_strict_parent_only requires the strict core switch"
            )
        if isinstance(num_cells, bool) or not isinstance(num_cells, int):
            raise TypeError("num_cells must be an integer")
        if num_cells < 1:
            raise ValueError("num_cells must be positive")
        if epsilon is not None:
            raise ValueError("strict Parent-only source does not accept epsilon")
        condition_summary = self.context_encoder._condition_summary(
            semantic_condition
        ).float()
        lookup = self.prior_bank.lookup(
            cov_celltype, cov_pert, cov_batch=cov_batch
        )
        control_mean = lookup.control_mean.to(
            device=condition_summary.device,
            dtype=condition_summary.dtype,
        )
        if control_mean.shape[0] != condition_summary.shape[0]:
            raise ValueError(
                "semantic condition and Parent lookup must share batch size"
            )
        parent_token = self.context_encoder.encode_expression(control_mean)
        perturbation_query = self.context_encoder.condition_encoder(
            condition_summary
        )
        context = StrictParentOnlyContext(
            parent_token=parent_token,
            perturbation_query=perturbation_query,
            parent_control=control_mean,
            parent_control_feature=parent_token,
        )
        parent = self.predict_parent_mean(
            context,
            cov_celltype,
            cov_pert,
            cov_batch=cov_batch,
        )

        real_control_sample = None
        if self.real_control_source_enabled:
            if self.real_control_sampling_mode != "global_marginal":
                raise RuntimeError(
                    "strict empirical source requires global_marginal sampling"
                )
            real_stochastic = (
                self.real_control_stochastic
                if stochastic is None
                else bool(stochastic)
            )
            real_control_sample = self.real_control_reservoir.sample_global(
                cov_celltype=cov_celltype,
                cov_batch=cov_batch,
                num_samples=num_cells,
                stochastic=real_stochastic,
                generator=generator,
                runtime_control_mean=parent.lookup.control_mean,
            )
            residual = real_control_sample.residual.to(parent.parent_mean)
            residual = residual * self.real_control_residual_scale
            source_value = parent.parent_mean.detach()[:, None, :] + residual
        else:
            residual = parent.parent_mean.new_zeros(
                parent.parent_mean.shape[0],
                num_cells,
                parent.parent_mean.shape[1],
            )
            source_value = parent.parent_mean.detach()[:, None, :].expand(
                -1, num_cells, -1
            )
        if not torch.isfinite(source_value).all():
            raise RuntimeError("strict Parent-only source contains non-finite values")
        source = ParentOnlyResidualSource(
            source=source_value,
            mean=source_value,
            residual=residual,
            std=torch.zeros_like(source_value),
        )
        prepared = PreparedParentOnlySource(
            source=source,
            parent=parent,
            real_control_sample=real_control_sample,
        )
        return context, prepared


    def build_prior_source(
        self,
        context,
        cov_celltype,
        cov_pert,
        child_stds,
        *,
        cov_batch=None,
        num_cells: Optional[int] = None,
        selected_indices: Optional[torch.Tensor] = None,
        stochastic: Optional[bool] = None,
        sampling_mode: str = "legacy",
        routing_group_ids: Optional[torch.Tensor] = None,
        epsilon: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PreparedPriorSource:
        if self.strict_parent_only_enabled:
            raise RuntimeError(
                "strict Parent-only mode forbids Child prior construction; "
                "use prepare_strict_parent_only"
            )
        child_stds = self._padded_child_stds(context, child_stds)
        if selected_indices is None:
            if num_cells is None:
                raise ValueError("num_cells is required when indices are not supplied")
            routed = self.route_prior(
                context,
                num_cells=int(num_cells),
                stochastic=stochastic,
                sampling_mode=sampling_mode,
                routing_group_ids=routing_group_ids,
                generator=generator,
            )
            selected_indices = routed["child_indices"]
        else:
            if num_cells is not None and int(num_cells) != selected_indices.shape[1]:
                raise ValueError(
                    "num_cells must equal selected_indices.shape[1] when both are supplied"
                )
            self._validate_selected_indices(context, selected_indices)
            routed = {
                "prior": self.source_selection_probabilities(context),
                "child_indices": selected_indices,
            }
            rows = torch.arange(
                selected_indices.shape[0], device=selected_indices.device
            )[:, None]
            routed["selected_child_tokens"] = context.child_tokens[
                rows, selected_indices
            ]
        parent = self.predict_parent_mean(
            context, cov_celltype, cov_pert, cov_batch=cov_batch
        )
        real_control_sample = None
        if self.real_control_source_enabled:
            if epsilon is not None:
                raise ValueError(
                    "epsilon is not used by the real-control residual source"
                )
            from src.models.real_control_residual_source import (
                build_real_control_parent_source,
            )

            real_stochastic = (
                self.real_control_stochastic
                if stochastic is None
                else bool(stochastic)
            )
            real_control_sample = self.real_control_reservoir.sample(
                cov_celltype=cov_celltype,
                cov_batch=cov_batch,
                selected_indices=selected_indices,
                stochastic=real_stochastic,
                generator=generator,
                runtime_child_mask=context.anchor_mask,
                runtime_control_mean=parent.lookup.control_mean,
                runtime_prototypes=(
                    None
                    if self.real_control_sampling_mode == "global_marginal"
                    else context.control_prototypes
                ),
                bank_inverse=(
                    None
                    if self.real_control_sampling_mode == "global_marginal"
                    else context.bank_inverse
                ),
            )
            bridge = build_real_control_parent_source(
                parent_mean=parent.parent_mean,
                sample=real_control_sample,
                prior_probs=routed["prior"].to(context.control_prototypes),
                child_mask=context.anchor_mask,
                scale=self.real_control_residual_scale,
            )
        else:
            bridge = build_parent_residual_prior_source(
                parent_mean=parent.parent_mean,
                prototypes=context.control_prototypes,
                stds=child_stds,
                prior_probs=routed["prior"].to(context.control_prototypes),
                bank_inverse=context.bank_inverse,
                selected_indices=selected_indices,
                child_mask=context.anchor_mask,
                beta=self.child_beta,
                beta_std=self.child_std_beta,
                epsilon=epsilon,
                generator=(None if epsilon is not None else generator),
            )
        return PreparedPriorSource(
            source=bridge,
            parent=parent,
            prior=routed["prior"],
            child_indices=selected_indices,
            selected_child_tokens=routed["selected_child_tokens"],
            real_control_sample=real_control_sample,
        )

    def predict_sparse_support(
        self,
        context,
        parent: ParentMeanPrediction,
        control_zero_rate: torch.Tensor,
    ):
        """Run the target-free S head without altering the Parent prediction."""

        if self.sparse_hurdle_adapter is None:
            raise RuntimeError("sparse support is not enabled")
        unified_sparse_program = getattr(
            parent, "unified_sparse_program", None
        )
        if (
            bool(getattr(self, "soft_gated_regression_enabled", False))
            and unified_sparse_program is None
        ):
            raise RuntimeError(
                "soft-gated sparse support requires its cached gene program"
            )
        if self.strict_parent_only_enabled:
            parent_condition = context.parent_control_feature
            routed_condition = torch.zeros_like(parent_condition)
        else:
            parent_condition = context.parent_token
            routed_condition = context.routed_token
        condition = torch.cat(
            (
                parent_condition,
                context.perturbation_query,
                routed_condition,
                parent.prior_feature,
            ),
            dim=-1,
        )
        if unified_sparse_program is None:
            base_parent_delta = (
                parent.parent_mean
                - parent.lookup.control_mean.to(parent.parent_mean)
            )
        else:
            if parent.legacy_delta is None:
                raise AssertionError(
                    "unified sparse program is missing its legacy delta"
                )
            base_parent_delta = parent.legacy_delta
        control_mean = parent.lookup.control_mean.to(base_parent_delta)
        control_zero_rate = control_zero_rate.to(base_parent_delta)
        detach_base_features = True
        if self.sparse_joint_context_enabled:
            # Joint S training reaches the target-free condition path only;
            # Parent outputs and control statistics remain a fixed floor.
            base_parent_delta = base_parent_delta.detach()
            control_mean = control_mean.detach()
            control_zero_rate = control_zero_rate.detach()
            detach_base_features = False
        return self.sparse_hurdle_adapter(
            condition=condition,
            base_parent_delta=base_parent_delta,
            control_mean=control_mean,
            control_zero_rate=control_zero_rate,
            alpha_mu=0.0,
            detach_base_features=detach_base_features,
            response_gene_identity=(
                self.gene_dit.main_tokenizer.gene_identity
                if bool(getattr(self, "gene_relation_enabled", False))
                else None
            ),
            program=unified_sparse_program,
        )

    def forward(self, x_input, x_control_input, t, self_condition=None):
        if self_condition is None:
            raise ValueError("Parent-Residual Gene-DiT requires self_condition")
        if x_input.ndim != 3 or x_input.shape[-1] != 2 * self.gene_dim:
            raise ValueError(f"x_input must have shape [B,S,{2 * self.gene_dim}]")
        if x_control_input.shape != x_input.shape:
            raise ValueError("x_control_input must match x_input")
        if self.strict_parent_only_enabled:
            required = ("parent_residual_parent_token",)
            forbidden_keys = (
                "parent_residual_child_tokens",
                "parent_residual_routed_token",
                "parent_residual_all_child_tokens",
                "parent_residual_child_mask",
                "parent_residual_perturbation_query",
            )
            present = [key for key in forbidden_keys if key in self_condition]
            if present:
                raise ValueError(
                    "strict Parent-only conditioning forbids "
                    + ", ".join(present)
                )
        else:
            required = (
                "parent_residual_parent_token",
                "parent_residual_child_tokens",
                "parent_residual_routed_token",
            )
            if self.strong_condition_memory_enabled:
                required = required + (
                    "parent_residual_all_child_tokens",
                    "parent_residual_child_mask",
                    "parent_residual_perturbation_query",
                )
        missing = [key for key in required if key not in self_condition]
        if missing:
            raise ValueError("missing Parent-Residual conditioning: " + ", ".join(missing))
        condition = self_condition.get("batch_emb")
        if condition is None:
            condition = x_input.new_zeros(
                x_input.shape[0],
                self.gene_dit.config.condition_dim,
            )
        normalized_time = t.float() / self.flow_time_scale
        population_only_field = self.field_context_mode == "population_only"
        return self.gene_dit(
            x_input[..., : self.gene_dim],
            x_control_input[..., : self.gene_dim],
            normalized_time,
            parent=self_condition["parent_residual_parent_token"],
            condition=condition,
            selected_child=(
                None
                if self.strict_parent_only_enabled or population_only_field
                else self_condition["parent_residual_child_tokens"]
            ),
            routed=(
                None
                if self.strict_parent_only_enabled or population_only_field
                else self_condition["parent_residual_routed_token"]
            ),
            all_child_tokens=(
                None
                if population_only_field
                else self_condition.get("parent_residual_all_child_tokens")
            ),
            child_mask=(
                None
                if population_only_field
                else self_condition.get("parent_residual_child_mask")
            ),
            perturb_query=(
                None
                if population_only_field
                else self_condition.get("parent_residual_perturbation_query")
            ),
            parent_only=self.strict_parent_only_enabled or population_only_field,
        )


__all__ = [
    "ParentMeanPrediction",
    "ParentResidualGeneDiTModel",
    "PreparedPriorSource",
]

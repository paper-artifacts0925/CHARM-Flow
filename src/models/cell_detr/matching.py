"""Independent H-DETR-style matching for individual response cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .context import CellDETRContext


@dataclass
class CellDETRMatch:
    selected_control: torch.Tensor
    selected_child_tokens: torch.Tensor
    positive_controls: torch.Tensor
    positive_delta: torch.Tensor
    positive_prediction: torch.Tensor
    child_indices: torch.Tensor
    positive_child_indices: torch.Tensor
    positive_mask: torch.Tensor
    child_distribution: torch.Tensor
    router_distribution: torch.Tensor
    delta_per_cell: torch.Tensor
    direction_per_cell: torch.Tensor
    magnitude_per_cell: torch.Tensor
    router_per_cell: torch.Tensor
    cost_per_cell: torch.Tensor
    delta_loss: torch.Tensor
    direction_loss: torch.Tensor
    magnitude_loss: torch.Tensor
    router_loss: torch.Tensor
    mean_cost: torch.Tensor
    positive_multiplicity: int
    effective_positive_multiplicity_min: torch.Tensor
    effective_positive_multiplicity_mean: torch.Tensor
    parent_control: torch.Tensor = None
    parent_delta: torch.Tensor = None
    parent_prediction: torch.Tensor = None
    parent_delta_per_cell: torch.Tensor = None
    parent_direction_per_cell: torch.Tensor = None
    parent_magnitude_per_cell: torch.Tensor = None
    parent_cost_per_cell: torch.Tensor = None
    child_quality_cost: torch.Tensor = None
    relative_gain: torch.Tensor = None
    gate_target: torch.Tensor = None
    selected_gate_probability: torch.Tensor = None
    gate_loss: torch.Tensor = None
    latent_match_loss: torch.Tensor = None
    balance_loss: torch.Tensor = None
    router_target_distribution: torch.Tensor = None
    router_confidence: torch.Tensor = None
    fallback_target_probability: torch.Tensor = None
    ot_enabled: torch.Tensor = None
    ot_column_l1: torch.Tensor = None
    ot_column_kl: torch.Tensor = None
    ot_effective_children: torch.Tensor = None
    ot_cost_scale: torch.Tensor = None
    ot_balanced: torch.Tensor = None


class PerCellHungarianMatcher:
    """Match each response cell independently against its own 128 queries.

    Repeating the single target R times gives an R-column Hungarian problem.
    Because those columns are identical, its exact solution is the R distinct
    minimum-cost queries. ``topk`` is therefore an exact vectorized solver for
    this special Hungarian matrix and avoids a CPU SciPy call for every cell.

    The optional cold-start hierarchy keeps this assignment target-only and
    detached from Router logits.  Parent-versus-Child gain teaches a separate
    inference-time gate; it never enters the actual FM path directly.
    """

    def __init__(
        self,
        endpoint_cost_weight=1.0,
        direction_cost_weight=0.25,
        magnitude_cost_weight=0.05,
        router_cost_weight=0.05,
        temperature=0.25,
        direction_reliability_scale=0.02,
        gain_cost_floor=1e-3,
        eps=1e-6,
    ):
        self.endpoint_cost_weight = float(endpoint_cost_weight)
        self.direction_cost_weight = float(direction_cost_weight)
        self.magnitude_cost_weight = float(magnitude_cost_weight)
        self.router_cost_weight = float(router_cost_weight)
        self.temperature = float(temperature)
        self.direction_reliability_scale = float(direction_reliability_scale)
        self.gain_cost_floor = float(gain_cost_floor)
        self.eps = float(eps)

    def _latent_biological_cost_components(self, control, predicted, target):
        control = control.float()
        predicted = predicted.float()
        target = target.float()
        observed = (target - control).detach()
        endpoint_cost = (observed - predicted).square().mean(dim=-1)
        direction_cost = 1.0 - F.cosine_similarity(
            observed, predicted, dim=-1, eps=self.eps
        )
        observed_magnitude = observed.square().mean(dim=-1).add(self.eps).sqrt()
        predicted_magnitude = predicted.square().mean(dim=-1).add(self.eps).sqrt()
        magnitude_cost = (
            observed_magnitude.log() - predicted_magnitude.log()
        ).abs()
        return endpoint_cost, direction_cost, magnitude_cost

    def _latent_biological_cost(
        self,
        control,
        predicted,
        target,
        cost_weight_multipliers=None,
    ):
        endpoint_cost, direction_cost, magnitude_cost = (
            self._latent_biological_cost_components(control, predicted, target)
        )
        weights = endpoint_cost.new_tensor(
            [
                self.endpoint_cost_weight,
                self.direction_cost_weight,
                self.magnitude_cost_weight,
            ]
        )
        if cost_weight_multipliers is not None:
            multipliers = cost_weight_multipliers.to(
                device=endpoint_cost.device,
                dtype=endpoint_cost.dtype,
            )
            expected = (endpoint_cost.shape[0], 3)
            if multipliers.shape != expected:
                raise ValueError(
                    "cost_weight_multipliers must have shape "
                    f"[B,3]={expected}, got {tuple(multipliers.shape)}"
                )
            if not torch.isfinite(multipliers).all() or (multipliers <= 0).any():
                raise ValueError(
                    "cost_weight_multipliers must be finite and strictly positive"
                )
            weights = weights[None, :] * multipliers
            endpoint_weight = weights[:, 0, None]
            direction_weight = weights[:, 1, None]
            magnitude_weight = weights[:, 2, None]
        else:
            endpoint_weight, direction_weight, magnitude_weight = weights
        biological = (
            endpoint_weight * endpoint_cost
            + direction_weight * direction_cost
            + magnitude_weight * magnitude_cost
        )
        return biological

    def _cost(
        self,
        context: CellDETRContext,
        target_features,
        include_router=True,
        cost_weight_multipliers=None,
    ):
        target = target_features[:, 0]
        biological = self._latent_biological_cost(
            context.control_features,
            context.predicted_delta_latent,
            target[:, None],
            cost_weight_multipliers=cost_weight_multipliers,
        )
        if include_router and self.router_cost_weight:
            router_cost = -F.log_softmax(context.router_logits.float(), dim=-1)
            biological = biological + self.router_cost_weight * router_cost
        return biological.masked_fill(~context.anchor_mask, torch.inf)

    def _gene_quality(self, predicted_delta, observed_delta):
        predicted = predicted_delta.float()
        observed = observed_delta.float()
        delta = (predicted - observed).square().mean(dim=-1)
        direction = 1.0 - F.cosine_similarity(
            predicted, observed, dim=-1, eps=self.eps
        )
        predicted_magnitude = predicted.square().mean(dim=-1).add(self.eps).sqrt()
        observed_magnitude = observed.square().mean(dim=-1).add(self.eps).sqrt()
        magnitude = (
            predicted_magnitude.log() - observed_magnitude.log()
        ).abs()
        reliability = (
            observed_magnitude
            / (observed_magnitude + self.direction_reliability_scale)
        ).detach()
        quality = (
            self.endpoint_cost_weight * delta
            + self.direction_cost_weight * reliability * direction
            + self.magnitude_cost_weight * magnitude
        )
        return delta, direction, magnitude, quality

    @staticmethod
    def _bank_rows(context, batch_size, device):
        if context.bank_inverse is None:
            return torch.arange(batch_size, device=device, dtype=torch.long)
        return context.bank_inverse.to(device=device, dtype=torch.long)

    def _router_probabilities(
        self,
        context,
        temperature,
        occupancy_logit_weight,
    ):
        logits = context.router_logits.float() / float(temperature)
        if occupancy_logit_weight:
            logits = logits + float(occupancy_logit_weight) * context.occupancy.float().clamp_min(
                1e-8
            ).log()
        logits = logits.masked_fill(~context.anchor_mask, -torch.inf)
        return torch.softmax(logits, dim=-1)

    def _balance_loss(self, context, probabilities, target_power):
        batch_size, num_anchors = probabilities.shape
        bank_rows = self._bank_rows(context, batch_size, probabilities.device)
        num_banks = context.control_prototypes.shape[0]
        group_probability = probabilities.new_zeros(num_banks, num_anchors)
        group_occupancy = probabilities.new_zeros(num_banks, num_anchors)
        group_count = probabilities.new_zeros(num_banks, 1)
        group_probability.index_add_(0, bank_rows, probabilities)
        group_occupancy.index_add_(0, bank_rows, context.occupancy.float())
        group_count.index_add_(
            0,
            bank_rows,
            probabilities.new_ones(batch_size, 1),
        )
        present = group_count[:, 0] > 0
        group_probability = group_probability[present] / group_count[present]
        group_occupancy = group_occupancy[present] / group_count[present]
        target = group_occupancy.clamp_min(0.0).pow(float(target_power))
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        group_probability = group_probability / group_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        return (
            target
            * (
                target.clamp_min(self.eps).log()
                - group_probability.clamp_min(self.eps).log()
            )
        ).sum(dim=-1).mean()

    def __call__(
        self,
        context: CellDETRContext,
        targets,
        target_features,
        decode_selected_delta,
        positive_multiplicity: int,
        supervision_mask=None,
        decode_parent_delta=None,
        gate_enabled=False,
        force_parent=False,
        relative_margin=0.30,
        gate_temperature=0.08,
        exploration_weight=0.10,
        router_temperature=None,
        router_occupancy_logit_weight=0.0,
        balance_target_power=1.0,
        semi_balanced_ot_enabled=False,
        semi_balanced_ot_group_ids=None,
        semi_balanced_ot_epsilon=0.1,
        semi_balanced_ot_rho=1.0,
        semi_balanced_ot_iterations=50,
        semi_balanced_ot_normalize_cost_by_median=True,
        balanced_ot_enabled=False,
        balanced_ot_column_tolerance=1.0e-3,
        positive_multiplicity_cap_to_active_children=False,
        cost_weight_multipliers=None,
    ):
        if targets.ndim != 3 or targets.shape[1] != 1:
            raise ValueError(
                "cell_detr matching requires targets [B,1,G]; cells never compete across batch"
            )
        if target_features.shape[:2] != targets.shape[:2]:
            raise ValueError("target_features must have shape [B,1,H]")
        batch_size = targets.shape[0]
        multiplicity = int(positive_multiplicity)
        if multiplicity < 1:
            raise ValueError("positive_multiplicity must be positive")
        active_count = context.anchor_mask.sum(dim=-1)
        if (active_count < 1).any():
            raise ValueError("matching requires at least one active child token per cell")
        cap_to_active = bool(positive_multiplicity_cap_to_active_children)
        if not cap_to_active and (active_count < multiplicity).any():
            raise ValueError(
                f"matching needs at least R={multiplicity} active child tokens per cell"
            )
        effective_multiplicity = active_count.clamp_max(multiplicity)
        positive_mask = (
            torch.arange(multiplicity, device=targets.device)[None, :]
            < effective_multiplicity[:, None]
        )
        effective_multiplicity_float = effective_multiplicity.to(
            device=targets.device,
            dtype=torch.float32,
        )
        effective_positive_multiplicity_min = effective_multiplicity_float.min()
        effective_positive_multiplicity_mean = effective_multiplicity_float.mean()
        if supervision_mask is None:
            supervision_mask = torch.ones(
                batch_size, 1, device=targets.device, dtype=torch.bool
            )
        supervision_mask = supervision_mask.to(device=targets.device, dtype=torch.bool)
        if supervision_mask.shape != (batch_size, 1):
            raise ValueError("supervision_mask must have shape [B,1]")

        # Router logits are intentionally excluded from the cold-start teacher.
        cost = self._cost(
            context,
            target_features,
            include_router=not bool(gate_enabled),
            cost_weight_multipliers=cost_weight_multipliers,
        )
        active_finite = context.anchor_mask & torch.isfinite(cost)
        invalid_finite_rows = supervision_mask[:, 0] & ~active_finite.any(dim=-1)
        if invalid_finite_rows.any():
            bad = invalid_finite_rows
            bad_indices = torch.nonzero(bad, as_tuple=False).flatten()[:8].tolist()

            def nonfinite_count(tensor):
                return int((~torch.isfinite(tensor[bad])).sum().item())

            raise ValueError(
                "matching produced no active finite Child cost before OT; "
                f"rows={bad_indices}, count={int(bad.sum().item())}, "
                f"target_nonfinite={nonfinite_count(target_features)}, "
                f"control_nonfinite={nonfinite_count(context.control_features)}, "
                f"delta_nonfinite={nonfinite_count(context.predicted_delta_latent)}, "
                f"router_nonfinite={nonfinite_count(context.router_logits)}"
            )
        first_active_index = context.anchor_mask.to(torch.long).argmax(
            dim=-1, keepdim=True
        )
        selection_width = min(multiplicity, cost.shape[1])

        def fixed_width_topk(scores, *, largest):
            indices = torch.topk(
                scores,
                k=selection_width,
                dim=-1,
                largest=largest,
                sorted=True,
            ).indices
            if selection_width < multiplicity:
                indices = torch.cat(
                    [
                        indices,
                        first_active_index.expand(
                            -1, multiplicity - selection_width
                        ),
                    ],
                    dim=-1,
                )
            if cap_to_active:
                indices = torch.where(
                    positive_mask,
                    indices,
                    first_active_index.expand_as(indices),
                )
            return indices

        legacy_positive_indices = fixed_width_topk(cost, largest=False)

        def hard_distribution(indices):
            hard = targets.new_zeros(
                batch_size, context.anchor_mask.shape[1]
            )
            if cap_to_active:
                mass = positive_mask.to(targets) / effective_multiplicity.to(
                    targets
                )[:, None].clamp_min(1.0)
            else:
                mass = targets.new_full(indices.shape, 1.0 / multiplicity)
            hard.scatter_add_(1, indices, mass)
            return hard
        ot_result = None
        if bool(balanced_ot_enabled) and not bool(semi_balanced_ot_enabled):
            raise ValueError("balanced OT requires the OT matching path")
        if bool(semi_balanced_ot_enabled):
            if semi_balanced_ot_group_ids is None:
                raise ValueError("semi-balanced OT matching requires condition group IDs")
            solver_kwargs = dict(
                group_ids=semi_balanced_ot_group_ids,
                child_mask=context.anchor_mask,
                row_mask=supervision_mask[:, 0],
                epsilon=float(semi_balanced_ot_epsilon),
                iterations=int(semi_balanced_ot_iterations),
                normalize_cost_by_median=bool(
                    semi_balanced_ot_normalize_cost_by_median
                ),
            )
            if bool(balanced_ot_enabled):
                from src.models.matched_set_transport.balanced_ot import (
                    solve_grouped_balanced_ot,
                )

                ot_result = solve_grouped_balanced_ot(
                    cost,
                    context.occupancy,
                    column_tolerance=float(balanced_ot_column_tolerance),
                    **solver_kwargs,
                )
            else:
                from src.models.matched_set_transport.semi_balanced_ot import (
                    solve_grouped_semi_balanced_ot,
                )

                ot_result = solve_grouped_semi_balanced_ot(
                    cost,
                    context.occupancy,
                    rho=float(semi_balanced_ot_rho),
                    **solver_kwargs,
                )
            distribution = ot_result.child_distribution.to(targets)
            selection_scores = distribution.masked_fill(~context.anchor_mask, -1.0)
            positive_indices = fixed_width_topk(
                selection_scores,
                largest=True,
            )
            invalid_rows = ~supervision_mask[:, 0]
            if invalid_rows.any():
                positive_indices = positive_indices.clone()
                positive_indices[invalid_rows] = legacy_positive_indices[invalid_rows]
                fallback = hard_distribution(legacy_positive_indices)
                distribution = torch.where(
                    invalid_rows[:, None], fallback, distribution
                )
        else:
            positive_indices = legacy_positive_indices
            distribution = hard_distribution(positive_indices)
        primary_indices = positive_indices[:, :1]
        batch = torch.arange(batch_size, device=targets.device)[:, None]
        bank_rows = self._bank_rows(context, batch_size, targets.device)[:, None]
        positive_controls = context.control_prototypes[bank_rows, positive_indices]
        positive_delta = decode_selected_delta(context, positive_indices)
        observed_delta = targets[:, :1] - positive_controls

        (
            delta_per_positive,
            direction_per_positive,
            magnitude_per_positive,
            child_quality_cost,
        ) = self._gene_quality(positive_delta, observed_delta)
        valid = supervision_mask.to(delta_per_positive.dtype).expand(-1, multiplicity)
        if cap_to_active:
            valid = valid * positive_mask.to(valid.dtype)
        denominator = valid.sum().clamp_min(1.0)
        router_valid = supervision_mask[:, 0].to(delta_per_positive.dtype)
        router_denominator = router_valid.sum().clamp_min(1.0)

        selected_control = context.control_prototypes[bank_rows, primary_indices]
        selected_child = context.child_tokens[batch, primary_indices]
        selected_cost = cost.gather(1, positive_indices)
        if cap_to_active:
            per_cell_denominator = effective_multiplicity.to(
                delta_per_positive
            ).clamp_min(1.0)
            positive_weight = positive_mask.to(delta_per_positive.dtype)
            delta_per_cell = (
                delta_per_positive * positive_weight
            ).sum(dim=-1) / per_cell_denominator
            direction_per_cell = (
                direction_per_positive * positive_weight
            ).sum(dim=-1) / per_cell_denominator
            magnitude_per_cell = (
                magnitude_per_positive * positive_weight
            ).sum(dim=-1) / per_cell_denominator
            cost_per_cell = (
                selected_cost * positive_weight
            ).sum(dim=-1) / per_cell_denominator
        else:
            delta_per_cell = delta_per_positive.mean(dim=-1)
            direction_per_cell = direction_per_positive.mean(dim=-1)
            magnitude_per_cell = magnitude_per_positive.mean(dim=-1)
            cost_per_cell = selected_cost.mean(dim=-1)
        ot_zero = delta_per_positive.new_zeros(())
        ot_enabled = delta_per_positive.new_tensor(
            float(bool(semi_balanced_ot_enabled))
        )
        ot_balanced = delta_per_positive.new_tensor(
            float(bool(balanced_ot_enabled))
        )
        ot_column_l1 = ot_zero if ot_result is None else ot_result.column_l1
        ot_column_kl = ot_zero if ot_result is None else ot_result.column_kl
        ot_effective_children = (
            ot_zero if ot_result is None else ot_result.effective_children
        )
        ot_cost_scale = ot_zero if ot_result is None else ot_result.cost_scale

        if not gate_enabled:
            router_log_probability = F.log_softmax(
                context.router_logits.float(), dim=-1
            )
            positive_log_probability = router_log_probability.gather(
                1, positive_indices
            )
            if cap_to_active:
                router_per_cell = -(
                    positive_log_probability
                    * positive_mask.to(positive_log_probability.dtype)
                ).sum(dim=-1) / effective_multiplicity.to(
                    positive_log_probability
                ).clamp_min(1.0)
            else:
                router_per_cell = -positive_log_probability.mean(dim=-1)
            router_distribution = torch.softmax(
                context.router_logits.float() / self.temperature, dim=-1
            )
            zero = delta_per_positive.new_zeros(())
            zeros_cell = delta_per_positive.new_zeros(batch_size)
            zeros_positive = delta_per_positive.new_zeros(
                batch_size, multiplicity
            )
            zeros_distribution = delta_per_positive.new_zeros(
                batch_size, context.anchor_mask.shape[1]
            )
            return CellDETRMatch(
                selected_control=selected_control,
                selected_child_tokens=selected_child,
                positive_controls=positive_controls,
                positive_delta=positive_delta,
                positive_prediction=positive_controls + positive_delta,
                child_indices=primary_indices,
                positive_child_indices=positive_indices,
                positive_mask=positive_mask,
                child_distribution=distribution,
                router_distribution=router_distribution,
                delta_per_cell=delta_per_cell,
                direction_per_cell=direction_per_cell,
                magnitude_per_cell=magnitude_per_cell,
                router_per_cell=router_per_cell,
                cost_per_cell=cost_per_cell,
                delta_loss=(delta_per_positive * valid).sum() / denominator,
                direction_loss=(direction_per_positive * valid).sum() / denominator,
                magnitude_loss=(magnitude_per_positive * valid).sum() / denominator,
                router_loss=(router_per_cell * router_valid).sum()
                / router_denominator,
                mean_cost=(selected_cost * valid).sum() / denominator,
                positive_multiplicity=multiplicity,
                effective_positive_multiplicity_min=(
                    effective_positive_multiplicity_min
                ),
                effective_positive_multiplicity_mean=(
                    effective_positive_multiplicity_mean
                ),
                gate_loss=zero,
                latent_match_loss=zero,
                balance_loss=zero,
                relative_gain=zeros_positive,
                gate_target=zeros_positive,
                selected_gate_probability=zeros_positive,
                router_target_distribution=zeros_distribution,
                router_confidence=zeros_cell,
                fallback_target_probability=zeros_cell,
                ot_enabled=ot_enabled,
                ot_column_l1=ot_column_l1,
                ot_column_kl=ot_column_kl,
                ot_effective_children=ot_effective_children,
                ot_cost_scale=ot_cost_scale,
                ot_balanced=ot_balanced,
            )

        required = (
            context.parent_control,
            context.parent_control_feature,
            context.predicted_parent_delta_latent,
            context.child_gate_logits,
            decode_parent_delta,
        )
        if any(item is None for item in required):
            raise ValueError(
                "cold-start matching requires Parent context, gate logits, and decoder"
            )
        parent_control = context.parent_control[:, None]
        parent_delta = decode_parent_delta(context)[:, None]
        parent_observed_delta = targets[:, :1] - parent_control
        (
            parent_delta_value,
            parent_direction,
            parent_magnitude,
            parent_quality_cost,
        ) = self._gene_quality(parent_delta, parent_observed_delta)
        parent_cost_per_cell = parent_quality_cost[:, 0]
        relative_gain = (
            (parent_quality_cost - child_quality_cost)
            / parent_quality_cost.detach().abs().clamp_min(self.gain_cost_floor)
        ).clamp(-2.0, 2.0)
        if force_parent:
            gate_target = torch.zeros_like(relative_gain)
        else:
            gate_target = torch.sigmoid(
                (relative_gain.detach() - float(relative_margin))
                / float(gate_temperature)
            )
        child_training_weight = float(exploration_weight) + (
            1.0 - float(exploration_weight)
        ) * gate_target
        child_training_weight = child_training_weight.detach()

        selected_gate_logits = context.child_gate_logits.gather(1, positive_indices)
        selected_gate_probability = torch.sigmoid(selected_gate_logits)
        gate_per_positive = F.binary_cross_entropy_with_logits(
            selected_gate_logits.float(),
            gate_target.float(),
            reduction="none",
        )
        gate_loss = (gate_per_positive * valid).sum() / denominator

        router_temperature = (
            self.temperature if router_temperature is None else router_temperature
        )
        router_distribution = self._router_probabilities(
            context,
            router_temperature,
            router_occupancy_logit_weight,
        )
        router_log_probability = router_distribution.clamp_min(self.eps).log()
        teacher_mass = gate_target * valid
        mass_sum = teacher_mass.sum(dim=-1, keepdim=True)
        normalized_mass = teacher_mass / mass_sum.clamp_min(self.eps)
        router_target = torch.zeros_like(router_distribution)
        router_target.scatter_add_(1, positive_indices, normalized_mass)
        router_confidence = (
            gate_target.masked_fill(~positive_mask, 0.0)
            .max(dim=-1)
            .values.detach()
            * router_valid
        )
        router_per_cell = -(
            router_target.detach() * router_log_probability
        ).sum(dim=-1)
        router_loss = (
            router_per_cell * router_confidence
        ).sum() / router_denominator
        balance_loss = self._balance_loss(
            context,
            router_distribution,
            balance_target_power,
        )

        parent_latent_cost = self._latent_biological_cost(
            context.parent_control_feature,
            context.predicted_parent_delta_latent,
            target_features[:, 0],
        )
        child_latent_loss = (
            selected_cost * child_training_weight * valid
        ).sum() / denominator
        parent_latent_loss = (
            parent_latent_cost * router_valid
        ).sum() / router_denominator
        latent_match_loss = child_latent_loss + parent_latent_loss

        child_weighted_valid = child_training_weight * valid
        return CellDETRMatch(
            selected_control=selected_control,
            selected_child_tokens=selected_child,
            positive_controls=positive_controls,
            positive_delta=positive_delta,
            positive_prediction=positive_controls + positive_delta,
            child_indices=primary_indices,
            positive_child_indices=positive_indices,
            positive_mask=positive_mask,
            child_distribution=distribution,
            router_distribution=router_distribution,
            delta_per_cell=delta_per_cell,
            direction_per_cell=direction_per_cell,
            magnitude_per_cell=magnitude_per_cell,
            router_per_cell=router_per_cell,
            cost_per_cell=cost_per_cell,
            delta_loss=(delta_per_positive * child_weighted_valid).sum()
            / denominator,
            direction_loss=(direction_per_positive * child_weighted_valid).sum()
            / denominator,
            magnitude_loss=(magnitude_per_positive * child_weighted_valid).sum()
            / denominator,
            router_loss=router_loss,
            mean_cost=(selected_cost * valid).sum() / denominator,
            positive_multiplicity=multiplicity,
            effective_positive_multiplicity_min=(
                effective_positive_multiplicity_min
            ),
            effective_positive_multiplicity_mean=(
                effective_positive_multiplicity_mean
            ),
            parent_control=parent_control,
            parent_delta=parent_delta,
            parent_prediction=parent_control + parent_delta,
            parent_delta_per_cell=parent_delta_value[:, 0],
            parent_direction_per_cell=parent_direction[:, 0],
            parent_magnitude_per_cell=parent_magnitude[:, 0],
            parent_cost_per_cell=parent_cost_per_cell,
            child_quality_cost=child_quality_cost,
            relative_gain=relative_gain,
            gate_target=gate_target,
            selected_gate_probability=selected_gate_probability,
            gate_loss=gate_loss,
            latent_match_loss=latent_match_loss,
            balance_loss=balance_loss,
            router_target_distribution=router_target,
            router_confidence=router_confidence,
            fallback_target_probability=1.0 - router_confidence,
            ot_enabled=ot_enabled,
            ot_column_l1=ot_column_l1,
            ot_column_kl=ot_column_kl,
            ot_effective_children=ot_effective_children,
            ot_cost_scale=ot_cost_scale,
            ot_balanced=ot_balanced,
        )

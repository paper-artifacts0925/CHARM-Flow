"""Zero-init, CV-gated Parent correction built from a safe few-shot artifact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .artifact import FewShotParentArtifact
from .config import FewShotParentAdapterConfig
from .episodes import build_leave_one_line_out_episode
from .ridge import (
    build_leakage_safe_equal_line_prior,
    fit_fewshot_ridge_candidate,
)


@dataclass(frozen=True)
class FewShotParentOutput:
    base_delta: torch.Tensor
    delta: torch.Tensor
    candidate_delta: torch.Tensor
    correction: torch.Tensor
    gate: torch.Tensor
    cv_eligible: torch.Tensor
    relative_cv_gain: torch.Tensor
    query_eligible: torch.Tensor
    audit_condition: torch.Tensor


class ZeroInitReliabilityGate(nn.Module):
    """A bounded straight-through scalar gate with exact zero initialization."""

    def __init__(self, maximum: float = 1.0):
        super().__init__()
        self.maximum = float(maximum)
        if not np.isfinite(self.maximum) or self.maximum < 0:
            raise ValueError("maximum must be finite and non-negative")
        self.raw_amplitude = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        relative_cv_gain: torch.Tensor,
        cv_eligible: torch.Tensor,
        query_eligible: torch.Tensor,
    ) -> torch.Tensor:
        if relative_cv_gain.shape != cv_eligible.shape or cv_eligible.shape != query_eligible.shape:
            raise ValueError("reliability tensors must share one shape")
        # Forward is rigorously bounded while the boundary keeps derivative one.
        clipped = self.raw_amplitude + (
            self.raw_amplitude.clamp(0.0, self.maximum) - self.raw_amplitude
        ).detach()
        # CV gain is logged as reliability evidence; eligibility is the hard
        # safety decision. Multiplying by the often tiny gain would impose an
        # unintended second shrink and prevent the transferred gate from ever
        # reaching the support-fitted candidate.
        _ = relative_cv_gain
        allowed = (cv_eligible & query_eligible).to(clipped)
        return clipped * allowed


def _adapter_cache_from_prior_bank(
    *,
    artifact: FewShotParentArtifact,
    prior_bank,
    config: FewShotParentAdapterConfig,
) -> dict[str, np.ndarray]:
    """Fit deterministic support-only adapters for every target line."""

    config.validate()
    if artifact.parent_sha256 != str(prior_bank.artifact_sha256):
        raise ValueError("few-shot and Parent artifacts do not share provenance")
    if int(config.rank) > int(artifact.max_rank):
        raise ValueError("requested few-shot rank exceeds artifact max_rank")
    target = prior_bank.train_target_delta.detach().cpu().numpy().astype(np.float32)
    train_mask = prior_bank.train_target_mask.detach().cpu().numpy().astype(bool)
    line_count, perturbation_count, gene_dim = target.shape
    if artifact.response_basis.shape[0] != line_count or artifact.response_basis.shape[2] != gene_dim:
        raise ValueError("few-shot artifact dimensions do not match Parent bank")
    if artifact.episodic_eligible_mask.shape != train_mask.shape:
        raise ValueError("few-shot episode mask does not match Parent bank")
    configured_deployment_line = (
        -1
        if config.heldout_deployment_line_id is None
        else int(config.heldout_deployment_line_id)
    )
    if configured_deployment_line != artifact.heldout_deployment_line_id:
        raise ValueError(
            "few-shot config/artifact held-out deployment line mismatch"
        )
    raw_base, prior_mask = build_leakage_safe_equal_line_prior(
        target,
        train_mask,
        artifact.donor_line_ids,
        artifact.donor_line_mask,
    )
    expected_adapter_train = (
        train_mask & ~artifact.audit_perturbation_mask[None, :]
    )
    if not np.array_equal(
        artifact.adapter_train_condition_mask, expected_adapter_train
    ):
        raise ValueError(
            "few-shot artifact adapter-training mask does not match Parent bank"
        )

    # Audit target labels must not be consumed even incidentally. Donor values
    # for an audit query remain valid cross-line predictor inputs; only the
    # target-line response is hidden from basis/ridge/support construction.
    fit_target = target.copy()
    fit_target[artifact.audit_condition_mask] = 0.0

    support_base = np.zeros_like(raw_base, dtype=np.float32)
    correction = np.zeros_like(raw_base, dtype=np.float32)
    query_mask = np.zeros((line_count, perturbation_count), dtype=bool)
    cv_eligible = np.zeros(line_count, dtype=bool)
    relative_gain = np.zeros(line_count, dtype=np.float32)
    base_cv_mse = np.full(line_count, np.nan, dtype=np.float32)
    adapted_cv_mse = np.full(line_count, np.nan, dtype=np.float32)
    selected_lambda = np.full(line_count, np.nan, dtype=np.float32)
    donor_weights = np.zeros((line_count, 3), dtype=np.float32)
    parent_calibration_scale = np.ones(
        (line_count, gene_dim), dtype=np.float32
    )
    parent_calibration_lambda = np.full(
        (line_count, gene_dim), np.nan, dtype=np.float32
    )
    support_ids = np.full((line_count, int(config.support_size)), -1, dtype=np.int64)
    for line in range(line_count):
        episode_train_mask = artifact.adapter_train_condition_mask.copy()
        if configured_deployment_line >= 0 and line != configured_deployment_line:
            episode_train_mask[configured_deployment_line] = False
        episode = build_leave_one_line_out_episode(
            train_mask=episode_train_mask,
            target_line=line,
            support_size=int(config.support_size),
            seed=int(config.seed),
            max_donors=int(config.max_donors),
            support_order=artifact.support_order(line),
        )
        donors = artifact.donor_line_ids[line, artifact.donor_line_mask[line]]
        if not len(donors):
            raise ValueError(f"target line {line} has no leakage-safe donor")
        donor_delta = target[donors].transpose(1, 0, 2)
        donor_mask = train_mask[donors].T
        line_config = FewShotParentAdapterConfig(
            **{
                **config.__dict__,
                "seed": int(config.seed) + line,
            }
        )
        line_target = fit_target[line].copy()
        if line == configured_deployment_line:
            support_mask = np.zeros(perturbation_count, dtype=bool)
            support_mask[episode.support_perturbations] = True
            line_target[~support_mask] = 0.0
        candidate = fit_fewshot_ridge_candidate(
            base_delta=raw_base[line],
            target_delta=line_target,
            donor_delta=donor_delta,
            donor_mask=donor_mask,
            support_perturbations=episode.support_perturbations,
            basis=artifact.response_basis[line, : int(config.rank)],
            config=line_config,
        )
        # Only non-support conditions can train/use the reliability gate.
        allowed_query = (
            ~np.isin(np.arange(perturbation_count), episode.support_perturbations)
            & candidate.source_available
            & prior_mask[line]
        )
        support_base[line] = candidate.base_delta
        correction[line] = candidate.delta - candidate.base_delta
        correction[line, ~allowed_query] = 0.0
        query_mask[line] = allowed_query
        cv_eligible[line] = candidate.fit.cv_eligible
        relative_gain[line] = candidate.fit.relative_cv_gain
        base_cv_mse[line] = candidate.fit.base_cv_mse
        adapted_cv_mse[line] = candidate.fit.adapted_cv_mse
        selected_lambda[line] = candidate.fit.selected_lambda
        donor_weights[line, : len(candidate.fit.donor_weights)] = (
            candidate.fit.donor_weights
        )
        parent_calibration_scale[line] = candidate.parent_calibration_scale
        parent_calibration_lambda[line] = candidate.parent_calibration_lambda
        support_ids[line] = episode.support_perturbations
    return {
        "base_delta": support_base,
        "correction": correction,
        "query_mask": query_mask,
        "cv_eligible": cv_eligible,
        "relative_gain": relative_gain,
        "base_cv_mse": base_cv_mse,
        "adapted_cv_mse": adapted_cv_mse,
        "selected_lambda": selected_lambda,
        "donor_weights": donor_weights,
        "parent_calibration_scale": parent_calibration_scale,
        "parent_calibration_lambda": parent_calibration_lambda,
        "support_ids": support_ids,
    }


class FewShotParentAdapter(nn.Module):
    """Apply a support-only analytic candidate through a learnable safe gate."""

    def __init__(
        self,
        artifact: FewShotParentArtifact,
        prior_bank,
        config: FewShotParentAdapterConfig,
    ):
        super().__init__()
        config.validate()
        cache = _adapter_cache_from_prior_bank(
            artifact=artifact,
            prior_bank=prior_bank,
            config=config,
        )
        self.config = config
        self.artifact_path = artifact.path
        self.artifact_sha256 = artifact.sha256
        self.register_buffer(
            "support_calibrated_base",
            torch.from_numpy(cache["base_delta"]),
            persistent=True,
        )
        self.register_buffer(
            "candidate_correction",
            torch.from_numpy(cache["correction"]),
            persistent=True,
        )
        self.register_buffer(
            "query_mask", torch.from_numpy(cache["query_mask"]), persistent=True
        )
        self.register_buffer(
            "audit_condition_mask",
            torch.from_numpy(artifact.audit_condition_mask.copy()),
            persistent=True,
        )
        self.register_buffer(
            "cv_eligible", torch.from_numpy(cache["cv_eligible"]), persistent=True
        )
        self.register_buffer(
            "relative_cv_gain",
            torch.from_numpy(cache["relative_gain"]),
            persistent=True,
        )
        self.register_buffer(
            "base_cv_mse", torch.from_numpy(cache["base_cv_mse"]), persistent=True
        )
        self.register_buffer(
            "adapted_cv_mse",
            torch.from_numpy(cache["adapted_cv_mse"]),
            persistent=True,
        )
        self.register_buffer(
            "selected_lambda",
            torch.from_numpy(cache["selected_lambda"]),
            persistent=True,
        )
        self.register_buffer(
            "donor_weights", torch.from_numpy(cache["donor_weights"]), persistent=True
        )
        self.register_buffer(
            "parent_calibration_scale",
            torch.from_numpy(cache["parent_calibration_scale"]),
            persistent=True,
        )
        self.register_buffer(
            "parent_calibration_lambda",
            torch.from_numpy(cache["parent_calibration_lambda"]),
            persistent=True,
        )
        # Support IDs are train-only bookkeeping and never needed for sampling.
        self.register_buffer(
            "support_perturbation_ids",
            torch.from_numpy(cache["support_ids"]),
            persistent=False,
        )
        support_condition_mask = torch.zeros_like(
            self.query_mask, dtype=torch.bool
        )
        support_condition_mask.scatter_(
            1, self.support_perturbation_ids, True
        )
        self.register_buffer(
            "support_condition_mask",
            support_condition_mask,
            persistent=True,
        )
        self.heldout_deployment_line_id = int(
            artifact.heldout_deployment_line_id
        )
        self.reliability_gate = ZeroInitReliabilityGate(config.gate_maximum)

    def is_audit(
        self,
        artifact_line_ids: torch.Tensor,
        artifact_perturbation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the frozen audit membership for artifact-aligned IDs."""

        if (
            artifact_line_ids.shape != artifact_perturbation_ids.shape
            or artifact_line_ids.ndim != 1
        ):
            raise ValueError("artifact condition IDs must share shape [B]")
        line = artifact_line_ids.to(
            device=self.audit_condition_mask.device, dtype=torch.long
        )
        perturbation = artifact_perturbation_ids.to(
            device=self.audit_condition_mask.device, dtype=torch.long
        )
        if ((line < 0) | (line >= self.audit_condition_mask.shape[0])).any():
            raise ValueError("artifact cell-line ID is out of range")
        if (
            (perturbation < 0)
            | (perturbation >= self.audit_condition_mask.shape[1])
        ).any():
            raise ValueError("artifact perturbation ID is out of range")
        return self.audit_condition_mask[line, perturbation]

    def training_forbidden(
        self,
        artifact_line_ids: torch.Tensor,
        artifact_perturbation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Audit and deployment non-support rows cannot enter training."""

        audit = self.is_audit(
            artifact_line_ids, artifact_perturbation_ids
        )
        if self.heldout_deployment_line_id < 0:
            return audit
        line = artifact_line_ids.to(
            device=self.support_condition_mask.device, dtype=torch.long
        )
        perturbation = artifact_perturbation_ids.to(
            device=self.support_condition_mask.device, dtype=torch.long
        )
        is_deployment = line == self.heldout_deployment_line_id
        is_support = self.support_condition_mask[line, perturbation]
        return audit | (is_deployment & ~is_support)

    def forward(
        self,
        base_delta: torch.Tensor,
        artifact_line_ids: torch.Tensor,
        artifact_perturbation_ids: torch.Tensor,
    ) -> FewShotParentOutput:
        if base_delta.ndim != 2:
            raise ValueError("base_delta must have shape [B,G]")
        if artifact_line_ids.shape != artifact_perturbation_ids.shape or artifact_line_ids.ndim != 1:
            raise ValueError("artifact condition IDs must share shape [B]")
        line = artifact_line_ids.to(device=base_delta.device, dtype=torch.long)
        perturbation = artifact_perturbation_ids.to(
            device=base_delta.device, dtype=torch.long
        )
        forbidden = self.training_forbidden(line, perturbation).to(
            base_delta.device
        )
        if self.training and forbidden.any():
            raise ValueError(
                "frozen audit or held-out deployment-line non-support "
                "condition entered few-shot adapter training"
            )
        candidate_correction = self.candidate_correction[line, perturbation].to(
            base_delta
        )
        support_base = self.support_calibrated_base[line, perturbation].to(
            base_delta
        )
        eligible = self.cv_eligible[line].to(base_delta.device)
        query = self.query_mask[line, perturbation].to(base_delta.device)
        audit = self.is_audit(line, perturbation).to(base_delta.device)
        gain = self.relative_cv_gain[line].to(base_delta)
        gate = self.reliability_gate(gain, eligible, query)
        correction = gate[:, None] * candidate_correction
        # Ineligible CV or support rows are bitwise base, independent of gate.
        delta = support_base + correction
        return FewShotParentOutput(
            base_delta=support_base,
            delta=delta,
            candidate_delta=support_base + candidate_correction,
            correction=correction,
            gate=gate,
            cv_eligible=eligible,
            relative_cv_gain=gain,
            query_eligible=query,
            audit_condition=audit,
        )


__all__ = [
    "FewShotParentAdapter",
    "FewShotParentOutput",
    "ZeroInitReliabilityGate",
]

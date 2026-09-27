"""Safe Child-to-Parent guidance primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Union

import torch
import torch.nn as nn


Scalar = Union[int, float]


@dataclass(frozen=True)
class PositiveParallelChildGuidance:
    """Auditable result of stop-gradient positive parallel guidance."""

    endpoint: torch.Tensor
    contribution: torch.Tensor
    raw_projection: torch.Tensor
    positive_projection: torch.Tensor
    multiplier: torch.Tensor


def _floating_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value


def _finite_nonnegative_scalar(name: str, value: Scalar) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _unit_scalar(name: str, value: Scalar) -> float:
    parsed = _finite_nonnegative_scalar(name, value)
    if parsed > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return parsed


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def positive_parallel_child_guidance(
    base_parent_mean: torch.Tensor,
    control_mean: torch.Tensor,
    child_proposal: torch.Tensor,
    *,
    gamma: float = 0.1,
    projection_cap: float = 1.0,
    coupling: float = 1.0,
    eps: float = 1.0e-12,
) -> PositiveParallelChildGuidance:
    """Use Child evidence only to increase Parent amplitude along its own direction.

    The Child proposal, Parent geometry used to measure its projection, and the
    resulting coefficient are all stop-gradient. The final Parent delta itself
    remains differentiable, so Parent objectives cannot update Child/Router
    tensors through this operation.
    """

    parent = _floating_tensor("base_parent_mean", base_parent_mean)
    control = _floating_tensor("control_mean", control_mean)
    proposal = _floating_tensor("child_proposal", child_proposal)
    if parent.ndim != 2 or parent.shape[0] < 1 or parent.shape[1] < 1:
        raise ValueError("base_parent_mean must have non-empty shape [B,G]")
    if control.shape != parent.shape or proposal.shape != parent.shape:
        raise ValueError(
            "control_mean and child_proposal must match base_parent_mean [B,G]"
        )
    if control.device != parent.device or proposal.device != parent.device:
        raise ValueError("all tensors must share one device")
    if control.dtype != parent.dtype or proposal.dtype != parent.dtype:
        raise TypeError("all tensors must share one floating-point dtype")
    for name, value in (
        ("base_parent_mean", parent),
        ("control_mean", control),
        ("child_proposal", proposal),
    ):
        _validate_finite(name, value)

    parsed_gamma = _unit_scalar("gamma", gamma)
    parsed_cap = _finite_nonnegative_scalar("projection_cap", projection_cap)
    parsed_coupling = _unit_scalar("coupling", coupling)
    parsed_eps = _finite_nonnegative_scalar("eps", eps)
    if parsed_eps == 0.0:
        raise ValueError("eps must be positive")

    # Only `parent_delta` below retains a graph. In particular, a Parent loss
    # cannot use the projection computation to rotate Parent geometry or update
    # the Child proposal (and therefore its decoder/shared context).
    parent_delta = parent - control.detach()
    geometry = parent_delta.detach()
    evidence = proposal.detach()
    accumulation_dtype = (
        torch.float32
        if parent.dtype in (torch.float16, torch.bfloat16)
        else parent.dtype
    )
    geometry_work = geometry.to(accumulation_dtype)
    evidence_work = evidence.to(accumulation_dtype)
    squared_norm = geometry_work.square().sum(dim=-1, keepdim=True)
    numerator = (evidence_work * geometry_work).sum(dim=-1, keepdim=True)
    raw_projection = torch.where(
        squared_norm < parsed_eps,
        torch.zeros_like(numerator),
        numerator / squared_norm.clamp_min(parsed_eps),
    ).to(parent.dtype)
    positive_projection = raw_projection.clamp(min=0.0, max=parsed_cap).detach()
    amplitude = parsed_coupling * parsed_gamma * positive_projection
    multiplier = (1.0 + amplitude).detach()
    contribution = amplitude * parent_delta
    endpoint = parent + contribution
    return PositiveParallelChildGuidance(
        endpoint=endpoint,
        contribution=contribution,
        raw_projection=raw_projection.detach(),
        positive_projection=positive_projection,
        multiplier=multiplier,
    )


def parent_fill_residual_proposal(
    parent_mean: torch.Tensor,
    child_endpoints: torch.Tensor,
    router_probabilities: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    detach_router_probabilities: bool = True,
) -> torch.Tensor:
    """Return a non-renormalized Top-K Child residual around ``parent_mean``.

    Shapes are ``parent_mean: [B,G]``, ``child_endpoints: [B,K,G]``, and
    ``router_probabilities: [B,K]``.  If ``S`` is the selected Top-K set, the
    proposal is

    ``sum(k in S) probability[k] * (child_endpoint[k] - parent_mean)``.

    This is equivalent to filling all omitted probability mass with the Parent
    itself.  Crucially, selected probabilities are not divided by their sum.
    Router probabilities are detached by default so Parent objectives cannot
    distort the matching policy unless that coupling is explicitly enabled.
    """

    parent_mean = _floating_tensor("parent_mean", parent_mean)
    child_endpoints = _floating_tensor("child_endpoints", child_endpoints)
    router_probabilities = _floating_tensor(
        "router_probabilities", router_probabilities
    )
    if parent_mean.ndim != 2:
        raise ValueError("parent_mean must have shape [B,G]")
    if child_endpoints.ndim != 3:
        raise ValueError("child_endpoints must have shape [B,K,G]")
    if router_probabilities.ndim != 2:
        raise ValueError("router_probabilities must have shape [B,K]")

    batch_size, genes = parent_mean.shape
    endpoint_batch, children, endpoint_genes = child_endpoints.shape
    if batch_size < 1 or genes < 1 or children < 1:
        raise ValueError("B, K, and G dimensions must all be non-empty")
    if endpoint_batch != batch_size or endpoint_genes != genes:
        raise ValueError(
            "child_endpoints must share B and G with parent_mean"
        )
    if router_probabilities.shape != (batch_size, children):
        raise ValueError("router_probabilities must have shape [B,K]")
    if child_endpoints.device != parent_mean.device:
        raise ValueError("all inputs must share one device")
    if router_probabilities.device != parent_mean.device:
        raise ValueError("all inputs must share one device")
    if child_endpoints.dtype != parent_mean.dtype:
        raise TypeError("all inputs must share one floating-point dtype")
    if router_probabilities.dtype != parent_mean.dtype:
        raise TypeError("all inputs must share one floating-point dtype")
    for name, value in (
        ("parent_mean", parent_mean),
        ("child_endpoints", child_endpoints),
        ("router_probabilities", router_probabilities),
    ):
        _validate_finite(name, value)
    if (router_probabilities < 0.0).any():
        raise ValueError("router_probabilities cannot be negative")
    if (router_probabilities > 1.0).any():
        raise ValueError("router_probabilities cannot exceed one")

    # Allow sub-probability rows: their missing mass is also filled by Parent.
    # The tolerance accommodates ordinary softmax summation error.
    tolerance = 32.0 * torch.finfo(parent_mean.dtype).eps * max(children, 1)
    if (router_probabilities.sum(dim=1) > 1.0 + tolerance).any():
        raise ValueError(
            "each router_probabilities row must have total mass at most one"
        )

    if not isinstance(detach_router_probabilities, bool):
        raise TypeError("detach_router_probabilities must be boolean")
    if top_k is None:
        selected_count = children
    else:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise TypeError("top_k must be an integer or None")
        selected_count = int(top_k)
        if not 1 <= selected_count <= children:
            raise ValueError("top_k must lie in [1,K]")

    probabilities = (
        router_probabilities.detach()
        if detach_router_probabilities
        else router_probabilities
    )
    if selected_count == children:
        selected_probabilities = probabilities
        selected_endpoints = child_endpoints
    else:
        selected_probabilities, selected_indices = torch.topk(
            probabilities,
            k=selected_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        selected_endpoints = child_endpoints.gather(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, genes),
        )

    return (
        selected_probabilities.unsqueeze(-1)
        * (selected_endpoints - parent_mean.unsqueeze(1))
    ).sum(dim=1)


def rms_trust_clip(
    proposal: torch.Tensor,
    reference_delta: torch.Tensor,
    max_rms_ratio: float,
    *,
    eps: float = 1.0e-12,
    detach_scale: bool = True,
) -> torch.Tensor:
    """Clip proposal RMS relative to a trusted reference delta, row by row.

    The returned tensor obeys approximately
    ``RMS(output) <= max_rms_ratio * RMS(reference_delta)``.  A zero reference
    therefore permits no Child correction.  By default the clipping scale is
    detached: the forward trust region remains exact while gradients keep the
    proposal's useful direction instead of differentiating through its norm.
    """

    proposal = _floating_tensor("proposal", proposal)
    reference_delta = _floating_tensor("reference_delta", reference_delta)
    if proposal.ndim < 1:
        raise ValueError("proposal must have at least one dimension")
    if proposal.shape != reference_delta.shape:
        raise ValueError("reference_delta must have the same shape as proposal")
    if proposal.shape[-1] < 1:
        raise ValueError("the gene dimension must be non-empty")
    if proposal.device != reference_delta.device:
        raise ValueError("proposal and reference_delta must share one device")
    if proposal.dtype != reference_delta.dtype:
        raise TypeError("proposal and reference_delta must share one dtype")
    _validate_finite("proposal", proposal)
    _validate_finite("reference_delta", reference_delta)
    ratio = _finite_nonnegative_scalar("max_rms_ratio", max_rms_ratio)
    epsilon = _finite_nonnegative_scalar("eps", eps)
    if epsilon == 0.0:
        raise ValueError("eps must be positive")
    if not isinstance(detach_scale, bool):
        raise TypeError("detach_scale must be boolean")
    if ratio == 0.0:
        return torch.zeros_like(proposal)

    accumulation_dtype = (
        torch.float32
        if proposal.dtype in (torch.float16, torch.bfloat16)
        else proposal.dtype
    )
    proposal_for_norm = proposal.to(dtype=accumulation_dtype)
    reference_for_norm = reference_delta.to(dtype=accumulation_dtype)
    proposal_rms = proposal_for_norm.square().mean(dim=-1, keepdim=True).sqrt()
    reference_rms = reference_for_norm.square().mean(dim=-1, keepdim=True).sqrt()
    allowed_rms = ratio * reference_rms
    scale = torch.where(
        proposal_rms <= allowed_rms,
        torch.ones_like(proposal_rms),
        allowed_rms / proposal_rms.clamp_min(epsilon),
    ).clamp(min=0.0, max=1.0)
    if detach_scale:
        scale = scale.detach()
    return proposal * scale.to(dtype=proposal.dtype)


class ConditionReliabilityGate(nn.Module):
    """Zero-output-initialized bounded condition-level reliability gate.

    The final projection is initialized to zero, hence every condition starts
    with ``alpha == 0`` and reproduces the historical Parent exactly.  The
    positive half of ``tanh`` bounds the learned reliability to
    ``[0, max_alpha)``.  A straight-through floor preserves that exact forward
    boundary while retaining the ``tanh`` derivative for negative logits; an
    temporarily closed condition can therefore reopen if later evidence says
    its Child proposal is useful.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 64,
        max_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral):
            raise TypeError("feature_dim must be an integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral):
            raise TypeError("hidden_dim must be an integer")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.max_alpha = _finite_nonnegative_scalar("max_alpha", max_alpha)
        if self.max_alpha == 0.0:
            raise ValueError("max_alpha must be positive")
        if self.max_alpha > 1.0:
            raise ValueError("max_alpha must lie in (0, 1]")

        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.activation = nn.SiLU()
        self.reliability_projection = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.reliability_projection.weight)
        nn.init.zeros_(self.reliability_projection.bias)

    def forward(self, condition_features: torch.Tensor) -> torch.Tensor:
        condition_features = _floating_tensor(
            "condition_features", condition_features
        )
        if condition_features.ndim < 2:
            raise ValueError(
                "condition_features must have shape [...,feature_dim]"
            )
        if condition_features.shape[-1] != self.feature_dim:
            raise ValueError(
                "condition_features last dimension must equal feature_dim"
            )
        _validate_finite("condition_features", condition_features)
        hidden = self.activation(
            self.feature_projection(self.feature_norm(condition_features))
        )
        logits = self.reliability_projection(hidden)
        raw_reliability = torch.tanh(logits)
        clipped_reliability = torch.clamp(
            raw_reliability, min=0.0, max=1.0
        )
        # Exact clipped value in the forward pass, smooth tanh derivative in
        # the backward pass.  Ordinary clamp would make every negative logit
        # permanently dead because its derivative is zero there.
        unit_reliability = raw_reliability + (
            clipped_reliability - raw_reliability
        ).detach()
        return self.max_alpha * unit_reliability


def coupling_at_step(
    step: int,
    *,
    start_step: int = 0,
    warmup_steps: int = 0,
    max_coupling: float = 1.0,
) -> float:
    """Return a deterministic linear Child-to-Parent coupling schedule."""

    parsed_step = _nonnegative_integer("step", step)
    parsed_start = _nonnegative_integer("start_step", start_step)
    parsed_warmup = _nonnegative_integer("warmup_steps", warmup_steps)
    target = _unit_scalar("max_coupling", max_coupling)
    if parsed_step < parsed_start:
        return 0.0
    if parsed_warmup == 0:
        return target
    progress = min(
        float(parsed_step - parsed_start) / float(parsed_warmup),
        1.0,
    )
    return target * progress


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient * ctx.scale, None


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Preserve exact forward values while scaling only the backward gradient."""

    if not torch.is_tensor(value):
        raise TypeError("value must be a torch.Tensor")
    parsed_scale = _unit_scalar("scale", scale)
    return _ScaleGradient.apply(value, parsed_scale)


def _condition_alpha(
    alpha: Union[float, torch.Tensor],
    parent_mean: torch.Tensor,
) -> Union[float, torch.Tensor]:
    if torch.is_tensor(alpha):
        alpha = _floating_tensor("alpha", alpha)
        if alpha.device != parent_mean.device:
            raise ValueError("alpha must share the Parent device")
        if alpha.dtype != parent_mean.dtype:
            raise TypeError("alpha must share the Parent dtype")
        _validate_finite("alpha", alpha)
        if ((alpha < 0.0) | (alpha > 1.0)).any():
            raise ValueError("alpha must lie in [0, 1]")
        batch_size = parent_mean.shape[0]
        if alpha.ndim == 0:
            return alpha
        if alpha.shape == (batch_size,):
            return alpha.unsqueeze(-1)
        if alpha.shape == (batch_size, 1):
            return alpha
        raise ValueError("alpha must be scalar, [B], or [B,1]")
    return _unit_scalar("alpha", alpha)


def compose_child_guided_parent(
    parent_mean: torch.Tensor,
    reference_parent_delta: Optional[torch.Tensor] = None,
    child_endpoints: Optional[torch.Tensor] = None,
    router_probabilities: Optional[torch.Tensor] = None,
    *,
    enabled: bool = False,
    alpha: Union[float, torch.Tensor] = 0.0,
    coupling: float = 0.0,
    top_k: Optional[int] = None,
    max_rms_ratio: float = 0.1,
    detach_router_probabilities: bool = True,
    gradient_scale: float = 1.0,
) -> torch.Tensor:
    """Conservatively compose a Child-guided Parent prediction.

    Disabled, scalar-zero ``alpha``, and scalar-zero ``coupling`` paths return
    ``parent_mean`` itself before inspecting any optional Child tensors.  A
    tensor-valued zero ``alpha`` keeps its computation graph, but still produces
    a Parent tensor exactly equal in value to the historical prediction.
    """

    parent_mean = _floating_tensor("parent_mean", parent_mean)
    if parent_mean.ndim != 2:
        raise ValueError("parent_mean must have shape [B,G]")
    if parent_mean.shape[0] < 1 or parent_mean.shape[1] < 1:
        raise ValueError("B and G dimensions must be non-empty")
    _validate_finite("parent_mean", parent_mean)
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    if not enabled:
        return parent_mean

    parsed_coupling = _unit_scalar("coupling", coupling)
    if parsed_coupling == 0.0:
        return parent_mean
    if not torch.is_tensor(alpha):
        parsed_alpha = _unit_scalar("alpha", alpha)
        if parsed_alpha == 0.0:
            return parent_mean
    else:
        parsed_alpha = _condition_alpha(alpha, parent_mean)

    if reference_parent_delta is None:
        raise ValueError("reference_parent_delta is required when enabled")
    if child_endpoints is None:
        raise ValueError("child_endpoints is required when enabled")
    if router_probabilities is None:
        raise ValueError("router_probabilities is required when enabled")
    reference_parent_delta = _floating_tensor(
        "reference_parent_delta", reference_parent_delta
    )
    if reference_parent_delta.shape != parent_mean.shape:
        raise ValueError(
            "reference_parent_delta must have the same [B,G] shape as parent_mean"
        )
    if reference_parent_delta.device != parent_mean.device:
        raise ValueError("reference_parent_delta must share the Parent device")
    if reference_parent_delta.dtype != parent_mean.dtype:
        raise TypeError("reference_parent_delta must share the Parent dtype")
    _validate_finite("reference_parent_delta", reference_parent_delta)
    parsed_gradient_scale = _unit_scalar("gradient_scale", gradient_scale)

    proposal = parent_fill_residual_proposal(
        parent_mean,
        child_endpoints,
        router_probabilities,
        top_k=top_k,
        detach_router_probabilities=detach_router_probabilities,
    )
    clipped_proposal = rms_trust_clip(
        proposal,
        reference_parent_delta,
        max_rms_ratio,
    )
    clipped_proposal = scale_gradient(clipped_proposal, parsed_gradient_scale)
    return parent_mean + parsed_coupling * parsed_alpha * clipped_proposal


__all__ = [
    "ConditionReliabilityGate",
    "PositiveParallelChildGuidance",
    "compose_child_guided_parent",
    "coupling_at_step",
    "parent_fill_residual_proposal",
    "positive_parallel_child_guidance",
    "rms_trust_clip",
    "scale_gradient",
]

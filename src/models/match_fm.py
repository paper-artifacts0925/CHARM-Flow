"""Core MATCH-FM components for latent control-state inference."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_distribution(values, mask, eps=1e-8):
    mask = mask.to(device=values.device, dtype=torch.bool)
    values = torch.where(mask, values.clamp_min(eps), torch.zeros_like(values))
    return values / values.sum(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def set_constrained_posterior(
    energy,
    inference_prior,
    source_mask=None,
    temperature=1.0,
    marginal_strength=1.0,
    iterations=12,
    damping=0.5,
    row_prior_scale=0.0,
    eps=1e-8,
):
    """Infer detached cell assignments with a set-level marginal constraint.

    The optimized normalized objective is the mean assignment energy plus
    entropy and ``marginal_strength * KL(mean_cell(Q) || inference_prior)``.
    Fixed-point updates are damped because the marginal correction depends on
    the assignments themselves.
    """
    if energy.ndim != 3:
        raise ValueError("energy must have shape [B,S,K]")
    if inference_prior.ndim != 2 or inference_prior.shape != (
        energy.shape[0],
        energy.shape[-1],
    ):
        raise ValueError("inference_prior must have shape [B,K]")
    if source_mask is None:
        source_mask = torch.ones_like(inference_prior, dtype=torch.bool)
    else:
        source_mask = source_mask.to(device=energy.device, dtype=torch.bool)
    if source_mask.shape != inference_prior.shape:
        raise ValueError("source_mask and inference_prior must have identical shapes")
    if not source_mask.any(dim=-1).all():
        raise ValueError("every set must contain at least one valid source")

    tau = max(float(temperature), eps)
    strength = max(float(marginal_strength), 0.0)
    damping = min(max(float(damping), 0.0), 0.999)
    prior = _masked_distribution(inference_prior.detach().to(energy), source_mask, eps)
    mask = source_mask.unsqueeze(1)
    base_logits = -energy.detach() / tau
    if float(row_prior_scale) != 0.0:
        base_logits = base_logits + float(row_prior_scale) * prior.clamp_min(eps).log().unsqueeze(1)
    base_logits = base_logits.masked_fill(~mask, -torch.inf)
    q = torch.softmax(base_logits, dim=-1)
    q = torch.where(mask, q, torch.zeros_like(q))

    correction_scale = strength / tau
    for _ in range(max(int(iterations), 0)):
        marginal = q.mean(dim=1)
        correction = correction_scale * (
            prior.clamp_min(eps).log() - marginal.clamp_min(eps).log()
        )
        updated = torch.softmax(
            (base_logits + correction.unsqueeze(1)).masked_fill(~mask, -torch.inf),
            dim=-1,
        )
        updated = torch.where(mask, updated, torch.zeros_like(updated))
        q = damping * q + (1.0 - damping) * updated
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q.detach()


def posterior_prior_kl(responsibilities, inference_prior, source_mask=None, eps=1e-8):
    """Distill a detached aggregate posterior into the inference prior."""
    if responsibilities.ndim != 3 or inference_prior.ndim != 2:
        raise ValueError("responsibilities and prior must be [B,S,K] and [B,K]")
    if responsibilities.shape[0] != inference_prior.shape[0] or responsibilities.shape[-1] != inference_prior.shape[-1]:
        raise ValueError("responsibilities and prior dimensions disagree")
    if source_mask is None:
        source_mask = torch.ones_like(inference_prior, dtype=torch.bool)
    target = _masked_distribution(
        responsibilities.detach().mean(dim=1),
        source_mask,
        eps,
    )
    prior = _masked_distribution(inference_prior, source_mask, eps)
    per_sample = (
        target
        * (target.clamp_min(eps).log() - prior.clamp_min(eps).log())
    ).sum(dim=-1)
    return per_sample, target


def distribution_kl(distribution, reference, source_mask=None, eps=1e-8):
    """Return KL(distribution || reference) over valid latent sources."""
    if distribution.ndim != 2 or reference.shape != distribution.shape:
        raise ValueError("distribution and reference must have shape [B,K]")
    if source_mask is None:
        source_mask = torch.ones_like(distribution, dtype=torch.bool)
    p = _masked_distribution(distribution, source_mask, eps)
    q = _masked_distribution(reference.to(p), source_mask, eps)
    return (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=-1)


@torch.no_grad()
def confidence_shrink_posterior(
    responsibilities,
    fallback_prior,
    source_mask=None,
    power=1.0,
    max_confidence=1.0,
    eps=1e-8,
):
    """Shrink uncertain cell assignments toward an inference-safe prior."""
    if responsibilities.ndim != 3 or fallback_prior.ndim != 2:
        raise ValueError("responsibilities and prior must be [B,S,K] and [B,K]")
    if responsibilities.shape[0] != fallback_prior.shape[0] or responsibilities.shape[-1] != fallback_prior.shape[-1]:
        raise ValueError("responsibilities and prior dimensions disagree")
    if source_mask is None:
        source_mask = torch.ones_like(fallback_prior, dtype=torch.bool)
    mask = source_mask.to(device=responsibilities.device, dtype=torch.bool)
    prior = _masked_distribution(fallback_prior.detach().to(responsibilities), mask, eps)
    q = torch.where(mask.unsqueeze(1), responsibilities.detach(), torch.zeros_like(responsibilities))
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(q * q.clamp_min(eps).log()).sum(dim=-1)
    valid_count = mask.sum(dim=-1).clamp_min(1).to(q.dtype)
    max_entropy = valid_count.log().unsqueeze(1)
    relative_entropy = torch.where(
        max_entropy > eps,
        entropy / max_entropy.clamp_min(eps),
        torch.zeros_like(entropy),
    )
    confidence = (1.0 - relative_entropy.clamp(0.0, 1.0)).pow(float(power))
    confidence = confidence.mul(float(max_confidence)).clamp(0.0, 1.0)
    shrunk = confidence.unsqueeze(-1) * q + (1.0 - confidence.unsqueeze(-1)) * prior.unsqueeze(1)
    shrunk = torch.where(mask.unsqueeze(1), shrunk, torch.zeros_like(shrunk))
    shrunk = shrunk / shrunk.sum(dim=-1, keepdim=True).clamp_min(eps)
    return shrunk.detach(), confidence.detach()


@torch.no_grad()
def deterministic_prior_assignments(prior, num_cells, source_mask=None, offset=None):
    """Allocate cells to sources deterministically while matching each prior."""
    if prior.ndim != 2:
        raise ValueError("prior must have shape [B,K]")
    if source_mask is None:
        source_mask = torch.ones_like(prior, dtype=torch.bool)
    prior = _masked_distribution(prior, source_mask, eps=1e-8)
    num_cells = int(num_cells)
    if num_cells < 1:
        raise ValueError("num_cells must be positive")
    grid = (torch.arange(num_cells, device=prior.device, dtype=prior.dtype) + 0.5) / num_cells
    grid = grid.unsqueeze(0).expand(prior.shape[0], -1)
    if offset is not None:
        offset = offset.to(device=prior.device, dtype=prior.dtype).reshape(-1, 1)
        grid = torch.remainder(grid + offset, 1.0)
    cdf = prior.cumsum(dim=-1)
    return torch.searchsorted(cdf.contiguous(), grid.contiguous(), right=False).clamp_max(prior.shape[-1] - 1)


class ControlStateSetEncoder(nn.Module):
    """Encode cluster token sets using mean, variance, and rare-token pooling."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, control_sets, token_mask=None):
        if control_sets.ndim != 4:
            raise ValueError("control_sets must have shape [B,K,T,G]")
        encoded = self.token_encoder(control_sets.float())
        if token_mask is None:
            token_mask = torch.ones(encoded.shape[:-1], device=encoded.device, dtype=torch.bool)
        token_mask = token_mask.to(device=encoded.device, dtype=torch.bool)
        if token_mask.shape != encoded.shape[:-1]:
            raise ValueError("token_mask must have shape [B,K,T]")
        weights = token_mask.unsqueeze(-1).to(encoded.dtype)
        count = weights.sum(dim=2).clamp_min(1.0)
        mean = (encoded * weights).sum(dim=2) / count
        variance = ((encoded - mean.unsqueeze(2)).square() * weights).sum(dim=2) / count
        masked = encoded.masked_fill(~token_mask.unsqueeze(-1), -torch.inf)
        rare_pool = masked.amax(dim=2)
        rare_pool = torch.where(torch.isfinite(rare_pool), rare_pool, torch.zeros_like(rare_pool))
        return self.output(torch.cat([mean, variance.clamp_min(1e-6).sqrt(), rare_pool], dim=-1))


class PerturbationConditionedPrior(nn.Module):
    """Learn r(z | perturbation, cell line, control-state set)."""

    def __init__(
        self,
        input_dim,
        num_perturbations,
        num_cell_types,
        num_cluster_embeddings=64,
        hidden_dim=64,
        empirical_prior_scale=1.0,
        semantic_context_dim=None,
        parent_context_dim=None,
        use_cluster_identity=True,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.empirical_prior_scale = float(empirical_prior_scale)
        self.use_cluster_identity = bool(use_cluster_identity)
        self.perturbation_embedding = nn.Embedding(int(num_perturbations) + 1, hidden_dim)
        self.cell_type_embedding = nn.Embedding(int(num_cell_types), hidden_dim)
        self.cluster_embedding = nn.Embedding(int(num_cluster_embeddings), hidden_dim)
        self.state_encoder = ControlStateSetEncoder(input_dim, hidden_dim)
        self.query = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.semantic_projection = (
            nn.Linear(int(semantic_context_dim), hidden_dim)
            if semantic_context_dim is not None else None
        )
        self.parent_projection = (
            nn.Linear(int(parent_context_dim), hidden_dim)
            if parent_context_dim is not None else None
        )
        self.content_query_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        perturbation_ids,
        cell_type_ids,
        control_sets,
        empirical_prior,
        source_mask,
        cluster_ids=None,
        token_mask=None,
        semantic_context=None,
        parent_context=None,
    ):
        if perturbation_ids.ndim > 1:
            perturbation_ids = perturbation_ids[:, 0]
        if cell_type_ids.ndim > 1:
            cell_type_ids = cell_type_ids[:, 0]
        perturbation_ids = perturbation_ids.long().clamp(
            min=0,
            max=self.perturbation_embedding.num_embeddings - 1,
        )
        cell_type_ids = cell_type_ids.long().clamp(
            min=0,
            max=self.cell_type_embedding.num_embeddings - 1,
        )
        state = self.state_encoder(control_sets, token_mask=token_mask)
        identity_query = self.query(
            torch.cat(
                [
                    self.perturbation_embedding(perturbation_ids),
                    self.cell_type_embedding(cell_type_ids),
                ],
                dim=-1,
            )
        )
        query = identity_query
        used_content = False
        if semantic_context is not None and self.semantic_projection is not None:
            semantic_context = semantic_context.to(device=state.device)
            if semantic_context.ndim == 3:
                semantic_context = semantic_context.mean(dim=1)
            if semantic_context.ndim != 2 or semantic_context.shape[0] != state.shape[0]:
                raise ValueError("semantic_context must have shape [B,D] or [B,S,D]")
            query = self.semantic_projection(
                semantic_context.to(dtype=self.semantic_projection.weight.dtype)
            ) + 0.0 * identity_query
            used_content = True
        if parent_context is not None and self.parent_projection is not None:
            parent_context = parent_context.to(device=state.device)
            if parent_context.ndim == 3 and parent_context.shape[1] == 1:
                parent_context = parent_context[:, 0]
            if parent_context.ndim != 2 or parent_context.shape[0] != state.shape[0]:
                raise ValueError("parent_context must have shape [B,H] or [B,1,H]")
            query = query + self.parent_projection(
                parent_context.to(dtype=self.parent_projection.weight.dtype)
            )
            used_content = True
        if used_content:
            query = self.content_query_norm(query)
        query = query.unsqueeze(1).expand(-1, state.shape[1], -1)
        if cluster_ids is None:
            cluster_ids = torch.arange(state.shape[1], device=state.device).unsqueeze(0).expand(state.shape[0], -1)
        cluster_ids = cluster_ids.long().clamp(min=0, max=self.cluster_embedding.num_embeddings - 1)
        cluster_feature = self.cluster_embedding(cluster_ids)
        if not self.use_cluster_identity:
            cluster_feature = 0.0 * cluster_feature
        learned_logits = self.score(
            torch.cat([query, state, cluster_feature], dim=-1)
        ).squeeze(-1)
        base = self.empirical_prior_scale * empirical_prior.to(learned_logits).clamp_min(1e-8).log()
        logits = (base + learned_logits).masked_fill(~source_mask.bool(), -torch.inf)
        return F.softmax(logits, dim=-1)


__all__ = [
    "ControlStateSetEncoder",
    "PerturbationConditionedPrior",
    "confidence_shrink_posterior",
    "deterministic_prior_assignments",
    "distribution_kl",
    "posterior_prior_kl",
    "set_constrained_posterior",
]

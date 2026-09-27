"""Hierarchical control-state context and child proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AdaptiveAnchorOutput:
    """Container returned by :class:`AdaptiveStateAnchor`."""

    candidates: torch.Tensor
    centroids: torch.Tensor
    prior: torch.Tensor
    mask: torch.Tensor
    cluster_ids: torch.Tensor
    objectness_logits: torch.Tensor
    objectness: torch.Tensor
    offsets: torch.Tensor
    empirical_parent_prior: torch.Tensor
    parent_index: torch.Tensor
    parent_context: torch.Tensor
    parent_tokens: torch.Tensor
    parent_centroids: torch.Tensor
    parent_memory_tokens: torch.Tensor | None
    parent_memory_mask: torch.Tensor | None
    perturbation_query: torch.Tensor
    child_keys: torch.Tensor
    barycenter_weights: torch.Tensor
    active_mask: torch.Tensor
    parent_ids: torch.Tensor
    leaf_ids: torch.Tensor
    child_ids: torch.Tensor
    child_slot_ids: torch.Tensor


class AdaptiveStateAnchor(nn.Module):
    """Build a cell-line context and an overcomplete child proposal bank.

    ``split_factor`` is retained as the public configuration name for backward
    compatibility, but semantically it is ``max_children_per_parent``.  Parent
    IDs and cell-line IDs are identity metadata only; they never enter the
    learned representation.  Thus a local cluster number cannot accidentally
    acquire a shared biological meaning across cell lines.

    For each parent, learned child queries attend to its observed control
    tokens.  The attention distribution defines both a content-derived child
    key and a control-token barycenter.  ``candidates`` contains a
    child-conditioned convex transport of the same real control tokens, while
    ``centroids`` contains the exact soft barycenter used as the child
    baseline.  Parent representations are returned separately and are not
    included among the treated matching candidates.
    """

    def __init__(
        self,
        gene_dim: int,
        num_perturbations: int,
        num_cell_types: int,
        num_cluster_embeddings: int = 64,
        hidden_dim: int = 128,
        offset_rank: int = 16,
        split_factor: int = 2,
        offset_scale: float = 0.1,
        objectness_threshold: float = 0.5,
        transport_identity_bias: float = 2.0,
        memory_num_patches: int = 0,
        memory_program_rank: int = 4,
    ) -> None:
        super().__init__()
        if split_factor < 1:
            raise ValueError("split_factor must be >= 1")
        self.gene_dim = int(gene_dim)
        self.hidden_dim = int(hidden_dim)
        self.offset_rank = int(offset_rank)
        self.split_factor = int(split_factor)
        self.max_children_per_parent = self.split_factor
        self.offset_scale = float(offset_scale)
        self.objectness_threshold = float(objectness_threshold)
        self.transport_identity_bias = float(transport_identity_bias)
        self.memory_num_patches = int(memory_num_patches)
        self.memory_program_rank = int(memory_program_rank)
        if self.memory_num_patches < 0:
            raise ValueError("memory_num_patches must be >= 0")
        if self.memory_num_patches > 0 and self.memory_program_rank < 1:
            raise ValueError("memory_program_rank must be >= 1")

        self.state_norm = nn.LayerNorm(self.gene_dim)
        self.state_encoder = nn.Linear(self.gene_dim, self.hidden_dim)
        # Keep cardinalities as metadata for API/checkpoint diagnostics.  IDs
        # are deliberately not embedded: identity is not biological content.
        self.num_cell_types = int(num_cell_types)
        self.num_cluster_embeddings = int(num_cluster_embeddings)
        self.perturbation_embedding = nn.Embedding(
            int(num_perturbations), self.hidden_dim
        )
        self.split_embedding = nn.Embedding(self.split_factor, self.hidden_dim)

        self.dispersion_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.parent_norm = nn.LayerNorm(self.hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.memory_num_patches > 0:
            # Each patch is a small learned gene program rather than a slice
            # of the arbitrary input-gene order. Mean and dispersion preserve
            # complementary views of the control distribution.
            self.gene_program_basis = nn.Parameter(
                torch.empty(
                    self.memory_num_patches,
                    self.memory_program_rank,
                    self.gene_dim,
                )
            )
            nn.init.normal_(
                self.gene_program_basis,
                std=1.0 / math.sqrt(float(self.gene_dim)),
            )
            self.memory_patch_embedding = nn.Parameter(
                torch.empty(self.memory_num_patches, self.hidden_dim)
            )
            nn.init.normal_(self.memory_patch_embedding, std=0.02)
            self.memory_patch_projection = nn.Sequential(
                nn.LayerNorm(2 * self.memory_program_rank),
                nn.Linear(2 * self.memory_program_rank, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.memory_token_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.register_parameter("gene_program_basis", None)
            self.register_parameter("memory_patch_embedding", None)
            self.memory_patch_projection = None
            self.memory_token_norm = None
        self.control_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.child_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.child_content_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.objectness_head = nn.Linear(self.hidden_dim, 1)
        self.route_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.route_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.route_scale = nn.Parameter(torch.tensor(0.1))
        self.match_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.split_objectness_bias = nn.Parameter(
            torch.linspace(1.0, -0.25, self.split_factor)
        )

        nn.init.zeros_(self.objectness_head.weight)
        nn.init.zeros_(self.objectness_head.bias)

    @staticmethod
    def _first_token_id(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            return values.long()
        return values[:, 0].long()

    def forward(
        self,
        candidates: torch.Tensor,
        empirical_prior: torch.Tensor,
        source_mask: torch.Tensor,
        perturbation_ids: torch.Tensor,
        cell_type_ids: torch.Tensor,
        cluster_ids: torch.Tensor | None = None,
        parent_ids: torch.Tensor | None = None,
    ) -> AdaptiveAnchorOutput:
        if candidates.ndim != 4:
            raise ValueError("candidates must have shape [B,K,S,G]")
        batch_size, num_parents, set_size, gene_dim = candidates.shape
        if gene_dim != self.gene_dim:
            raise ValueError(
                f"candidate gene dimension {gene_dim} != configured {self.gene_dim}"
            )
        if empirical_prior.shape != (batch_size, num_parents):
            raise ValueError("empirical_prior must have shape [B,K]")
        if source_mask.shape != (batch_size, num_parents):
            raise ValueError("source_mask must have shape [B,K]")

        valid_parent = source_mask.to(device=candidates.device, dtype=torch.bool)
        if not valid_parent.any(dim=-1).all():
            raise ValueError("every sample needs at least one control-state seed")
        parent_prior = empirical_prior.to(candidates).masked_fill(~valid_parent, 0.0)
        parent_prior = parent_prior / parent_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        candidates_float = candidates.float()
        leaf_centroids = candidates_float.mean(dim=2)
        if cluster_ids is None:
            leaf_ids = torch.arange(
                num_parents, device=candidates.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            if cluster_ids.shape != (batch_size, num_parents):
                raise ValueError("cluster_ids must have shape [B,K]")
            leaf_ids = cluster_ids.to(device=candidates.device, dtype=torch.long)
        if parent_ids is None:
            coarse_parent_ids = leaf_ids
        else:
            if parent_ids.shape != (batch_size, num_parents):
                raise ValueError("parent_ids must have shape [B,K]")
            coarse_parent_ids = parent_ids.to(device=candidates.device, dtype=torch.long)
        pert_ids = self._first_token_id(perturbation_ids).to(candidates.device)
        pert_ids = pert_ids.clamp(
            min=0, max=self.perturbation_embedding.num_embeddings - 1
        )
        # Validate the legacy argument without making IDs semantic features.
        cell_ids = self._first_token_id(cell_type_ids).to(candidates.device)
        if cell_ids.shape != (batch_size,):
            raise ValueError("cell_type_ids must have batch dimension B")

        token_feature = self.state_encoder(self.state_norm(candidates_float))
        token_feature = token_feature.to(dtype=self.state_encoder.weight.dtype)
        leaf_mean = token_feature.mean(dim=2)
        leaf_dispersion = token_feature.var(dim=2, unbiased=False).add(1e-6).sqrt()
        leaf_tokens = self.parent_norm(
            leaf_mean + self.dispersion_projection(leaf_dispersion)
        )
        leaf_tokens = leaf_tokens.masked_fill(~valid_parent.unsqueeze(-1), 0.0)

        # Aggregate leaf summaries sharing a coarse hierarchy prefix.
        same_parent = coarse_parent_ids.unsqueeze(2).eq(coarse_parent_ids.unsqueeze(1))
        same_parent = same_parent & (
            valid_parent.unsqueeze(2) & valid_parent.unsqueeze(1)
        )
        aggregation_weights = (
            same_parent.to(parent_prior) * parent_prior.unsqueeze(1)
        )
        aggregation_weights = aggregation_weights / (
            aggregation_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        parent_tokens = torch.einsum("bkj,bjh->bkh", aggregation_weights, leaf_tokens)
        parent_centroids = torch.einsum(
            "bkj,bjg->bkg", aggregation_weights.float(), leaf_centroids
        )

        # A weighted DeepSets-style root token describes the whole control
        # landscape and is invariant to parent ordering.
        parent_context = (
            parent_tokens * parent_prior.unsqueeze(-1).to(parent_tokens)
        ).sum(dim=1)
        parent_context = self.context_projection(parent_context)

        parent_memory_tokens = None
        parent_memory_mask = None
        if self.memory_num_patches > 0:
            leaf_second_moment = candidates_float.square().mean(dim=2)
            control_mean = torch.einsum(
                "bk,bkg->bg", parent_prior.float(), leaf_centroids
            )
            control_second_moment = torch.einsum(
                "bk,bkg->bg", parent_prior.float(), leaf_second_moment
            )
            control_dispersion = (
                control_second_moment - control_mean.square()
            ).clamp_min(1e-6).sqrt()
            program_basis = F.normalize(
                self.gene_program_basis.float(), dim=-1, eps=1e-6
            )
            mean_coefficients = torch.einsum(
                "bg,prg->bpr", control_mean, program_basis
            )
            dispersion_coefficients = torch.einsum(
                "bg,prg->bpr", control_dispersion, program_basis
            )
            patch_statistics = torch.cat(
                [mean_coefficients, dispersion_coefficients], dim=-1
            ).to(dtype=self.memory_patch_projection[1].weight.dtype)
            patch_tokens = self.memory_patch_projection(patch_statistics)
            patch_tokens = self.memory_token_norm(
                patch_tokens + self.memory_patch_embedding.unsqueeze(0)
            )
            cls_token = self.memory_token_norm(parent_context).unsqueeze(1)
            parent_memory_tokens = torch.cat([cls_token, patch_tokens], dim=1)
            parent_memory_mask = torch.ones(
                batch_size,
                self.memory_num_patches + 1,
                device=candidates.device,
                dtype=torch.bool,
            )

        split_ids = torch.arange(
            self.split_factor, device=candidates.device, dtype=torch.long
        )
        proposal_seed = (
            leaf_tokens.unsqueeze(2)
            + parent_tokens.unsqueeze(2)
            + parent_context[:, None, None, :]
            + self.split_embedding(split_ids)[None, None, :, :]
        )
        proposal_seed = F.silu(self.feature_norm(proposal_seed))

        # Each child matches control tokens inside its parent, so its baseline
        # remains inside the convex hull of observed controls.
        token_keys = F.normalize(self.control_key(token_feature), dim=-1, eps=1e-6)
        proposal_queries = F.normalize(
            self.child_query(proposal_seed), dim=-1, eps=1e-6
        )
        match_logits = torch.einsum(
            "bkmh,bksh->bkms", proposal_queries, token_keys
        )
        match_logits = match_logits * self.match_logit_scale.exp().clamp(max=100.0)
        barycenter_weights = F.softmax(match_logits, dim=-1)
        refined_centroids = torch.einsum(
            "bkms,bksg->bkmg", barycenter_weights.float(), candidates_float
        )

        matched_content = torch.einsum(
            "bkms,bksh->bkmh", barycenter_weights, token_feature
        )
        feature = F.silu(
            self.feature_norm(
                proposal_seed + self.child_content_projection(matched_content)
            )
        )
        child_keys = F.normalize(self.route_key(feature), dim=-1, eps=1e-6)

        # Preserve cell variability with a child-specific convex transport of
        # observed controls; the identity bias keeps individual control cells.
        log_barycenter_weights = barycenter_weights.float().clamp_min(1e-8).log()
        identity = torch.eye(
            set_size, device=candidates.device, dtype=log_barycenter_weights.dtype
        )
        transport_logits = (
            log_barycenter_weights.unsqueeze(-2)
            + self.transport_identity_bias * identity[None, None, None, :, :]
        )
        transport = F.softmax(transport_logits, dim=-1)
        refined_candidates = torch.einsum(
            "bkmst,bktg->bkmsg", transport, candidates_float
        )

        offsets = refined_centroids - parent_centroids.unsqueeze(2)

        objectness_logits = self.objectness_head(feature).squeeze(-1)
        objectness_logits = objectness_logits + self.split_objectness_bias[None, None]
        objectness = torch.sigmoid(objectness_logits)
        route_query = F.normalize(
            self.route_query(self.perturbation_embedding(pert_ids)), dim=-1, eps=1e-6
        ).unsqueeze(1).unsqueeze(2)
        route_logits = (route_query * child_keys).sum(dim=-1)
        route_logits = self.route_scale.tanh() * route_logits

        expanded_mask = valid_parent.unsqueeze(-1).expand(-1, -1, self.split_factor)
        slot_weight = (
            parent_prior.unsqueeze(-1)
            * objectness
            * torch.exp(route_logits.clamp(min=-5.0, max=5.0))
        )
        slot_weight = slot_weight.masked_fill(~expanded_mask, 0.0)
        slot_prior = slot_weight / slot_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)

        # Thresholding controls effective K; top-1 fallback prevents a valid
        # parent from losing every child.  Structural mask stays DDP-stable.
        threshold_active = expanded_mask & objectness.ge(self.objectness_threshold)
        best_child = objectness.masked_fill(~expanded_mask, -1.0).argmax(dim=2)
        fallback_active = F.one_hot(
            best_child, num_classes=self.split_factor
        ).to(dtype=torch.bool)
        fallback_active = fallback_active & valid_parent.unsqueeze(-1)
        active_mask = threshold_active | fallback_active

        out_k = num_parents * self.split_factor
        parent_index = torch.arange(
            num_parents, device=candidates.device, dtype=torch.long
        ).view(1, num_parents, 1).expand(batch_size, -1, self.split_factor)
        expanded_parent_ids = coarse_parent_ids.unsqueeze(-1).expand(
            -1, -1, self.split_factor
        )
        expanded_leaf_ids = leaf_ids.unsqueeze(-1).expand(
            -1, -1, self.split_factor
        )
        expanded_child_slot_ids = split_ids.view(1, 1, -1).expand(
            batch_size, num_parents, -1
        )
        stable_child_ids = (
            expanded_leaf_ids * self.split_factor + expanded_child_slot_ids
        )
        return AdaptiveAnchorOutput(
            candidates=refined_candidates.to(dtype=candidates.dtype).reshape(
                batch_size, out_k, set_size, gene_dim
            ),
            centroids=refined_centroids.to(dtype=candidates.dtype).reshape(
                batch_size, out_k, gene_dim
            ),
            prior=slot_prior.reshape(batch_size, out_k),
            mask=expanded_mask.reshape(batch_size, out_k),
            cluster_ids=stable_child_ids.reshape(batch_size, out_k),
            objectness_logits=objectness_logits.reshape(batch_size, out_k),
            objectness=objectness.reshape(batch_size, out_k),
            offsets=offsets.to(dtype=candidates.dtype).reshape(
                batch_size, out_k, gene_dim
            ),
            empirical_parent_prior=parent_prior,
            parent_index=parent_index.reshape(batch_size, out_k),
            parent_context=parent_context,
            parent_tokens=parent_tokens,
            parent_centroids=parent_centroids.to(dtype=candidates.dtype),
            parent_memory_tokens=parent_memory_tokens,
            parent_memory_mask=parent_memory_mask,
            perturbation_query=self.perturbation_embedding(pert_ids),
            child_keys=child_keys.reshape(batch_size, out_k, self.hidden_dim),
            barycenter_weights=barycenter_weights.reshape(
                batch_size, out_k, set_size
            ),
            active_mask=active_mask.reshape(batch_size, out_k),
            parent_ids=expanded_parent_ids.reshape(batch_size, out_k),
            leaf_ids=expanded_leaf_ids.reshape(batch_size, out_k),
            child_ids=stable_child_ids.reshape(batch_size, out_k),
            child_slot_ids=expanded_child_slot_ids.reshape(batch_size, out_k),
        )


def masked_cell_mean(values: torch.Tensor, valid_cell_mask: torch.Tensor) -> torch.Tensor:
    """Reduce a ``[B,S,...]`` tensor over valid cells only."""
    valid = valid_cell_mask.to(device=values.device, dtype=values.dtype)
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def safe_root_mean_square(values: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Compute RMS with a finite derivative for exactly zero vectors."""
    squared_mean = values.float().square().mean(dim=dim)
    return (squared_mean + float(eps)).sqrt()

def posterior_prior_distillation(
    responsibilities: torch.Tensor,
    adaptive_prior: torch.Tensor,
    valid_cell_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill detached posterior occupancy into the learned anchor prior."""
    posterior = masked_cell_mean(responsibilities.detach(), valid_cell_mask)
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prior = adaptive_prior.clamp_min(1e-8)
    loss = (posterior * (posterior.clamp_min(1e-8).log() - prior.log())).sum(dim=-1)
    return loss.mean(), posterior


def anchor_parent_prior_kl(output: AdaptiveAnchorOutput) -> torch.Tensor:
    """Keep total child occupancy close to each empirical parent occupancy."""
    batch_size, num_slots = output.prior.shape
    num_parents = output.empirical_parent_prior.shape[-1]
    parent_mass = output.prior.new_zeros(batch_size, num_parents)
    parent_mass.scatter_add_(1, output.parent_index, output.prior)
    empirical = output.empirical_parent_prior.clamp_min(1e-8)
    return (
        empirical * (empirical.log() - parent_mass.clamp_min(1e-8).log())
    ).sum(dim=-1).mean()


def anchor_sibling_diversity(
    output: AdaptiveAnchorOutput,
    split_factor: int,
    margin: float = 0.005,
) -> torch.Tensor:
    """Prevent sibling offsets from remaining exact duplicates."""
    if int(split_factor) <= 1:
        return output.offsets.new_zeros(())
    batch_size, num_slots, gene_dim = output.offsets.shape
    num_parents = num_slots // int(split_factor)
    offsets = output.offsets.reshape(batch_size, num_parents, int(split_factor), gene_dim)
    distances = []
    for left in range(int(split_factor)):
        for right in range(left + 1, int(split_factor)):
            distances.append(
                safe_root_mean_square(
                    offsets[:, :, left] - offsets[:, :, right], dim=-1
                )
            )
    distance = torch.stack(distances, dim=-1)
    parent_valid = output.mask.reshape(
        batch_size, num_parents, int(split_factor)
    ).any(dim=-1)
    penalty = F.relu(float(margin) - distance).mean(dim=-1)
    valid = parent_valid.to(dtype=penalty.dtype)
    return (penalty * valid).sum() / valid.sum().clamp_min(1.0)


__all__ = [
    "AdaptiveAnchorOutput",
    "AdaptiveStateAnchor",
    "anchor_parent_prior_kl",
    "anchor_sibling_diversity",
    "masked_cell_mean",
    "posterior_prior_distillation",
    "safe_root_mean_square",
]

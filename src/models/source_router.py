"""Learned perturbation-to-source router for broken symmetry in soft-EM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceRouter(nn.Module):
    """Predict source routing logits from perturbation expression + DiT embedding.

    Design rationale
    ----------------
    The detached-loss EM path alone can get stuck in a uniform equilibrium
    when the backbone's source conditioning is too weak.  The Router provides
    a second, complementary gradient path that breaks symmetry:

      perturbation expression → pert_repr → router_logits → gumbel_topk_q
      detached candidate_loss  → em_scores → topk_softmax → em_q

    The two are fused and calibrated through a bidirectional KL loss.
    """

    def __init__(
        self,
        gene_dim: int = 2000,
        hidden_dim: int = 768,
        num_sources: int = 8,
        class_emb_dim: int = None,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.hidden_dim = hidden_dim
        self.class_emb_dim = hidden_dim if class_emb_dim is None else int(class_emb_dim)

        # perturbation expression → compact representation
        self.pert_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # combined (pert_repr + class_embedding) → source logits
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + self.class_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_sources),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [self.pert_encoder, self.router]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # final router layer: zero init for stable start
        last_linear = None
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, pert_cells, class_emb):
        """
        Parameters
        ----------
        pert_cells : (B, set_size, gene_dim)
            Perturbation expression cells (used for mean-pool).
        class_emb : (B, hidden_dim), (B, set_size, hidden_dim), or source-expanded
            (B * num_sources, class_emb_dim)/(B * num_sources, set_size, class_emb_dim).
            DiT class_embedding output (perturbation + cell_line context).

        Returns
        -------
        router_logits : (B, num_sources)
        """
        batch_size = pert_cells.shape[0]
        pert_repr = pert_cells.mean(dim=1)  # (B, gene_dim)
        pert_repr = self.pert_encoder(pert_repr)  # (B, hidden_dim)

        if class_emb.ndim == 3:
            class_emb = class_emb.mean(dim=1)
        if class_emb.ndim != 2:
            raise ValueError("class_emb must be 2D or 3D")
        if class_emb.shape[0] == batch_size * self.num_sources:
            class_emb = class_emb.reshape(
                batch_size,
                self.num_sources,
                class_emb.shape[-1],
            ).mean(dim=1)
        elif class_emb.shape[0] != batch_size:
            raise ValueError(
                "class_emb batch dimension must match pert_cells or "
                "be source-expanded by num_sources"
            )

        combined = torch.cat([pert_repr, class_emb], dim=-1)
        return self.router(combined)  # (B, num_sources)


def gumbel_softmax_topk(
    logits,
    k,
    temperature=1.0,
    hard=False,
    mask=None,
    eps=1e-8,
):
    """Differentiable top-k selection via Gumbel-Softmax.

    Returns a probability vector where only the top-k positions are non-zero.
    Uses the Straight-Through Gumbel estimator for gradient flow.

    Parameters
    ----------
    logits : (..., K) — unnormalized log-probabilities
    k : int — number of elements to select
    temperature : float — Gumbel softmax temperature
    hard : bool — if True, return one-hot; soft otherwise
    mask : (..., K) or None — boolean mask for valid sources
    eps : float — numerical epsilon

    Returns
    -------
    q : (..., K) — normalized top-k probabilities
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.inf)

    # Gumbel noise
    gumbel = -torch.empty_like(logits).exponential_().log()
    gumbel = (logits + gumbel) / max(temperature, 1e-6)

    # Argmax top-k
    _, topk_indices = torch.topk(gumbel, k=k, dim=-1)

    if hard:
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(dim=-1, index=topk_indices, value=1.0)
        y_hard = y_hard - logits.detach() + logits  # straight-through
        q = y_hard / k
    else:
        # Soft top-k: Gumbel softmax for top-k positions only
        y_soft = torch.zeros_like(logits)
        topk_values = torch.gather(gumbel, dim=-1, index=topk_indices)
        topk_soft = F.softmax(topk_values, dim=-1).to(dtype=y_soft.dtype)
        y_soft.scatter_(dim=-1, index=topk_indices, src=topk_soft)
        q = y_soft

    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q


def compute_router_loss(router_logits, target_q, mask=None, eps=1e-8):
    """Bidirectional KL between router prediction and target responsibility.

    Parameters
    ----------
    router_logits : (B, K)
    target_q : (B, K) — target distribution (e.g. fused q)
    mask : (B, K) or None

    Returns
    -------
    L_router : scalar
    """
    router_probs = F.softmax(router_logits, dim=-1)
    target_q = target_q.detach()

    if mask is not None:
        router_probs = router_probs * mask
        router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        target_q = target_q * mask
        target_q = target_q / target_q.sum(dim=-1, keepdim=True).clamp_min(eps)

    # KL(router || target) + KL(target || router)
    kl_rt = (target_q * (target_q.clamp_min(eps).log() - router_probs.clamp_min(eps).log())).sum(dim=-1)
    kl_tr = (router_probs * (router_probs.clamp_min(eps).log() - target_q.clamp_min(eps).log())).sum(dim=-1)
    return (kl_rt + kl_tr).mean()


def compute_usage_loss(q, num_sources, eps=1e-8):
    """Entropy-based penalty for source under-utilisation.

    Returns KL(uniform || per_source_usage).
    Zero when all sources are used equally.

    Parameters
    ----------
    q : (B, K) — responsibility matrix
    num_sources : int

    Returns
    -------
    L_usage : scalar
    """
    per_source_usage = q.mean(dim=0)  # (K,)
    uniform = torch.full_like(per_source_usage, 1.0 / num_sources)
    return (per_source_usage * (per_source_usage.clamp_min(eps).log() - uniform.log())).sum()

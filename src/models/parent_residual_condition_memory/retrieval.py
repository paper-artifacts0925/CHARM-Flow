"""Numerically stable one-query retrieval over a complete Child bank."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ChunkedQueryChildAttention(nn.Module):
    """Exact multi-head attention from one perturbation query to all Children.

    Attention normalization is accumulated online in float32.  This keeps the
    result exact across chunks, avoids a ``[B, H, Q, K]`` activation, and is
    safe when the surrounding module runs in bf16.  Attention maps are only
    materialized when ``return_weights=True``.

    The returned token is ``query + residual(attended_children)``.  The final
    residual projection is zero initialized, so attaching this module cannot
    perturb a warm-started query before it has learned a useful retrieval.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        chunk_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.chunk_size = int(chunk_size)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        self.query_norm = nn.LayerNorm(self.hidden_dim, eps=float(norm_eps))
        self.child_norm = nn.LayerNorm(self.hidden_dim, eps=float(norm_eps))
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.reset_residual_projection()

    def reset_residual_projection(self) -> None:
        """Restore an exact identity warm start for the retrieval residual."""

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _keys_values(self, children: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = children.shape
        children = self.child_norm(children)
        keys = self.key_projection(children).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        values = self.value_projection(children).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        return keys.transpose(1, 2), values.transpose(1, 2)

    def _chunk_logits(
        self,
        query_heads: torch.Tensor,
        children: torch.Tensor,
    ) -> torch.Tensor:
        keys, _ = self._keys_values(children)
        scale = 1.0 / math.sqrt(float(self.head_dim))
        return (query_heads.unsqueeze(2).float() * keys.float()).sum(dim=-1) * scale

    def forward(
        self,
        query: torch.Tensor,
        children: torch.Tensor,
        child_mask: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.ndim != 2 or query.shape[1] != self.hidden_dim:
            raise ValueError(f"query must have shape [B,{self.hidden_dim}]")
        if children.ndim != 3 or children.shape[0] != query.shape[0]:
            raise ValueError("children must have shape [B,K,D]")
        if children.shape[2] != self.hidden_dim:
            raise ValueError(f"children must end in hidden_dim={self.hidden_dim}")
        if child_mask.dtype != torch.bool or child_mask.shape != children.shape[:2]:
            raise ValueError("child_mask must be boolean with shape [B,K]")
        if not child_mask.any(dim=1).all():
            raise ValueError("every batch row needs at least one active Child")
        if query.device != children.device or query.device != child_mask.device:
            raise ValueError("query, children, and child_mask must share one device")
        if query.dtype != children.dtype:
            raise ValueError("query and children must share one dtype")

        batch, child_count, _ = children.shape
        query_heads = self.query_projection(self.query_norm(query)).reshape(
            batch, self.num_heads, self.head_dim
        )

        # Online log-sum-exp attention.  Only one Child chunk is promoted to
        # float32 at a time, keeping bf16 training stable without a large copy.
        running_max = torch.full(
            (batch, self.num_heads),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            dtype=torch.float32,
            device=query.device,
        )
        scale = 1.0 / math.sqrt(float(self.head_dim))

        for start in range(0, child_count, self.chunk_size):
            stop = min(start + self.chunk_size, child_count)
            active = child_mask[:, start:stop]
            keys, values = self._keys_values(children[:, start:stop])
            logits = (
                query_heads.unsqueeze(2).float() * keys.float()
            ).sum(dim=-1) * scale
            logits = logits.masked_fill(~active.unsqueeze(1), -torch.inf)
            chunk_max = logits.amax(dim=-1)
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - new_max),
                torch.zeros_like(new_max),
            )
            shifted = logits - new_max.unsqueeze(-1)
            chunk_weights = torch.where(
                active.unsqueeze(1),
                torch.exp(shifted),
                torch.zeros_like(shifted),
            )
            numerator = numerator * old_scale.unsqueeze(-1)
            numerator = numerator + torch.einsum(
                "bhk,bhkd->bhd", chunk_weights, values.float()
            )
            denominator = denominator * old_scale + chunk_weights.sum(dim=-1)
            running_max = new_max

        attended = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        attended = attended.to(query.dtype).reshape(batch, self.hidden_dim)
        retrieved = query + self.output_projection(attended)

        weights = None
        if return_weights:
            # Diagnostics are deliberately a second pass.  Normal training
            # avoids retaining K-wide logits in the autograd graph.
            logits_chunks = []
            for start in range(0, child_count, self.chunk_size):
                stop = min(start + self.chunk_size, child_count)
                logits_chunks.append(
                    self._chunk_logits(query_heads, children[:, start:stop])
                )
            logits = torch.cat(logits_chunks, dim=-1)
            logits = logits.masked_fill(~child_mask.unsqueeze(1), -torch.inf)
            weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return retrieved, weights


__all__ = ["ChunkedQueryChildAttention"]

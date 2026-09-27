"""Factorized Parent correction tied to Gene-DiT response-gene identities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TiedGeneParentCorrection(nn.Module):
    """Decode a condition hidden state through shared response-gene keys.

    The query projection is the sole zero-output boundary.  Gene keys start
    nonzero, so the query weight receives a useful gradient on the first step;
    key and (when live) gene-identity gradients open after the query moves.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        gene_identity_dim: int,
        rank: int = 64,
        detach_gene_identity: bool = True,
        norm_eps: float = 1.0e-6,
    ):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("gene_identity_dim", gene_identity_dim),
            ("rank", rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(detach_gene_identity, bool):
            raise TypeError("detach_gene_identity must be boolean")
        if not math.isfinite(float(norm_eps)) or float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be finite and positive")

        self.hidden_dim = int(hidden_dim)
        self.gene_identity_dim = int(gene_identity_dim)
        self.rank = int(rank)
        self.detach_gene_identity = detach_gene_identity
        self.norm_eps = float(norm_eps)
        self.gene_identity_norm = nn.LayerNorm(
            self.gene_identity_dim,
            eps=self.norm_eps,
            elementwise_affine=False,
        )
        self.key_projection = nn.Linear(
            self.gene_identity_dim,
            self.rank,
            bias=False,
        )
        self.query_projection = nn.Linear(
            self.hidden_dim,
            self.rank,
            bias=False,
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Restore the exact zero raw-correction boundary."""

        nn.init.zeros_(self.query_projection.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        gene_identity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(hidden)
            or not hidden.is_floating_point()
            or hidden.ndim != 2
            or hidden.shape[1] != self.hidden_dim
        ):
            raise ValueError(
                f"hidden must be floating point with shape [B,{self.hidden_dim}]"
            )
        if (
            not torch.is_tensor(gene_identity)
            or not gene_identity.is_floating_point()
            or gene_identity.ndim != 2
            or gene_identity.shape[1] != self.gene_identity_dim
        ):
            raise ValueError(
                "gene_identity must be floating point with shape "
                f"[G,{self.gene_identity_dim}]"
            )
        if hidden.device != gene_identity.device:
            raise ValueError("hidden and gene_identity must share one device")
        if not torch.isfinite(hidden).all() or not torch.isfinite(
            gene_identity
        ).all():
            raise ValueError("hidden and gene_identity must be finite")

        identity = (
            gene_identity.detach()
            if self.detach_gene_identity
            else gene_identity
        )
        identity = identity.to(dtype=self.key_projection.weight.dtype)
        key = self.key_projection(self.gene_identity_norm(identity))
        # RMS normalization keeps every key at L2~=sqrt(rank).  Combined with
        # the standard 1/sqrt(rank) dot-product scale this avoids both an
        # uncontrolled key norm and the extra 1/sqrt(rank) shrinkage caused by
        # unit-L2 normalization.
        key = key.float()
        key_rms = key.square().mean(dim=-1, keepdim=True).add(
            self.norm_eps
        ).sqrt()
        key = key / key_rms
        query = self.query_projection(
            hidden.to(dtype=self.query_projection.weight.dtype)
        )
        raw = torch.einsum("br,gr->bg", query.float(), key)
        return (raw * (1.0 / math.sqrt(float(self.rank)))).to(hidden)


__all__ = ["TiedGeneParentCorrection"]

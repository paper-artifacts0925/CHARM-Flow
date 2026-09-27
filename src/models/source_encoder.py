"""Encoder for latent control-source distribution statistics."""

import torch
import torch.nn as nn


class SourceEncoder(nn.Module):
    """Encode centroid, diagonal variance, and prior into source tokens."""

    def __init__(self, gene_dim=2000, hidden_dim=768):
        super().__init__()
        self.gene_dim = int(gene_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.gene_dim + 1, 1024),
            nn.GELU(),
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.cell_line_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )

    def forward(self, centroid, variance, prior, cell_line_context=None):
        if centroid.shape != variance.shape:
            raise ValueError("centroid and variance must have identical shapes")
        if centroid.shape[-1] != self.gene_dim:
            raise ValueError(
                f"expected gene_dim={self.gene_dim}, got {centroid.shape[-1]}"
            )
        if prior.shape != centroid.shape[:-1]:
            raise ValueError(
                f"prior shape {tuple(prior.shape)} does not match "
                f"source shape {tuple(centroid.shape[:-1])}"
            )
        stats = torch.cat(
            [centroid, variance, prior.unsqueeze(-1)],
            dim=-1,
        )
        token = self.encoder(stats)
        if cell_line_context is not None:
            context = cell_line_context
            while context.ndim < token.ndim:
                context = context.unsqueeze(-2)
            token = token + self.cell_line_projection(context)
        return token


__all__ = ["SourceEncoder"]

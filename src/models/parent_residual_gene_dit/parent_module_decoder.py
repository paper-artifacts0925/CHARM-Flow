"""Child-aware gene-module decoder for the analytic Parent correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import SwiGLU
from .tokenizer import SparseModuleGeneDecoder


@dataclass(frozen=True)
class ChildAwareParentModuleDecoderConfig:
    """Dimensions for :class:`ChildAwareParentModuleDecoder`."""

    gene_dim: int = 2000
    num_modules: int = 128
    context_dim: int = 128
    hidden_dim: int = 256
    num_latent_queries: int = 4
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    decoder_chunk_size: int = 256
    norm_eps: float = 1.0e-6
    router_log_probability_beta: float = 0.5

    def validate(self) -> "ChildAwareParentModuleDecoderConfig":
        for name in (
            "gene_dim",
            "num_modules",
            "context_dim",
            "hidden_dim",
            "num_latent_queries",
            "depth",
            "num_heads",
            "decoder_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(self.norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive")
        if (
            not torch.isfinite(torch.tensor(float(self.router_log_probability_beta)))
            or float(self.router_log_probability_beta) < 0.0
        ):
            raise ValueError(
                "router_log_probability_beta must be finite and non-negative"
            )
        return self


class _PreNormCrossAttentionBlock(nn.Module):
    """Cross-attention followed by a pre-norm SwiGLU residual."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        norm_eps: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.memory_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.num_heads = int(num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.mlp = SwiGLU(hidden_dim, mlp_ratio, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_mask = None
        if attention_bias is not None:
            expected = (query.shape[0], query.shape[1], memory.shape[1])
            if tuple(attention_bias.shape) != expected:
                raise ValueError(
                    "attention_bias must have shape [B,Q,K]="
                    f"{list(expected)}"
                )
            attention_mask = attention_bias.repeat_interleave(
                self.num_heads, dim=0
            )
        padding_mask = memory_key_padding_mask
        if padding_mask is not None and attention_mask is not None:
            padding_mask = torch.zeros_like(
                padding_mask, dtype=attention_mask.dtype
            ).masked_fill(padding_mask, -torch.inf)
        update = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        query = query + self.residual_dropout(update)
        return query + self.residual_dropout(self.mlp(self.mlp_norm(query)))


class ChildAwareParentModuleDecoder(nn.Module):
    """Decode a bounded raw Parent correction from all valid Child states.

    Four perturbation-conditioned latent queries retrieve the full Child bank.
    The resulting consensus joins the four semantic tokens used by the legacy
    correction head (Parent, perturbation, routed, analytic-prior feature).
    Fixed module queries then read this memory through two cross-attention
    blocks.  :class:`SparseModuleGeneDecoder` reuses the artifact module IDs,
    and its sole gene output projection starts at exact zero.
    """

    _SEMANTIC_TOKEN_COUNT = 4

    def __init__(
        self,
        module_ids: torch.Tensor,
        config: ChildAwareParentModuleDecoderConfig | None = None,
        *,
        detach_child_context: bool = True,
    ):
        super().__init__()
        if not isinstance(detach_child_context, bool):
            raise TypeError("detach_child_context must be boolean")
        self.config = (
            ChildAwareParentModuleDecoderConfig() if config is None else config
        ).validate()
        self.detach_child_context = detach_child_context
        cfg = self.config

        self.child_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.occupancy_projection = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.latent_seed_projection = nn.Linear(
            3 * cfg.context_dim, cfg.hidden_dim
        )
        self.latent_queries = nn.Parameter(
            torch.empty(cfg.num_latent_queries, cfg.hidden_dim)
        )
        self.child_consensus = _PreNormCrossAttentionBlock(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            norm_eps=cfg.norm_eps,
        )

        self.semantic_projections = nn.ModuleList(
            [
                nn.Linear(cfg.context_dim, cfg.hidden_dim)
                for _ in range(self._SEMANTIC_TOKEN_COUNT)
            ]
        )
        self.semantic_type_embedding = nn.Parameter(
            torch.empty(self._SEMANTIC_TOKEN_COUNT, cfg.hidden_dim)
        )
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.module_queries = nn.Parameter(
            torch.empty(cfg.num_modules, cfg.hidden_dim)
        )
        self.module_decoder_blocks = nn.ModuleList(
            [
                _PreNormCrossAttentionBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    norm_eps=cfg.norm_eps,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.gene_decoder = SparseModuleGeneDecoder(
            module_ids,
            gene_dim=cfg.gene_dim,
            num_modules=cfg.num_modules,
            hidden_dim=cfg.hidden_dim,
            chunk_size=cfg.decoder_chunk_size,
            norm_eps=cfg.norm_eps,
        )
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02)
        nn.init.normal_(self.module_queries, std=0.02)

    def reset_output_projection(self) -> None:
        """Restore the one function-preserving boundary to exact zero."""

        self.gene_decoder.reset_output_projection()

    @staticmethod
    def _validate_context_feature(
        value: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        context_dim: int,
        device: torch.device,
    ) -> None:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if tuple(value.shape) != (batch_size, context_dim):
            raise ValueError(f"{name} must have shape [B,{context_dim}]")
        if value.device != device:
            raise ValueError("all Parent decoder inputs must share one device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

    def _validate_inputs(
        self,
        *,
        parent_token: torch.Tensor,
        perturbation_query: torch.Tensor,
        routed_token: torch.Tensor,
        prior_feature: torch.Tensor,
        child_tokens: torch.Tensor,
        child_mask: torch.Tensor,
        occupancy: torch.Tensor,
        router_probabilities: torch.Tensor,
    ) -> None:
        cfg = self.config
        if not torch.is_tensor(child_tokens) or not child_tokens.is_floating_point():
            raise TypeError("child_tokens must be a floating-point tensor")
        if child_tokens.ndim != 3 or child_tokens.shape[2] != cfg.context_dim:
            raise ValueError(
                f"child_tokens must have shape [B,K,{cfg.context_dim}]"
            )
        batch_size, child_count, _ = child_tokens.shape
        if child_count < 1:
            raise ValueError("child_tokens must contain at least one Child")
        if child_mask.dtype != torch.bool or tuple(child_mask.shape) != (
            batch_size,
            child_count,
        ):
            raise ValueError("child_mask must be boolean with shape [B,K]")
        if not torch.is_tensor(occupancy) or not occupancy.is_floating_point():
            raise TypeError("occupancy must be a floating-point tensor")
        if tuple(occupancy.shape) != (batch_size, child_count):
            raise ValueError("occupancy must have shape [B,K]")
        if (
            not torch.is_tensor(router_probabilities)
            or not router_probabilities.is_floating_point()
        ):
            raise TypeError("router_probabilities must be a floating-point tensor")
        if tuple(router_probabilities.shape) != (batch_size, child_count):
            raise ValueError("router_probabilities must have shape [B,K]")
        device = child_tokens.device
        if (
            child_mask.device != device
            or occupancy.device != device
            or router_probabilities.device != device
        ):
            raise ValueError("all Parent decoder inputs must share one device")
        if not child_mask.any(dim=1).all():
            raise ValueError("every row must contain at least one valid Child")
        if not torch.isfinite(child_tokens).all():
            raise ValueError("child_tokens must be finite")
        if not torch.isfinite(occupancy).all() or (occupancy < 0.0).any():
            raise ValueError("occupancy must be finite and non-negative")
        if (
            not torch.isfinite(router_probabilities).all()
            or (router_probabilities < 0.0).any()
        ):
            raise ValueError(
                "router_probabilities must be finite and non-negative"
            )
        if (occupancy.masked_select(~child_mask) > 1.0e-7).any():
            raise ValueError("occupancy cannot place mass on a padded Child")
        active_mass = occupancy.masked_fill(~child_mask, 0.0).sum(dim=1)
        if (active_mass <= 0.0).any():
            raise ValueError("each row must place positive mass on valid Children")
        if (router_probabilities.masked_select(~child_mask) > 1.0e-7).any():
            raise ValueError(
                "router_probabilities cannot place mass on a padded Child"
            )
        router_mass = router_probabilities.masked_fill(
            ~child_mask, 0.0
        ).sum(dim=1)
        if (router_mass <= 0.0).any():
            raise ValueError(
                "each row must route positive mass to valid Children"
            )
        for name, value in (
            ("parent_token", parent_token),
            ("perturbation_query", perturbation_query),
            ("routed_token", routed_token),
            ("prior_feature", prior_feature),
        ):
            self._validate_context_feature(
                value,
                name=name,
                batch_size=batch_size,
                context_dim=cfg.context_dim,
                device=device,
            )

    def forward(
        self,
        *,
        parent_token: torch.Tensor,
        perturbation_query: torch.Tensor,
        routed_token: torch.Tensor,
        prior_feature: torch.Tensor,
        child_tokens: torch.Tensor,
        child_mask: torch.Tensor,
        occupancy: torch.Tensor,
        router_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            parent_token=parent_token,
            perturbation_query=perturbation_query,
            routed_token=routed_token,
            prior_feature=prior_feature,
            child_tokens=child_tokens,
            child_mask=child_mask,
            occupancy=occupancy,
            router_probabilities=router_probabilities,
        )
        cfg = self.config
        batch_size, child_count, _ = child_tokens.shape
        if self.detach_child_context:
            child_tokens = child_tokens.detach()
            occupancy = occupancy.detach()
        # Matching/router probabilities are always a detached teacher for the
        # Parent consensus; the correction loss must not optimize the Router.
        router_probabilities = router_probabilities.detach()

        active_occupancy = occupancy.masked_fill(~child_mask, 0.0)
        active_occupancy = active_occupancy / active_occupancy.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        # Relative log occupancy preserves the useful long tail while clipping
        # an empty/rare state's numerical leverage before the learned embedding.
        log_occupancy = (
            active_occupancy.mul(float(child_count))
            .clamp_min(1.0e-8)
            .log()
            .clamp(min=-8.0, max=8.0)
        )
        child_memory = self.child_projection(child_tokens)
        child_memory = child_memory + self.occupancy_projection(
            log_occupancy.unsqueeze(-1).to(child_memory)
        )
        child_memory = child_memory.masked_fill(~child_mask.unsqueeze(-1), 0.0)

        active_router = router_probabilities.masked_fill(~child_mask, 0.0)
        active_router = active_router / active_router.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        router_attention_bias = (
            float(cfg.router_log_probability_beta)
            * active_router.clamp_min(1.0e-8).log()
        )
        router_attention_bias = router_attention_bias[:, None, :].expand(
            -1, cfg.num_latent_queries, -1
        )

        seed_condition = self.latent_seed_projection(
            torch.cat(
                [parent_token, perturbation_query, prior_feature], dim=-1
            )
        )
        latent = self.latent_queries.to(seed_condition).unsqueeze(0)
        latent = latent.expand(batch_size, -1, -1) + seed_condition.unsqueeze(1)
        latent = self.child_consensus(
            latent,
            child_memory,
            memory_key_padding_mask=~child_mask,
            attention_bias=router_attention_bias.to(child_memory),
        )

        semantic_values = (
            parent_token,
            perturbation_query,
            routed_token,
            prior_feature,
        )
        semantic_tokens = torch.stack(
            [
                projection(value)
                for projection, value in zip(
                    self.semantic_projections, semantic_values
                )
            ],
            dim=1,
        )
        semantic_tokens = semantic_tokens + self.semantic_type_embedding.to(
            semantic_tokens
        ).unsqueeze(0)
        memory = self.memory_norm(torch.cat([semantic_tokens, latent], dim=1))

        module_tokens = self.module_queries.to(memory).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        for block in self.module_decoder_blocks:
            module_tokens = block(module_tokens, memory)
        return self.gene_decoder(module_tokens)


__all__ = [
    "ChildAwareParentModuleDecoder",
    "ChildAwareParentModuleDecoderConfig",
]

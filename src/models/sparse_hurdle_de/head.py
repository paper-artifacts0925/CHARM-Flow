"""Condition-level DE-support and zero-rate prediction head."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import SparseHurdleDEConfig
from .output import SparseHurdleHeadOutput


class ConditionSparseHurdleHead(nn.Module):
    """Predict residual support, zero shift, and a condition DE budget.

    The default condition-only output projections are exactly zero initialized.
    The opt-in gene-relation arm replaces the dense support projection with a
    low-rank condition-query/response-gene-key bilinear score.  Its key starts
    from the complete learned Gene-DiT response-gene identity plus a trainable
    per-gene residual, so it needs no incomplete external gene artifact.
    """

    def __init__(self, config: Optional[SparseHurdleDEConfig] = None):
        super().__init__()
        self.config = (
            SparseHurdleDEConfig() if config is None else config
        ).validate()
        cfg = self.config
        self.input_norm = nn.LayerNorm(cfg.condition_dim, eps=cfg.norm_eps)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.condition_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps),
            nn.Dropout(cfg.dropout),
        )
        self.de_support_projection = None
        self.gene_relation_query_projection = None
        self.gene_relation_key_norm = None
        self.gene_relation_key_projection = None
        self.gene_relation_key_residual = None
        self.gene_relation_bias = None
        if cfg.gene_relation_enabled:
            rank = int(cfg.gene_relation_rank)
            identity_dim = int(cfg.gene_relation_identity_dim)
            self.gene_relation_query_projection = nn.Linear(
                cfg.hidden_dim, rank, bias=False
            )
            self.gene_relation_key_norm = nn.LayerNorm(
                identity_dim, eps=cfg.norm_eps
            )
            self.gene_relation_key_projection = nn.Linear(
                identity_dim, rank, bias=False
            )
            self.gene_relation_key_residual = nn.Parameter(
                torch.empty(cfg.gene_dim, rank)
            )
            self.gene_relation_bias = nn.Parameter(torch.zeros(cfg.gene_dim))
            nn.init.normal_(self.gene_relation_key_residual, std=0.02)
        else:
            # This remains byte-for-byte the historical module allocation on
            # the default-false path; no relation parameters or RNG draws exist.
            self.de_support_projection = nn.Linear(
                cfg.hidden_dim, cfg.gene_dim
            )
        self.zero_rate_projection = nn.Linear(cfg.hidden_dim, cfg.gene_dim)
        self.condition_budget_projection = nn.Linear(cfg.hidden_dim, 1)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Restore the function-neutral boundaries that can be zeroed safely."""

        projections = (
            self.de_support_projection,
            self.zero_rate_projection,
            self.condition_budget_projection,
        )
        for projection in projections:
            if projection is not None:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Permit an all-or-nothing warm start from a pre-head checkpoint."""

        has_any_head_key = any(key.startswith(prefix) for key in state_dict)
        if not has_any_head_key:
            for name, value in self.state_dict().items():
                state_dict[prefix + name] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _gene_relation_logits(
        self,
        hidden: torch.Tensor,
        response_gene_identity: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.config
        if response_gene_identity is None:
            raise ValueError(
                "response_gene_identity is required when the gene relation "
                "is enabled"
            )
        if (
            not torch.is_tensor(response_gene_identity)
            or not response_gene_identity.is_floating_point()
            or tuple(response_gene_identity.shape)
            != (cfg.gene_dim, cfg.gene_relation_identity_dim)
        ):
            raise ValueError(
                "response_gene_identity must have shape "
                f"[{cfg.gene_dim},{cfg.gene_relation_identity_dim}]"
            )
        if response_gene_identity.device != hidden.device:
            raise ValueError(
                "response_gene_identity must share the condition device"
            )
        if not torch.isfinite(response_gene_identity).all():
            raise ValueError("response_gene_identity must be finite")

        query = self.gene_relation_query_projection(hidden)
        identity = response_gene_identity.to(
            dtype=self.gene_relation_key_norm.weight.dtype
        )
        key = self.gene_relation_key_projection(
            self.gene_relation_key_norm(identity)
        )
        key = key + self.gene_relation_key_residual.to(key)
        logits = torch.einsum("br,gr->bg", query, key.to(query))
        logits = logits * (1.0 / math.sqrt(float(cfg.gene_relation_rank)))
        return logits + self.gene_relation_bias.to(logits).unsqueeze(0)

    def forward(
        self,
        condition: torch.Tensor,
        *,
        response_gene_identity: Optional[torch.Tensor] = None,
    ) -> SparseHurdleHeadOutput:
        cfg = self.config
        if not torch.is_tensor(condition) or not condition.is_floating_point():
            raise TypeError("condition must be a floating-point tensor")
        if condition.ndim != 2 or tuple(condition.shape[1:]) != (
            cfg.condition_dim,
        ):
            raise ValueError(
                f"condition must have shape [B,{cfg.condition_dim}]"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition must be finite")

        parameter_dtype = self.input_norm.weight.dtype
        hidden = self.trunk(
            self.input_norm(condition.to(dtype=parameter_dtype))
        )
        if cfg.gene_relation_enabled:
            support_logits = self._gene_relation_logits(
                hidden, response_gene_identity
            )
        else:
            support_logits = self.de_support_projection(hidden)
        zero_logits = self.zero_rate_projection(hidden)
        budget_shift = self.condition_budget_projection(hidden)
        temperature = float(cfg.zero_rate_temperature)
        support_probability = torch.sigmoid(support_logits.float()).to(
            support_logits
        )
        zero_probability = torch.sigmoid(
            zero_logits.float() / temperature
        ).to(zero_logits)
        return SparseHurdleHeadOutput(
            de_support_logits=support_logits,
            zero_rate_logits=zero_logits,
            de_support_probability=support_probability,
            zero_rate_probability=zero_probability,
            condition_budget_shift=budget_shift,
        )


__all__ = ["ConditionSparseHurdleHead"]

"""Strong Parent/Child/perturbation memory for Parent-Residual GeneDiT."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StrongConditionMemoryConfig
from .output import StrongConditionMemoryOutput
from .retrieval import ChunkedQueryChildAttention


class StrongConditionMemory(nn.Module):
    """Build full Child KV memory and a separate AdaLN condition.

    Inputs remain on the condition-batch axis; no cell-set axis is required.
    All active Child tokens are retained, while a perturbation query also
    performs exact multi-head retrieval over the complete bank.  The output
    token order is stable and documented by :class:`StrongConditionMemoryOutput`.

    The query-conditioned Child update and Parent/retrieval fusion use zero
    initialized residual gates.  The independent AdaLN path has a zero
    initialized final projection.  These choices make the component safe to
    attach to a warm-started DiT without an initial context shock.
    """

    _NUM_MEMORY_TYPES = 5

    def __init__(self, config: Optional[StrongConditionMemoryConfig] = None):
        super().__init__()
        self.config = (
            StrongConditionMemoryConfig() if config is None else config
        ).validate()
        cfg = self.config
        self.parent_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.child_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.perturb_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.condition_projection = nn.Linear(
            cfg.resolved_condition_dim, cfg.hidden_dim
        )

        self.retriever = ChunkedQueryChildAttention(
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.child_chunk_size,
            cfg.norm_eps,
        )

        self.child_interaction_norm = nn.LayerNorm(
            cfg.hidden_dim, eps=cfg.norm_eps
        )
        self.query_interaction_norm = nn.LayerNorm(
            cfg.hidden_dim, eps=cfg.norm_eps
        )
        interaction_hidden = 2 * cfg.hidden_dim
        self.child_interaction_in = nn.Linear(
            3 * cfg.hidden_dim, interaction_hidden
        )
        self.child_interaction_out = nn.Linear(
            interaction_hidden, cfg.hidden_dim
        )
        self.child_query_gate = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.parent_retrieval_gate = nn.Parameter(torch.zeros(cfg.hidden_dim))

        self.memory_type_embedding = nn.Parameter(
            torch.empty(self._NUM_MEMORY_TYPES, cfg.hidden_dim)
        )
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)

        adaln_hidden = max(
            int(round(4 * cfg.hidden_dim * cfg.adaln_hidden_ratio)),
            cfg.hidden_dim,
        )
        self.adaln_input_norms = nn.ModuleList(
            [
                nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)
                for _ in range(4)
            ]
        )
        self.adaln_fusion = nn.Sequential(
            nn.Linear(4 * cfg.hidden_dim, adaln_hidden),
            nn.SiLU(),
            nn.Linear(adaln_hidden, cfg.hidden_dim),
        )

        nn.init.normal_(
            self.memory_type_embedding, std=float(cfg.type_embedding_std)
        )
        self.reset_zero_initialized_paths()

    def reset_zero_initialized_paths(self) -> None:
        """Reset every warm-start residual/gate to exact zero."""

        with torch.no_grad():
            self.child_query_gate.zero_()
            self.parent_retrieval_gate.zero_()
        self.retriever.reset_residual_projection()
        nn.init.zeros_(self.adaln_fusion[-1].weight)
        nn.init.zeros_(self.adaln_fusion[-1].bias)

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
        """Allow a complete legacy checkpoint to warm-start this new branch.

        PyTorch recursively calls ``_load_from_state_dict`` even when this
        module is nested inside a Lightning wrapper.  Only the all-or-nothing
        legacy case is filled from the constructor initialization; a partially
        present StrongConditionMemory still reports missing keys normally.
        """

        has_any_strong_key = any(key.startswith(prefix) for key in state_dict)
        if not has_any_strong_key:
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

    def _validate_inputs(
        self,
        parent: torch.Tensor,
        children: torch.Tensor,
        child_mask: torch.Tensor,
        perturb: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> None:
        cfg = self.config
        if not torch.is_tensor(parent) or not parent.is_floating_point():
            raise TypeError("parent must be a floating-point tensor")
        expected = (parent.shape[0], cfg.context_dim)
        if parent.ndim != 2 or tuple(parent.shape) != expected:
            raise ValueError(f"parent must have shape [B,{cfg.context_dim}]")
        if children.ndim != 3 or children.shape[0] != parent.shape[0]:
            raise ValueError("children must have shape [B,K,Hc]")
        if children.shape[1] < 1 or children.shape[2] != cfg.context_dim:
            raise ValueError(
                f"children must have K>=1 and end in context_dim={cfg.context_dim}"
            )
        if child_mask.dtype != torch.bool or child_mask.shape != children.shape[:2]:
            raise ValueError("child_mask must be boolean with shape [B,K]")
        if not child_mask.any(dim=1).all():
            raise ValueError("every batch row needs at least one active Child")
        if perturb.shape != parent.shape:
            raise ValueError(f"perturb must have shape [B,{cfg.context_dim}]")
        expected_condition = (parent.shape[0], cfg.resolved_condition_dim)
        if condition is not None and tuple(condition.shape) != expected_condition:
            raise ValueError(
                "condition must have shape "
                f"[B,{cfg.resolved_condition_dim}]"
            )
        floating = [parent, children, perturb]
        if condition is not None:
            floating.append(condition)
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("all context inputs must be floating point")
        if any(value.device != parent.device for value in floating):
            raise ValueError("all context inputs and child_mask must share one device")
        if child_mask.device != parent.device:
            raise ValueError("all context inputs and child_mask must share one device")
        if any(value.dtype != parent.dtype for value in floating):
            raise ValueError("all context inputs must share one dtype")

    def _project_children(self, children: torch.Tensor) -> torch.Tensor:
        chunks = []
        chunk_size = self.config.child_chunk_size
        for start in range(0, children.shape[1], chunk_size):
            chunks.append(self.child_projection(children[:, start : start + chunk_size]))
        return torch.cat(chunks, dim=1)

    def _condition_children(
        self,
        children: torch.Tensor,
        perturb: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        chunk_size = self.config.child_chunk_size
        query = self.query_interaction_norm(perturb)
        gate = torch.tanh(self.child_query_gate).to(children)
        for start in range(0, children.shape[1], chunk_size):
            child = children[:, start : start + chunk_size]
            child_normalized = self.child_interaction_norm(child)
            expanded_query = query.unsqueeze(1).expand(-1, child.shape[1], -1)
            interaction = torch.cat(
                [child_normalized, expanded_query, child_normalized * expanded_query],
                dim=-1,
            )
            update = self.child_interaction_out(
                F.silu(self.child_interaction_in(interaction))
            )
            chunks.append(child + gate.view(1, 1, -1) * update)
        return torch.cat(chunks, dim=1)

    def _adaln_condition(
        self,
        parent: torch.Tensor,
        perturb: torch.Tensor,
        retrieved: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        components = [parent, perturb, retrieved, condition]
        fused = torch.cat(
            [norm(value) for norm, value in zip(self.adaln_input_norms, components)],
            dim=-1,
        )
        return self.adaln_fusion(fused)

    def forward(
        self,
        parent: torch.Tensor,
        children: torch.Tensor,
        child_mask: torch.Tensor,
        perturb: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        *,
        return_retrieval_weights: bool = False,
    ) -> StrongConditionMemoryOutput:
        self._validate_inputs(parent, children, child_mask, perturb, condition)

        # Padded values can safely be NaN or stale storage: remove them before
        # any projection so they cannot contaminate retrieval or gradients.
        children = children.masked_fill(~child_mask.unsqueeze(-1), 0.0)
        parent_token = self.parent_projection(parent)
        perturb_token = self.perturb_projection(perturb)
        child_tokens = self._project_children(children)

        retrieved_token, retrieval_weights = self.retriever(
            perturb_token,
            child_tokens,
            child_mask,
            return_weights=return_retrieval_weights,
        )
        parent_token = parent_token + torch.tanh(
            self.parent_retrieval_gate
        ).to(parent_token) * retrieved_token
        child_tokens = self._condition_children(child_tokens, perturb_token)

        condition_is_missing = condition is None
        if condition_is_missing:
            condition_token = torch.zeros_like(parent_token)
        else:
            condition_token = self.condition_projection(condition)

        adaln_condition = self._adaln_condition(
            parent_token,
            perturb_token,
            retrieved_token,
            condition_token,
        )

        special = torch.stack(
            [parent_token, perturb_token, retrieved_token, condition_token], dim=1
        )
        special_types = self.memory_type_embedding[:4].to(special).unsqueeze(0)
        child_type = self.memory_type_embedding[4].to(child_tokens).view(1, 1, -1)
        memory = torch.cat(
            [special + special_types, child_tokens + child_type], dim=1
        )
        memory = self.memory_norm(memory)

        batch = parent.shape[0]
        special_mask = torch.zeros(
            batch, 4, dtype=torch.bool, device=parent.device
        )
        if condition_is_missing:
            special_mask[:, 3] = True
        memory_key_padding_mask = torch.cat(
            [special_mask, ~child_mask], dim=1
        )
        memory = memory.masked_fill(memory_key_padding_mask.unsqueeze(-1), 0.0)
        return StrongConditionMemoryOutput(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            adaln_condition=adaln_condition,
            retrieval_weights=retrieval_weights,
        )


__all__ = ["StrongConditionMemory"]

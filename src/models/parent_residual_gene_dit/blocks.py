"""Dual-stream gene-module DiT blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with an explicit output projection."""

    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        intermediate = max(int(float(hidden_dim) * float(mlp_ratio)), 1)
        self.input_projection = nn.Linear(hidden_dim, 2 * intermediate)
        self.output_projection = nn.Linear(intermediate, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.input_projection(value).chunk(2, dim=-1)
        value = F.silu(gate) * content
        return self.output_projection(self.dropout(value))


class GeneModuleDiTBlock(nn.Module):
    """One dual-stream module block with shared contextual memory.

    Residual order:
      1. independent main/control self-attention;
      2. main-query to control-key/value cross-attention;
      3. main and control queries to the masked context memory;
      4. independent SwiGLU updates.

    Every update is AdaLN-Zero gated, so a new stack starts as an identity.
    """

    _NUM_RESIDUALS = 7

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        source_reference_gate_limit: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        attention_kwargs = {
            "embed_dim": hidden_dim,
            "num_heads": int(num_heads),
            "dropout": float(dropout),
            "batch_first": True,
        }
        self.main_self_attention = nn.MultiheadAttention(**attention_kwargs)
        self.control_self_attention = nn.MultiheadAttention(**attention_kwargs)
        self.main_control_attention = nn.MultiheadAttention(**attention_kwargs)
        self.main_memory_attention = nn.MultiheadAttention(**attention_kwargs)
        self.control_memory_attention = nn.MultiheadAttention(**attention_kwargs)

        def norm():
            return nn.LayerNorm(hidden_dim, eps=float(norm_eps), elementwise_affine=False)

        self.main_self_norm = norm()
        self.control_self_norm = norm()
        self.main_control_norm = norm()
        self.control_key_norm = norm()
        self.main_memory_norm = norm()
        self.control_memory_norm = norm()
        self.memory_norm = norm()
        self.main_mlp_norm = norm()
        self.control_mlp_norm = norm()
        self.main_mlp = SwiGLU(hidden_dim, mlp_ratio, dropout)
        self.control_mlp = SwiGLU(hidden_dim, mlp_ratio, dropout)
        self.residual_dropout = nn.Dropout(float(dropout))
        if source_reference_gate_limit is not None:
            source_reference_gate_limit = float(source_reference_gate_limit)
            if (
                not math.isfinite(source_reference_gate_limit)
                or source_reference_gate_limit <= 0.0
            ):
                raise ValueError(
                    "source_reference_gate_limit must be finite and positive"
                )
        self.source_reference_gate_limit = source_reference_gate_limit

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                self._NUM_RESIDUALS * 3 * hidden_dim,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def _parameters_for_residuals(self, condition: torch.Tensor):
        batch, hidden = condition.shape
        values = self.adaln_modulation(condition)
        return values.reshape(batch, self._NUM_RESIDUALS, 3, hidden)

    def _residual(
        self,
        value: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return value + gate.unsqueeze(1) * self.residual_dropout(update)

    def _bound_source_reference_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Bound only the persistent source-reference residual gate.

        ``limit * tanh(gate / limit)`` retains unit slope at zero, preserving
        the AdaLN-Zero warm start while preventing a learned source read from
        acquiring an unbounded residual multiplier.
        """

        if self.source_reference_gate_limit is None:
            return gate
        limit = float(self.source_reference_gate_limit)
        return limit * torch.tanh(gate / limit)

    def forward(
        self,
        main: torch.Tensor,
        control: torch.Tensor,
        memory: torch.Tensor,
        condition: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
        update_control: bool = True,
        use_source_reference_attention: bool = True,
    ):
        if not isinstance(update_control, bool):
            raise TypeError("update_control must be boolean")
        if not isinstance(use_source_reference_attention, bool):
            raise TypeError("use_source_reference_attention must be boolean")
        if main.shape != control.shape:
            raise ValueError("main and control module streams must share one shape")
        if main.ndim != 3 or condition.shape != (main.shape[0], main.shape[2]):
            raise ValueError("invalid module-stream or AdaLN condition shape")
        if memory.ndim != 3 or memory.shape[0] != main.shape[0]:
            raise ValueError("memory must have shape [N,T,D]")
        if memory_key_padding_mask is not None:
            expected_mask = (memory.shape[0], memory.shape[1])
            if (
                memory_key_padding_mask.dtype != torch.bool
                or tuple(memory_key_padding_mask.shape) != expected_mask
            ):
                raise ValueError(
                    "memory_key_padding_mask must be boolean with shape [N,T]"
                )
            if memory_key_padding_mask.device != memory.device:
                raise ValueError(
                    "memory_key_padding_mask must share the memory device"
                )
            if memory_key_padding_mask.all(dim=1).any():
                raise ValueError("context memory cannot be fully masked")

        parameters = self._parameters_for_residuals(condition)

        shift, scale, gate = parameters[:, 0].unbind(dim=1)
        query = _modulate(self.main_self_norm(main), shift, scale)
        update = self.main_self_attention(
            query, query, query, need_weights=False
        )[0]
        main = self._residual(main, update, gate)

        if update_control:
            shift, scale, gate = parameters[:, 1].unbind(dim=1)
            query = _modulate(self.control_self_norm(control), shift, scale)
            update = self.control_self_attention(
                query, query, query, need_weights=False
            )[0]
            control = self._residual(control, update, gate)

        if use_source_reference_attention:
            shift, scale, gate = parameters[:, 2].unbind(dim=1)
            gate = self._bound_source_reference_gate(gate)
            query = _modulate(self.main_control_norm(main), shift, scale)
            control_key = self.control_key_norm(control)
            update = self.main_control_attention(
                query, control_key, control_key, need_weights=False
            )[0]
            main = self._residual(main, update, gate)

        normalized_memory = self.memory_norm(memory)
        shift, scale, gate = parameters[:, 3].unbind(dim=1)
        query = _modulate(self.main_memory_norm(main), shift, scale)
        update = self.main_memory_attention(
            query,
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        main = self._residual(main, update, gate)

        if update_control:
            shift, scale, gate = parameters[:, 4].unbind(dim=1)
            query = _modulate(self.control_memory_norm(control), shift, scale)
            update = self.control_memory_attention(
                query,
                normalized_memory,
                normalized_memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
            control = self._residual(control, update, gate)

        shift, scale, gate = parameters[:, 5].unbind(dim=1)
        update = self.main_mlp(
            _modulate(self.main_mlp_norm(main), shift, scale)
        )
        main = self._residual(main, update, gate)

        if update_control:
            shift, scale, gate = parameters[:, 6].unbind(dim=1)
            update = self.control_mlp(
                _modulate(self.control_mlp_norm(control), shift, scale)
            )
            control = self._residual(control, update, gate)
        return main, control


__all__ = ["GeneModuleDiTBlock", "SwiGLU"]

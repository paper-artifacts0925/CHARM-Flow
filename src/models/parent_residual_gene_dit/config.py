"""Configuration for the isolated Parent-Residual gene-module DiT core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParentResidualGeneDiTConfig:
    """Static dimensions for :class:`ParentResidualGeneDiT`.

    The defaults describe the intended full model.  Tests and ablations can use
    smaller values without changing the implementation.
    """

    gene_dim: int = 2000
    num_modules: int = 128
    hidden_dim: int = 384
    depth: int = 12
    num_heads: int = 8
    context_dim: int = 128
    condition_dim: int = 2000
    time_embedding_dim: int = 256
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    decoder_chunk_size: int = 256
    norm_eps: float = 1e-6
    strong_condition_memory_enabled: bool = False
    strong_condition_num_heads: int = 8
    strong_condition_child_chunk_size: int = 32
    strong_condition_adaln_hidden_ratio: float = 2.0
    source_reference_attention_enabled: bool = True
    source_reference_gate_limit: Optional[float] = None
    velocity_output_fp32: bool = False
    global_local_memory_split_enabled: bool = False
    local_child_memory_last_n_layers: int = 0
    response_only_last_n_layers: int = 0

    def validate(self) -> "ParentResidualGeneDiTConfig":
        integer_fields = {
            "gene_dim": self.gene_dim,
            "num_modules": self.num_modules,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "context_dim": self.context_dim,
            "condition_dim": self.condition_dim,
            "time_embedding_dim": self.time_embedding_dim,
            "decoder_chunk_size": self.decoder_chunk_size,
            "strong_condition_num_heads": self.strong_condition_num_heads,
            "strong_condition_child_chunk_size": (
                self.strong_condition_child_chunk_size
            ),
        }
        for name, value in integer_fields.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(self.norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive")
        if not isinstance(self.strong_condition_memory_enabled, bool):
            raise TypeError("strong_condition_memory_enabled must be boolean")
        if not isinstance(self.source_reference_attention_enabled, bool):
            raise TypeError("source_reference_attention_enabled must be boolean")
        if self.source_reference_gate_limit is not None:
            if isinstance(self.source_reference_gate_limit, bool):
                raise TypeError(
                    "source_reference_gate_limit must be a positive float or None"
                )
            gate_limit = float(self.source_reference_gate_limit)
            if not math.isfinite(gate_limit) or gate_limit <= 0.0:
                raise ValueError(
                    "source_reference_gate_limit must be finite and positive"
                )
        if not isinstance(self.velocity_output_fp32, bool):
            raise TypeError("velocity_output_fp32 must be boolean")
        if (
            self.strong_condition_memory_enabled
            and int(self.hidden_dim) % int(self.strong_condition_num_heads) != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible by strong_condition_num_heads"
            )
        if float(self.strong_condition_adaln_hidden_ratio) <= 0.0:
            raise ValueError(
                "strong_condition_adaln_hidden_ratio must be positive"
            )
        if not isinstance(self.global_local_memory_split_enabled, bool):
            raise TypeError("global_local_memory_split_enabled must be boolean")
        local_layers = int(self.local_child_memory_last_n_layers)
        if local_layers < 0:
            raise ValueError("local_child_memory_last_n_layers must be non-negative")
        if self.global_local_memory_split_enabled:
            if not 1 <= local_layers <= int(self.depth):
                raise ValueError(
                    "local_child_memory_last_n_layers must lie in [1, depth] "
                    "when global/local memory split is enabled"
                )
            if self.strong_condition_memory_enabled:
                raise ValueError(
                    "the first global/local memory split implementation requires "
                    "strong_condition_memory_enabled=false"
                )
        elif local_layers != 0:
            raise ValueError(
                "local_child_memory_last_n_layers must be zero when global/local "
                "memory split is disabled"
            )
        response_only_layers = self.response_only_last_n_layers
        if isinstance(response_only_layers, bool) or not isinstance(
            response_only_layers, int
        ):
            raise TypeError("response_only_last_n_layers must be an integer")
        if not 0 <= response_only_layers < int(self.depth):
            raise ValueError(
                "response_only_last_n_layers must lie in [0, depth)"
            )
        if response_only_layers and self.strong_condition_memory_enabled:
            raise ValueError(
                "response-only tail layers are incompatible with strong condition "
                "memory"
            )
        return self


__all__ = ["ParentResidualGeneDiTConfig"]

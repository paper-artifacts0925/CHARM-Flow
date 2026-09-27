"""Configuration for the stand-alone Parent/Child condition memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StrongConditionMemoryConfig:
    """Static dimensions for :class:`StrongConditionMemory`.

    ``child_chunk_size`` bounds temporary tensors while projecting and fusing
    the Child bank.  The final ``[B, K, D]`` Child memory is intentionally not
    compressed: every active anchor remains available to downstream
    cross-attention.
    """

    context_dim: int = 128
    condition_dim: Optional[int] = None
    hidden_dim: int = 384
    num_heads: int = 8
    child_chunk_size: int = 32
    adaln_hidden_ratio: float = 2.0
    norm_eps: float = 1e-6
    type_embedding_std: float = 0.02

    def validate(self) -> "StrongConditionMemoryConfig":
        integer_fields = {
            "context_dim": self.context_dim,
            "condition_dim": (
                self.context_dim
                if self.condition_dim is None
                else self.condition_dim
            ),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "child_chunk_size": self.child_chunk_size,
        }
        for name, value in integer_fields.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(self.adaln_hidden_ratio) <= 0.0:
            raise ValueError("adaln_hidden_ratio must be positive")
        if float(self.norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive")
        if float(self.type_embedding_std) < 0.0:
            raise ValueError("type_embedding_std cannot be negative")
        return self

    @property
    def resolved_condition_dim(self) -> int:
        """Input width of the optional semantic-condition token."""

        if self.condition_dim is None:
            return int(self.context_dim)
        return int(self.condition_dim)


__all__ = ["StrongConditionMemoryConfig"]

"""Typed outputs shared by the condition-memory module and its helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StrongConditionMemoryOutput:
    """Condition tensors ready for Transformer cross-attention.

    Attributes:
        memory: ``[B, K + 4, D]`` tokens ordered as Parent, perturbation query,
            retrieved Child summary, condition/null, and every Child anchor.
        memory_key_padding_mask: ``[B, K + 4]`` boolean mask using PyTorch
            ``MultiheadAttention`` semantics: ``True`` means ignore the token.
        adaln_condition: ``[B, D]`` context produced by a path independent of
            memory pooling.  Its output projection is exactly zero initialized.
        retrieval_weights: Optional diagnostic ``[B, H, K]`` float32 tensor.
            It is omitted by default to avoid retaining attention maps.
    """

    memory: torch.Tensor
    memory_key_padding_mask: torch.Tensor
    adaln_condition: torch.Tensor
    retrieval_weights: Optional[torch.Tensor] = None

    @property
    def parent_token(self) -> torch.Tensor:
        return self.memory[:, 0]

    @property
    def perturb_token(self) -> torch.Tensor:
        return self.memory[:, 1]

    @property
    def retrieved_token(self) -> torch.Tensor:
        return self.memory[:, 2]

    @property
    def condition_token(self) -> torch.Tensor:
        return self.memory[:, 3]

    @property
    def child_memory(self) -> torch.Tensor:
        return self.memory[:, 4:]


@dataclass
class FlatConditionMemory:
    """A ``B*S`` view/copy suitable for the current independent-cell core."""

    memory: torch.Tensor
    memory_key_padding_mask: torch.Tensor
    adaln_condition: torch.Tensor
    batch_indices: torch.Tensor


__all__ = ["FlatConditionMemory", "StrongConditionMemoryOutput"]

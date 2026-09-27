"""Opt-in, function-preserving full-bank memory for Parent-Residual GeneDiT."""

from .config import StrongConditionMemoryConfig
from .helpers import expand_for_cell_set, iter_cell_set_chunks
from .memory import StrongConditionMemory
from .output import FlatConditionMemory, StrongConditionMemoryOutput
from .residual import MainOnlyMemoryResidualAdapter, StrongMemoryResidualAdapter
from .retrieval import ChunkedQueryChildAttention

__all__ = [
    "ChunkedQueryChildAttention",
    "FlatConditionMemory",
    "MainOnlyMemoryResidualAdapter",
    "StrongConditionMemory",
    "StrongConditionMemoryConfig",
    "StrongConditionMemoryOutput",
    "StrongMemoryResidualAdapter",
    "expand_for_cell_set",
    "iter_cell_set_chunks",
]

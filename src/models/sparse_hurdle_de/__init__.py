"""Standalone, opt-in sparse hurdle and DE-support primitives."""

from .adapter import SparseHurdleDEAdapter
from .config import SparseHurdleDEConfig
from .head import ConditionSparseHurdleHead
from .losses import (
    SparseHurdleDELossConfig,
    SparseHurdleDELossOutput,
    sparse_hurdle_de_loss,
)
from .output import (
    SparseHurdleDEAdapterOutput,
    SparseHurdleGeneProgramOutput,
    SparseHurdleHeadOutput,
    SparseHurdleProjection,
)
from .projector import project_ranked_hard_zero_fixed_parent_mean

__all__ = [
    "ConditionSparseHurdleHead",
    "SparseHurdleDEAdapter",
    "SparseHurdleDEAdapterOutput",
    "SparseHurdleDEConfig",
    "SparseHurdleDELossConfig",
    "SparseHurdleDELossOutput",
    "SparseHurdleGeneProgramOutput",
    "SparseHurdleHeadOutput",
    "SparseHurdleProjection",
    "project_ranked_hard_zero_fixed_parent_mean",
    "sparse_hurdle_de_loss",
]

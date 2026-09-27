"""P2 train-only DE-aware residual objectives."""

from .artifact import (
    SCHEMA_VERSION,
    TrainDELabelBank,
    TrainDELookup,
    build_train_de_label_artifact,
    sha256_file,
)
from .config import DEAwareResidualConfig
from .losses import (
    DEAwareResidualLoss,
    cap_auxiliary_gradient,
    grouped_de_aware_residual_loss,
    warmup_fraction,
)

__all__ = [
    "DEAwareResidualConfig",
    "DEAwareResidualLoss",
    "SCHEMA_VERSION",
    "TrainDELabelBank",
    "TrainDELookup",
    "build_train_de_label_artifact",
    "cap_auxiliary_gradient",
    "grouped_de_aware_residual_loss",
    "sha256_file",
    "warmup_fraction",
]

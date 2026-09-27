"""Local CSV logger that records metrics without serializing runtime objects."""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import CSVLogger


class MetricsOnlyCSVLogger(CSVLogger):
    """CSVLogger with a no-op hyperparameter serializer."""

    def log_hyperparams(self, params: Any) -> None:
        del params



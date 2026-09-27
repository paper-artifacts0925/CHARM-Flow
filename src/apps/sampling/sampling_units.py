"""Unit transforms shared by sampling serialization paths."""

from __future__ import annotations

from typing import Optional

import numpy as np


def scale_saved_expression_units(
    samples: np.ndarray,
    truths: np.ndarray,
    normalize_counts,
    *,
    parent_projection_references: Optional[np.ndarray] = None,
) -> None:
    """Scale expression and its Parent reference together, in place.

    Parent-lock projection references are expression means. They must undergo
    exactly the same serialization-unit transform as predictions; otherwise an
    in-memory valid endpoint fails the independent saved-H5AD mean audit.
    Legacy fixed-source sampling passes no Parent reference.
    """

    if normalize_counts is None or normalize_counts is False:
        return
    samples *= normalize_counts
    truths *= normalize_counts
    if parent_projection_references is not None:
        parent_projection_references *= normalize_counts


__all__ = ["scale_saved_expression_units"]

"""Integration helpers that do not depend on Lightning or a DiT core."""

from __future__ import annotations

from typing import Iterator

import torch

from .output import FlatConditionMemory, StrongConditionMemoryOutput


def _validate_set_size(set_size: int) -> int:
    set_size = int(set_size)
    if set_size < 1:
        raise ValueError("set_size must be positive")
    return set_size


def expand_for_cell_set(
    output: StrongConditionMemoryOutput,
    set_size: int,
) -> FlatConditionMemory:
    """Repeat condition tensors in the ``B0,B0,...,B1,B1,...`` cell order."""

    set_size = _validate_set_size(set_size)
    batch = output.memory.shape[0]
    indices = torch.arange(batch, device=output.memory.device).repeat_interleave(
        set_size
    )
    return FlatConditionMemory(
        memory=output.memory.index_select(0, indices),
        memory_key_padding_mask=output.memory_key_padding_mask.index_select(0, indices),
        adaln_condition=output.adaln_condition.index_select(0, indices),
        batch_indices=indices,
    )


def iter_cell_set_chunks(
    output: StrongConditionMemoryOutput,
    set_size: int,
    max_cells: int,
) -> Iterator[FlatConditionMemory]:
    """Yield bounded ``B*S`` condition chunks instead of one large expansion."""

    set_size = _validate_set_size(set_size)
    max_cells = int(max_cells)
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    batch = output.memory.shape[0]
    total = batch * set_size
    for start in range(0, total, max_cells):
        stop = min(start + max_cells, total)
        flat_indices = torch.arange(start, stop, device=output.memory.device)
        indices = torch.div(flat_indices, set_size, rounding_mode="floor")
        yield FlatConditionMemory(
            memory=output.memory.index_select(0, indices),
            memory_key_padding_mask=output.memory_key_padding_mask.index_select(
                0, indices
            ),
            adaln_condition=output.adaln_condition.index_select(0, indices),
            batch_indices=indices,
        )


__all__ = ["expand_for_cell_set", "iter_cell_set_chunks"]

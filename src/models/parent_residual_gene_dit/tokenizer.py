"""Sparse fixed-assignment gene-module tokenization and decoding."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _validate_module_ids(
    module_ids: torch.Tensor,
    gene_dim: int,
    num_modules: int,
) -> torch.Tensor:
    if not torch.is_tensor(module_ids):
        module_ids = torch.as_tensor(module_ids)
    if module_ids.ndim != 1 or module_ids.numel() != int(gene_dim):
        raise ValueError(f"module_ids must have shape [{int(gene_dim)}]")
    if module_ids.is_floating_point() or module_ids.dtype == torch.bool:
        raise TypeError("module_ids must use an integer dtype")
    module_ids = module_ids.detach().to(device="cpu", dtype=torch.long).clone()
    if ((module_ids < 0) | (module_ids >= int(num_modules))).any():
        raise ValueError("module_ids contains an out-of-range module")
    counts = torch.bincount(module_ids, minlength=int(num_modules))
    if (counts == 0).any():
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"module_ids leaves empty modules: {empty}")
    return module_ids


def _padded_membership(
    module_ids: torch.Tensor,
    num_modules: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a compact fixed membership table instead of a dense MxG matrix."""

    counts = torch.bincount(module_ids, minlength=int(num_modules))
    max_members = max(int(counts.max().item()), 1)
    indices = torch.zeros(int(num_modules), max_members, dtype=torch.long)
    mask = torch.zeros(int(num_modules), max_members, dtype=torch.bool)
    for module_index in range(int(num_modules)):
        members = torch.nonzero(module_ids == module_index, as_tuple=False).flatten()
        if members.numel():
            indices[module_index, : members.numel()] = members
            mask[module_index, : members.numel()] = True
    return indices, mask, counts


class SparseGeneModuleTokenizer(nn.Module):
    """Map whole-gene expression to fixed gene-module tokens.

    Each gene owns a learned identity vector.  Its scalar expression value
    weights that vector before aggregation into its one fixed module.  The
    padded membership table contains only actual module members (plus masked
    padding), avoiding a learned or dense GxM assignment.
    """

    def __init__(
        self,
        module_ids: torch.Tensor,
        gene_dim: int,
        num_modules: int,
        hidden_dim: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        module_ids = _validate_module_ids(module_ids, gene_dim, num_modules)
        membership, membership_mask, counts = _padded_membership(
            module_ids, num_modules
        )
        self.gene_dim = int(gene_dim)
        self.num_modules = int(num_modules)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("module_ids", module_ids, persistent=True)
        self.register_buffer("module_gene_indices", membership, persistent=True)
        self.register_buffer(
            "module_gene_mask", membership_mask, persistent=True
        )
        self.register_buffer("module_counts", counts, persistent=True)

        self.gene_identity = nn.Parameter(
            torch.empty(self.gene_dim, self.hidden_dim)
        )
        self.module_identity = nn.Parameter(
            torch.empty(self.num_modules, self.hidden_dim)
        )
        self.value_statistics = nn.Linear(2, self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(norm_eps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gene_identity, std=0.02)
        nn.init.normal_(self.module_identity, std=0.02)
        nn.init.xavier_uniform_(self.value_statistics.weight)
        nn.init.zeros_(self.value_statistics.bias)

    def assigned_gene_indices(self, module_index: int) -> torch.Tensor:
        """Return the immutable gene indices assigned to one module."""

        module_index = int(module_index)
        if not 0 <= module_index < self.num_modules:
            raise IndexError("module_index is out of range")
        return self.module_gene_indices[module_index][
            self.module_gene_mask[module_index]
        ]

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.gene_dim:
            raise ValueError(
                f"expression must have shape [N,{self.gene_dim}]"
            )
        if not expression.is_floating_point():
            raise TypeError("expression must be floating point")

        module_ids = self.module_ids
        counts = self.module_counts.to(expression.dtype)
        value_sum = expression.new_zeros(expression.shape[0], self.num_modules)
        value_sum.index_add_(1, module_ids, expression)
        square_sum = expression.new_zeros(expression.shape[0], self.num_modules)
        square_sum.index_add_(1, module_ids, expression.square())
        mean = value_sum / counts.view(1, -1)
        mean_square = square_sum / counts.view(1, -1)
        rms = mean_square.clamp_min(0.0).add(1e-8).sqrt()
        statistics = self.value_statistics(torch.stack([mean, rms], dim=-1))

        # Packed segment reductions are O(N*G*D).  A padded [M,Lmax] gather
        # silently becomes O(N*M*Lmax*D) for uneven biological modules.  The
        # statistics Linear also gives us autocast's compute dtype; align raw
        # Parameters/elementwise paths to it instead of promoting activations
        # back to fp32 under bf16-mixed training.
        compute_expression = expression.to(dtype=statistics.dtype)
        weighted_identity = (
            compute_expression.unsqueeze(-1)
            * self.gene_identity.to(compute_expression).unsqueeze(0)
        )
        identity_value = compute_expression.new_zeros(
            expression.shape[0], self.num_modules, self.hidden_dim
        )
        identity_value.index_add_(1, module_ids, weighted_identity)
        compute_counts = self.module_counts.to(compute_expression)
        identity_value = identity_value / compute_counts.sqrt().view(1, -1, 1)

        # Keep absolute mean/RMS outside LayerNorm.  Perturbation magnitude is
        # a primary endpoint metric and must not become scale-invariant.
        identity_tokens = self.output_norm(
            self.module_identity.to(identity_value).unsqueeze(0) + identity_value
        ).to(statistics)
        return identity_tokens + statistics.to(identity_tokens)


class SparseModuleGeneDecoder(nn.Module):
    """Decode module tokens back to genes through the same fixed assignment."""

    def __init__(
        self,
        module_ids: torch.Tensor,
        gene_dim: int,
        num_modules: int,
        hidden_dim: int,
        chunk_size: int = 256,
        norm_eps: float = 1e-6,
        output_fp32: bool = False,
    ):
        super().__init__()
        module_ids = _validate_module_ids(module_ids, gene_dim, num_modules)
        self.gene_dim = int(gene_dim)
        self.num_modules = int(num_modules)
        self.hidden_dim = int(hidden_dim)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not isinstance(output_fp32, bool):
            raise TypeError("output_fp32 must be boolean")
        self.output_fp32 = output_fp32
        self.register_buffer("module_ids", module_ids, persistent=True)
        self.gene_query = nn.Parameter(torch.empty(self.gene_dim, self.hidden_dim))
        self.output_weight = nn.Parameter(
            torch.zeros(self.gene_dim, self.hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.gene_dim))
        self.hidden_norm = nn.LayerNorm(self.hidden_dim, eps=float(norm_eps))
        nn.init.normal_(self.gene_query, std=0.02)

    def reset_output_projection(self):
        """Restore an exact zero-output warm-start boundary."""

        nn.init.zeros_(self.output_weight)
        nn.init.zeros_(self.output_bias)

    def forward(self, module_tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.num_modules, self.hidden_dim)
        if module_tokens.ndim != 3 or tuple(module_tokens.shape[1:]) != expected:
            raise ValueError(
                "module_tokens must have shape "
                f"[N,{self.num_modules},{self.hidden_dim}]"
            )
        outputs = []
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        for start in range(0, self.gene_dim, self.chunk_size):
            end = min(start + self.chunk_size, self.gene_dim)
            assigned = module_tokens.index_select(
                1, self.module_ids[start:end]
            )
            hidden = self.hidden_norm(
                assigned
                + self.gene_query[start:end].to(assigned).unsqueeze(0)
            ).to(assigned) + assigned
            if self.output_fp32:
                # The group-common velocity can be much larger than the
                # centred cell-to-cell signal. Perform the final gene-wise
                # reduction before any bf16 rounding so subsequent group
                # centring does not lose that smaller signal to cancellation.
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    value = torch.einsum(
                        "ncd,cd->nc",
                        hidden.float(),
                        self.output_weight[start:end].float(),
                    )
                    value = value * scale + self.output_bias[start:end].float()
            else:
                value = (
                    hidden
                    * self.output_weight[start:end].to(hidden).unsqueeze(0)
                ).sum(dim=-1).to(hidden)
                value = (
                    value * scale
                    + self.output_bias[start:end].to(value).unsqueeze(0)
                )
            outputs.append(value)
        return torch.cat(outputs, dim=1)


__all__ = ["SparseGeneModuleTokenizer", "SparseModuleGeneDecoder"]

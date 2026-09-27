"""Function-preserving residual access to the full Child memory bank."""

from __future__ import annotations

import torch
import torch.nn as nn


class StrongMemoryResidualAdapter(nn.Module):
    """Add full-bank cross-attention behind exact-zero learnable gates.

    The legacy four-token attention remains inside :class:`GeneModuleDiTBlock`.
    This adapter is applied after that unchanged block and therefore cannot
    alter a warm-started checkpoint while both gates are zero. A single
    attention projection is shared by the main/control streams; independent
    query norms and gates let the two streams open at different rates.

    Attention dropout is intentionally fixed at zero. Besides being a stable
    conditioning path, this means an enabled, zero-gated adapter does not
    consume random numbers and shift the legacy block dropout sequence during
    training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim < 1 or num_heads < 1:
            raise ValueError("hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive")

        self.hidden_dim = hidden_dim
        self.main_query_norm = nn.LayerNorm(
            hidden_dim, eps=float(norm_eps), elementwise_affine=False
        )
        self.control_query_norm = nn.LayerNorm(
            hidden_dim, eps=float(norm_eps), elementwise_affine=False
        )
        self.memory_norm = nn.LayerNorm(
            hidden_dim, eps=float(norm_eps), elementwise_affine=False
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.main_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.control_gate = nn.Parameter(torch.zeros(hidden_dim))

    def reset_zero_initialized_paths(self) -> None:
        """Restore exact function preservation without resetting features."""

        with torch.no_grad():
            self.main_gate.zero_()
            self.control_gate.zero_()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Permit strict warm-start from a checkpoint predating this branch."""

        has_any_adapter_key = any(key.startswith(prefix) for key in state_dict)
        if not has_any_adapter_key:
            for name, value in self.state_dict().items():
                state_dict[prefix + name] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _validate(
        main: torch.Tensor,
        control: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        if main.shape != control.shape or main.ndim != 3:
            raise ValueError("main and control must share shape [N,M,D]")
        if main.shape[-1] != hidden_dim:
            raise ValueError(f"main/control must end in hidden_dim={hidden_dim}")
        if memory.ndim != 3 or memory.shape[0] != main.shape[0]:
            raise ValueError("memory must have shape [N,T,D]")
        if memory.shape[-1] != hidden_dim:
            raise ValueError(f"memory must end in hidden_dim={hidden_dim}")
        if (
            memory_key_padding_mask.dtype != torch.bool
            or tuple(memory_key_padding_mask.shape) != tuple(memory.shape[:2])
        ):
            raise ValueError(
                "memory_key_padding_mask must be boolean with shape [N,T]"
            )
        if memory_key_padding_mask.all(dim=1).any():
            raise ValueError("strong memory cannot be fully masked")
        tensors = (control, memory, memory_key_padding_mask)
        if any(value.device != main.device for value in tensors):
            raise ValueError("streams, memory, and mask must share one device")
        if control.dtype != main.dtype:
            raise ValueError("main and control must share one dtype")

    def forward(
        self,
        main: torch.Tensor,
        control: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        self._validate(
            main,
            control,
            memory,
            memory_key_padding_mask,
            self.hidden_dim,
        )
        normalized_memory = self.memory_norm(memory).to(main)
        main_update = self.attention(
            self.main_query_norm(main),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        control_update = self.attention(
            self.control_query_norm(control),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        main_gate = torch.tanh(self.main_gate).to(main).view(1, 1, -1)
        control_gate = torch.tanh(self.control_gate).to(control).view(1, 1, -1)
        return (
            main + main_gate * main_update,
            control + control_gate * control_update,
        )


class MainOnlyMemoryResidualAdapter(nn.Module):
    """Add local Child memory to the response stream behind an exact-zero gate.

    Unlike :class:`StrongMemoryResidualAdapter`, this module has no
    control-stream input or parameters. The API therefore makes it impossible
    for local Child state to rewrite the control reference stream.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim < 1 or num_heads < 1:
            raise ValueError("hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive")

        self.hidden_dim = hidden_dim
        self.query_norm = nn.LayerNorm(
            hidden_dim, eps=float(norm_eps), elementwise_affine=False
        )
        self.memory_norm = nn.LayerNorm(
            hidden_dim, eps=float(norm_eps), elementwise_affine=False
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(hidden_dim))

    def reset_zero_initialized_paths(self) -> None:
        with torch.no_grad():
            self.gate.zero_()

    def forward(
        self,
        main: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if main.ndim != 3 or main.shape[-1] != self.hidden_dim:
            raise ValueError(f"main must have shape [N,M,{self.hidden_dim}]")
        if (
            memory.ndim != 3
            or memory.shape[0] != main.shape[0]
            or memory.shape[-1] != self.hidden_dim
        ):
            raise ValueError(f"memory must have shape [N,T,{self.hidden_dim}]")
        if memory.device != main.device:
            raise ValueError("main and memory must share one device")
        if memory_key_padding_mask is not None:
            if (
                memory_key_padding_mask.dtype != torch.bool
                or tuple(memory_key_padding_mask.shape) != tuple(memory.shape[:2])
            ):
                raise ValueError(
                    "memory_key_padding_mask must be boolean with shape [N,T]"
                )
            if memory_key_padding_mask.device != main.device:
                raise ValueError("memory mask must share the main device")
            if memory_key_padding_mask.all(dim=1).any():
                raise ValueError("local Child memory cannot be fully masked")

        normalized_memory = self.memory_norm(memory).to(main)
        update = self.attention(
            self.query_norm(main),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        gate = torch.tanh(self.gate).to(main).view(1, 1, -1)
        return main + gate * update


__all__ = ["MainOnlyMemoryResidualAdapter", "StrongMemoryResidualAdapter"]

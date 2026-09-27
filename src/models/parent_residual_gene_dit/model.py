"""Parent-Residual dual-stream DiT over fixed gene modules."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..parent_residual_condition_memory import (
    MainOnlyMemoryResidualAdapter,
    StrongConditionMemory,
    StrongConditionMemoryConfig,
    StrongMemoryResidualAdapter,
    expand_for_cell_set,
)
from .blocks import GeneModuleDiTBlock
from .config import ParentResidualGeneDiTConfig
from .tokenizer import SparseGeneModuleTokenizer, SparseModuleGeneDecoder


class ScalarTimeEmbedding(nn.Module):
    """Sinusoidal scalar-time embedding followed by a two-layer MLP."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("flattened time must have shape [N]")
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half, 1)
        )
        phase = time.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([phase.cos(), phase.sin()], dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.cat(
                [embedding, embedding.new_zeros(embedding.shape[0], 1)], dim=-1
            )
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class ParentResidualGeneDiT(nn.Module):
    """Gene-module dual-stream velocity core.

    Public tensors use ``[B,S,G]``.  Internally ``B*S`` is flattened into
    independent cells and the Transformer sequence is the fixed module axis M.
    Consequently changing S changes only the number of independent examples;
    it never introduces untrained cross-cell attention.

    Args:
        module_ids: Fixed length-G integer vector assigning every gene to one
            module in ``[0, M)``.  It is registered as a buffer, never learned.
        config: Model dimensions.  Defaults are G=2000, M=128, D=384, L=12,
            H=8.

    Forward returns a dictionary with ``x`` and ``x_control``, both [B,S,G].
    Decoder output projections are exactly zero initialized for a stable
    velocity-model warm start.
    """

    def __init__(
        self,
        module_ids: torch.Tensor,
        config: Optional[ParentResidualGeneDiTConfig] = None,
    ):
        super().__init__()
        self.config = (
            ParentResidualGeneDiTConfig() if config is None else config
        ).validate()
        cfg = self.config
        self.main_tokenizer = SparseGeneModuleTokenizer(
            module_ids,
            gene_dim=cfg.gene_dim,
            num_modules=cfg.num_modules,
            hidden_dim=cfg.hidden_dim,
            norm_eps=cfg.norm_eps,
        )
        self.control_tokenizer = SparseGeneModuleTokenizer(
            module_ids,
            gene_dim=cfg.gene_dim,
            num_modules=cfg.num_modules,
            hidden_dim=cfg.hidden_dim,
            norm_eps=cfg.norm_eps,
        )

        self.parent_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.child_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.routed_projection = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.condition_projection = nn.Linear(cfg.condition_dim, cfg.hidden_dim)
        self.memory_type_embedding = nn.Parameter(
            torch.empty(4, cfg.hidden_dim)
        )
        self.memory_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.adaln_context = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.strong_condition_memory = None
        self.strong_memory_adapters = None
        self.local_child_adapters = None
        if cfg.strong_condition_memory_enabled:
            self.strong_condition_memory = StrongConditionMemory(
                StrongConditionMemoryConfig(
                    context_dim=cfg.context_dim,
                    condition_dim=cfg.condition_dim,
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.strong_condition_num_heads,
                    child_chunk_size=cfg.strong_condition_child_chunk_size,
                    adaln_hidden_ratio=(
                        cfg.strong_condition_adaln_hidden_ratio
                    ),
                    norm_eps=cfg.norm_eps,
                )
            )
            self.strong_memory_adapters = nn.ModuleList(
                [
                    StrongMemoryResidualAdapter(
                        hidden_dim=cfg.hidden_dim,
                        num_heads=cfg.strong_condition_num_heads,
                        norm_eps=cfg.norm_eps,
                    )
                    for _ in range(cfg.depth)
                ]
            )
        if cfg.global_local_memory_split_enabled:
            self.local_child_adapters = nn.ModuleList(
                [
                    MainOnlyMemoryResidualAdapter(
                        hidden_dim=cfg.hidden_dim,
                        num_heads=cfg.num_heads,
                        norm_eps=cfg.norm_eps,
                    )
                    for _ in range(cfg.local_child_memory_last_n_layers)
                ]
            )
        self.time_embedding = ScalarTimeEmbedding(
            cfg.hidden_dim, cfg.time_embedding_dim
        )
        self.blocks = nn.ModuleList(
            [
                GeneModuleDiTBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    norm_eps=cfg.norm_eps,
                    source_reference_gate_limit=(
                        cfg.source_reference_gate_limit
                    ),
                )
                for _ in range(cfg.depth)
            ]
        )
        self.main_decoder = SparseModuleGeneDecoder(
            module_ids,
            gene_dim=cfg.gene_dim,
            num_modules=cfg.num_modules,
            hidden_dim=cfg.hidden_dim,
            chunk_size=cfg.decoder_chunk_size,
            norm_eps=cfg.norm_eps,
            output_fp32=cfg.velocity_output_fp32,
        )
        self.control_decoder = SparseModuleGeneDecoder(
            module_ids,
            gene_dim=cfg.gene_dim,
            num_modules=cfg.num_modules,
            hidden_dim=cfg.hidden_dim,
            chunk_size=cfg.decoder_chunk_size,
            norm_eps=cfg.norm_eps,
            output_fp32=cfg.velocity_output_fp32,
        )
        nn.init.normal_(self.memory_type_embedding, std=0.02)

    @property
    def module_ids(self) -> torch.Tensor:
        return self.main_tokenizer.module_ids

    def reset_output_projection(self):
        """Reset both gene decoders to exact zero output."""

        self.main_decoder.reset_output_projection()
        self.control_decoder.reset_output_projection()

    def reset_strong_condition_residual(self) -> None:
        """Reset only the opt-in branch function-preserving boundaries."""

        if self.strong_condition_memory is None:
            return
        self.strong_condition_memory.reset_zero_initialized_paths()
        for adapter in self.strong_memory_adapters:
            adapter.reset_zero_initialized_paths()

    def reset_local_child_residual(self) -> None:
        """Close every opt-in local Child adapter without resetting features."""

        if self.local_child_adapters is None:
            return
        for adapter in self.local_child_adapters:
            adapter.reset_zero_initialized_paths()

    @property
    def strong_condition_memory_enabled(self) -> bool:
        return self.strong_condition_memory is not None

    @property
    def global_local_memory_split_enabled(self) -> bool:
        return self.local_child_adapters is not None

    def encode_gene_modules(
        self,
        expression: torch.Tensor,
        stream: str = "main",
    ) -> torch.Tensor:
        """Encode ``[..., G]`` while preserving all leading axes."""

        if expression.ndim < 2 or expression.shape[-1] != self.config.gene_dim:
            raise ValueError(
                f"expression must end in gene_dim={self.config.gene_dim}"
            )
        if stream == "main":
            tokenizer = self.main_tokenizer
        elif stream == "control":
            tokenizer = self.control_tokenizer
        else:
            raise ValueError("stream must be 'main' or 'control'")
        leading_shape = expression.shape[:-1]
        tokens = tokenizer(expression.reshape(-1, self.config.gene_dim))
        return tokens.reshape(
            *leading_shape,
            self.config.num_modules,
            self.config.hidden_dim,
        )

    def decode_gene_modules(
        self,
        module_tokens: torch.Tensor,
        stream: str = "main",
    ) -> torch.Tensor:
        """Decode ``[..., M, D]`` through the fixed module-to-gene map."""

        expected = (self.config.num_modules, self.config.hidden_dim)
        if module_tokens.ndim < 3 or tuple(module_tokens.shape[-2:]) != expected:
            raise ValueError(
                f"module_tokens must end in [M,D]={list(expected)}"
            )
        if stream == "main":
            decoder = self.main_decoder
        elif stream == "control":
            decoder = self.control_decoder
        else:
            raise ValueError("stream must be 'main' or 'control'")
        leading_shape = module_tokens.shape[:-2]
        expression = decoder(
            module_tokens.reshape(
                -1,
                self.config.num_modules,
                self.config.hidden_dim,
            )
        )
        return expression.reshape(*leading_shape, self.config.gene_dim)

    @staticmethod
    def _flatten_context(
        value: torch.Tensor,
        name: str,
        batch_size: int,
        set_size: int,
        feature_dim: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if value.ndim == 2:
            if tuple(value.shape) != (batch_size, feature_dim):
                raise ValueError(
                    f"{name} must have shape [B,{feature_dim}] or [B,S,{feature_dim}]"
                )
            value = value.unsqueeze(1).expand(-1, set_size, -1)
        elif value.ndim == 3:
            if value.shape[0] != batch_size or value.shape[2] != feature_dim:
                raise ValueError(
                    f"{name} must have shape [B,{feature_dim}] or [B,S,{feature_dim}]"
                )
            if value.shape[1] == 1:
                value = value.expand(-1, set_size, -1)
            elif value.shape[1] != set_size:
                raise ValueError(f"{name} set axis must be 1 or S={set_size}")
        else:
            raise ValueError(
                f"{name} must have shape [B,{feature_dim}] or [B,S,{feature_dim}]"
            )
        return value.reshape(batch_size * set_size, feature_dim)

    @staticmethod
    def _flatten_time(
        time: torch.Tensor,
        batch_size: int,
        set_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(time) or not time.is_floating_point():
            raise TypeError("time must be a floating-point tensor")
        if time.ndim == 1 and time.shape[0] == batch_size:
            time = time.unsqueeze(1).expand(-1, set_size)
        elif time.ndim == 2 and time.shape[0] == batch_size:
            if time.shape[1] == 1:
                time = time.expand(-1, set_size)
            elif time.shape[1] != set_size:
                raise ValueError("time set axis must be 1 or S")
        elif time.ndim == 3 and tuple(time.shape) == (
            batch_size,
            set_size,
            1,
        ):
            time = time[..., 0]
        else:
            raise ValueError("time must have shape [B], [B,1], [B,S], or [B,S,1]")
        return time.reshape(batch_size * set_size)

    def _build_memory(
        self,
        parent: torch.Tensor,
        selected_child: torch.Tensor,
        routed: torch.Tensor,
        condition: torch.Tensor,
        batch_size: int,
        set_size: int,
    ) -> torch.Tensor:
        cfg = self.config
        parent = self._flatten_context(
            parent, "parent", batch_size, set_size, cfg.context_dim
        )
        selected_child = self._flatten_context(
            selected_child,
            "selected_child",
            batch_size,
            set_size,
            cfg.context_dim,
        )
        routed = self._flatten_context(
            routed, "routed", batch_size, set_size, cfg.context_dim
        )
        condition = self._flatten_context(
            condition,
            "condition",
            batch_size,
            set_size,
            cfg.condition_dim,
        )
        memory = torch.stack(
            [
                self.parent_projection(parent),
                self.child_projection(selected_child),
                self.routed_projection(routed),
                self.condition_projection(condition),
            ],
            dim=1,
        )
        memory = memory + self.memory_type_embedding.to(memory).unsqueeze(0)
        return self.memory_norm(memory).to(memory)
    def _build_parent_only_memory(
        self,
        parent: torch.Tensor,
        condition: torch.Tensor,
        batch_size: int,
        set_size: int,
    ) -> torch.Tensor:
        """Build strict memory without evaluating Child/routed projections."""

        cfg = self.config
        parent = self._flatten_context(
            parent, "parent", batch_size, set_size, cfg.context_dim
        )
        condition = self._flatten_context(
            condition,
            "condition",
            batch_size,
            set_size,
            cfg.condition_dim,
        )
        memory = torch.stack(
            [
                self.parent_projection(parent),
                self.condition_projection(condition),
            ],
            dim=1,
        )
        type_embedding = self.memory_type_embedding[[0, 3]].to(memory)
        memory = memory + type_embedding.unsqueeze(0)
        return self.memory_norm(memory).to(memory)

    def _build_local_memory(
        self,
        selected_child: torch.Tensor,
        routed: torch.Tensor,
        batch_size: int,
        set_size: int,
    ) -> torch.Tensor:
        """Build local state memory without Parent/condition leakage."""

        cfg = self.config
        selected_child = self._flatten_context(
            selected_child,
            "selected_child",
            batch_size,
            set_size,
            cfg.context_dim,
        )
        routed = self._flatten_context(
            routed, "routed", batch_size, set_size, cfg.context_dim
        )
        memory = torch.stack(
            [
                self.child_projection(selected_child),
                self.routed_projection(routed),
            ],
            dim=1,
        )
        type_embedding = self.memory_type_embedding[[1, 2]].to(memory)
        memory = memory + type_embedding.unsqueeze(0)
        return self.memory_norm(memory).to(memory)



    @staticmethod
    def _batch_condition(
        condition: torch.Tensor,
        batch_size: int,
        set_size: int,
        feature_dim: int,
    ) -> torch.Tensor:
        """Reduce a repeated optional cell-set condition to the B axis."""

        if not torch.is_tensor(condition) or not condition.is_floating_point():
            raise TypeError("condition must be a floating-point tensor")
        if condition.ndim == 2:
            if tuple(condition.shape) != (batch_size, feature_dim):
                raise ValueError(
                    f"condition must have shape [B,{feature_dim}] or [B,S,{feature_dim}]"
                )
            return condition
        if condition.ndim == 3 and tuple(condition.shape) == (
            batch_size,
            set_size,
            feature_dim,
        ):
            return condition.mean(dim=1)
        if condition.ndim == 3 and tuple(condition.shape) == (
            batch_size,
            1,
            feature_dim,
        ):
            return condition[:, 0]
        raise ValueError(
            f"condition must have shape [B,{feature_dim}] or [B,S,{feature_dim}]"
        )

    def _build_strong_memory(
        self,
        *,
        parent: torch.Tensor,
        all_child_tokens: torch.Tensor,
        child_mask: torch.Tensor,
        perturb_query: torch.Tensor,
        condition: torch.Tensor,
        batch_size: int,
        set_size: int,
    ):
        if self.strong_condition_memory is None:
            raise RuntimeError("StrongConditionMemory is not enabled")
        condition = self._batch_condition(
            condition,
            batch_size,
            set_size,
            self.config.condition_dim,
        )
        # The context encoder runs under Lightning autocast while the raw
        # covariate condition can be prepared outside that autocast region.
        # Consequently, real training may provide bf16 Parent/Child/query
        # tensors together with an fp32 condition even though unit tests that
        # construct every input directly do not. StrongConditionMemory keeps a
        # strict single-dtype contract, so normalize these small context inputs
        # to the module parameter dtype at this explicit model boundary. The
        # following Linear layers still use autocast's compute dtype.
        context_dtype = self.strong_condition_memory.parent_projection.weight.dtype
        parent = parent.to(dtype=context_dtype)
        all_child_tokens = all_child_tokens.to(dtype=context_dtype)
        perturb_query = perturb_query.to(dtype=context_dtype)
        condition = condition.to(dtype=context_dtype)
        output = self.strong_condition_memory(
            parent=parent,
            children=all_child_tokens,
            child_mask=child_mask,
            perturb=perturb_query,
            condition=condition,
        )
        return output, expand_for_cell_set(output, set_size)

    def forward(
        self,
        x: torch.Tensor,
        control: torch.Tensor,
        time: torch.Tensor,
        *,
        parent: torch.Tensor,
        condition: torch.Tensor,
        selected_child: Optional[torch.Tensor] = None,
        routed: Optional[torch.Tensor] = None,
        all_child_tokens: Optional[torch.Tensor] = None,
        child_mask: Optional[torch.Tensor] = None,
        perturb_query: Optional[torch.Tensor] = None,
        parent_only: bool = False,
        return_module_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        if x.ndim != 3 or x.shape[2] != cfg.gene_dim:
            raise ValueError(f"x must have shape [B,S,{cfg.gene_dim}]")
        if control.shape != x.shape:
            raise ValueError("control must have the same [B,S,G] shape as x")
        if not x.is_floating_point() or not control.is_floating_point():
            raise TypeError("x and control must be floating point")
        if x.device != control.device or x.dtype != control.dtype:
            raise ValueError("x and control must share device and dtype")
        batch_size, set_size, _ = x.shape
        flattened_time = self._flatten_time(time, batch_size, set_size)
        strong_output = None
        strong_memory = None
        strong_memory_key_padding_mask = None
        strong_adaln_condition = None
        local_memory = None
        if bool(parent_only):
            forbidden = [
                name
                for name, value in (
                    ("selected_child", selected_child),
                    ("routed", routed),
                    ("all_child_tokens", all_child_tokens),
                    ("child_mask", child_mask),
                    ("perturb_query", perturb_query),
                )
                if value is not None
            ]
            if forbidden:
                raise ValueError(
                    "strict Parent-only memory forbids " + ", ".join(forbidden)
                )
            memory = self._build_parent_only_memory(
                parent,
                condition,
                batch_size,
                set_size,
            )
        else:
            if selected_child is None or routed is None:
                raise ValueError(
                    "legacy memory requires selected_child and routed"
                )
            if self.global_local_memory_split_enabled:
                memory = self._build_parent_only_memory(
                    parent,
                    condition,
                    batch_size,
                    set_size,
                )
                local_memory = self._build_local_memory(
                    selected_child,
                    routed,
                    batch_size,
                    set_size,
                )
            else:
                # The checkpoint-compatible four-token memory remains unchanged.
                memory = self._build_memory(
                    parent,
                    selected_child,
                    routed,
                    condition,
                    batch_size,
                    set_size,
                )
        if self.strong_condition_memory_enabled and not bool(parent_only):
            missing = [
                name
                for name, value in (
                    ("all_child_tokens", all_child_tokens),
                    ("child_mask", child_mask),
                    ("perturb_query", perturb_query),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "StrongConditionMemory requires " + ", ".join(missing)
                )
            strong_output, flat_context = self._build_strong_memory(
                parent=parent,
                all_child_tokens=all_child_tokens,
                child_mask=child_mask,
                perturb_query=perturb_query,
                condition=condition,
                batch_size=batch_size,
                set_size=set_size,
            )
            strong_memory = flat_context.memory
            strong_memory_key_padding_mask = (
                flat_context.memory_key_padding_mask
            )
            strong_adaln_condition = flat_context.adaln_condition
        if memory.device != x.device:
            raise ValueError("context memory and expression must share one device")

        main_tokens = self.encode_gene_modules(x, stream="main").reshape(
            batch_size * set_size,
            cfg.num_modules,
            cfg.hidden_dim,
        )
        control_tokens = self.encode_gene_modules(
            control, stream="control"
        ).reshape(
            batch_size * set_size,
            cfg.num_modules,
            cfg.hidden_dim,
        )
        adaln_condition = self.time_embedding(flattened_time.to(x.device))
        adaln_condition = adaln_condition + self.adaln_context(
            memory.mean(dim=1)
        )
        if strong_adaln_condition is not None:
            # Its final projection is exactly zero initialized, making this
            # an additive function-preserving AdaLN residual at warm start.
            adaln_condition = adaln_condition + strong_adaln_condition.to(
                adaln_condition
            )
        first_response_only_block = (
            cfg.depth - cfg.response_only_last_n_layers
        )
        for block_index, block in enumerate(self.blocks):
            if block_index >= first_response_only_block:
                main_tokens, control_tokens = block(
                    main_tokens,
                    control_tokens,
                    memory,
                    adaln_condition,
                    update_control=False,
                    use_source_reference_attention=(
                        cfg.source_reference_attention_enabled
                    ),
                )
            else:
                # Preserve the literal historical call path for all layers
                # when response_only_last_n_layers == 0.
                main_tokens, control_tokens = block(
                    main_tokens,
                    control_tokens,
                    memory,
                    adaln_condition,
                    use_source_reference_attention=(
                        cfg.source_reference_attention_enabled
                    ),
                )
            if self.local_child_adapters is not None:
                first_local_block = (
                    cfg.depth - cfg.local_child_memory_last_n_layers
                )
                if block_index >= first_local_block:
                    if local_memory is None:
                        raise RuntimeError(
                            "local Child memory is unavailable for an enabled adapter"
                        )
                    adapter_index = block_index - first_local_block
                    main_tokens = self.local_child_adapters[adapter_index](
                        main_tokens,
                        local_memory,
                    )
            if (
                self.strong_memory_adapters is not None
                and not bool(parent_only)
            ):
                main_tokens, control_tokens = self.strong_memory_adapters[
                    block_index
                ](
                    main_tokens,
                    control_tokens,
                    strong_memory,
                    strong_memory_key_padding_mask,
                )

        main_output = self.decode_gene_modules(
            main_tokens.reshape(
                batch_size,
                set_size,
                cfg.num_modules,
                cfg.hidden_dim,
            ),
            stream="main",
        )
        control_output = self.decode_gene_modules(
            control_tokens.reshape(
                batch_size,
                set_size,
                cfg.num_modules,
                cfg.hidden_dim,
            ),
            stream="control",
        )
        result = {"x": main_output, "x_control": control_output}
        if return_module_tokens:
            result["main_module_tokens"] = main_tokens
            result["control_module_tokens"] = control_tokens
            result["context_memory"] = memory
            if local_memory is not None:
                result["local_context_memory"] = local_memory
            if strong_memory_key_padding_mask is not None:
                result["strong_context_memory"] = strong_memory
                result["strong_context_memory_key_padding_mask"] = (
                    strong_memory_key_padding_mask
                )
                result["context_adaln_condition"] = strong_adaln_condition
                result["batch_context_memory"] = strong_output.memory
        return result


__all__ = ["ParentResidualGeneDiT", "ScalarTimeEmbedding"]

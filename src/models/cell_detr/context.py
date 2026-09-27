"""Parent-as-global-memory and Child-as-control-state tokens."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values, mask, dim):
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass
class CellDETRContext:
    """The t-independent context used by matching and the unchanged FM path."""

    control_prototypes: torch.Tensor
    control_features: torch.Tensor
    anchor_mask: torch.Tensor
    occupancy: torch.Tensor
    child_tokens: torch.Tensor
    parent_token: torch.Tensor
    perturbation_query: torch.Tensor
    routed_token: torch.Tensor
    predicted_delta_latent: torch.Tensor
    router_logits: torch.Tensor
    compactness_loss: torch.Tensor
    separation_loss: torch.Tensor
    bank_inverse: torch.Tensor = None
    parent_control: torch.Tensor = None
    parent_control_feature: torch.Tensor = None
    predicted_parent_delta_latent: torch.Tensor = None
    child_gate_logits: torch.Tensor = None


class CellDETRContextEncoder(nn.Module):
    """Encode a cell-line-specific KxS control bank into 128 state queries.

    The Parent is a global CLS summary of the full cell line. Child tokens are
    permutation-equivariant state clusters, not contiguous slices of cells.
    Perturbation semantics act as the query over Parent-conditioned Children.
    """

    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 128,
        anchor_capacity: int = 128,
        num_heads: int = 8,
        parent_depth: int = 2,
        dropout: float = 0.0,
        max_child_similarity: float = 0.85,
        cold_start_gate_enabled: bool = False,
        gate_initial_probability: float = 0.02,
        child_encoding_mode: str = "joint",
    ):
        super().__init__()
        self.gene_dim = int(gene_dim)
        self.hidden_dim = int(hidden_dim)
        self.anchor_capacity = int(anchor_capacity)
        self.max_child_similarity = float(max_child_similarity)
        self.cold_start_gate_enabled = bool(cold_start_gate_enabled)
        self.gate_initial_probability = float(gate_initial_probability)
        if child_encoding_mode not in {"joint", "independent"}:
            raise ValueError("child_encoding_mode must be joint or independent")
        self.child_encoding_mode = child_encoding_mode
        self.num_heads = int(num_heads)

        self.expression_norm = nn.LayerNorm(self.gene_dim)
        self.expression_encoder = nn.Sequential(
            nn.Linear(self.gene_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.child_builder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.occupancy_projector = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.parent_transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(parent_depth),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.parent_cls = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.parent_cls, std=0.02)
        self.child_parent_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(self.gene_dim),
            nn.Linear(self.gene_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.perturbation_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.routed_fusion = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        candidate_dim = 3 * self.hidden_dim
        self.delta_latent_head = nn.Sequential(
            nn.Linear(candidate_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.router_head = nn.Sequential(
            nn.Linear(candidate_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Gene-space decoding is deliberately applied only to selected tokens.
        self.delta_gene_head = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.gene_dim),
        )
        self.child_gate_head = None
        if self.cold_start_gate_enabled:
            self.child_gate_head = nn.Sequential(
                nn.Linear(candidate_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.reset_cold_start_parameters()

    def reset_cold_start_parameters(self):
        """Restore the safe Parent-first initialization after any global reset."""
        if not self.cold_start_gate_enabled:
            return
        for head in (self.delta_latent_head, self.delta_gene_head, self.router_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.child_gate_head[-1].weight)
        initial = self.gate_initial_probability
        nn.init.constant_(
            self.child_gate_head[-1].bias,
            math.log(initial / (1.0 - initial)),
        )

    def encode_expression(self, expression):
        return self.expression_encoder(self.expression_norm(expression.float()))

    @staticmethod
    def _condition_summary(condition):
        if condition.ndim == 2:
            return condition
        if condition.ndim == 3:
            # All cells in a sampling set share perturbation/cell-line semantics.
            # Averaging also makes the direct model robust to repeated set rows.
            return condition.mean(dim=1)
        raise ValueError("cell_detr condition must have shape [B,G] or [B,S,G]")

    def _pad(self, tensor, size, value=0.0):
        count = self.anchor_capacity - size
        if count < 0:
            raise ValueError(
                f"control bank K={size} exceeds capacity={self.anchor_capacity}"
            )
        if count == 0:
            return tensor
        shape = list(tensor.shape)
        shape[1] = count
        padding = tensor.new_full(shape, value)
        return torch.cat([tensor, padding], dim=1)

    def _structure_losses(
        self,
        token_features,
        token_mask,
        means,
        anchor_mask,
        bank_inverse,
    ):
        centered = token_features - means.unsqueeze(2)
        compact_per_bank = _masked_mean(
            centered.square().mean(dim=-1), token_mask, dim=(1, 2)
        )
        compact = compact_per_bank.index_select(0, bank_inverse).mean()

        normalized = F.normalize(means, dim=-1, eps=1e-6)
        similarity = torch.einsum("bkh,blh->bkl", normalized, normalized)
        pair_mask = anchor_mask[:, :, None] & anchor_mask[:, None, :]
        diagonal = torch.eye(
            anchor_mask.shape[1], device=anchor_mask.device, dtype=torch.bool
        )
        pair_mask = pair_mask & ~diagonal.unsqueeze(0)
        penalty = F.relu(similarity - self.max_child_similarity).square()
        separation_sum = (penalty * pair_mask.to(penalty.dtype)).sum(dim=(1, 2))
        separation_count = pair_mask.sum(dim=(1, 2))
        separation = separation_sum.index_select(0, bank_inverse).sum()
        separation = separation / separation_count.index_select(
            0, bank_inverse
        ).sum().clamp_min(1)
        return compact, separation

    def _independent_child_mask(self, active_children):
        """Global CLS reads the bank; each active Child reads only itself.

        Inactive query rows may read CLS to avoid fully masked softmax rows.
        They remain excluded as keys and their returned Child tokens are zero.
        No trainable parameters or child-index embeddings are introduced.
        """
        batch, capacity = active_children.shape
        size = capacity + 1
        blocked = ~torch.eye(size, dtype=torch.bool, device=active_children.device)
        blocked = blocked.unsqueeze(0).expand(batch, -1, -1).clone()
        blocked[:, 0, :] = False
        blocked[:, 1:, 0] = active_children
        return blocked.repeat_interleave(self.num_heads, dim=0)

    def forward(
        self,
        control_sets,
        anchor_mask,
        occupancy,
        semantic_condition,
        token_mask=None,
        bank_inverse=None,
    ):
        if control_sets.ndim != 4:
            raise ValueError("control_sets must have shape [U,K,S,G]")
        bank_batch_size, num_anchors, set_size, gene_dim = control_sets.shape
        if gene_dim != self.gene_dim:
            raise ValueError(f"expected G={self.gene_dim}, got {gene_dim}")

        condition_summary = self._condition_summary(semantic_condition).float()
        cell_batch_size = condition_summary.shape[0]
        if bank_inverse is None:
            if cell_batch_size != bank_batch_size:
                raise ValueError(
                    "without bank_inverse, control-bank and cell batch sizes must match"
                )
            bank_inverse = torch.arange(
                bank_batch_size, device=control_sets.device, dtype=torch.long
            )
        else:
            bank_inverse = bank_inverse.to(
                device=control_sets.device, dtype=torch.long
            )
            if bank_inverse.shape != (cell_batch_size,):
                raise ValueError("bank_inverse must have shape [B]")

        anchor_mask = anchor_mask.to(device=control_sets.device, dtype=torch.bool)
        if anchor_mask.shape != (bank_batch_size, num_anchors):
            raise ValueError("anchor_mask must have shape [U,K]")
        if not anchor_mask.any(dim=-1).all():
            raise ValueError("every cell must have at least one active Child")
        if token_mask is None:
            token_mask = anchor_mask.unsqueeze(-1).expand(-1, -1, set_size)
        token_mask = token_mask.to(device=control_sets.device, dtype=torch.bool)
        token_mask = token_mask & anchor_mask.unsqueeze(-1)
        if token_mask.shape != (bank_batch_size, num_anchors, set_size):
            raise ValueError("token_mask must have shape [U,K,S]")

        occupancy = occupancy.to(device=control_sets.device, dtype=control_sets.dtype)
        occupancy = torch.where(anchor_mask, occupancy.clamp_min(0.0), 0.0)
        occupancy = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        control_prototypes = _masked_mean(control_sets, token_mask, dim=2)
        token_features = self.encode_expression(control_sets)
        token_features = torch.where(token_mask.unsqueeze(-1), token_features, 0.0)
        control_features = _masked_mean(token_features, token_mask, dim=2)
        second = _masked_mean(token_features.square(), token_mask, dim=2)
        child_std = (second - control_features.square()).clamp_min(0.0).add(1e-6).sqrt()
        initial_child = self.child_builder(torch.cat([control_features, child_std], dim=-1))
        initial_child = initial_child + self.occupancy_projector(
            occupancy.clamp_min(1e-8).log().unsqueeze(-1)
        )
        initial_child = torch.where(anchor_mask.unsqueeze(-1), initial_child, 0.0)
        compactness, separation = self._structure_losses(
            token_features,
            token_mask,
            control_features,
            anchor_mask,
            bank_inverse,
        )

        padded_child = self._pad(initial_child, num_anchors)
        padded_mask = self._pad(anchor_mask, num_anchors, value=False)
        padded_occupancy = self._pad(occupancy, num_anchors)
        padded_controls = self._pad(control_prototypes, num_anchors)
        padded_features = self._pad(control_features, num_anchors)
        bank_parent_control = (
            padded_controls * padded_occupancy.unsqueeze(-1)
        ).sum(dim=1)
        bank_parent_feature = (
            padded_features * padded_occupancy.unsqueeze(-1)
        ).sum(dim=1)
        cls = self.parent_cls.expand(bank_batch_size, -1, -1)
        attention_kwargs = {}
        if getattr(self, "child_encoding_mode", "joint") == "independent":
            attention_kwargs["mask"] = self._independent_child_mask(padded_mask)
        memory = self.parent_transformer(
            torch.cat([cls, padded_child], dim=1),
            src_key_padding_mask=torch.cat(
                [
                    torch.zeros(
                        bank_batch_size,
                        1,
                        device=control_sets.device,
                        dtype=torch.bool,
                    ),
                    ~padded_mask,
                ],
                dim=1,
            ),
            **attention_kwargs,
        )
        bank_parent = memory[:, 0]
        bank_parent_per_child = bank_parent.unsqueeze(1).expand(
            -1, self.anchor_capacity, -1
        )
        # In the independent arm, replace the broadcast population feature
        # with this Child's own pre-Transformer feature. Both halves of the
        # existing fusion layer remain in use, with identical parameter shapes.
        fusion_reference = (
            padded_child
            if getattr(self, "child_encoding_mode", "joint") == "independent"
            else bank_parent_per_child
        )
        bank_child = self.child_parent_fusion(
            torch.cat([memory[:, 1:], fusion_reference], dim=-1)
        )
        bank_child = torch.where(padded_mask.unsqueeze(-1), bank_child, 0.0)

        cell_mask = padded_mask.index_select(0, bank_inverse)
        cell_occupancy = padded_occupancy.index_select(0, bank_inverse)
        control_features = padded_features.index_select(0, bank_inverse)
        parent_control = bank_parent_control.index_select(0, bank_inverse)
        parent_control_feature = bank_parent_feature.index_select(0, bank_inverse)
        parent = bank_parent.index_select(0, bank_inverse)
        child = bank_child.index_select(0, bank_inverse)
        parent_per_child = parent.unsqueeze(1).expand(
            -1, self.anchor_capacity, -1
        )

        perturbation_query = self.condition_encoder(condition_summary)
        attended, _ = self.perturbation_attention(
            perturbation_query.unsqueeze(1),
            child,
            child,
            key_padding_mask=~cell_mask,
            need_weights=False,
        )
        routed = self.routed_fusion(
            torch.cat([parent, perturbation_query, attended[:, 0]], dim=-1)
        )
        query_per_child = perturbation_query.unsqueeze(1).expand_as(child)
        candidate = torch.cat([child, parent_per_child, query_per_child], dim=-1)
        predicted_delta_latent = self.delta_latent_head(candidate)
        router_logits = self.router_head(candidate).squeeze(-1)
        predicted_delta_latent = torch.where(
            cell_mask.unsqueeze(-1), predicted_delta_latent, 0.0
        )
        router_logits = router_logits.masked_fill(~cell_mask, -torch.inf)
        predicted_parent_delta_latent = None
        child_gate_logits = None
        if self.cold_start_gate_enabled:
            parent_candidate = torch.cat(
                [parent, parent, perturbation_query], dim=-1
            )
            predicted_parent_delta_latent = self.delta_latent_head(parent_candidate)
            child_gate_logits = self.child_gate_head(candidate).squeeze(-1)
            child_gate_logits = child_gate_logits.masked_fill(~cell_mask, -torch.inf)

        return CellDETRContext(
            control_prototypes=padded_controls,
            control_features=control_features,
            anchor_mask=cell_mask,
            occupancy=cell_occupancy,
            child_tokens=child,
            parent_token=parent,
            perturbation_query=perturbation_query,
            routed_token=routed,
            predicted_delta_latent=predicted_delta_latent,
            router_logits=router_logits,
            compactness_loss=compactness,
            separation_loss=separation,
            bank_inverse=bank_inverse,
            parent_control=parent_control,
            parent_control_feature=parent_control_feature,
            predicted_parent_delta_latent=predicted_parent_delta_latent,
            child_gate_logits=child_gate_logits,
        )

    def decode_selected_delta(self, context: CellDETRContext, child_indices):
        if child_indices.ndim != 2:
            raise ValueError("child_indices must have shape [B,R]")
        batch = torch.arange(child_indices.shape[0], device=child_indices.device)[:, None]
        child = context.child_tokens[batch, child_indices]
        repeats = child_indices.shape[1]
        parent = context.parent_token[:, None].expand(-1, repeats, -1)
        query = context.perturbation_query[:, None].expand(-1, repeats, -1)
        routed = context.routed_token[:, None].expand(-1, repeats, -1)
        return self.delta_gene_head(torch.cat([child, parent, query, routed], dim=-1))

    def decode_weighted_delta(
        self,
        context: CellDETRContext,
        child_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the exact weighted mean of all Child gene-space deltas.

        This equals decoding every Child and taking its weighted mean, while
        applying the expensive final gene projection only once.
        """

        if not torch.is_tensor(child_weights):
            raise TypeError("child_weights must be a torch.Tensor")
        if not child_weights.is_floating_point():
            raise TypeError("child_weights must be floating point")
        if child_weights.shape != context.anchor_mask.shape:
            raise ValueError("child_weights must have shape [B,K]")
        if child_weights.device != context.child_tokens.device:
            raise ValueError("child_weights must share the context device")
        if not torch.isfinite(child_weights).all():
            raise ValueError("child_weights must be finite")
        if (child_weights < 0.0).any():
            raise ValueError("child_weights cannot be negative")
        active_mask = context.anchor_mask.to(
            device=child_weights.device, dtype=torch.bool
        )
        inactive_weights = child_weights.masked_select(~active_mask)
        if inactive_weights.numel() and (inactive_weights > 1.0e-7).any():
            raise ValueError("child_weights place mass on a padded Child")
        mass = child_weights.sum(dim=-1, keepdim=True)
        if not torch.allclose(
            mass,
            torch.ones_like(mass),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("each child_weights row must sum to one")

        batch_size, _, hidden_dim = context.child_tokens.shape
        for name, value in (
            ("parent_token", context.parent_token),
            ("perturbation_query", context.perturbation_query),
            ("routed_token", context.routed_token),
        ):
            if value.shape != (batch_size, hidden_dim):
                raise ValueError(f"context.{name} must have shape [B,H]")

        # The last linear projection commutes with a unit-mass weighted sum.
        # Splitting the first matrix avoids a large [B,K,4H] concatenation.
        input_projection = self.delta_gene_head[0]
        activation = self.delta_gene_head[1]
        output_projection = self.delta_gene_head[2]
        child_weight, parent_weight, query_weight, routed_weight = (
            input_projection.weight.split(hidden_dim, dim=1)
        )
        preactivation = torch.nn.functional.linear(
            context.child_tokens,
            child_weight,
            input_projection.bias,
        )
        shared = (
            torch.nn.functional.linear(
                context.parent_token, parent_weight, None
            )
            + torch.nn.functional.linear(
                context.perturbation_query, query_weight, None
            )
            + torch.nn.functional.linear(
                context.routed_token, routed_weight, None
            )
        )
        hidden = activation(preactivation + shared[:, None, :])
        weights = child_weights.to(hidden)
        weighted_hidden = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        weighted_delta = torch.nn.functional.linear(
            weighted_hidden, output_projection.weight, None
        )
        if output_projection.bias is not None:
            weighted_delta = (
                weighted_delta
                + mass.to(weighted_delta) * output_projection.bias
            )
        return weighted_delta

    def decode_parent_delta(self, context: CellDETRContext):
        if not self.cold_start_gate_enabled:
            raise RuntimeError("Parent delta decoding requires cold_start_gate_enabled")
        return self.delta_gene_head(
            torch.cat(
                [
                    context.parent_token,
                    context.parent_token,
                    context.perturbation_query,
                    context.routed_token,
                ],
                dim=-1,
            )
        )

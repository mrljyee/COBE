from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def chebyshev_basis(x: torch.Tensor, degree: int) -> tuple[torch.Tensor, ...]:
    """Return T_0(x), ..., T_degree(x) using the stable Chebyshev recurrence."""

    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}.")
    basis = [torch.ones_like(x)]
    if degree == 0:
        return tuple(basis)
    basis.append(x)
    for _ in range(2, degree + 1):
        basis.append(2.0 * x * basis[-1] - basis[-2])
    return tuple(basis)


@dataclass(frozen=True)
class RelationDiagnostics:
    local: torch.Tensor
    relation: torch.Tensor
    attention: torch.Tensor
    gate: torch.Tensor


class NonlocalPolynomialRelation1D(nn.Module):
    """Attention-weighted, low-rank polynomial interactions between sequence positions.

    For each head, this module implements the separable relation

        r_ij = sum_{p,q} c[p,q] T_p(a_i) * T_q(b_j)
        R_i  = sum_j attention[i,j] r_ij.

    The implementation evaluates it as

        R_i = sum_{p,q} c[p,q] T_p(a_i)
              * (sum_j attention[i,j] T_q(b_j)),

    which is algebraically identical but avoids materializing a [L,L,D] tensor.
    """

    def __init__(
        self,
        channels: int,
        relation_dim: int = 16,
        num_heads: int = 2,
        degree: int = 3,
        relation_mode: str = "joint",
        nonlocal_radius: int = 2,
        max_relative_distance: int = 512,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if relation_dim <= 0 or relation_dim % num_heads != 0:
            raise ValueError("relation_dim must be positive and divisible by num_heads.")
        if degree < 2:
            raise ValueError("degree must be at least 2 to contain a cross term.")
        valid_modes = {
            "joint",
            "attention_only",
            "polynomial_only",
            "shuffled_joint",
            "uniform_global",
            "position_only",
        }
        if relation_mode not in valid_modes:
            raise ValueError(
                f"relation_mode must be one of {sorted(valid_modes)}, got {relation_mode!r}."
            )
        if nonlocal_radius < 0:
            raise ValueError("nonlocal_radius must be non-negative.")
        if max_relative_distance < 1:
            raise ValueError("max_relative_distance must be positive.")

        self.channels = int(channels)
        self.relation_dim = int(relation_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.relation_dim // self.num_heads
        self.degree = int(degree)
        self.relation_mode = relation_mode
        self.nonlocal_radius = int(nonlocal_radius)
        self.max_relative_distance = int(max_relative_distance)

        self.norm = nn.LayerNorm(self.channels)
        self.query = nn.Linear(self.channels, self.relation_dim, bias=False)
        self.key = nn.Linear(self.channels, self.relation_dim, bias=False)
        self.left = nn.Linear(self.channels, self.relation_dim, bias=False)
        self.right = nn.Linear(self.channels, self.relation_dim, bias=False)
        self.relative_bias = nn.Embedding(
            2 * self.max_relative_distance + 1,
            self.num_heads,
        )
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.output = nn.Linear(self.relation_dim, self.channels, bias=False)

        pair_mask = torch.zeros(self.degree + 1, self.degree + 1, dtype=torch.bool)
        for p in range(1, self.degree + 1):
            for q in range(1, self.degree + 1):
                if p + q <= self.degree:
                    pair_mask[p, q] = True
        self.register_buffer("pair_mask", pair_mask, persistent=True)

        coefficients = torch.zeros(
            self.num_heads,
            self.degree + 1,
            self.degree + 1,
        )
        coefficients[:, 1, 1] = 1.0
        self.coefficients = nn.Parameter(coefficients)

        nn.init.zeros_(self.relative_bias.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _relative_position_bias(
        self,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]
        relative = relative.clamp(
            -self.max_relative_distance,
            self.max_relative_distance,
        )
        relative = relative + self.max_relative_distance
        bias = self.relative_bias(relative)
        return bias.permute(2, 0, 1)

    def _local_exclusion_mask(
        self,
        length: int,
        device: torch.device,
        excluded_offsets: Sequence[int] | None = None,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]
        if excluded_offsets is not None:
            exclusion = torch.zeros_like(relative, dtype=torch.bool)
            for offset in excluded_offsets:
                if int(offset) != 0:
                    exclusion |= relative == int(offset)
            return exclusion
        distance = relative.abs()
        # Keep the diagonal as a stable fallback, but exclude nearby neighbours that
        # are already covered by the local convolution branch.
        return (distance > 0) & (distance <= self.nonlocal_radius)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        key_valid_mask: torch.Tensor | None = None,
        excluded_offsets: Sequence[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(
                f"Expected [B,C,L] features, received shape {tuple(features.shape)}."
            )
        if features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {features.shape[1]}."
            )

        x = features.transpose(1, 2)
        x = self.norm(x)
        a = torch.tanh(self._split_heads(self.left(x)))
        b = torch.tanh(self._split_heads(self.right(x)))

        if self.relation_mode == "polynomial_only":
            attention = None
            if return_attention:
                attention = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
                attention = attention[None, None].expand(
                    x.shape[0], self.num_heads, x.shape[1], x.shape[1]
                )
        else:
            length = x.shape[1]
            exclusion = self._local_exclusion_mask(
                length,
                x.device,
                excluded_offsets=excluded_offsets,
            )
            allowed = (~exclusion)[None, None].expand(
                x.shape[0], self.num_heads, length, length
            )
            if key_valid_mask is not None:
                valid = key_valid_mask.to(device=x.device, dtype=torch.bool)
                if valid.ndim == 1:
                    valid = valid[None].expand(x.shape[0], -1)
                if valid.shape != (x.shape[0], length):
                    raise ValueError(
                        "key_valid_mask must have shape [L] or [B,L], got "
                        f"{tuple(key_valid_mask.shape)} for B={x.shape[0]}, L={length}."
                    )
                valid_keys = valid[:, None, None, :]
                diagonal = torch.eye(length, dtype=torch.bool, device=x.device)[None, None]
                # Invalid positions may still query reliable positions. Their diagonal is
                # retained only as a guaranteed numerical fallback.
                allowed = allowed & (valid_keys | diagonal)
            if self.relation_mode == "uniform_global":
                attention = allowed.to(dtype=x.dtype)
                attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                relative_bias = self._relative_position_bias(length, x.device)[None]
                if self.relation_mode == "position_only":
                    scores = relative_bias.expand(
                        x.shape[0], self.num_heads, length, length
                    )
                else:
                    q = self._split_heads(self.query(x))
                    k = self._split_heads(self.key(x))
                    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                    scores = scores + relative_bias
                scores = scores.masked_fill(~allowed, float("-inf"))
                attention = torch.softmax(scores, dim=-1)
                attention = self.attention_dropout(attention)

        if self.relation_mode == "shuffled_joint":
            # Deliberately break the selected-key/value pairing while preserving
            # tensor shapes, parameter count, and attention computation.
            b = torch.roll(b, shifts=max(1, x.shape[1] // 3), dims=2)

        basis_a = chebyshev_basis(a, self.degree)
        basis_b = chebyshev_basis(b, self.degree)
        if self.relation_mode == "attention_only":
            relation = torch.matmul(attention, basis_b[1])
            batch, _, length, _ = relation.shape
            relation = relation.transpose(1, 2).reshape(batch, length, self.relation_dim)
            relation = self.output(relation).transpose(1, 2)
            if return_attention:
                return relation, attention
            return relation

        context_b = [None]
        for q_degree in range(1, self.degree + 1):
            if self.relation_mode == "polynomial_only":
                context_b.append(basis_b[q_degree])
            else:
                context_b.append(torch.matmul(attention, basis_b[q_degree]))

        relation = torch.zeros_like(a)
        active_coefficients = self.coefficients * self.pair_mask[None]
        for p in range(1, self.degree + 1):
            for q_degree in range(1, self.degree + 1):
                if not bool(self.pair_mask[p, q_degree]):
                    continue
                coefficient = active_coefficients[:, p, q_degree][None, :, None, None]
                relation = relation + coefficient * basis_a[p] * context_b[q_degree]

        batch, _, length, _ = relation.shape
        relation = relation.transpose(1, 2).reshape(batch, length, self.relation_dim)
        relation = self.output(relation).transpose(1, 2)
        if return_attention:
            return relation, attention
        return relation


class RelationConv1D(nn.Module):
    """A conventional Conv1D augmented by a gated nonlocal polynomial relation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int | None = None,
        activation: str = "relu",
        relation_dim: int = 16,
        num_heads: int = 2,
        degree: int = 3,
        relation_mode: str = "joint",
        nonlocal_radius: int = 2,
        max_relative_distance: int = 512,
        attention_dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.local_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        if activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}.")

        self.relation = NonlocalPolynomialRelation1D(
            channels=out_channels,
            relation_dim=relation_dim,
            num_heads=num_heads,
            degree=degree,
            relation_mode=relation_mode,
            nonlocal_radius=nonlocal_radius,
            max_relative_distance=max_relative_distance,
            attention_dropout=attention_dropout,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RelationDiagnostics]:
        local = self.activation(self.local_conv(x))
        if return_diagnostics:
            relation, attention = self.relation(local, return_attention=True)
        else:
            relation = self.relation(local)
            attention = None
        gate = torch.tanh(self.gate)
        output = local + gate * relation
        if return_diagnostics:
            diagnostics = RelationDiagnostics(
                local=local,
                relation=relation,
                attention=attention,
                gate=gate,
            )
            return output, diagnostics
        return output


def relation_parameter_summary(module: nn.Module) -> dict[str, Any]:
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "relation_gates": {
            name: float(torch.tanh(child.gate).detach().cpu())
            for name, child in module.named_modules()
            if isinstance(child, RelationConv1D)
        },
    }

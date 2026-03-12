"""Actor networks for SmartMARL: GATv2 and MLP alternatives."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Actor(nn.Module):
    """
    True GATv2-style dynamic attention actor.

    e_ij = a(W * concat(h_i, h_j)), then softmax over incoming neighbors.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 2,
        head_dim: int = 64,
        num_phases: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_phases = num_phases

        self.pair_linears = nn.ModuleList([nn.Linear(2 * input_dim, head_dim) for _ in range(num_heads)])
        self.attn_linears = nn.ModuleList([nn.Linear(head_dim, 1, bias=False) for _ in range(num_heads)])
        self.value_linears = nn.ModuleList([nn.Linear(input_dim, head_dim, bias=False) for _ in range(num_heads)])

        self.out_proj = nn.Linear(num_heads * head_dim, input_dim)
        self.phase_head = nn.Linear(input_dim, num_phases)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.last_attention: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _add_self_loops(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        loops = torch.arange(num_nodes, device=device)
        self_edges = torch.stack([loops, loops], dim=0)
        if edge_index.numel() == 0:
            return self_edges
        return torch.cat([edge_index, self_edges], dim=1)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        out = torch.zeros_like(scores)
        for seg in range(num_segments):
            mask = index == seg
            if torch.any(mask):
                out[mask] = torch.softmax(scores[mask], dim=0)
        return out

    def compute_attention(
        self,
        h_int: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        device = h_int.device
        n = h_int.shape[0]
        edges = self._add_self_loops(edge_index.long().to(device), n, device)
        src, dst = edges[0], edges[1]

        attention_per_head: Dict[int, torch.Tensor] = {}
        for head in range(self.num_heads):
            pair = torch.cat([h_int[src], h_int[dst]], dim=-1)
            logits = self.attn_linears[head](self.leaky_relu(self.pair_linears[head](pair))).squeeze(-1)
            alpha = self._segment_softmax(logits, dst, n)
            attention_per_head[head] = alpha
        return attention_per_head

    def forward(
        self,
        h_int: torch.Tensor,
        edge_index: torch.Tensor,
        feasible_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        device = h_int.device
        n = h_int.shape[0]
        edges = self._add_self_loops(edge_index.long().to(device), n, device)
        src, dst = edges[0], edges[1]

        head_outputs = []
        self.last_attention = {}

        for head in range(self.num_heads):
            pair = torch.cat([h_int[src], h_int[dst]], dim=-1)
            logits = self.attn_linears[head](self.leaky_relu(self.pair_linears[head](pair))).squeeze(-1)
            alpha = self._segment_softmax(logits, dst, n)
            self.last_attention[head] = alpha.detach()

            msg = self.value_linears[head](h_int[src]) * alpha.unsqueeze(-1)
            out = torch.zeros((n, self.head_dim), device=device, dtype=h_int.dtype)
            out.index_add_(0, dst, msg)
            head_outputs.append(out)

        h = torch.cat(head_outputs, dim=-1)
        h = torch.relu(self.out_proj(h))
        logits = self.phase_head(h)

        if feasible_mask is not None:
            mask = feasible_mask.to(dtype=torch.bool)
            logits = logits.masked_fill(~mask, -1e9)

        probs = torch.softmax(logits, dim=-1)
        if return_attention:
            return probs, self.last_attention
        return probs


class MLPActor(nn.Module):
    """Ablation actor: 2-layer MLP (128->64->num_phases)."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_phases: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases),
        )

    def forward(self, h_int: torch.Tensor, *_args, feasible_mask: Optional[torch.Tensor] = None, **_kwargs):
        logits = self.net(h_int)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask.bool(), -1e9)
        return torch.softmax(logits, dim=-1)

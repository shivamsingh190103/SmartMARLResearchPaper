"""GPLight-style grouped homogeneous graph baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicGroupAssigner:
    positions: np.ndarray
    num_groups: int = 4
    num_iters: int = 6

    def assign(self, traffic_features: np.ndarray) -> np.ndarray:
        if traffic_features.ndim != 2:
            raise ValueError(f"Expected 2D traffic features, got {traffic_features.shape}")

        n = traffic_features.shape[0]
        k = max(1, min(int(self.num_groups), n))
        pos = np.asarray(self.positions, dtype=np.float32)
        pos = pos / np.maximum(np.max(pos, axis=0, keepdims=True), 1.0)

        dyn = np.asarray(traffic_features, dtype=np.float32)
        dyn_std = np.std(dyn, axis=0, keepdims=True)
        dyn = (dyn - np.mean(dyn, axis=0, keepdims=True)) / np.where(dyn_std < 1e-6, 1.0, dyn_std)

        feats = np.concatenate([pos, dyn], axis=1)
        init_idx = np.linspace(0, n - 1, num=k, dtype=int)
        centers = feats[init_idx].copy()
        labels = np.zeros(n, dtype=np.int64)

        for _ in range(self.num_iters):
            dist2 = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1)
            for g in range(k):
                mask = labels == g
                if np.any(mask):
                    centers[g] = feats[mask].mean(axis=0)

        return labels


class GPLightEncoder(nn.Module):
    """Homogeneous spatial encoder used by the grouped baseline."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_proj = nn.LazyLinear(hidden_dim)
        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.neighbor_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

    @staticmethod
    def _aggregate(messages: torch.Tensor, target_idx: torch.Tensor, num_targets: int) -> torch.Tensor:
        out = torch.zeros((num_targets, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
        count = torch.zeros((num_targets, 1), device=messages.device, dtype=messages.dtype)
        out.index_add_(0, target_idx, messages)
        ones = torch.ones((target_idx.shape[0], 1), device=messages.device, dtype=messages.dtype)
        count.index_add_(0, target_idx, ones)
        return out / count.clamp_min(1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        if edge_index.numel() == 0:
            for layer in range(self.num_layers):
                h = F.elu(self.self_linears[layer](h))
            return h

        src_idx = edge_index[0].long().to(h.device)
        dst_idx = edge_index[1].long().to(h.device)
        for layer in range(self.num_layers):
            msg = self.neighbor_linears[layer](h[src_idx])
            agg = self._aggregate(msg, dst_idx, h.shape[0])
            h = F.elu(self.self_linears[layer](h) + agg)
        return h


class GPLightActor(nn.Module):
    """Group-specific policy heads over a shared spatial encoder."""

    def __init__(self, input_dim: int = 128, num_phases: int = 4, num_groups: int = 4) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_phases = num_phases
        self.group_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_phases),
                )
                for _ in range(num_groups)
            ]
        )

    def forward(
        self,
        h_int: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        feasible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = torch.zeros((h_int.shape[0], self.num_phases), device=h_int.device, dtype=h_int.dtype)
        group_ids = group_ids.long().to(h_int.device)

        for group_idx, head in enumerate(self.group_heads):
            mask = group_ids == group_idx
            if torch.any(mask):
                logits[mask] = head(h_int[mask])

        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask.bool(), -1e9)
        return torch.softmax(logits, dim=-1)


__all__ = ["DynamicGroupAssigner", "GPLightActor", "GPLightEncoder"]

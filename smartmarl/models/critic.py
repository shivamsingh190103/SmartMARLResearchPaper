"""Centralized critic used only during training (CTDE)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents: int = 25, embed_dim: int = 128) -> None:
        super().__init__()
        input_dim = num_agents * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, h_all_agents: torch.Tensor) -> torch.Tensor:
        if h_all_agents.ndim == 2:
            h_all_agents = h_all_agents.reshape(1, -1)
        elif h_all_agents.ndim == 3:
            h_all_agents = h_all_agents.reshape(h_all_agents.shape[0], -1)
        return self.net(h_all_agents)

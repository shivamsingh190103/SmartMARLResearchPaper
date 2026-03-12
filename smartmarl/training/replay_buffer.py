"""Trajectory storage for on-policy MA2C updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class TrajectoryBuffer:
    log_probs: List[torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    embeddings: List[torch.Tensor] = field(default_factory=list)

    def add_step(
        self,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        embedding: torch.Tensor,
    ) -> None:
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.rewards.append(float(reward))
        self.values.append(value)
        self.embeddings.append(embedding)

    def clear(self) -> None:
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()
        self.values.clear()
        self.embeddings.clear()

    def discounted_returns(self, gamma: float, device: torch.device) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(self.rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.rewards)

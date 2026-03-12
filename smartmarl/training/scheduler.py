"""Episode-based learning rate scheduler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeLRScheduler:
    optimizer: any
    decay_factor: float = 0.99
    decay_every_n_episodes: int = 100

    def step(self, episode_idx: int) -> None:
        if episode_idx <= 0:
            return
        if episode_idx % self.decay_every_n_episodes != 0:
            return

        for group in self.optimizer.param_groups:
            group["lr"] *= self.decay_factor

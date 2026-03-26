"""Rule-based traffic signal baselines for paper comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class GridTopology:
    grid_size: int
    num_intersections: int

    def downstream(self, index: int) -> List[int]:
        g = self.grid_size
        r, c = divmod(index, g)
        out: List[int] = []
        if c + 1 < g:
            out.append(index + 1)
        if r + 1 < g:
            out.append(index + g)
        return out


def fixed_time_actions(
    *,
    current_step: int,
    num_intersections: int,
    num_phases: int,
    cycle_length: int = 60,
) -> np.ndarray:
    phase_duration = max(1, int(cycle_length) // max(1, int(num_phases)))
    actions = np.zeros(num_intersections, dtype=np.int64)
    for i in range(num_intersections):
        local_step = current_step + (i % phase_duration)
        actions[i] = int((local_step // phase_duration) % num_phases)
    return actions


def maxpressure_actions(
    *,
    obs: Dict,
    current_step: int,
    topology: GridTopology,
    num_phases: int,
) -> np.ndarray:
    queue = np.asarray(obs["queue_per_intersection"], dtype=np.float32)
    n = topology.num_intersections
    actions = np.zeros(n, dtype=np.int64)
    threshold = 1.0

    for idx in range(n):
        downstream = topology.downstream(idx)
        downstream_mean = float(np.mean(queue[downstream])) if downstream else 0.0
        pressure = float(queue[idx] - downstream_mean)

        if num_phases <= 2:
            actions[idx] = 0 if pressure >= 0.0 else 1
        else:
            if pressure > threshold:
                actions[idx] = 0
            elif pressure < -threshold:
                actions[idx] = 2 if num_phases > 2 else 1
            else:
                actions[idx] = 1 if (current_step + idx) % 2 == 0 else min(3, num_phases - 1)
    return actions


def evaluate_policy(
    *,
    env,
    action_fn: Callable[[Dict, int], np.ndarray],
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Dict:
    att = []
    awt = []
    tput = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for step in range(steps_per_episode):
            actions = action_fn(obs, step)
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        completed = max(int(env.stats.completed_vehicles), 1)
        att.append(float(env.stats.total_travel_time / completed))
        awt.append(float(env.stats.total_waiting_time / completed))
        tput.append(float(env.stats.completed_vehicles / (steps_per_episode / 3600.0)))

    return {
        "ATT": float(np.mean(att)) if att else 0.0,
        "AWT": float(np.mean(awt)) if awt else 0.0,
        "Throughput": float(np.mean(tput)) if tput else 0.0,
        "ATT_runs": att,
        "AWT_runs": awt,
        "Throughput_runs": tput,
    }

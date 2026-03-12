"""Traffic metrics for SmartMARL experiments."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def average_travel_time(trips: Sequence[Tuple[float, float]]) -> float:
    if not trips:
        return 0.0
    durations = [arr - dep for dep, arr in trips if arr >= dep]
    if not durations:
        return 0.0
    return float(np.mean(durations))


def average_waiting_time(waiting_times: Sequence[float]) -> float:
    if not waiting_times:
        return 0.0
    return float(np.mean(waiting_times))


def throughput_per_hour(completed_vehicles: int, sim_seconds: float) -> float:
    if sim_seconds <= 0:
        return 0.0
    return float(completed_vehicles / (sim_seconds / 3600.0))


def aukf_retention_rate(mse_clear: float, mse_condition: float) -> float:
    if mse_condition <= 1e-12:
        return 0.0
    return float(mse_clear / mse_condition)


def compute_metrics(
    completed_vehicles: int,
    total_waiting_time: float,
    total_travel_time: float,
    sim_seconds: float,
) -> dict:
    completed = max(int(completed_vehicles), 1)
    att = float(total_travel_time / completed)
    awt = float(total_waiting_time / completed)
    throughput = throughput_per_hour(completed_vehicles, sim_seconds)
    return {"ATT": att, "AWT": awt, "Throughput": throughput}

"""77 GHz radar perception noise model used in SmartMARL simulation."""

from __future__ import annotations

from typing import List

import numpy as np

from .noise_injection import (
    add_radar_spurious_returns,
    apply_radar_noise,
    condition_radar_position_sigma,
    condition_radar_velocity_sigma,
)


class RadarPerceptionModel:
    def __init__(self, condition: str = "clear", seed: int = 0) -> None:
        self.condition = condition
        self.rng = np.random.default_rng(seed)

    def set_condition(self, condition: str) -> None:
        self.condition = condition

    def process(self, vehicle_positions: np.ndarray) -> List[dict]:
        vehicle_positions = np.asarray(vehicle_positions, dtype=np.float32)
        if vehicle_positions.size == 0:
            return []

        ranges = np.linalg.norm(vehicle_positions, axis=1)
        noisy_ranges = apply_radar_noise(ranges, self.condition, self.rng)
        velocity_sigma = condition_radar_velocity_sigma(self.condition)
        position_sigma = condition_radar_position_sigma(self.condition)
        radial_vel = self.rng.normal(0.0, velocity_sigma, size=len(vehicle_positions))

        detections: List[dict] = []
        for (x, y), r, v in zip(vehicle_positions, noisy_ranges, radial_vel):
            detections.append(
                {
                    "x": float(x + self.rng.normal(0.0, position_sigma)),
                    "y": float(y + self.rng.normal(0.0, position_sigma)),
                    "range": float(max(r, 0.0)),
                    "velocity": float(v),
                    "spurious": False,
                }
            )

        return add_radar_spurious_returns(detections, self.condition, self.rng)


class RadarProcessor(RadarPerceptionModel):
    """Backward-compatible alias for the legacy class name."""


__all__ = ["RadarPerceptionModel", "RadarProcessor"]

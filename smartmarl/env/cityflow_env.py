"""Simplified CityFlow wrapper that mirrors the SUMO environment API."""

from __future__ import annotations

from typing import Any, Optional

from .sumo_env import SumoTrafficEnv


class CityFlowTrafficEnv(SumoTrafficEnv):
    """
    A light wrapper around the mock dynamics used for SUMO.

    This keeps API parity for fine-tuning while allowing replacement with a real
    CityFlow backend later.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        scenario: str = "standard",
        episode_length_seconds: int = 3600,
        num_intersections: int = 25,
        num_phases: int = 4,
        min_green_time_seconds: int = 5,
        seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__(
            config_path=config_path,
            scenario=scenario,
            episode_length_seconds=episode_length_seconds,
            num_intersections=num_intersections,
            num_phases=num_phases,
            min_green_time_seconds=min_green_time_seconds,
            seed=seed,
            use_traci=False,
        )

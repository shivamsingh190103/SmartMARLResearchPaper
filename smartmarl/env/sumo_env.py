"""SUMO environment wrapper with TraCI integration and a robust mock fallback."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import shutil
import socket
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = object  # type: ignore
    spaces = None  # type: ignore

try:
    import traci  # type: ignore
except Exception:  # pragma: no cover
    traci = None  # type: ignore


@dataclass
class EpisodeStats:
    completed_vehicles: int = 0
    total_waiting_time: float = 0.0
    total_travel_time: float = 0.0


class SumoTrafficEnv(gym.Env):
    """
    SUMO TraCI wrapper for multi-intersection signal control.

    If TraCI or SUMO binaries are unavailable, it transparently falls back to a
    deterministic mock simulator so training scripts still run end-to-end.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: Optional[str] = None,
        scenario: str = "standard",
        episode_length_seconds: int = 3600,
        num_intersections: int = 25,
        num_phases: int = 4,
        min_green_time_seconds: int = 5,
        seed: int = 0,
        use_traci: bool = True,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.scenario = scenario
        self.episode_length_seconds = int(episode_length_seconds)
        self.num_intersections = int(num_intersections)
        self.num_phases = int(num_phases)
        self.min_green_time_seconds = int(min_green_time_seconds)
        self.use_traci_requested = use_traci
        self._use_traci = False

        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.reward_mode = "normal"
        self.ev_active = False
        self._ev_timer = 0
        grid_side = max(1, int(np.sqrt(self.num_intersections)))
        mid_row = max(0, grid_side // 2)
        self.ev_corridor = [
            mid_row * grid_side + col for col in range(grid_side) if mid_row * grid_side + col < self.num_intersections
        ]
        self.ev_start_step = 300
        self.ev_duration_steps = 60
        self.ev_phase = 0
        self._scheduled_ev = False
        self.stats = EpisodeStats()

        self.phase = np.zeros(self.num_intersections, dtype=np.int64)
        self.elapsed_green = np.zeros(self.num_intersections, dtype=np.float32)
        self.queue = np.zeros(self.num_intersections, dtype=np.float32)
        self.delay = np.zeros(self.num_intersections, dtype=np.float32)
        self.vehicle_count = np.zeros(self.num_intersections, dtype=np.float32)
        self.speed = np.zeros(self.num_intersections, dtype=np.float32)
        self.occupancy = np.zeros(self.num_intersections, dtype=np.float32)
        self.incident_elapsed = np.zeros(self.num_intersections, dtype=np.float32)

        self._trl_ids: list[str] = []
        self._intersection_lanes: list[list[str]] = []
        self.mock_mode = True
        self._vehicle_depart_time: dict[str, float] = {}
        self._vehicle_wait_time: dict[str, float] = {}

        if spaces is not None:
            self.action_space = spaces.MultiDiscrete([self.num_phases] * self.num_intersections)
            self.observation_space = spaces.Dict(
                {
                    "intersection_features": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_intersections, 3),
                        dtype=np.float32,
                    ),
                    "lane_features": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_intersections, 4),
                        dtype=np.float32,
                    ),
                    "incidents": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_intersections, 4),
                        dtype=np.float32,
                    ),
                }
            )

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if not self.use_traci_requested:
            self._use_traci = False
            self.mock_mode = True
            return

        if traci is None:
            warnings.warn("TraCI not available; using SmartMARL mock SUMO backend.")
            self._use_traci = False
            self.mock_mode = True
            return

        if not self.config_path or not Path(self.config_path).exists():
            warnings.warn("SUMO config not found; using SmartMARL mock SUMO backend.")
            self._use_traci = False
            self.mock_mode = True
            return

        sumo_binary = self._resolve_sumo_binary()
        command = [
            sumo_binary,
            "-c",
            str(self.config_path),
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        try:
            traci.start(command, port=self._get_free_port())
            self._use_traci = True
            self.mock_mode = False
            self._trl_ids = list(traci.trafficlight.getIDList())[: self.num_intersections]
            self._intersection_lanes = [
                list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
                for tl_id in self._trl_ids
            ]
            traci.close()
        except Exception as exc:  # pragma: no cover - depends on local SUMO install
            warnings.warn(f"Unable to start TraCI ({exc}); using mock backend.")
            self._use_traci = False
            self.mock_mode = True

    @staticmethod
    def _resolve_sumo_binary() -> str:
        env_binary = os.environ.get("SUMO_BINARY", "").strip()
        if env_binary:
            return env_binary

        local_binary = Path(".venv/bin/sumo")
        if local_binary.exists():
            return str(local_binary)

        which_sumo = shutil.which("sumo")
        if which_sumo:
            return which_sumo

        return "sumo"

    @staticmethod
    def _get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return int(s.getsockname()[1])

    def set_reward_mode(self, mode: str) -> None:
        if mode not in {"normal", "ev"}:
            raise ValueError("reward mode must be 'normal' or 'ev'")
        self.reward_mode = mode

    def configure_ev_corridor(
        self,
        corridor_indices: Optional[list[int]] = None,
        start_step: int = 300,
        duration_steps: int = 60,
        preferred_phase: int = 0,
    ) -> None:
        if corridor_indices is not None:
            self.ev_corridor = [idx for idx in corridor_indices if 0 <= idx < self.num_intersections]
        self.ev_start_step = max(0, int(start_step))
        self.ev_duration_steps = max(1, int(duration_steps))
        self.ev_phase = int(preferred_phase)
        self._scheduled_ev = True

    def recommended_ev_actions(self, base_actions: Optional[np.ndarray] = None) -> np.ndarray:
        if base_actions is None:
            actions = np.zeros(self.num_intersections, dtype=np.int64)
        else:
            actions = np.asarray(base_actions, dtype=np.int64).copy()
        for idx in self.ev_corridor:
            actions[idx] = self.ev_phase
        return actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.stats = EpisodeStats()
        self.phase.fill(0)
        self.elapsed_green.fill(0.0)
        self.queue = self.rng.uniform(2.0, 8.0, size=self.num_intersections).astype(np.float32)
        self.delay = self.rng.uniform(5.0, 15.0, size=self.num_intersections).astype(np.float32)
        self.vehicle_count = self.queue + self.rng.uniform(2.0, 10.0, size=self.num_intersections).astype(np.float32)
        self.speed = np.clip(14.0 - 0.5 * self.queue, 0.2, 15.0).astype(np.float32)
        self.occupancy = np.clip(self.queue / 30.0, 0.0, 1.0).astype(np.float32)
        self.incident_elapsed.fill(0.0)
        self._vehicle_depart_time.clear()
        self._vehicle_wait_time.clear()
        self.ev_active = False
        self._ev_timer = 0

        if self._use_traci:
            self._reset_traci(seed)
            self._sync_from_traci()

        obs = self._build_observation()
        return obs, {"backend": "traci" if self._use_traci else "mock"}

    def _reset_traci(self, seed: Optional[int]) -> None:
        if not self.config_path:
            return
        sumo_binary = self._resolve_sumo_binary()
        command = [
            sumo_binary,
            "-c",
            str(self.config_path),
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])

        try:
            if traci.isLoaded():
                traci.close()
            traci.start(command, port=self._get_free_port())
            self._trl_ids = list(traci.trafficlight.getIDList())[: self.num_intersections]
            self._intersection_lanes = [
                list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
                for tl_id in self._trl_ids
            ]
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"TraCI reset failed ({exc}); falling back to mock backend.")
            self._use_traci = False
            self.mock_mode = True

    def step(self, actions: np.ndarray):
        action_arr = np.asarray(actions, dtype=np.int64).reshape(self.num_intersections)

        if self._use_traci:
            self._step_traci(action_arr)
            self._sync_from_traci()
        else:
            self._step_mock(action_arr)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length_seconds
        truncated = False

        rewards = self._compute_rewards()
        obs = self._build_observation()
        info = {
            "ev_active": self.ev_active,
            "ev_corridor": list(self.ev_corridor),
            "ev_travel_time": float(40.0 + 0.8 * np.mean(self.queue)),
            "network_penalty": float(np.mean(self.delay)),
            "completed_vehicles": self.stats.completed_vehicles,
            "total_waiting_time": self.stats.total_waiting_time,
            "total_travel_time": self.stats.total_travel_time,
        }
        return obs, rewards.astype(np.float32), terminated, truncated, info

    def _step_traci(self, actions: np.ndarray) -> None:  # pragma: no cover - depends on SUMO install
        for idx, desired_phase in enumerate(actions):
            if idx >= len(self._trl_ids):
                break
            self.elapsed_green[idx] += 1.0
            if self.elapsed_green[idx] >= self.min_green_time_seconds:
                self.phase[idx] = int(desired_phase)
                self.elapsed_green[idx] = 0.0
            tl_id = self._trl_ids[idx]
            try:
                traci.trafficlight.setPhase(tl_id, int(self.phase[idx]))
            except Exception:
                continue

        traci.simulationStep()
        self._update_traci_trip_stats()

    def _step_mock(self, actions: np.ndarray) -> None:
        self.elapsed_green += 1.0
        can_switch = self.elapsed_green >= self.min_green_time_seconds
        switched = can_switch & (actions != self.phase)
        self.phase = np.where(switched, actions, self.phase)
        self.elapsed_green = np.where(switched, 0.0, self.elapsed_green)

        if self._scheduled_ev:
            ev_end = self.ev_start_step + self.ev_duration_steps
            self.ev_active = self.ev_start_step <= self.current_step < ev_end
            self._ev_timer = max(0, ev_end - self.current_step) if self.ev_active else 0
        elif self.reward_mode == "ev":
            if self._ev_timer <= 0 and self.rng.random() < 0.015:
                self.ev_active = True
                self._ev_timer = int(self.rng.integers(20, 80))
            else:
                self._ev_timer -= 1
                if self._ev_timer <= 0:
                    self.ev_active = False
        else:
            self.ev_active = False

        demand = self.rng.poisson(2.0, size=self.num_intersections).astype(np.float32)
        phase_efficiency = 0.9 + 0.2 * (self.phase == 0) + 0.2 * (self.phase == 2)
        if self.ev_active:
            corridor_mask = np.zeros(self.num_intersections, dtype=np.float32)
            corridor_mask[self.ev_corridor] = 1.0
            phase_efficiency += 0.25 * corridor_mask * (self.phase == self.ev_phase)

        service = np.clip(phase_efficiency + self.rng.normal(0, 0.25, size=self.num_intersections), 0.2, 4.0)

        prev_queue = self.queue.copy()
        self.queue = np.clip(self.queue + demand - service, 0.0, None)
        departed = np.clip(prev_queue + demand - self.queue, 0.0, None)

        self.delay = np.clip(self.delay + 0.25 * self.queue + self.rng.normal(0, 0.5, size=self.num_intersections), 0.0, None)
        self.vehicle_count = np.clip(self.queue + self.rng.uniform(2.0, 10.0, size=self.num_intersections), 0.0, None)
        self.speed = np.clip(14.0 - 0.45 * self.queue + self.rng.normal(0, 0.3, size=self.num_intersections), 0.1, 15.0)
        self.occupancy = np.clip(self.queue / 35.0 + self.rng.normal(0, 0.02, size=self.num_intersections), 0.0, 1.0)

        incident_trigger = self.rng.random(self.num_intersections) < 0.01
        self.incident_elapsed = np.where(incident_trigger, 1.0, np.maximum(0.0, self.incident_elapsed - 1.0))

        completed = int(np.sum(departed))
        self.stats.completed_vehicles += completed
        if completed > 0:
            # Calibrated aggregate metrics for realistic ATT/AWT magnitudes in mock mode.
            avg_wait = max(6.0, 12.0 + 0.4 * float(np.mean(self.queue)))
            delay_term = min(120.0, float(np.mean(self.delay)))
            avg_trip = max(60.0, 120.0 + 0.3 * float(np.mean(self.queue)) + 0.05 * delay_term)
            self.stats.total_waiting_time += completed * avg_wait
            self.stats.total_travel_time += completed * avg_trip

    def _sync_from_traci(self) -> None:  # pragma: no cover - depends on SUMO install
        if not self._trl_ids:
            return
        queue = np.zeros(self.num_intersections, dtype=np.float32)
        delay = np.zeros(self.num_intersections, dtype=np.float32)
        speed = np.zeros(self.num_intersections, dtype=np.float32)
        count = np.zeros(self.num_intersections, dtype=np.float32)

        for idx, lanes in enumerate(self._intersection_lanes):
            if idx >= self.num_intersections:
                break
            if not lanes:
                continue
            lane_queue = []
            lane_delay = []
            lane_speed = []
            lane_count = []
            for lane_id in lanes:
                try:
                    lane_queue.append(float(traci.lane.getLastStepHaltingNumber(lane_id)))
                    lane_delay.append(float(traci.lane.getWaitingTime(lane_id)))
                    lane_speed.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
                    lane_count.append(float(traci.lane.getLastStepVehicleNumber(lane_id)))
                except Exception:
                    continue

            if lane_queue:
                queue[idx] = float(np.sum(lane_queue))
                delay[idx] = float(np.mean(lane_delay))
                speed[idx] = float(np.mean(lane_speed))
                count[idx] = float(np.sum(lane_count))

        self.queue = queue
        self.delay = delay
        self.speed = np.clip(speed, 0.0, 15.0)
        self.vehicle_count = count
        self.occupancy = np.clip(queue / 35.0, 0.0, 1.0)

    def _update_traci_trip_stats(self) -> None:  # pragma: no cover - depends on SUMO install
        sim_time = float(traci.simulation.getTime())
        departed_ids = list(traci.simulation.getDepartedIDList())
        arrived_ids = list(traci.simulation.getArrivedIDList())
        active_ids = list(traci.vehicle.getIDList())

        for vid in departed_ids:
            self._vehicle_depart_time[vid] = sim_time
            self._vehicle_wait_time[vid] = 0.0

        for vid in active_ids:
            try:
                self._vehicle_wait_time[vid] = float(traci.vehicle.getAccumulatedWaitingTime(vid))
            except Exception:
                continue

        for vid in arrived_ids:
            dep = self._vehicle_depart_time.pop(vid, None)
            if dep is not None:
                self.stats.completed_vehicles += 1
                self.stats.total_travel_time += max(0.0, sim_time - dep)
            self.stats.total_waiting_time += self._vehicle_wait_time.pop(vid, 0.0)

    def _compute_rewards(self) -> np.ndarray:
        if self.reward_mode == "ev" and self.ev_active:
            t_ev = 40.0 + 0.8 * self.queue
            network_penalty = np.mean(self.delay)
            reward = 0.85 * (-t_ev) + 0.15 * (-network_penalty)
            return reward.astype(np.float32)

        reward = -(0.6 * self.queue + 0.4 * self.delay)
        return reward.astype(np.float32)

    def _build_sensor_measurements(self) -> Dict[str, np.ndarray]:
        base = np.stack([self.queue, self.vehicle_count, self.speed, self.occupancy], axis=1)
        camera_noise = self.rng.normal(0.0, 0.18 if self.scenario == "indian_hetero" else 0.12, size=base.shape)
        radar_noise = self.rng.normal(0.0, 0.10, size=base.shape)

        z_camera = np.clip(base + camera_noise, 0.0, None).astype(np.float32)
        z_radar = np.clip(base + radar_noise, 0.0, None).astype(np.float32)
        return {"camera": z_camera, "radar": z_radar}

    def _build_incident_features(self) -> np.ndarray:
        active = (self.incident_elapsed > 0).astype(np.float32)
        incident_type = active
        severity = active * self.rng.uniform(0.3, 1.0, size=self.num_intersections)
        confidence = active * self.rng.uniform(0.5, 0.95, size=self.num_intersections)
        elapsed = self.incident_elapsed / max(self.min_green_time_seconds, 1)
        return np.stack([incident_type, severity, confidence, elapsed], axis=1).astype(np.float32)

    def _mock_vehicle_positions(self) -> np.ndarray:
        total = int(np.clip(np.sum(self.vehicle_count), 20, 400))
        grid_side = int(np.sqrt(self.num_intersections))
        positions = np.zeros((total, 2), dtype=np.float32)
        for idx in range(total):
            inter = int(self.rng.integers(0, self.num_intersections))
            row, col = divmod(inter, grid_side)
            x = col * 400.0 + self.rng.normal(0, 40)
            y = row * 400.0 + self.rng.normal(0, 40)
            positions[idx] = (x, y)
        return positions

    def _build_observation(self) -> Dict[str, Any]:
        intersection_features = np.stack(
            [
                self.phase.astype(np.float32),
                self.elapsed_green.astype(np.float32),
                self.delay.astype(np.float32),
            ],
            axis=1,
        )

        lane_features = np.stack(
            [
                self.queue.astype(np.float32),
                self.vehicle_count.astype(np.float32),
                self.speed.astype(np.float32),
                self.occupancy.astype(np.float32),
            ],
            axis=1,
        )

        obs = {
            "intersection_features": intersection_features,
            "lane_features": lane_features,
            "sensor_measurements": self._build_sensor_measurements(),
            "incidents": self._build_incident_features(),
            "vehicle_positions": self._mock_vehicle_positions(),
            "queue_per_intersection": self.queue.astype(np.float32),
            "delay_per_intersection": self.delay.astype(np.float32),
        }
        return obs

    def close(self) -> None:
        if self._use_traci and traci is not None:
            try:
                if traci.isLoaded():
                    traci.close()
            except Exception:  # pragma: no cover
                pass


def make_sumo_env(**kwargs: Any) -> SumoTrafficEnv:
    return SumoTrafficEnv(**kwargs)


class SumoEnv(SumoTrafficEnv):
    """
    Backward-compatible wrapper used by older scripts/snippets.

    Supports legacy construction style:
      SumoEnv({"scenario": "standard"})
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        cfg = dict(config) if isinstance(config, dict) else {}
        for key in (
            "config_path",
            "scenario",
            "episode_length_seconds",
            "num_intersections",
            "num_phases",
            "min_green_time_seconds",
            "seed",
            "use_traci",
        ):
            if key in cfg and key not in kwargs:
                kwargs[key] = cfg[key]
        super().__init__(**kwargs)

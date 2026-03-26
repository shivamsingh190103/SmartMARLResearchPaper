"""Emergency-vehicle corridor experiment for SmartMARL."""

from __future__ import annotations

from typing import Dict

import yaml

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer
from smartmarl.utils.metrics import compute_metrics


def _scheduled_ev_active(env: SumoTrafficEnv) -> bool:
    return bool(env.ev_start_step <= env.current_step < env.ev_start_step + env.ev_duration_steps)


def _make_env(cfg: Dict, seed: int) -> SumoTrafficEnv:
    env = SumoTrafficEnv(
        config_path="smartmarl/configs/grid5x5/grid5x5.sumocfg",
        scenario="standard",
        episode_length_seconds=int(cfg["episode_length_seconds"]),
        num_intersections=int(cfg["num_intersections"]),
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=seed,
        use_traci=True,
    )
    env.configure_ev_corridor(start_step=250, duration_steps=90, preferred_phase=0)
    return env


def _evaluate_with_strategy(
    trainer: MA2CTrainer,
    env: SumoTrafficEnv,
    *,
    strategy: str,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    att = []
    awt = []
    tput = []

    for episode in range(num_episodes):
        trainer.reset_aukfs()
        obs, _ = env.reset(seed=trainer.seed + 20000 + episode)

        for _ in range(steps_per_episode):
            actions = trainer.inference_policy(obs)
            if strategy == "fixed_preemption" and _scheduled_ev_active(env):
                actions = env.recommended_ev_actions(actions)
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        metrics = compute_metrics(
            completed_vehicles=int(env.stats.completed_vehicles),
            total_waiting_time=float(env.stats.total_waiting_time),
            total_travel_time=float(env.stats.total_travel_time),
            sim_seconds=steps_per_episode,
        )
        att.append(metrics["ATT"])
        awt.append(metrics["AWT"])
        tput.append(metrics["Throughput"])

    return {
        "strategy": strategy,
        "ATT": float(sum(att) / len(att)) if att else 0.0,
        "AWT": float(sum(awt) / len(awt)) if awt else 0.0,
        "Throughput": float(sum(tput) / len(tput)) if tput else 0.0,
        "ATT_runs": att,
        "AWT_runs": awt,
        "Throughput_runs": tput,
    }


def run_ev_experiment(
    config_path: str = "smartmarl/configs/default.yaml",
    seed: int = 0,
    strategy: str = "learned_adaptive",
    episodes: int = 50,
) -> Dict:
    if strategy not in {"no_preemption", "fixed_preemption", "learned_adaptive"}:
        raise ValueError("strategy must be one of: no_preemption, fixed_preemption, learned_adaptive")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = _make_env(cfg, seed)
    ablation = "full" if strategy == "learned_adaptive" else "no_ev"
    env.set_reward_mode("ev" if strategy == "learned_adaptive" else "normal")

    trainer = MA2CTrainer(env=env, config=cfg, ablation=ablation, seed=seed)
    trainer.train(num_episodes=episodes, progress=False)
    metrics = _evaluate_with_strategy(
        trainer,
        env,
        strategy=strategy,
        num_episodes=5,
        steps_per_episode=int(cfg.get("mock_training_steps", cfg["episode_length_seconds"])),
    )
    env.close()
    return metrics


if __name__ == "__main__":
    for strategy_name in ("no_preemption", "fixed_preemption", "learned_adaptive"):
        print(strategy_name, run_ev_experiment(strategy=strategy_name))

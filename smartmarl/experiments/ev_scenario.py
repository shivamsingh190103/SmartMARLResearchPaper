"""Emergency vehicle scenario evaluation."""

from __future__ import annotations

from typing import Dict

import yaml

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer


def run_ev_experiment(
    config_path: str = "smartmarl/configs/default.yaml",
    seed: int = 0,
    use_ev_reward: bool = True,
    episodes: int = 50,
) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ablation = "full" if use_ev_reward else "no_ev"
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
    trainer = MA2CTrainer(env=env, config=cfg, ablation=ablation, seed=seed)
    trainer.train(num_episodes=episodes, progress=False)
    metrics = trainer.evaluate(num_episodes=5)
    env.close()
    return metrics


if __name__ == "__main__":
    ev = run_ev_experiment(use_ev_reward=True)
    baseline = run_ev_experiment(use_ev_reward=False)
    print("EV reward mode:", ev)
    print("Normal reward during EV:", baseline)

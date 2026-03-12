"""Evaluation entry point for SmartMARL checkpoints."""

from __future__ import annotations

import argparse

import yaml

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SmartMARL checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", choices=["standard", "indian_hetero"], default="standard")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open("smartmarl/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_path = (
        "smartmarl/configs/grid5x5/grid5x5_indian.sumocfg"
        if args.scenario == "indian_hetero"
        else "smartmarl/configs/grid5x5/grid5x5.sumocfg"
    )

    env = SumoTrafficEnv(
        config_path=config_path,
        scenario=args.scenario,
        episode_length_seconds=int(cfg["episode_length_seconds"]),
        num_intersections=int(cfg["num_intersections"]),
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=args.seed,
        use_traci=True,
    )
    trainer = MA2CTrainer(env=env, config=cfg, ablation=args.ablation, seed=args.seed)
    trainer.load_checkpoint(args.checkpoint)
    metrics = trainer.evaluate(num_episodes=args.episodes)
    print(metrics)
    env.close()


if __name__ == "__main__":
    main()

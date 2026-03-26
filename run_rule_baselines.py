"""Run FixedTime and MaxPressure baselines for SmartMARL scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from smartmarl.baselines.rule_based import (
    GridTopology,
    evaluate_policy,
    fixed_time_actions,
    maxpressure_actions,
)
from smartmarl.env.sumo_env import SumoTrafficEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based traffic signal baselines.")
    parser.add_argument("--scenario", choices=["standard", "indian_hetero"], default="standard")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=29)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _sumocfg_for_scenario(scenario: str) -> str:
    if scenario == "indian_hetero":
        return "smartmarl/configs/grid5x5/grid5x5_indian.sumocfg"
    return "smartmarl/configs/grid5x5/grid5x5.sumocfg"


def _run_variant(
    *,
    scenario: str,
    seed: int,
    variant: str,
    eval_episodes: int,
    steps_per_episode: int,
) -> dict:
    with open("smartmarl/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = SumoTrafficEnv(
        config_path=_sumocfg_for_scenario(scenario),
        scenario=scenario,
        episode_length_seconds=int(cfg["episode_length_seconds"]),
        num_intersections=int(cfg["num_intersections"]),
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=seed,
        use_traci=True,
    )
    env.set_reward_mode("normal")

    topo = GridTopology(grid_size=int(cfg["grid_size"]), num_intersections=int(cfg["num_intersections"]))
    if variant == "fixed_time":
        action_fn = lambda obs, step: fixed_time_actions(  # noqa: E731
            current_step=step,
            num_intersections=topo.num_intersections,
            num_phases=int(cfg["num_phases"]),
            cycle_length=60,
        )
    elif variant == "maxpressure":
        action_fn = lambda obs, step: maxpressure_actions(  # noqa: E731
            obs=obs,
            current_step=step,
            topology=topo,
            num_phases=int(cfg["num_phases"]),
        )
    else:
        env.close()
        raise ValueError(f"Unsupported variant: {variant}")

    eval_metrics = evaluate_policy(
        env=env,
        action_fn=action_fn,
        num_episodes=int(eval_episodes),
        steps_per_episode=int(steps_per_episode),
        seed=seed + 10000,
    )
    env.close()

    att = float(eval_metrics["ATT"])
    awt = float(eval_metrics["AWT"])
    throughput = float(eval_metrics["Throughput"])
    return {
        "scenario": scenario,
        "seed": int(seed),
        "ablation": variant,
        "variant": variant,
        "backend": "traci" if not env.mock_mode else "mock",
        "mock_mode": bool(env.mock_mode),
        "episodes": 0,
        "steps_per_episode": int(steps_per_episode),
        "checkpoint": "",
        "metrics_csv": "",
        "train_metrics": {},
        "eval_metrics": eval_metrics,
        "att": att,
        "final_att": att,
        "awt": awt,
        "throughput": throughput,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ("fixed_time", "maxpressure")
    for seed in range(int(args.seed_start), int(args.seed_end) + 1):
        for variant in variants:
            out_path = out_dir / f"{args.scenario}_{variant}_seed{seed}.json"
            if args.skip_existing and out_path.exists():
                print(f"Skipping existing: {out_path}")
                continue

            print(f"Running {variant} seed={seed} scenario={args.scenario}")
            payload = _run_variant(
                scenario=args.scenario,
                seed=seed,
                variant=variant,
                eval_episodes=int(args.eval_episodes),
                steps_per_episode=int(args.steps_per_episode),
            )
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

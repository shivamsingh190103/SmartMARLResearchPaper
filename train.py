"""Main training entry point for SmartMARL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from setup_network import ensure_sumo_assets
from smartmarl.env.cityflow_env import CityFlowTrafficEnv
from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer, default_checkpoint_path


def load_config(scenario: str) -> dict:
    with open("smartmarl/configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if scenario == "indian_hetero":
        with open("smartmarl/configs/indian_hetero.yaml", "r", encoding="utf-8") as f:
            cfg["indian_hetero"] = yaml.safe_load(f)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmartMARL")
    parser.add_argument("--scenario", choices=["standard", "indian_hetero"], default="standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablation",
        choices=["full", "no_ctde", "no_aukf", "no_hetgnn", "no_incident", "no_ev", "yolov5", "mlp", "l7", "gplight"],
        default="full",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Override simulation steps per episode")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--metrics_csv", type=str, default="", help="Path to per-episode training metrics CSV")
    parser.add_argument("--result_json", type=str, default="", help="Path to save final train/eval metrics as JSON")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If result_json already exists, skip run and exit successfully.",
    )
    parser.add_argument("--cityflow", action="store_true", help="Use CityFlow mock backend")
    parser.add_argument(
        "--allow_mock",
        action="store_true",
        help="Allow long runs in mock backend. By default, long runs in mock mode are blocked.",
    )
    parser.add_argument(
        "--force_regenerate_sumo_assets",
        action="store_true",
        help="Regenerate SUMO network and route assets with SUMO tools before training.",
    )
    return parser.parse_args()


def sumocfg_for_scenario(scenario: str) -> str:
    if scenario == "indian_hetero":
        return "smartmarl/configs/grid5x5/grid5x5_indian.sumocfg"
    return "smartmarl/configs/grid5x5/grid5x5.sumocfg"


def infer_start_episode(metrics_csv: str) -> int:
    path = Path(metrics_csv)
    if not path.exists():
        return 0

    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) <= 1:
            return 0
        last = lines[-1]
        first_field = last.split(",")[0].strip()
        return max(0, int(first_field))
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    result_json = args.result_json.strip()
    if args.skip_existing and result_json and Path(result_json).exists():
        print(f"Skipping existing result: {result_json}")
        return

    cfg = load_config(args.scenario)

    if not args.cityflow and not args.allow_mock:
        ensure_sumo_assets(
            force_regenerate=bool(args.force_regenerate_sumo_assets),
            strict_tools=bool(args.force_regenerate_sumo_assets),
        )

    env_cls = CityFlowTrafficEnv if args.cityflow else SumoTrafficEnv
    env = env_cls(
        config_path=sumocfg_for_scenario(args.scenario),
        scenario=args.scenario,
        episode_length_seconds=int(cfg["episode_length_seconds"]),
        num_intersections=int(cfg["num_intersections"]),
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=args.seed,
        use_traci=not args.cityflow,
    )
    print("Mock mode:", getattr(env, "mock_mode", True))

    # Protect against accidental paper-scale training on mock backend.
    planned_episodes = int(args.episodes or cfg["training_episodes_sumo"])
    if (
        bool(getattr(env, "mock_mode", True))
        and not args.cityflow
        and not args.eval_only
        and planned_episodes >= 100
        and not args.allow_mock
    ):
        env.close()
        raise SystemExit(
            "Refusing long training in mock mode. Install/enable real SUMO backend first, "
            "or explicitly pass --allow_mock for development-only runs."
        )

    trainer = MA2CTrainer(env=env, config=cfg, ablation=args.ablation, seed=args.seed)

    ckpt = args.checkpoint.strip() or default_checkpoint_path(
        results_dir=cfg.get("results_dir", "results"),
        variant=args.ablation,
        scenario=args.scenario,
        seed=args.seed,
    )
    default_metrics_csv = (
        Path(cfg.get("results_dir", "results"))
        / "training_logs"
        / f"{args.scenario}_{args.ablation}_seed{args.seed}.csv"
    )
    metrics_csv = args.metrics_csv.strip() or str(default_metrics_csv)

    train_metrics = {}
    if args.eval_only:
        trainer.load_checkpoint(ckpt)
    else:
        if args.resume and Path(ckpt).exists():
            trainer.load_checkpoint(ckpt)
            print(f"Loaded checkpoint: {ckpt}")
        elif not args.resume:
            metrics_path_obj = Path(metrics_csv)
            if metrics_path_obj.exists():
                metrics_path_obj.unlink()

        episodes = planned_episodes
        start_episode = infer_start_episode(metrics_csv) if args.resume else 0
        if args.resume and start_episode > 0:
            print(f"Resuming from episode {start_episode + 1}")

        train_metrics = trainer.train(
            num_episodes=episodes,
            progress=True,
            start_episode=start_episode,
            steps_per_episode=args.steps_per_episode,
            checkpoint_every=max(0, int(args.checkpoint_every)),
            checkpoint_path=ckpt,
            metrics_csv_path=metrics_csv,
        )
        print("Train metrics:", train_metrics)
        trainer.save_checkpoint(ckpt)
        print(f"Saved checkpoint: {ckpt}")
        print(f"Metrics CSV: {metrics_csv}")

    eval_metrics = trainer.evaluate(num_episodes=5, steps_per_episode=args.steps_per_episode)
    print("Eval metrics:", eval_metrics)

    if result_json:
        out = Path(result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        att = float(eval_metrics.get("ATT", 0.0))
        awt = float(eval_metrics.get("AWT", 0.0))
        throughput = float(eval_metrics.get("Throughput", 0.0))
        payload = {
            "scenario": args.scenario,
            "seed": int(args.seed),
            "ablation": args.ablation,
            "variant": args.ablation,
            "backend": "mock" if bool(getattr(env, "mock_mode", True)) else "traci",
            "mock_mode": bool(getattr(env, "mock_mode", True)),
            "episodes": int(args.episodes or cfg["training_episodes_sumo"]),
            "steps_per_episode": int(args.steps_per_episode or 0),
            "checkpoint": str(ckpt),
            "metrics_csv": str(metrics_csv),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "att": att,
            "final_att": att,
            "awt": awt,
            "throughput": throughput,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved result JSON: {out}")
    env.close()


if __name__ == "__main__":
    main()

"""Run SmartMARL training sequentially across a seed range with resume support."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SmartMARL over multiple seeds")
    p.add_argument("--scenario", choices=["standard", "indian_hetero"], default="standard")
    p.add_argument("--ablation", default="full")
    p.add_argument("--seed_start", type=int, required=True)
    p.add_argument("--seed_end", type=int, required=True, help="Exclusive end")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--steps_per_episode", type=int, default=300)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    return p.parse_args()


def run_one(args: argparse.Namespace, seed: int) -> int:
    result_json = Path("results/raw") / f"{args.scenario}_{args.ablation}_seed{seed}.json"
    if args.skip_existing and result_json.exists():
        print(f"Skipping seed {seed}; existing result {result_json}")
        return 0

    cmd = [
        sys.executable,
        "train.py",
        "--scenario",
        args.scenario,
        "--ablation",
        args.ablation,
        "--seed",
        str(seed),
        "--episodes",
        str(args.episodes),
        "--steps_per_episode",
        str(args.steps_per_episode),
        "--checkpoint_every",
        str(args.checkpoint_every),
        "--result_json",
        str(result_json),
    ]
    if args.resume:
        cmd.append("--resume")

    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    return subprocess.run(cmd).returncode


def main() -> None:
    args = parse_args()

    Path("logs").mkdir(exist_ok=True)

    failures = []
    for seed in range(args.seed_start, args.seed_end):
        rc = run_one(args, seed)
        if rc != 0:
            failures.append(seed)
            print(f"Seed {seed} failed with exit code {rc}")
            if args.stop_on_error:
                break

    if failures:
        print(f"Completed with failures: {failures}")
        raise SystemExit(1)

    print("All seeds completed successfully")


if __name__ == "__main__":
    main()

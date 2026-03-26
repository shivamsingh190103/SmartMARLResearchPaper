"""Run the GPLight-style grouped homogeneous baseline over multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SmartMARL GPLight-style baseline.")
    parser.add_argument("--scenario", choices=["standard", "indian_hetero"], default="standard")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in range(int(args.seed_start), int(args.seed_end) + 1):
        cmd = [
            sys.executable,
            "train.py",
            "--scenario",
            args.scenario,
            "--ablation",
            "gplight",
            "--seed",
            str(seed),
            "--episodes",
            str(args.episodes),
            "--steps_per_episode",
            str(args.steps_per_episode),
            "--result_json",
            f"results/raw/{args.scenario}_gplight_seed{seed}.json",
        ]
        if args.skip_existing:
            cmd.append("--skip_existing")
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

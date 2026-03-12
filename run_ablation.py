"""Run all SmartMARL ablations and generate Table 8 outputs."""

from __future__ import annotations

import argparse

from smartmarl.experiments.ablation import format_table, run_all_ablations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmartMARL ablations")
    parser.add_argument("--config", default="smartmarl/configs/default.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--scenario", default="standard")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_all_ablations(
        config_path=args.config,
        output_dir=args.output,
        scenario=args.scenario,
        episodes_override=args.episodes,
        num_seeds_override=args.num_seeds,
    )
    print(format_table(df.to_dict("records")))


if __name__ == "__main__":
    main()

"""Fit simple demand priors from public trajectory datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


def fit_profile(df: pd.DataFrame) -> Dict:
    required = {"vehicle_id", "timestamp", "speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    first_obs = df.sort_values(["vehicle_id", "timestamp"]).groupby("vehicle_id", as_index=False).first()
    arrivals_per_min = float(len(first_obs) / max((df["timestamp"].max() - df["timestamp"].min()) / 60.0, 1e-6))

    profile = {
        "arrival_rate_per_min": arrivals_per_min,
        "speed_mean": float(df["speed"].mean()),
        "speed_std": float(df["speed"].std(ddof=0)),
        "num_unique_vehicles": int(df["vehicle_id"].nunique()),
        "time_span_seconds": float(df["timestamp"].max() - df["timestamp"].min()),
    }

    if "vehicle_type" in df.columns:
        dist = df[["vehicle_id", "vehicle_type"]].drop_duplicates()["vehicle_type"].value_counts(normalize=True)
        profile["vehicle_type_distribution"] = {str(k): float(v) for k, v in dist.to_dict().items()}

    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate demand priors from a public trajectory dataset.")
    parser.add_argument("--input", required=True, help="CSV with at least vehicle_id,timestamp,speed columns")
    parser.add_argument("--output", required=True, help="YAML file to write calibration profile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    profile = fit_profile(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

"""Quick monitor for SmartMARL training CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(csv_path: Path) -> str:
    if not csv_path.exists():
        return f"No file: {csv_path}"

    df = pd.read_csv(csv_path)
    if df.empty:
        return f"Empty CSV: {csv_path}"

    df = df.sort_values("episode")
    last_ep = int(df.iloc[-1]["episode"])

    att = df["att"].to_numpy(dtype=float)

    def ma(k: int) -> float:
        if len(att) == 0:
            return float("nan")
        return float(np.mean(att[-min(k, len(att)) :]))

    k = min(50, len(att))
    if k >= 2:
        y = att[-k:]
        x = np.arange(k, dtype=float)
        slope = float(np.polyfit(x, y, deg=1)[0])
    else:
        slope = 0.0

    trend = "down" if slope < -0.02 else "up" if slope > 0.02 else "flat"

    return (
        f"episodes={len(att)} last_episode={last_ep} "
        f"att_last={att[-1]:.2f} att_ma10={ma(10):.2f} "
        f"att_ma25={ma(25):.2f} att_ma50={ma(50):.2f} "
        f"slope50={slope:.4f} ({trend})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/training_logs/standard_full_seed0.csv")
    args = parser.parse_args()

    print(summarize(Path(args.csv)))


if __name__ == "__main__":
    main()

"""Numerical validation of AUKF robustness under increasing sensor noise."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from smartmarl.perception.aukf import AdaptiveUKF
from smartmarl.perception.noise_injection import (
    condition_radar_position_sigma,
    condition_radar_velocity_sigma,
)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def run_noise_sweep(
    sigma_scales: List[float] | None = None,
    steps: int = 250,
    seed: int = 0,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    sigma_scales = sigma_scales or [0.0, 0.25, 0.5, 0.75, 1.0]
    rows: List[Dict] = []

    # Camera is stronger on occupancy/count cues; radar is stronger on kinematics.
    base_cam_sigma = np.array([0.25, 0.25, 0.45, 0.05], dtype=np.float64)
    base_rad_sigma = np.array(
        [
            0.45,
            0.45,
            condition_radar_velocity_sigma("clear"),
            0.02,
        ],
        dtype=np.float64,
    )

    for scale in sigma_scales:
        filt = AdaptiveUKF(beta=0.02)
        truth = np.array([10.0, 12.0, 8.0, 0.35], dtype=np.float64)
        truth_hist = []
        cam_hist = []
        rad_hist = []
        fused_hist = []

        for _ in range(steps):
            truth = np.clip(
                truth + rng.normal(0.0, [0.15, 0.15, 0.10, 0.01], size=4),
                [0.0, 0.0, 0.0, 0.0],
                [50.0, 80.0, 30.0, 1.0],
            )
            z_cam = truth + rng.normal(0.0, base_cam_sigma * (1.0 + scale), size=4)
            z_rad = truth + rng.normal(0.0, base_rad_sigma * (1.0 + scale), size=4)
            fused, _ = filt.update(z_cam, z_rad)

            truth_hist.append(truth.copy())
            cam_hist.append(z_cam.copy())
            rad_hist.append(z_rad.copy())
            fused_hist.append(fused.copy())

        truth_arr = np.asarray(truth_hist)
        cam_arr = np.asarray(cam_hist)
        rad_arr = np.asarray(rad_hist)
        fused_arr = np.asarray(fused_hist)

        metric_slice = slice(max(20, steps // 5), None)
        rows.append(
            {
                "sigma_scale": float(scale),
                "camera_rmse": _rmse(cam_arr[metric_slice], truth_arr[metric_slice]),
                "radar_rmse": _rmse(rad_arr[metric_slice], truth_arr[metric_slice]),
                "aukf_rmse": _rmse(fused_arr[metric_slice], truth_arr[metric_slice]),
            }
        )
    return rows


def save_rows(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SmartMARL AUKF noise sweep.")
    parser.add_argument("--output", default="results/aukf_noise_sweep.csv")
    parser.add_argument("--steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_noise_sweep(steps=int(args.steps))
    save_rows(rows, Path(args.output))
    print("sigma_scale,camera_rmse,radar_rmse,aukf_rmse")
    for row in rows:
        print(f"{row['sigma_scale']:.2f},{row['camera_rmse']:.4f},{row['radar_rmse']:.4f},{row['aukf_rmse']:.4f}")


if __name__ == "__main__":
    main()

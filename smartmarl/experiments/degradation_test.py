"""Table 1-style degradation testing for SmartMARL perception."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smartmarl.perception.aukf import AdaptiveUKF
from smartmarl.perception.noise_injection import apply_camera_measurement_noise, apply_radar_noise
from smartmarl.utils.metrics import aukf_retention_rate


def run_condition(condition: str, steps: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    aukf = AdaptiveUKF()

    mse_acc = []
    sigma_acc = []

    true_state = np.array([8.0, 14.0, 9.0, 0.35], dtype=np.float32)
    for _ in range(steps):
        drift = rng.normal(0.0, [0.15, 0.2, 0.1, 0.01])
        true_state = np.clip(true_state + drift, [0, 0, 0, 0], [50, 80, 30, 1]).astype(np.float32)

        z_cam = true_state + rng.normal(0.0, [0.25, 0.25, 0.2, 0.02], size=4)
        z_rad = true_state + rng.normal(0.0, [0.1, 0.1, 0.12, 0.015], size=4)

        z_cam = apply_camera_measurement_noise(z_cam, condition, rng)
        z_rad[0] = apply_radar_noise(np.array([z_rad[0]], dtype=np.float32), condition, rng)[0]

        state, sigma2 = aukf.update(z_cam, z_rad)
        mse_acc.append(float((state[0] - true_state[0]) ** 2))
        sigma_acc.append(float(np.mean(sigma2)))

    return {
        "condition": condition,
        "mse_queue": float(np.mean(mse_acc)),
        "mean_sigma2_r": float(np.mean(sigma_acc)),
    }


def run_degradation_test(output_dir: str = "results") -> pd.DataFrame:
    conditions = ["clear", "rain", "night", "radar_multipath"]
    rows = [run_condition(c, seed=idx) for idx, c in enumerate(conditions)]

    clear_mse = next(r["mse_queue"] for r in rows if r["condition"] == "clear")
    for row in rows:
        row["retention_rate"] = aukf_retention_rate(clear_mse, row["mse_queue"])

    df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "degradation_table.csv", index=False)
    return df


if __name__ == "__main__":
    result = run_degradation_test(output_dir="results")
    print(result.to_string(index=False))

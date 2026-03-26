import numpy as np

from smartmarl.experiments.aukf_noise_sweep import run_noise_sweep
from smartmarl.perception.aukf import AdaptiveUKF
from smartmarl.perception.noise_injection import apply_camera_measurement_noise


def test_aukf_r_matrix_adapts_over_time():
    rng = np.random.default_rng(123)
    filt = AdaptiveUKF()

    r0 = filt.R.copy()
    truth = np.array([10.0, 12.0, 8.0, 0.4])

    for _ in range(100):
        z_cam = truth + rng.normal(0.0, [0.3, 0.3, 0.2, 0.03], size=4)
        z_rad = truth + rng.normal(0.0, [0.1, 0.1, 0.1, 0.02], size=4)
        filt.update(z_cam, z_rad)

    r1 = filt.R.copy()
    assert not np.allclose(r0, r1), "AUKF R matrix did not adapt"
    assert np.linalg.norm(r1 - r0) > 1e-3


def _mean_sigma_for_condition(condition: str, steps: int = 200) -> float:
    rng = np.random.default_rng(7)
    filt = AdaptiveUKF()
    truth = np.array([9.0, 11.0, 7.0, 0.35], dtype=np.float32)

    sigmas = []
    for _ in range(steps):
        truth = np.clip(truth + rng.normal(0, [0.1, 0.1, 0.1, 0.01]), [0, 0, 0, 0], [50, 80, 30, 1])
        z_cam = truth + rng.normal(0.0, [0.25, 0.25, 0.2, 0.02], size=4)
        z_rad = truth + rng.normal(0.0, [0.1, 0.1, 0.12, 0.02], size=4)

        z_cam = apply_camera_measurement_noise(z_cam, condition, rng)
        _, sigma2 = filt.update(z_cam, z_rad)
        sigmas.append(float(np.mean(sigma2)))

    return float(np.mean(sigmas[-50:]))


def test_sigma2_r_increases_under_rain_and_night():
    clear_sigma = _mean_sigma_for_condition("clear")
    rain_sigma = _mean_sigma_for_condition("rain")
    night_sigma = _mean_sigma_for_condition("night")

    assert rain_sigma > clear_sigma
    assert night_sigma > clear_sigma


def test_aukf_noise_sweep_fusion_beats_camera_at_nominal_noise():
    rows = run_noise_sweep(sigma_scales=[0.5], steps=150, seed=3)
    row = rows[0]
    assert row["aukf_rmse"] < row["camera_rmse"]

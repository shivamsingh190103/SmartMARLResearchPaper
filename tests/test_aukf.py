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


def test_aukf_fusion_weights_favor_lower_variance_sensor():
    filt = AdaptiveUKF()
    z_cam = np.array([100.0, 100.0, 20.0, 0.9], dtype=np.float64)
    z_rad = np.array([10.0, 10.0, 5.0, 0.2], dtype=np.float64)
    fused = filt._fuse_measurements(z_cam, z_rad)

    # Queue/count should stay closer to camera (camera variance lower on these dims).
    assert abs(fused[0] - z_cam[0]) < abs(fused[0] - z_rad[0])
    assert abs(fused[1] - z_cam[1]) < abs(fused[1] - z_rad[1])

    # Speed/occupancy should stay closer to radar (radar variance lower on these dims).
    assert abs(fused[2] - z_rad[2]) < abs(fused[2] - z_cam[2])
    assert abs(fused[3] - z_rad[3]) < abs(fused[3] - z_cam[3])

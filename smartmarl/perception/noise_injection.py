"""Noise injection utilities for SmartMARL degradation experiments."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


VALID_CONDITIONS = {"clear", "rain", "night", "radar_multipath"}

CAMERA_BACKBONE_SPECS = {
    # Software-only perception model calibrated to published small-backbone detector behavior.
    "yolov8n": {"base_confidence": 0.78, "localization_sigma_m": 0.70},
    "yolov5_tiny": {"base_confidence": 0.68, "localization_sigma_m": 1.00},
}

RADAR_RANGE_SIGMA_M = {
    "clear": 0.05,
    "rain": 0.06,
    "night": 0.05,
    "radar_multipath": 0.18,
}

RADAR_VELOCITY_SIGMA_MS = {
    "clear": 0.10,
    "rain": 0.12,
    "night": 0.10,
    "radar_multipath": 0.35,
}

RADAR_POSITION_SIGMA_M = {
    "clear": 0.05,
    "rain": 0.06,
    "night": 0.05,
    "radar_multipath": 0.20,
}


def _validate_condition(condition: str) -> None:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unsupported condition '{condition}'. Expected one of {sorted(VALID_CONDITIONS)}")


def apply_camera_confidence_noise(
    confidences: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    _validate_condition(condition)
    conf = np.asarray(confidences, dtype=np.float32).copy()

    if condition == "rain":
        conf += rng.normal(0.0, 0.12, size=conf.shape)
        conf = np.clip(conf, 0.0, 1.0)
    elif condition == "night":
        drops = rng.random(conf.shape) < 0.18
        conf[drops] = 0.0
    return conf


def camera_backbone_specs(backbone: str) -> dict:
    if backbone not in CAMERA_BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of {sorted(CAMERA_BACKBONE_SPECS)}")
    return dict(CAMERA_BACKBONE_SPECS[backbone])


def apply_camera_measurement_noise(
    measurements: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    _validate_condition(condition)
    m = np.asarray(measurements, dtype=np.float32).copy()

    if condition == "clear":
        return m
    if condition == "rain":
        # Stronger camera corruption for rain to reflect degraded visibility.
        return np.clip(m + rng.normal(0.0, 0.45, size=m.shape), 0.0, None)
    if condition == "night":
        masked = m.copy()
        drop_mask = rng.random(masked.shape) < 0.18
        masked[drop_mask] = 0.0
        return masked
    return m


def apply_radar_noise(
    ranges: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    _validate_condition(condition)
    r = np.asarray(ranges, dtype=np.float32).copy()

    sigma = RADAR_RANGE_SIGMA_M[condition]
    r += rng.normal(0.0, sigma, size=r.shape)
    return r


def add_radar_spurious_returns(
    detections: List[dict],
    condition: str,
    rng: np.random.Generator,
) -> List[dict]:
    _validate_condition(condition)
    output = list(detections)

    if condition != "radar_multipath":
        return output

    n_spurious = int(np.ceil(max(1, 0.03 * max(1, len(detections)))))
    for _ in range(n_spurious):
        output.append(
            {
                "x": float(rng.uniform(0.0, 2000.0)),
                "y": float(rng.uniform(0.0, 2000.0)),
                "range": float(rng.uniform(1.0, 300.0)),
                "velocity": float(rng.normal(0.0, 1.0)),
                "spurious": True,
            }
        )
    return output


def condition_radar_sigma(condition: str) -> float:
    _validate_condition(condition)
    return float(RADAR_RANGE_SIGMA_M[condition])


def condition_radar_velocity_sigma(condition: str) -> float:
    _validate_condition(condition)
    return float(RADAR_VELOCITY_SIGMA_MS[condition])


def condition_radar_position_sigma(condition: str) -> float:
    _validate_condition(condition)
    return float(RADAR_POSITION_SIGMA_M[condition])


def clear_measurements_from_detections(detections: Sequence[dict]) -> np.ndarray:
    if not detections:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[d["x"], d["y"]] for d in detections], dtype=np.float32)

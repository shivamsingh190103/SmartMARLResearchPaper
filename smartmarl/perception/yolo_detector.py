"""Camera perception noise model used in SmartMARL simulation."""

from __future__ import annotations

from typing import List

import numpy as np

from .noise_injection import apply_camera_confidence_noise, camera_backbone_specs


class CameraPerceptionModel:
    """Parameterized camera detector surrogate for software-only experiments."""

    def __init__(
        self,
        backbone: str = "yolov8n",
        confidence_threshold: float = 0.45,
        condition: str = "clear",
        seed: int = 0,
    ) -> None:
        self.backbone = backbone
        self.confidence_threshold = float(confidence_threshold)
        self.condition = condition
        self.rng = np.random.default_rng(seed)
        self.spec = camera_backbone_specs(backbone)

    def set_condition(self, condition: str) -> None:
        self.condition = condition

    def detect(self, vehicle_positions: np.ndarray) -> List[dict]:
        vehicle_positions = np.asarray(vehicle_positions, dtype=np.float32)
        if vehicle_positions.size == 0:
            return []

        loc_sigma = float(self.spec["localization_sigma_m"])
        base_conf = float(self.spec["base_confidence"])

        centers = vehicle_positions + self.rng.normal(0.0, loc_sigma, size=vehicle_positions.shape)
        confs = np.clip(base_conf + self.rng.normal(0.0, 0.08, size=len(centers)), 0.0, 1.0)
        confs = apply_camera_confidence_noise(confs, self.condition, self.rng)

        detections: List[dict] = []
        for (x, y), conf in zip(centers, confs):
            if conf < self.confidence_threshold:
                continue
            w = float(np.clip(self.rng.normal(1.8, 0.2), 1.0, 2.6))
            h = float(np.clip(self.rng.normal(1.6, 0.25), 0.8, 2.6))
            detections.append(
                {
                    "bbox": [float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)],
                    "confidence": float(conf),
                    "x": float(x),
                    "y": float(y),
                }
            )
        return detections


class YOLODetector(CameraPerceptionModel):
    """Backward-compatible alias for the legacy class name."""


__all__ = ["CameraPerceptionModel", "YOLODetector"]

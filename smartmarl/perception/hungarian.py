"""Hungarian matching between camera and radar detections."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def _extract_xy(detections: List[dict]) -> np.ndarray:
    if not detections:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[d["x"], d["y"]] for d in detections], dtype=np.float32)


def associate_detections(
    camera_detections: List[dict],
    radar_detections: List[dict],
    max_distance: float = 2.0,
) -> Dict[str, list]:
    cam_xy = _extract_xy(camera_detections)
    rad_xy = _extract_xy(radar_detections)

    if len(cam_xy) == 0 or len(rad_xy) == 0:
        return {
            "matches": [],
            "unmatched_camera": list(range(len(camera_detections))),
            "unmatched_radar": list(range(len(radar_detections))),
        }

    diff = cam_xy[:, None, :] - rad_xy[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    matched_cam = set()
    matched_rad = set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_distance:
            matches.append((int(r), int(c), float(cost[r, c])))
            matched_cam.add(int(r))
            matched_rad.add(int(c))

    unmatched_camera = [i for i in range(len(camera_detections)) if i not in matched_cam]
    unmatched_radar = [i for i in range(len(radar_detections)) if i not in matched_rad]

    return {
        "matches": matches,
        "unmatched_camera": unmatched_camera,
        "unmatched_radar": unmatched_radar,
    }

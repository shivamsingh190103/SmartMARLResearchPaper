"""Tests for perception layer: Hungarian matching, noise injection,
radar processor, and camera (YOLO) detector."""

from __future__ import annotations

import numpy as np
import pytest

from smartmarl.perception.hungarian import associate_detections
from smartmarl.perception.noise_injection import (
    VALID_CONDITIONS,
    add_radar_spurious_returns,
    apply_camera_confidence_noise,
    apply_camera_measurement_noise,
    apply_radar_noise,
    camera_backbone_specs,
    clear_measurements_from_detections,
    condition_radar_position_sigma,
    condition_radar_sigma,
    condition_radar_velocity_sigma,
)
from smartmarl.perception.radar_processor import RadarPerceptionModel, RadarProcessor
from smartmarl.perception.yolo_detector import CameraPerceptionModel, YOLODetector


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

class TestAssociateDetections:
    def _make_cam(self, coords):
        return [{"x": float(x), "y": float(y)} for x, y in coords]

    def _make_rad(self, coords):
        return [{"x": float(x), "y": float(y)} for x, y in coords]

    def test_empty_camera_returns_all_radar_unmatched(self):
        rad = self._make_rad([(0, 0), (1, 1)])
        result = associate_detections([], rad)
        assert result["matches"] == []
        assert result["unmatched_camera"] == []
        assert set(result["unmatched_radar"]) == {0, 1}

    def test_empty_radar_returns_all_camera_unmatched(self):
        cam = self._make_cam([(0, 0), (1, 1)])
        result = associate_detections(cam, [])
        assert result["matches"] == []
        assert set(result["unmatched_camera"]) == {0, 1}
        assert result["unmatched_radar"] == []

    def test_both_empty(self):
        result = associate_detections([], [])
        assert result == {"matches": [], "unmatched_camera": [], "unmatched_radar": []}

    def test_perfect_one_to_one_match(self):
        cam = self._make_cam([(0.0, 0.0), (5.0, 5.0)])
        rad = self._make_rad([(0.1, 0.1), (5.1, 5.1)])
        result = associate_detections(cam, rad, max_distance=1.0)
        assert len(result["matches"]) == 2
        assert result["unmatched_camera"] == []
        assert result["unmatched_radar"] == []

    def test_distance_threshold_filters_far_pairs(self):
        cam = self._make_cam([(0.0, 0.0)])
        rad = self._make_rad([(10.0, 10.0)])  # far away
        result = associate_detections(cam, rad, max_distance=2.0)
        assert result["matches"] == []
        assert 0 in result["unmatched_camera"]
        assert 0 in result["unmatched_radar"]

    def test_match_tuple_structure(self):
        cam = self._make_cam([(1.0, 1.0)])
        rad = self._make_rad([(1.5, 1.5)])
        result = associate_detections(cam, rad, max_distance=2.0)
        assert len(result["matches"]) == 1
        cam_idx, rad_idx, dist = result["matches"][0]
        assert cam_idx == 0
        assert rad_idx == 0
        assert dist > 0.0

    def test_asymmetric_sizes_unmatched_are_correct(self):
        cam = self._make_cam([(0.0, 0.0), (100.0, 100.0)])
        rad = self._make_rad([(0.1, 0.1)])
        result = associate_detections(cam, rad, max_distance=1.0)
        assert len(result["matches"]) == 1
        assert 1 in result["unmatched_camera"]
        assert result["unmatched_radar"] == []


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

class TestNoiseInjectionValidation:
    def test_invalid_condition_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unsupported condition"):
            apply_camera_confidence_noise(np.array([0.8]), "fog", rng)

    def test_invalid_condition_for_radar(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            apply_radar_noise(np.array([10.0]), "snow", rng)

    def test_invalid_condition_for_camera_measurement(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            apply_camera_measurement_noise(np.array([5.0, 5.0]), "blizzard", rng)

    def test_condition_sigma_invalid(self):
        with pytest.raises(ValueError):
            condition_radar_sigma("snow")


class TestApplyCameraConfidenceNoise:
    def test_clear_condition_no_change(self):
        rng = np.random.default_rng(0)
        confs = np.array([0.8, 0.9, 0.7], dtype=np.float32)
        result = apply_camera_confidence_noise(confs, "clear", rng)
        np.testing.assert_array_equal(result, confs)

    def test_rain_clips_to_0_1(self):
        rng = np.random.default_rng(0)
        confs = np.array([0.95, 0.05], dtype=np.float32)
        result = apply_camera_confidence_noise(confs, "rain", rng)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_night_can_zero_some_confidences(self):
        # With enough samples, at least one drop is expected under night conditions
        rng = np.random.default_rng(0)
        confs = np.ones(200, dtype=np.float32) * 0.8
        result = apply_camera_confidence_noise(confs, "night", rng)
        assert np.any(result == 0.0), "Night condition should zero out some detections"

    def test_output_is_copy_not_inplace(self):
        rng = np.random.default_rng(0)
        confs = np.array([0.8, 0.9], dtype=np.float32)
        original = confs.copy()
        apply_camera_confidence_noise(confs, "rain", rng)
        np.testing.assert_array_equal(confs, original)


class TestApplyCameraMeasurementNoise:
    def test_clear_returns_unchanged(self):
        rng = np.random.default_rng(0)
        m = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_camera_measurement_noise(m, "clear", rng)
        np.testing.assert_array_equal(result, m)

    def test_rain_adds_noise_and_clips_non_negative(self):
        rng = np.random.default_rng(0)
        m = np.zeros(50, dtype=np.float32)
        result = apply_camera_measurement_noise(m, "rain", rng)
        assert np.all(result >= 0.0)

    def test_night_zeroes_some_entries(self):
        rng = np.random.default_rng(42)
        m = np.ones(200, dtype=np.float32) * 5.0
        result = apply_camera_measurement_noise(m, "night", rng)
        assert np.any(result == 0.0)

    def test_radar_multipath_returns_unchanged(self):
        # radar_multipath is a valid condition but camera noise should be identity
        rng = np.random.default_rng(0)
        m = np.array([1.0, 2.0], dtype=np.float32)
        result = apply_camera_measurement_noise(m, "radar_multipath", rng)
        np.testing.assert_array_equal(result, m)


class TestApplyRadarNoise:
    def test_output_shape_preserved(self):
        rng = np.random.default_rng(0)
        ranges = np.linspace(10.0, 100.0, 20, dtype=np.float32)
        for cond in VALID_CONDITIONS:
            result = apply_radar_noise(ranges, cond, rng)
            assert result.shape == ranges.shape, f"Shape mismatch for condition {cond}"

    def test_noise_is_nonzero_statistically(self):
        rng = np.random.default_rng(1)
        ranges = np.ones(1000, dtype=np.float32) * 50.0
        result = apply_radar_noise(ranges, "clear", rng)
        assert not np.allclose(result, ranges), "Radar noise should modify the range values"


class TestAddRadarSpuriousReturns:
    def test_non_multipath_returns_same_length(self):
        rng = np.random.default_rng(0)
        dets = [{"x": 1.0, "y": 2.0, "range": 5.0, "velocity": 0.5, "spurious": False}]
        for cond in ("clear", "rain", "night"):
            result = add_radar_spurious_returns(dets, cond, rng)
            assert len(result) == len(dets)

    def test_multipath_adds_spurious_returns(self):
        rng = np.random.default_rng(0)
        dets = [{"x": 1.0, "y": 2.0, "range": 5.0, "velocity": 0.5, "spurious": False}] * 10
        result = add_radar_spurious_returns(dets, "radar_multipath", rng)
        assert len(result) > len(dets)
        spurious = [d for d in result if d.get("spurious")]
        assert len(spurious) >= 1

    def test_empty_detections_multipath_still_adds_spurious(self):
        rng = np.random.default_rng(0)
        result = add_radar_spurious_returns([], "radar_multipath", rng)
        assert len(result) >= 1


class TestConditionSigmaLookups:
    @pytest.mark.parametrize("cond", list(VALID_CONDITIONS))
    def test_radar_sigma_is_positive(self, cond):
        assert condition_radar_sigma(cond) > 0.0

    @pytest.mark.parametrize("cond", list(VALID_CONDITIONS))
    def test_radar_velocity_sigma_is_positive(self, cond):
        assert condition_radar_velocity_sigma(cond) > 0.0

    @pytest.mark.parametrize("cond", list(VALID_CONDITIONS))
    def test_radar_position_sigma_is_positive(self, cond):
        assert condition_radar_position_sigma(cond) > 0.0

    def test_radar_multipath_has_highest_sigma(self):
        sigmas = {c: condition_radar_sigma(c) for c in VALID_CONDITIONS}
        assert sigmas["radar_multipath"] == max(sigmas.values())


class TestCameraBackboneSpecs:
    def test_yolov8n_returns_expected_keys(self):
        spec = camera_backbone_specs("yolov8n")
        assert "base_confidence" in spec
        assert "localization_sigma_m" in spec

    def test_yolov5_tiny_returns_expected_keys(self):
        spec = camera_backbone_specs("yolov5_tiny")
        assert spec["base_confidence"] < camera_backbone_specs("yolov8n")["base_confidence"]

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            camera_backbone_specs("resnet50")

    def test_returns_copy(self):
        spec1 = camera_backbone_specs("yolov8n")
        spec2 = camera_backbone_specs("yolov8n")
        spec1["base_confidence"] = 0.0
        assert spec2["base_confidence"] != 0.0


class TestClearMeasurementsFromDetections:
    def test_empty_returns_empty_array(self):
        result = clear_measurements_from_detections([])
        assert result.shape == (0, 2)

    def test_extracts_xy_correctly(self):
        dets = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        result = clear_measurements_from_detections(dets)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0])
        np.testing.assert_allclose(result[1], [3.0, 4.0])


# ---------------------------------------------------------------------------
# RadarPerceptionModel
# ---------------------------------------------------------------------------

class TestRadarPerceptionModel:
    def test_empty_positions_returns_empty_list(self):
        model = RadarPerceptionModel(condition="clear", seed=0)
        result = model.process(np.zeros((0, 2), dtype=np.float32))
        assert result == []

    def test_returns_detection_list_with_correct_keys(self):
        model = RadarPerceptionModel(condition="clear", seed=0)
        positions = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        result = model.process(positions)
        assert len(result) == 2
        for det in result:
            assert "x" in det and "y" in det and "range" in det and "velocity" in det

    def test_range_is_non_negative(self):
        model = RadarPerceptionModel(condition="clear", seed=0)
        positions = np.random.default_rng(0).random((20, 2)).astype(np.float32) * 100
        for det in model.process(positions):
            assert det["range"] >= 0.0

    def test_multipath_adds_spurious_detections(self):
        model = RadarPerceptionModel(condition="radar_multipath", seed=0)
        positions = np.array([[10.0, 20.0]] * 10, dtype=np.float32)
        result = model.process(positions)
        spurious = [d for d in result if d.get("spurious")]
        assert len(spurious) >= 1

    def test_set_condition_changes_condition(self):
        model = RadarPerceptionModel(condition="clear", seed=0)
        model.set_condition("rain")
        assert model.condition == "rain"

    def test_radar_processor_is_alias(self):
        # RadarProcessor should be identical to RadarPerceptionModel
        model = RadarProcessor(condition="clear", seed=0)
        positions = np.array([[5.0, 5.0]], dtype=np.float32)
        result = model.process(positions)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# CameraPerceptionModel (YOLODetector)
# ---------------------------------------------------------------------------

class TestCameraPerceptionModel:
    def test_empty_positions_returns_empty_list(self):
        model = CameraPerceptionModel(backbone="yolov8n", seed=0)
        result = model.detect(np.zeros((0, 2), dtype=np.float32))
        assert result == []

    def test_returns_detection_dicts_with_expected_keys(self):
        model = CameraPerceptionModel(backbone="yolov8n", condition="clear", seed=0)
        positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        result = model.detect(positions)
        for det in result:
            assert "bbox" in det
            assert "confidence" in det
            assert "x" in det and "y" in det

    def test_confidence_in_range(self):
        model = CameraPerceptionModel(backbone="yolov8n", condition="clear", seed=42)
        positions = np.ones((50, 2), dtype=np.float32) * 5.0
        for det in model.detect(positions):
            assert 0.0 <= det["confidence"] <= 1.0

    def test_night_reduces_detections(self):
        # With many positions, night should produce fewer detections than clear
        # (the 18% drop probability ensures this with high probability at n=500)
        positions = np.random.default_rng(0).random((500, 2)).astype(np.float32) * 50
        clear_model = CameraPerceptionModel(backbone="yolov8n", condition="clear", seed=0)
        night_model = CameraPerceptionModel(backbone="yolov8n", condition="night", seed=0)
        clear_count = len(clear_model.detect(positions))
        night_count = len(night_model.detect(positions))
        assert night_count < clear_count, (
            f"Night ({night_count}) should detect fewer vehicles than clear ({clear_count})"
        )

    def test_yolov5_has_lower_base_confidence(self):
        clear_model_v8 = CameraPerceptionModel(backbone="yolov8n", condition="clear", seed=0)
        clear_model_v5 = CameraPerceptionModel(backbone="yolov5_tiny", condition="clear", seed=0)
        assert clear_model_v5.spec["base_confidence"] < clear_model_v8.spec["base_confidence"]

    def test_set_condition_changes_condition(self):
        model = CameraPerceptionModel(backbone="yolov8n", condition="clear", seed=0)
        model.set_condition("rain")
        assert model.condition == "rain"

    def test_yolo_detector_is_alias(self):
        model = YOLODetector(backbone="yolov8n", seed=0)
        positions = np.array([[5.0, 5.0]], dtype=np.float32)
        # Should not raise and should return a list
        result = model.detect(positions)
        assert isinstance(result, list)

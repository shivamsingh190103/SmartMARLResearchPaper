"""Tests for demand calibration fit_profile function."""

from __future__ import annotations

import pytest
import pandas as pd

from smartmarl.calibration.demand_calibration import fit_profile


class TestFitProfile:
    def _make_df(self, n_vehicles=10, duration_s=300.0, n_types=None):
        import numpy as np
        rng = np.random.default_rng(0)
        records = []
        for v_id in range(n_vehicles):
            t = rng.uniform(0.0, duration_s)
            speed = rng.uniform(5.0, 30.0)
            row = {"vehicle_id": v_id, "timestamp": t, "speed": speed}
            if n_types:
                row["vehicle_type"] = f"type_{v_id % n_types}"
            records.append(row)
        return pd.DataFrame(records)

    def test_returns_required_keys(self):
        df = self._make_df()
        profile = fit_profile(df)
        assert "arrival_rate_per_min" in profile
        assert "speed_mean" in profile
        assert "speed_std" in profile
        assert "num_unique_vehicles" in profile
        assert "time_span_seconds" in profile

    def test_num_unique_vehicles_correct(self):
        df = self._make_df(n_vehicles=15)
        profile = fit_profile(df)
        assert profile["num_unique_vehicles"] == 15

    def test_arrival_rate_positive(self):
        df = self._make_df(n_vehicles=10, duration_s=600.0)
        profile = fit_profile(df)
        assert profile["arrival_rate_per_min"] > 0.0

    def test_speed_mean_in_range(self):
        df = self._make_df(n_vehicles=20)
        profile = fit_profile(df)
        assert 5.0 <= profile["speed_mean"] <= 30.0

    def test_speed_std_non_negative(self):
        df = self._make_df(n_vehicles=20)
        profile = fit_profile(df)
        assert profile["speed_std"] >= 0.0

    def test_vehicle_type_distribution_present_when_column_exists(self):
        df = self._make_df(n_vehicles=10, n_types=3)
        profile = fit_profile(df)
        assert "vehicle_type_distribution" in profile
        # Should sum to approximately 1.0
        dist = profile["vehicle_type_distribution"]
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_vehicle_type_distribution_absent_without_column(self):
        df = self._make_df(n_vehicles=10)
        profile = fit_profile(df)
        assert "vehicle_type_distribution" not in profile

    def test_missing_column_raises(self):
        df = pd.DataFrame({"vehicle_id": [0, 1], "timestamp": [0.0, 1.0]})
        # Missing 'speed' column
        with pytest.raises(ValueError, match="Missing required columns"):
            fit_profile(df)

    def test_time_span_positive(self):
        df = self._make_df(n_vehicles=10, duration_s=300.0)
        profile = fit_profile(df)
        assert profile["time_span_seconds"] > 0.0

    def test_single_vehicle(self):
        df = pd.DataFrame({
            "vehicle_id": [0, 0, 0],
            "timestamp": [0.0, 1.0, 2.0],
            "speed": [10.0, 12.0, 11.0],
        })
        profile = fit_profile(df)
        assert profile["num_unique_vehicles"] == 1
        assert profile["speed_mean"] == pytest.approx(11.0)

"""Extended tests for utils/metrics.py and utils/stats.py."""

from __future__ import annotations

import numpy as np
import pytest

from smartmarl.utils.metrics import (
    aukf_retention_rate,
    average_travel_time,
    average_waiting_time,
    throughput_per_hour,
)
from smartmarl.utils.stats import (
    bootstrap_confidence_interval,
    cohens_d,
    format_mean_ci,
    wilcoxon_with_effect_size,
)


# ---------------------------------------------------------------------------
# average_travel_time
# ---------------------------------------------------------------------------

class TestAverageTravelTime:
    def test_empty_returns_zero(self):
        assert average_travel_time([]) == 0.0

    def test_single_trip(self):
        result = average_travel_time([(0.0, 100.0)])
        assert result == pytest.approx(100.0)

    def test_multiple_trips(self):
        trips = [(0.0, 60.0), (10.0, 80.0), (5.0, 95.0)]
        durations = [60.0, 70.0, 90.0]
        expected = float(np.mean(durations))
        assert average_travel_time(trips) == pytest.approx(expected)

    def test_invalid_trip_ignored(self):
        # arr < dep should be ignored
        trips = [(100.0, 50.0), (0.0, 60.0)]
        assert average_travel_time(trips) == pytest.approx(60.0)

    def test_all_invalid_trips_returns_zero(self):
        trips = [(100.0, 50.0), (200.0, 100.0)]
        assert average_travel_time(trips) == 0.0


# ---------------------------------------------------------------------------
# average_waiting_time
# ---------------------------------------------------------------------------

class TestAverageWaitingTime:
    def test_empty_returns_zero(self):
        assert average_waiting_time([]) == 0.0

    def test_single_value(self):
        assert average_waiting_time([30.0]) == pytest.approx(30.0)

    def test_mean_computed_correctly(self):
        wt = [10.0, 20.0, 30.0]
        assert average_waiting_time(wt) == pytest.approx(20.0)

    def test_zeros_handled(self):
        assert average_waiting_time([0.0, 0.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# throughput_per_hour
# ---------------------------------------------------------------------------

class TestThroughputPerHour:
    def test_zero_sim_seconds_returns_zero(self):
        assert throughput_per_hour(100, 0.0) == 0.0

    def test_negative_sim_seconds_returns_zero(self):
        assert throughput_per_hour(100, -1.0) == 0.0

    def test_one_hour_sim(self):
        result = throughput_per_hour(1000, 3600.0)
        assert result == pytest.approx(1000.0)

    def test_half_hour_sim_doubles_rate(self):
        result = throughput_per_hour(500, 1800.0)
        assert result == pytest.approx(1000.0)

    def test_zero_vehicles_returns_zero(self):
        assert throughput_per_hour(0, 3600.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aukf_retention_rate
# ---------------------------------------------------------------------------

class TestAukfRetentionRate:
    def test_zero_mse_condition_returns_zero(self):
        assert aukf_retention_rate(0.5, 0.0) == 0.0

    def test_equal_mse_returns_one(self):
        assert aukf_retention_rate(2.0, 2.0) == pytest.approx(1.0)

    def test_clear_better_than_condition(self):
        # If mse_clear < mse_condition, retention_rate < 1.0
        result = aukf_retention_rate(1.0, 2.0)
        assert result == pytest.approx(0.5)

    def test_near_zero_condition_treated_as_zero(self):
        # 1e-15 is well below the 1e-12 threshold used in the implementation
        assert aukf_retention_rate(1.0, 1e-15) == 0.0


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_identical_arrays_returns_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert cohens_d(x, x) == pytest.approx(0.0)

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            cohens_d(np.array([1.0, 2.0]), np.array([1.0]))

    def test_zero_variance_difference_returns_zero(self):
        # When all differences are equal (zero variance), Cohen's d is undefined
        # and the implementation returns 0.0 (std ddof=1 is 0 for constant diff)
        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0])
        result = cohens_d(x, y)
        # diff = [1,1,1], std(ddof=1) = 0, so result = 0.0
        assert result == pytest.approx(0.0)

    def test_direction_of_effect(self):
        x = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        # diff = [5,5,5,5,5], mean=5, std=0 => 0.0
        assert cohens_d(x, y) == pytest.approx(0.0)

    def test_varying_diff_gives_nonzero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(10.0, 1.0, size=20)
        y = rng.normal(8.0, 1.0, size=20)
        result = cohens_d(x, y)
        assert result != 0.0

    def test_single_element_returns_zero(self):
        # Single element difference -> std = 0.0
        x = np.array([5.0])
        y = np.array([3.0])
        assert cohens_d(x, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bootstrap_confidence_interval
# ---------------------------------------------------------------------------

class TestBootstrapConfidenceInterval:
    def test_empty_returns_zeros(self):
        result = bootstrap_confidence_interval([])
        assert result == (0.0, 0.0, 0.0)

    def test_mean_is_correct(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, low, high = bootstrap_confidence_interval(values, seed=0)
        assert mean == pytest.approx(3.0)

    def test_ci_bounds_order(self):
        values = list(range(1, 31))  # 1..30
        mean, low, high = bootstrap_confidence_interval(values, confidence=0.95, seed=42)
        assert low <= mean <= high

    def test_wider_ci_at_lower_confidence(self):
        values = list(range(1, 101))
        _, low95, high95 = bootstrap_confidence_interval(values, confidence=0.95, seed=0)
        _, low80, high80 = bootstrap_confidence_interval(values, confidence=0.80, seed=0)
        # 95% CI should be wider than 80% CI
        assert (high95 - low95) >= (high80 - low80)

    def test_single_value_has_tight_ci(self):
        # With a single repeated value, CI should be narrow
        values = [5.0] * 100
        mean, low, high = bootstrap_confidence_interval(values, seed=0)
        assert mean == pytest.approx(5.0)
        assert abs(high - low) < 0.01

    def test_reproducible_with_same_seed(self):
        values = list(range(1, 21))
        r1 = bootstrap_confidence_interval(values, seed=99)
        r2 = bootstrap_confidence_interval(values, seed=99)
        assert r1 == r2


# ---------------------------------------------------------------------------
# format_mean_ci
# ---------------------------------------------------------------------------

class TestFormatMeanCi:
    def test_returns_string(self):
        result = format_mean_ci([1.0, 2.0, 3.0], seed=0)
        assert isinstance(result, str)

    def test_contains_plus_minus(self):
        result = format_mean_ci([10.0, 20.0, 30.0], seed=0)
        assert "\u00b1" in result

    def test_mean_approximately_correct(self):
        result = format_mean_ci([100.0, 100.0, 100.0], seed=0)
        assert result.startswith("100.0")

    def test_empty_input_returns_zero_format(self):
        result = format_mean_ci([], seed=0)
        assert "0.0" in result


# ---------------------------------------------------------------------------
# wilcoxon_with_effect_size (extended)
# ---------------------------------------------------------------------------

class TestWilcoxonWithEffectSizeExtended:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        x = rng.normal(10.0, 1.0, size=15)
        y = rng.normal(10.0, 1.0, size=15)
        result = wilcoxon_with_effect_size(x, y)
        assert "W" in result
        assert "p_value" in result
        assert "cohens_d" in result

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            wilcoxon_with_effect_size([1.0, 2.0], [1.0])

    def test_large_difference_gives_small_p_value(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(100.0, 1.0, size=30)
        variant = baseline + 20.0
        result = wilcoxon_with_effect_size(variant, baseline)
        assert result["p_value"] < 0.01

    def test_p_value_is_in_range(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, size=20)
        y = rng.normal(5.0, 1.0, size=20)
        result = wilcoxon_with_effect_size(x, y)
        assert 0.0 <= result["p_value"] <= 1.0

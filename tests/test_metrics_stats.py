import numpy as np

from smartmarl.utils.metrics import compute_metrics
from smartmarl.utils.stats import wilcoxon_with_effect_size


def test_att_metric_sanity_range_seconds():
    metrics = compute_metrics(
        completed_vehicles=100,
        total_waiting_time=5000.0,
        total_travel_time=14000.0,
        sim_seconds=3600,
    )
    assert 100.0 <= metrics["ATT"] <= 200.0


def test_wilcoxon_detects_large_difference():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=138.5, scale=1.0, size=30)
    variant = baseline + 15.0

    stats = wilcoxon_with_effect_size(variant, baseline)
    assert stats["p_value"] < 0.05

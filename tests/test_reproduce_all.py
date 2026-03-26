from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_result(path: Path, att: float, backend: str = "traci") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "att": float(att),
        "final_att": float(att),
        "backend": backend,
        "mock_mode": backend == "mock",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reproduce_all_builds_expected_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "repro"

    # Standard scenario baselines
    _write_result(raw / "standard_full_seed0.json", 140.0)
    _write_result(raw / "standard_full_seed1.json", 138.0)
    _write_result(raw / "standard_gplight_seed0.json", 145.0)
    _write_result(raw / "standard_gplight_seed1.json", 144.0)
    _write_result(raw / "standard_maxpressure_seed0.json", 150.0)
    _write_result(raw / "standard_fixed_time_seed0.json", 160.0)

    # Standard ablations
    _write_result(raw / "standard_no_aukf_seed0.json", 150.0)
    _write_result(raw / "standard_no_hetgnn_seed0.json", 147.0)
    _write_result(raw / "standard_l7_seed0.json", 149.0)

    # Indian scenario comparison
    _write_result(raw / "indian_hetero_full_seed0.json", 130.0)
    _write_result(raw / "indian_hetero_gplight_seed0.json", 142.0)

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_all.py",
            "--raw_dir",
            str(raw),
            "--out_dir",
            str(out),
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    expected = [
        "seed_metrics.csv",
        "method_summary.csv",
        "ablation_summary.csv",
        "paper_claims_audit.json",
        "paper_claims_audit.md",
        "dashboard.png",
    ]
    for name in expected:
        assert (out / name).exists(), name

    method_df = pd.read_csv(out / "method_summary.csv")
    assert set(method_df["variant"].tolist()) >= {"full", "gplight"}

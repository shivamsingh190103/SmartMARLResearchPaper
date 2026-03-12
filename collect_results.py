"""Collect SmartMARL per-seed JSON outputs into ablation summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LABELS = {
    "full": "Full SmartMARL",
    "no_ctde": "-CTDE (->IndepQL)",
    "no_aukf": "-AUKF (->raw counts)",
    "no_hetgnn": "-HetGNN (->hom. GAT)",
    "l7": "-Vsens only (L7)",
    "no_incident": "-Incident nodes",
    "no_ev": "-EV mode (normal)",
    "yolov5": "YOLOv5->YOLOv8n",
    "mlp": "MLP->GATv2 actor",
}

ORDER = ["full", "no_ctde", "no_aukf", "no_hetgnn", "l7", "no_incident", "no_ev", "yolov5", "mlp"]


def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    return "n.s."


def bootstrap_margin(values: np.ndarray, n_boot: int = 5000) -> float:
    if len(values) < 2:
        return 0.0
    rng = np.random.default_rng(0)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return (hi - lo) / 2.0


def load_variant(raw_dir: Path, scenario: str, variant: str) -> np.ndarray:
    pattern = f"{scenario}_{variant}_seed*.json"
    paths = sorted(raw_dir.glob(pattern))
    values = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            values.append(float(data["att"]))
        except Exception:
            continue
    return np.asarray(values, dtype=float)


def build_rows(raw_dir: Path, scenario: str) -> List[Dict]:
    full = load_variant(raw_dir, scenario, "full")
    full_mean = float(np.mean(full)) if len(full) else 0.0

    rows: List[Dict] = []
    for v in ORDER:
        vals = load_variant(raw_dir, scenario, v)
        if len(vals) == 0:
            continue

        mean = float(np.mean(vals))
        margin = float(bootstrap_margin(vals))
        att_fmt = f"{mean:.1f}\u00b1{margin:.1f}"

        if v == "full":
            delta = "-"
            p_fmt = "-"
            p = np.nan
        else:
            delta_v = mean - full_mean
            pct = (delta_v / full_mean * 100.0) if full_mean else 0.0
            delta = f"{delta_v:+.1f}s ({pct:+.1f}%)"
            if len(vals) >= 2 and len(full) >= 2:
                n = min(len(vals), len(full))
                p = float(wilcoxon(vals[:n], full[:n]).pvalue)
                p_fmt = format_pvalue(p)
            else:
                p = np.nan
                p_fmt = "n.s."

        rows.append(
            {
                "variant": v,
                "label": LABELS.get(v, v),
                "n": int(len(vals)),
                "ATT_mean": mean,
                "ATT_fmt": att_fmt,
                "delta": delta,
                "p_value": p,
                "p_fmt": p_fmt,
            }
        )
    return rows


def save_outputs(rows: List[Dict], out_csv: Path, out_txt: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    lines = [
        "Variant              | ATT (s)      | ΔATT               | p-value",
        "---------------------|--------------|--------------------|--------",
    ]
    for r in rows:
        lines.append(f"{r['label']:<21}| {r['ATT_fmt']:<12} | {r['delta']:<18} | {r['p_fmt']}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/raw")
    p.add_argument("--scenario", default="standard")
    p.add_argument("--out_csv", default="results/ablation_table.csv")
    p.add_argument("--out_txt", default="results/ablation_table.txt")
    args = p.parse_args()

    rows = build_rows(Path(args.raw_dir), args.scenario)
    if not rows:
        raise SystemExit("No result JSON files found.")
    save_outputs(rows, Path(args.out_csv), Path(args.out_txt))
    print(Path(args.out_txt).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

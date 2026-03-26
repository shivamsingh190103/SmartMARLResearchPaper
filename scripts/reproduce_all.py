"""Generate reproducible SmartMARL result artifacts from seed files.

Outputs:
- results/repro/seed_metrics.csv
- results/repro/method_summary.csv
- results/repro/ablation_summary.csv
- results/repro/paper_claims_audit.json
- results/repro/paper_claims_audit.md
- results/repro/dashboard.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Ensure local imports work when invoked as `python scripts/reproduce_all.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from collect_results import LABELS, ORDER, bootstrap_margin, load_variant_entries


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


METHODS: Sequence[MethodSpec] = (
    MethodSpec("full", "SmartMARL"),
    MethodSpec("gplight", "GPLight"),
    MethodSpec("maxpressure", "MaxPressure"),
    MethodSpec("fixed_time", "FixedTime"),
)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _wilcoxon_overlap(
    left: Sequence[Dict[str, object]],
    right: Sequence[Dict[str, object]],
) -> Tuple[float, int]:
    lmap = {int(x["seed"]): float(x["att"]) for x in left}
    rmap = {int(x["seed"]): float(x["att"]) for x in right}
    overlap = sorted(set(lmap).intersection(rmap))
    if len(overlap) < 2:
        return float("nan"), len(overlap)
    lvals = np.asarray([lmap[s] for s in overlap], dtype=float)
    rvals = np.asarray([rmap[s] for s in overlap], dtype=float)
    if np.allclose(lvals - rvals, 0.0):
        return 1.0, len(overlap)
    try:
        return float(wilcoxon(lvals, rvals, zero_method="zsplit", method="auto").pvalue), len(overlap)
    except Exception:
        return float("nan"), len(overlap)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}"


def _collect_seed_rows(
    raw_dir: Path,
    scenarios: Sequence[str],
    variants: Sequence[str],
    include_mock: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        for variant in variants:
            entries = load_variant_entries(raw_dir, scenario, variant, include_mock=include_mock)
            for e in entries:
                rows.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "seed": int(e["seed"]),
                        "att": float(e["att"]),
                        "backend": e.get("backend"),
                        "source_path": str(e.get("source_path", "")),
                    }
                )
    return rows


def _build_method_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in sorted(seed_df["scenario"].unique()):
        for method in METHODS:
            s = seed_df[(seed_df["scenario"] == scenario) & (seed_df["variant"] == method.key)]
            values = s["att"].astype(float).tolist()
            rows.append(
                {
                    "scenario": scenario,
                    "variant": method.key,
                    "method": method.label,
                    "n": int(len(values)),
                    "att_mean": _mean(values),
                    "att_margin": float(bootstrap_margin(np.asarray(values, dtype=float))) if len(values) >= 2 else 0.0,
                    "backends": ",".join(sorted({str(v) for v in s["backend"].dropna().unique()})),
                }
            )
    return pd.DataFrame(rows)


def _build_ablation_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    standard_df = seed_df[seed_df["scenario"] == "standard"]
    full_entries = standard_df[standard_df["variant"] == "full"].to_dict("records")
    full_mean = _mean([float(x["att"]) for x in full_entries])

    for variant in ORDER:
        v_entries = standard_df[standard_df["variant"] == variant].to_dict("records")
        if not v_entries:
            continue
        vals = np.asarray([float(x["att"]) for x in v_entries], dtype=float)
        mean = float(np.mean(vals))
        margin = float(bootstrap_margin(vals)) if len(vals) >= 2 else 0.0
        delta_s = mean - full_mean if np.isfinite(full_mean) else float("nan")
        delta_pct = (delta_s / full_mean * 100.0) if np.isfinite(full_mean) and full_mean else float("nan")
        p, n_overlap = (float("nan"), 0) if variant == "full" else _wilcoxon_overlap(v_entries, full_entries)
        rows.append(
            {
                "scenario": "standard",
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "n": int(len(v_entries)),
                "n_overlap_full": int(n_overlap),
                "att_mean": mean,
                "att_margin": margin,
                "delta_s": delta_s,
                "delta_pct": delta_pct,
                "p_value_vs_full": p,
                "p_fmt_vs_full": "-" if variant == "full" else _fmt_p(p),
            }
        )
    return pd.DataFrame(rows)


def _value_for(summary_df: pd.DataFrame, scenario: str, variant: str) -> float:
    row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["variant"] == variant)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["att_mean"])


def _write_claim_audit(
    out_json: Path,
    out_md: Path,
    method_summary: pd.DataFrame,
    seed_df: pd.DataFrame,
) -> None:
    claims: Dict[str, object] = {}

    for scenario in ("standard", "indian_hetero"):
        smart = _value_for(method_summary, scenario, "full")
        gplight = _value_for(method_summary, scenario, "gplight")
        smart_entries = seed_df[(seed_df["scenario"] == scenario) & (seed_df["variant"] == "full")].to_dict("records")
        gp_entries = seed_df[(seed_df["scenario"] == scenario) & (seed_df["variant"] == "gplight")].to_dict("records")
        p, overlap = _wilcoxon_overlap(smart_entries, gp_entries)
        reduction_pct = ((gplight - smart) / gplight * 100.0) if np.isfinite(smart) and np.isfinite(gplight) and gplight else float("nan")
        claims[scenario] = {
            "smartmarl_att": smart,
            "gplight_att": gplight,
            "reduction_pct": reduction_pct,
            "wilcoxon_p": p,
            "wilcoxon_p_fmt": _fmt_p(p),
            "n_overlap": overlap,
        }

    # Backend integrity summary.
    backend_counts = (
        seed_df.groupby(["scenario", "variant", "backend"]).size().reset_index(name="count").to_dict("records")
        if not seed_df.empty
        else []
    )
    claims["backend_counts"] = backend_counts
    claims["notes"] = [
        "Claims are computed only from artifacts present under results/raw.",
        "If n_overlap is low or missing, paper claims are not statistically auditable yet.",
        "Mock-backend rows are excluded unless --include-mock is passed.",
    ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(claims, indent=2), encoding="utf-8")

    lines = [
        "# SmartMARL Paper Claims Audit",
        "",
        "This report is auto-generated from `results/raw` artifacts.",
        "",
        "| Scenario | SmartMARL ATT | GPLight ATT | Reduction | Wilcoxon p | Overlap N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, label in (("standard", "Standard"), ("indian_hetero", "Indian Heterogeneous")):
        c = claims.get(scenario, {})
        lines.append(
            "| "
            f"{label} | "
            f"{c.get('smartmarl_att', float('nan')):.3f} | "
            f"{c.get('gplight_att', float('nan')):.3f} | "
            f"{c.get('reduction_pct', float('nan')):.3f}% | "
            f"{c.get('wilcoxon_p_fmt', 'n/a')} | "
            f"{int(c.get('n_overlap', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- These numbers are auditable only if corresponding seed artifacts exist.",
            "- Run this script again after adding more seeds to refresh all figures and stats.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_dashboard(
    out_png: Path,
    method_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle("SmartMARL Reproducibility Dashboard (from results/raw)", fontsize=13)

    # Panel 1: standard method comparison
    ax = axes[0]
    s_df = method_summary[method_summary["scenario"] == "standard"]
    methods = [m.label for m in METHODS]
    vals = [
        float(
            s_df[s_df["variant"] == m.key]["att_mean"].iloc[0]
            if not s_df[s_df["variant"] == m.key].empty
            else np.nan
        )
        for m in METHODS
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    ax.bar(methods, np.nan_to_num(vals, nan=0.0), color=colors, alpha=0.85)
    for i, val in enumerate(vals):
        txt = "NA" if not np.isfinite(val) else f"{val:.1f}"
        y = 0.5 if not np.isfinite(val) else val + 0.5
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=9)
    ax.set_title("Standard Scenario ATT")
    ax.set_ylabel("ATT (s)")
    ax.tick_params(axis="x", labelrotation=18)

    # Panel 2: indian method comparison
    ax = axes[1]
    i_df = method_summary[method_summary["scenario"] == "indian_hetero"]
    vals = [
        float(
            i_df[i_df["variant"] == m.key]["att_mean"].iloc[0]
            if not i_df[i_df["variant"] == m.key].empty
            else np.nan
        )
        for m in METHODS
    ]
    ax.bar(methods, np.nan_to_num(vals, nan=0.0), color=colors, alpha=0.85)
    for i, val in enumerate(vals):
        txt = "NA" if not np.isfinite(val) else f"{val:.1f}"
        y = 0.5 if not np.isfinite(val) else val + 0.5
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=9)
    ax.set_title("Indian Heterogeneous ATT")
    ax.set_ylabel("ATT (s)")
    ax.tick_params(axis="x", labelrotation=18)

    # Panel 3: ablation
    ax = axes[2]
    if ablation_summary.empty:
        ax.text(0.5, 0.5, "No ablation data found", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = ablation_summary["label"].tolist()
        means = ablation_summary["att_mean"].astype(float).to_numpy()
        margins = ablation_summary["att_margin"].astype(float).to_numpy()
        y = np.arange(len(labels))
        colors = ["#2ca02c" if v == "full" else "#d62728" for v in ablation_summary["variant"].tolist()]
        ax.barh(y, means, xerr=margins, color=colors, alpha=0.9, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("ATT (s)")
        ax.set_title("Standard Ablation ATT")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate reproducible SmartMARL result artifacts.")
    p.add_argument("--raw_dir", default="results/raw")
    p.add_argument("--out_dir", default="results/repro")
    p.add_argument(
        "--include_mock",
        action="store_true",
        help="Include entries marked as mock backend.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ("standard", "indian_hetero")
    variants = list({*ORDER, "gplight", "maxpressure", "fixed_time"})
    seed_rows = _collect_seed_rows(raw_dir=raw_dir, scenarios=scenarios, variants=variants, include_mock=args.include_mock)
    seed_df = pd.DataFrame(seed_rows)

    if seed_df.empty:
        raise SystemExit("No seed artifacts found to reproduce outputs.")

    seed_df = seed_df.sort_values(["scenario", "variant", "seed"]).reset_index(drop=True)
    seed_csv = out_dir / "seed_metrics.csv"
    seed_df.to_csv(seed_csv, index=False)

    method_summary = _build_method_summary(seed_df)
    method_csv = out_dir / "method_summary.csv"
    method_summary.to_csv(method_csv, index=False)

    ablation_summary = _build_ablation_summary(seed_df)
    ablation_csv = out_dir / "ablation_summary.csv"
    ablation_summary.to_csv(ablation_csv, index=False)

    claims_json = out_dir / "paper_claims_audit.json"
    claims_md = out_dir / "paper_claims_audit.md"
    _write_claim_audit(claims_json, claims_md, method_summary, seed_df)

    dashboard_png = out_dir / "dashboard.png"
    _render_dashboard(dashboard_png, method_summary, ablation_summary)

    print(f"Wrote {seed_csv}")
    print(f"Wrote {method_csv}")
    print(f"Wrote {ablation_csv}")
    print(f"Wrote {claims_json}")
    print(f"Wrote {claims_md}")
    print(f"Wrote {dashboard_png}")


if __name__ == "__main__":
    main()

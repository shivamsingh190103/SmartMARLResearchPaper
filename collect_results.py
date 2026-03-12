"""Collect SmartMARL per-seed outputs into ablation summary tables.

Supports both JSON and CSV seed files and multiple historical filename styles.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

VARIANT_ALIASES: Dict[str, List[str]] = {
    "full": ["full", "full_smartmarl"],
    "no_ctde": ["no_ctde"],
    "no_aukf": ["no_aukf"],
    "no_hetgnn": ["no_hetgnn"],
    "l7": ["l7", "l7_ablation"],
    "no_incident": ["no_incident", "no_incident_nodes"],
    "no_ev": ["no_ev", "no_ev_mode"],
    "yolov5": ["yolov5", "yolov5_backbone"],
    "mlp": ["mlp", "mlp_actor"],
}

SEED_RE = re.compile(r"seed(?P<seed>\d+)\.(json|csv)$")


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


def _extract_seed(path: Path) -> int | None:
    m = SEED_RE.search(path.name)
    if not m:
        return None
    return int(m.group("seed"))


def _normalize_backend(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"traci", "sumo", "real"}:
        return "traci"
    if text in {"mock", "cityflow"}:
        return "mock"
    if text in {"true", "1"}:
        return "mock"
    if text in {"false", "0"}:
        return "traci"
    return None


def _read_att_from_json(path: Path) -> Tuple[float, Optional[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("att", "final_att", "ATT"):
        if key in data:
            att = float(data[key])
            backend = _normalize_backend(data.get("backend"))
            if backend is None and "mock_mode" in data:
                backend = "mock" if bool(data["mock_mode"]) else "traci"
            return att, backend
    raise KeyError(f"No ATT key in {path.name}")


def _read_att_from_csv(path: Path) -> Tuple[float, Optional[str]]:
    df = pd.read_csv(path)
    backend: Optional[str] = None
    if "backend" in df.columns and len(df["backend"]) > 0:
        backend = _normalize_backend(df["backend"].iloc[-1])
    elif "mock_mode" in df.columns and len(df["mock_mode"]) > 0:
        backend = "mock" if bool(df["mock_mode"].iloc[-1]) else "traci"
    for key in ("ATT", "att", "final_att"):
        if key in df.columns and len(df[key]) > 0:
            return float(df[key].iloc[-1]), backend
    raise KeyError(f"No ATT column in {path.name}")


def _candidate_paths(raw_dir: Path, scenario: str, alias: str, ext: str) -> List[Path]:
    patterns = [
        f"{scenario}_{alias}_seed*.{ext}",
        f"{alias}_{scenario}_seed*.{ext}",
        f"{alias}_seed*.{ext}",
    ]
    out: List[Path] = []
    seen = set()
    for pat in patterns:
        for p in sorted(raw_dir.glob(pat)):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _backend_quality(backend: Optional[str]) -> int:
    if backend == "traci":
        return 2
    if backend is None:
        return 1
    return 0


def load_variant_records(
    raw_dir: Path,
    scenario: str,
    variant: str,
    include_mock: bool = False,
) -> Dict[int, Tuple[float, Optional[str]]]:
    aliases = VARIANT_ALIASES.get(variant, [variant])
    # seed -> (value, backend, rank_tuple)
    # rank_tuple order: backend_quality, format_priority, mtime
    # format_priority: json=1, csv=0
    acc: Dict[int, Tuple[float, Optional[str], Tuple[int, int, float]]] = {}

    for alias in aliases:
        for p in _candidate_paths(raw_dir, scenario, alias, "csv"):
            seed = _extract_seed(p)
            if seed is None:
                continue
            try:
                val, backend = _read_att_from_csv(p)
            except Exception:
                continue
            if backend == "mock" and not include_mock:
                continue
            mtime = p.stat().st_mtime
            rank = (_backend_quality(backend), 0, mtime)
            prev = acc.get(seed)
            if prev is None or rank > prev[2]:
                acc[seed] = (val, backend, rank)

        for p in _candidate_paths(raw_dir, scenario, alias, "json"):
            seed = _extract_seed(p)
            if seed is None:
                continue
            try:
                val, backend = _read_att_from_json(p)
            except Exception:
                continue
            if backend == "mock" and not include_mock:
                continue
            mtime = p.stat().st_mtime
            rank = (_backend_quality(backend), 1, mtime)
            prev = acc.get(seed)
            if prev is None or rank > prev[2]:
                acc[seed] = (val, backend, rank)

    return {seed: (tup[0], tup[1]) for seed, tup in acc.items()}


def build_rows(raw_dir: Path, scenario: str, include_mock: bool = False) -> List[Dict]:
    full_records = load_variant_records(raw_dir, scenario, "full", include_mock=include_mock)
    full_values = np.asarray([full_records[s][0] for s in sorted(full_records)], dtype=float)
    full_mean = float(np.mean(full_values)) if len(full_values) else np.nan

    rows: List[Dict] = []
    for variant in ORDER:
        records = load_variant_records(raw_dir, scenario, variant, include_mock=include_mock)
        if not records:
            continue

        seeds = sorted(records)
        values = np.asarray([records[s][0] for s in seeds], dtype=float)
        mean = float(np.mean(values))
        margin = float(bootstrap_margin(values))
        att_fmt = f"{mean:.1f}+/-{margin:.1f}"

        overlap = sorted(set(records).intersection(full_records))
        if variant == "full":
            delta = "-"
            p = np.nan
            p_fmt = "-"
        else:
            if np.isfinite(full_mean) and full_mean != 0.0:
                delta_v = mean - full_mean
                pct = delta_v / full_mean * 100.0
                delta = f"{delta_v:+.1f}s ({pct:+.1f}%)"
            else:
                delta = "n/a"

            if len(overlap) >= 2:
                x = np.asarray([records[s][0] for s in overlap], dtype=float)
                y = np.asarray([full_records[s][0] for s in overlap], dtype=float)
                diff = x - y
                if np.allclose(diff, 0.0, atol=1e-12, rtol=0.0):
                    p = 1.0
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            p = float(
                                wilcoxon(
                                    x,
                                    y,
                                    zero_method="zsplit",
                                    alternative="two-sided",
                                    method="auto",
                                ).pvalue
                            )
                    except Exception:
                        p = np.nan
                p_fmt = format_pvalue(p)
            else:
                p = np.nan
                p_fmt = "n.s."

        rows.append(
            {
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "n": int(len(values)),
                "n_overlap_full": int(len(overlap)),
                "ATT_mean": mean,
                "ATT_margin": margin,
                "ATT_fmt": att_fmt,
                "delta": delta,
                "p_value": p,
                "p_fmt": p_fmt,
            }
        )
    return rows


def save_outputs(rows: List[Dict], out_csv: Path, out_txt: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    lines = [
        "Variant                | ATT (s)      | DeltaATT            | p-value | n | n_overlap",
        "-----------------------|--------------|---------------------|---------|---|----------",
    ]
    for r in rows:
        lines.append(
            f"{r['label']:<23}| {r['ATT_fmt']:<12} | {r['delta']:<19} | {r['p_fmt']:<7} | "
            f"{r['n']:>2} | {r['n_overlap_full']:>8}"
        )
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="results/raw")
    parser.add_argument("--scenario", default="standard")
    parser.add_argument("--out_csv", default="results/ablation_table.csv")
    parser.add_argument("--out_txt", default="results/ablation_table.txt")
    parser.add_argument(
        "--include_mock",
        action="store_true",
        help="Include files explicitly marked as mock backend.",
    )
    args = parser.parse_args()

    rows = build_rows(Path(args.raw_dir), args.scenario, include_mock=args.include_mock)
    if not rows:
        raise SystemExit("No result JSON/CSV files found (after backend filtering).")
    save_outputs(rows, Path(args.out_csv), Path(args.out_txt))
    print(Path(args.out_txt).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

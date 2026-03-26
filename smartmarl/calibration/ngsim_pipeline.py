"""NGSIM-oriented calibration helper for SmartMARL."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from smartmarl.calibration.demand_calibration import fit_profile


def _download_csv(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def _normalize_ngsim_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in {"vehicle_id", "vehicleid", "id"}:
            rename_map[col] = "vehicle_id"
        elif low in {"global_time", "timestamp", "time"}:
            rename_map[col] = "timestamp"
        elif low in {"v_vel", "speed", "velocity"}:
            rename_map[col] = "speed"
        elif low in {"v_class", "vehicle_type", "class"}:
            rename_map[col] = "vehicle_type"
    out = df.rename(columns=rename_map)
    required = {"vehicle_id", "timestamp", "speed"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required normalized columns: {sorted(missing)}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/normalize NGSIM CSV and produce SmartMARL calibration profile.")
    p.add_argument("--input_csv", default="", help="Existing NGSIM CSV path (if already downloaded)")
    p.add_argument("--download_url", default="", help="Optional URL to download NGSIM CSV")
    p.add_argument("--download_to", default="data/ngsim/ngsim.csv")
    p.add_argument("--output", default="results/calibration/ngsim_profile.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser() if args.input_csv else None

    if input_csv is None and args.download_url:
        input_csv = _download_csv(args.download_url, Path(args.download_to))
    if input_csv is None:
        raise SystemExit("Provide --input_csv or --download_url.")
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    norm = _normalize_ngsim_columns(df)
    profile = fit_profile(norm)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    out.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

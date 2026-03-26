"""Build and package reproducibility artifacts into a single zip bundle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SmartMARL reproducibility bundle.")
    p.add_argument("--raw_dir", default="results/raw")
    p.add_argument("--out_dir", default="results/repro")
    p.add_argument("--bundle", default="results/repro_bundle.zip")
    p.add_argument("--include_mock", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "reproduce_all.py"),
        "--raw_dir",
        args.raw_dir,
        "--out_dir",
        str(out_dir),
    ]
    if args.include_mock:
        cmd.append("--include_mock")
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    bundle_path = Path(args.bundle)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(
        str(bundle_path.with_suffix("")),
        "zip",
        root_dir=str(out_dir.parent),
        base_dir=str(out_dir.name),
    )
    print(f"Wrote bundle: {bundle_path}")


if __name__ == "__main__":
    main()

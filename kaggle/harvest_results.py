"""Download and validate Kaggle kernel result JSON files into results/raw/."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.common import (
    KERNEL_SPECS,
    ROOT,
    RESULTS_RAW,
    atomic_write_text,
    kaggle_bin,
    log_line,
    validate_result,
)

HARVEST_LOG = ROOT / 'kaggle' / 'harvest_log.txt'
OUTPUT_ROOT = ROOT / 'kaggle' / 'kernel_outputs'


def _download_kernel_output(slug: str, dry_run: bool) -> Tuple[bool, str]:
    kb = kaggle_bin()
    out_dir = OUTPUT_ROOT / slug.split('/', 1)[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [kb, 'kernels', 'output', slug, '-p', str(out_dir), '--force']
    if dry_run:
        return True, f"DRY-RUN {' '.join(cmd)}"

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or 'unknown error').strip()
        return False, msg
    return True, (proc.stdout or '').strip()


def _safe_copy_json(src: Path, dst: Path) -> None:
    text = src.read_text(encoding='utf-8')
    atomic_write_text(dst, text)


def harvest_once(dry_run: bool = False) -> Dict[str, int]:
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    counters = {
        'downloaded': 0,
        'copied': 0,
        'skipped_existing': 0,
        'invalid': 0,
        'errors': 0,
    }

    for spec in KERNEL_SPECS:
        ok, msg = _download_kernel_output(spec.slug, dry_run=dry_run)
        if not ok:
            counters['errors'] += 1
            log_line(HARVEST_LOG, f'ERROR download {spec.slug}: {msg}')
            continue

        counters['downloaded'] += 1
        log_line(HARVEST_LOG, f'DOWNLOAD {spec.slug}: {msg}')

        if dry_run:
            continue

        kernel_dir = OUTPUT_ROOT / spec.slug.split('/', 1)[1]
        for src in sorted(kernel_dir.rglob('*.json')):
            dst = RESULTS_RAW / src.name
            if dst.exists():
                counters['skipped_existing'] += 1
                continue

            # Validate source first. Normalize in source copy, then copy atomically.
            valid, reason = validate_result(src, normalize=True)
            if not valid:
                counters['invalid'] += 1
                log_line(HARVEST_LOG, f'INVALID {src}: {reason}')
                continue

            _safe_copy_json(src, dst)
            counters['copied'] += 1
            log_line(HARVEST_LOG, f'COPIED {src.name} -> {dst}')

    return counters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Harvest Kaggle kernel outputs into results/raw/')
    p.add_argument('--dry-run', action='store_true', help='Only perform status/output calls, no file writes')
    return p.parse_args()


def main() -> None:
    # Graceful PATH handling.
    os.environ['PATH'] = f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"
    args = parse_args()

    try:
        counters = harvest_once(dry_run=args.dry_run)
        print(
            'harvest_results:',
            f"downloaded={counters['downloaded']}",
            f"copied={counters['copied']}",
            f"skipped_existing={counters['skipped_existing']}",
            f"invalid={counters['invalid']}",
            f"errors={counters['errors']}",
        )
    except FileNotFoundError as exc:
        print(f'kaggle CLI not found: {exc}')
    except Exception as exc:  # pragma: no cover - CLI/network dependent
        print(f'harvest failed: {exc}')


if __name__ == '__main__':
    main()

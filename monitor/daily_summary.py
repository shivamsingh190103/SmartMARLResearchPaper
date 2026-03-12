"""Daily SmartMARL status summary generator."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.common import (
    KERNEL_SPECS,
    ROOT,
    atomic_append_line,
    atomic_write_text,
    completed_seed_count,
    ensure_kaggle_auth_env,
    kernel_status_with_output,
    load_att_series,
)

DAILY_LOG = ROOT / 'monitor' / 'daily_log.txt'
TODAY_TXT = ROOT / 'monitor' / 'today.txt'


def _local_snapshot() -> Tuple[str, int, float, bool]:
    csv = ROOT / 'results' / 'training_logs' / 'standard_full_seed0.csv'
    episodes, atts = load_att_series(csv)
    if not episodes:
        return 'seed0 no data', 0, 0.0, False

    ep = episodes[-1]
    att = atts[-1]
    improving = False
    if len(atts) >= 120:
        recent = sum(atts[-20:]) / 20.0
        prev = sum(atts[-120:-100]) / 20.0
        improving = recent < prev
    return f'seed0 ep{ep} ATT{att:.0f}s', ep, att, improving


def _kaggle_running_count() -> int:
    ensure_kaggle_auth_env()
    running = 0
    for spec in KERNEL_SPECS:
        status, _raw = kernel_status_with_output(spec.slug)
        if 'RUNNING' in status or 'QUEUED' in status:
            running += 1
    return running


def _day_index() -> int:
    if not DAILY_LOG.exists():
        return 1
    lines = [ln for ln in DAILY_LOG.read_text(encoding='utf-8').splitlines() if ln.strip()]
    return len(lines) + 1


def _status_label(improving: bool, running_kernels: int) -> str:
    if improving and running_kernels >= 2:
        return 'ON TRACK'
    if running_kernels == 0:
        return 'ACTION REQUIRED'
    return 'MONITOR'


def generate_summary() -> str:
    now = datetime.now()
    local_text, ep, att, improving = _local_snapshot()
    running = _kaggle_running_count()
    full_done = completed_seed_count('standard', 'full')
    status = _status_label(improving, running)

    one_line = (
        f"{now.strftime('%Y-%m-%d %H:%M')} | Local: {local_text} | "
        f"Kaggle: {running} running | Results: {full_done}/30 | Status: {status}"
    )
    atomic_append_line(DAILY_LOG, one_line)

    day = _day_index()
    est_days_left = max(1, 5 - min(4, day - 1))

    body_lines: List[str] = [
        '=== SmartMARL Daily Summary ===',
        f'Day {day} of ~5',
        '',
        (
            f'LOCAL: {local_text}. '
            + ('ATT decreasing. Healthy.' if improving else 'Trend not clearly decreasing yet.')
        ),
        (
            f'KAGGLE: {running} kernels running. '
            + ('No immediate timeout concern.' if running >= 2 else 'Check kernel states now.')
        ),
        f'RESULTS: {full_done} seeds collected so far.',
        '',
        'WHAT YOU NEED TO DO TODAY: Nothing.',
        'WHAT YOU NEED TO DO TOMORROW MORNING:',
        '  Check kaggle.com for any timed-out kernels.',
        '  Run: python monitor/health_check.py',
        '',
        f'Estimated completion: ~{est_days_left} days from now.',
        '================================',
        '',
    ]
    text = '\n'.join(body_lines)
    atomic_write_text(TODAY_TXT, text)
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate daily SmartMARL summary files.')
    return p.parse_args()


def main() -> None:
    parse_args()
    print(generate_summary())


if __name__ == '__main__':
    main()

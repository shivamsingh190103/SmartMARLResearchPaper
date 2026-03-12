"""SmartMARL training health monitor."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.common import (
    KERNEL_SPECS,
    ROOT,
    RESULTS_RAW,
    completed_seeds,
    ensure_kaggle_auth_env,
    kernel_status_with_output,
    load_att_series,
    safe_run,
    validate_result,
)


@dataclass
class LocalRunStatus:
    label: str
    csv_path: Path
    pid_file: Optional[Path]
    episode: int
    att: float
    state: str
    warnings: List[str]


def _trend_state(episodes: List[int], atts: List[float]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if len(episodes) < 120:
        return 'STARTING', warnings

    ep = episodes[-1]
    now = float(np.mean(atts[-25:]))
    prev = float(np.mean(atts[-125:-100])) if len(atts) >= 125 else float(atts[0])
    delta = now - prev

    if ep >= 500 and delta > 0.8:
        warnings.append(f'ATT increasing over last 100 episodes ({delta:+.2f}s)')
        return 'CRITICAL', warnings
    if ep >= 300 and delta >= 0.0:
        warnings.append(f'ATT not decreasing after ep300 ({delta:+.2f}s)')
        return 'WARN', warnings
    return 'HEALTHY', warnings


def _arrow_for_state(state: str) -> str:
    if state == 'HEALTHY':
        return '↓'
    if state == 'WARN':
        return '→'
    if state == 'CRITICAL':
        return '↑'
    return '·'


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _extract_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding='utf-8').strip())
    except Exception:
        return None


def _local_runs() -> List[LocalRunStatus]:
    mapping = [
        ('standard/full  seed0', ROOT / 'results' / 'training_logs' / 'standard_full_seed0.csv', ROOT / 'logs' / 'standard_seed0.pid'),
        ('standard/l7    seed0', ROOT / 'results' / 'training_logs' / 'standard_l7_seed0.csv', None),
        ('indian/full    seed0', ROOT / 'results' / 'training_logs' / 'indian_hetero_full_seed0.csv', None),
    ]
    out: List[LocalRunStatus] = []
    for label, csv_path, pid_file in mapping:
        episodes, atts = load_att_series(csv_path)
        if not episodes:
            out.append(LocalRunStatus(label, csv_path, pid_file, 0, 0.0, 'NO_DATA', ['no CSV data yet']))
            continue
        state, warns = _trend_state(episodes, atts)
        episode = episodes[-1]
        att = atts[-1]

        if pid_file is not None and pid_file.exists():
            pid = _extract_pid(pid_file)
            if pid is not None and not _is_pid_alive(pid) and episode < 3000:
                warns.append(f'PID {pid} is not alive before completion')
                if state == 'HEALTHY':
                    state = 'WARN'

        out.append(LocalRunStatus(label, csv_path, pid_file, episode, att, state, warns))
    return out


def _pid_file_warnings() -> List[str]:
    warnings: List[str] = []
    logs_dir = ROOT / 'logs'
    if not logs_dir.exists():
        return warnings
    for pid_file in sorted(logs_dir.glob('*.pid')):
        pid = _extract_pid(pid_file)
        if pid is None:
            continue
        if not _is_pid_alive(pid):
            warnings.append(f'PID file {pid_file.name} points to dead process ({pid})')
    return warnings


def _kernel_last_update_minutes(raw: str) -> Optional[int]:
    m = re.search(r'(\d+)\s+minutes?\s+ago', raw)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)\s+hours?\s+ago', raw)
    if m2:
        return int(m2.group(1)) * 60
    return None


def _kernel_report() -> Tuple[List[str], List[str], int]:
    warnings: List[str] = []
    lines: List[str] = []
    running = 0

    ensure_kaggle_auth_env()
    rc, list_out = safe_run(['kaggle', 'kernels', 'list', '--mine'], timeout=120)
    if rc != 0:
        warnings.append('kaggle kernels list --mine failed; verify CLI auth')
        lines.append('  (Kaggle CLI unavailable or unauthenticated)')
    else:
        # Show only first few lines of CLI output in case user needs direct visibility.
        preview = '\n'.join(list_out.splitlines()[:6]).strip()
        if preview:
            lines.append('  list --mine: OK')

    for spec in KERNEL_SPECS:
        status, raw = kernel_status_with_output(spec.slug)
        short = spec.slug.split('/')[-1].replace('smartmarl-', '')
        mins = _kernel_last_update_minutes(raw)
        if 'RUNNING' in status or 'QUEUED' in status:
            running += 1
            if mins is not None:
                lines.append(f'  {short}: RUNNING (last update {mins}min ago)')
            else:
                lines.append(f'  {short}: RUNNING')
        elif status == 'UNKNOWN':
            lines.append(f'  {short}: UNKNOWN')
            warnings.append(f'Unable to resolve status for {spec.slug}')
        else:
            lines.append(f'  {short}: {status.replace("KernelWorkerStatus.", "")}')

    if running < 2:
        warnings.append(f'Only {running} Kaggle kernels running (expected at least 2).')

    return lines, warnings, running


def _results_report() -> Tuple[List[str], List[str], int, int, int]:
    lines: List[str] = []
    warnings: List[str] = []

    valid = 0
    invalid = 0
    for path in sorted(RESULTS_RAW.glob('*.json')):
        ok, _reason = validate_result(path, normalize=False)
        if ok:
            valid += 1
        else:
            invalid += 1

    full_done = completed_seeds('standard', 'full', 0, 29)
    l7_done = completed_seeds('standard', 'l7', 0, 29)
    full_n = len(full_done)
    l7_n = len(l7_done)
    missing = (30 - full_n) + (30 - l7_n)

    lines.append(f'  full/standard: {full_n}/30 seeds done')
    lines.append(f'  l7/standard:   {l7_n}/30 seeds done')
    lines.append(f'  valid JSON files: {valid}, invalid: {invalid}, missing target slots: {missing}')

    if invalid > 0:
        warnings.append(f'{invalid} invalid result JSON files detected.')

    return lines, warnings, valid, missing, invalid


def _convergence_lines(local_runs: List[LocalRunStatus]) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    warnings: List[str] = []
    target_ep = 3000
    for run in local_runs:
        if run.episode < 1000 or not run.csv_path.exists():
            continue
        episodes, atts = load_att_series(run.csv_path)
        if len(episodes) < 50:
            continue
        tail_n = min(200, len(episodes))
        x = np.asarray(episodes[-tail_n:], dtype=float)
        y = np.asarray(atts[-tail_n:], dtype=float)
        # Linear trend for extrapolation.
        slope, intercept = np.polyfit(x, y, 1)
        est = slope * float(target_ep) + intercept
        lines.append(
            f'  {run.label}: Estimated final ATT {est:.1f}s '
            f'(currently ep{run.episode}: {run.att:.1f}s)'
        )
        if est > 150.0:
            warnings.append(f'{run.label} estimated final ATT {est:.1f}s (>150s)')
    return lines, warnings


def build_report() -> str:
    local = _local_runs()
    kernel_lines, kernel_warnings, running_kernels = _kernel_report()
    result_lines, result_warnings, _valid, _missing, _invalid = _results_report()
    conv_lines, conv_warnings = _convergence_lines(local)

    critical: List[str] = []
    warnings: List[str] = []

    for run in local:
        if run.state == 'CRITICAL':
            critical.extend(run.warnings or [f'{run.label} marked critical'])
        elif run.state == 'WARN':
            warnings.extend(run.warnings)

    warnings.extend(kernel_warnings)
    warnings.extend(result_warnings)
    warnings.extend(conv_warnings)
    warnings.extend(_pid_file_warnings())

    lines: List[str] = []
    lines.append('============ SmartMARL Health Report ============')
    lines.append(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append('LOCAL TRAINING:')
    for run in local:
        arrow = _arrow_for_state(run.state)
        if run.episode > 0:
            lines.append(
                f'  {run.label}: episode {run.episode}, ATT={run.att:.1f}s {arrow} {run.state}'
            )
        else:
            lines.append(f'  {run.label}: no data yet ({run.state})')
    lines.append('')
    lines.append('KAGGLE KERNELS:')
    lines.extend(kernel_lines)
    lines.append('')
    lines.append('RESULTS COLLECTED:')
    lines.extend(result_lines)
    if running_kernels >= 2:
        lines.append('  Expected completion: ~4 days')
    else:
        lines.append('  Expected completion: delayed (check kernels)')
    lines.append('')

    if conv_lines:
        lines.append('ATT CONVERGENCE ESTIMATE:')
        lines.extend(conv_lines)
        lines.append('')

    lines.append(f'CRITICAL ISSUES: {", ".join(critical) if critical else "None"}')
    lines.append(f'WARNINGS: {", ".join(warnings) if warnings else "None"}')
    lines.append('')
    if critical:
        next_action = 'Immediate debugging needed for critical items.'
    elif warnings:
        next_action = 'Review warnings and rerun health check in a few hours.'
    else:
        next_action = 'Nothing needed. Check again tomorrow.'
    lines.append(f'NEXT ACTION: {next_action}')
    lines.append('=================================================')
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SmartMARL health report')
    p.add_argument('--watch', action='store_true', help='Repeat every 30 minutes')
    return p.parse_args()


def run_once() -> None:
    print(build_report())


def main() -> None:
    args = parse_args()
    if not args.watch:
        run_once()
        return
    while True:
        run_once()
        time.sleep(30 * 60)


if __name__ == '__main__':
    main()

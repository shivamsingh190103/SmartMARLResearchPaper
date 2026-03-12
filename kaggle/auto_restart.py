"""Kaggle kernel auto-restarter for long SmartMARL training runs.

Checks tracked kernels every hour and restarts timed-out terminal sessions
unless all expected seed result files are already present in results/raw/.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'kaggle') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'kaggle'))

from monitor.common import (
    KERNEL_SPECS,
    ROOT,
    all_expected_done,
    ensure_kaggle_auth_env,
    kernel_status_with_output,
    load_kaggle_auth,
    log_line,
)

RESTART_LOG = ROOT / 'kaggle' / 'restart_log.txt'
RESTARTABLE_STATUS_HINTS = ('COMPLETE', 'ERROR', 'CANCEL')
RUNNING_STATUS_HINTS = ('RUNNING',)
from harvest_results import harvest_once  # noqa: E402


def _status_is_running(status: str) -> bool:
    upper = status.upper()
    return any(h in upper for h in RUNNING_STATUS_HINTS)


def _status_is_restartable(status: str) -> bool:
    upper = status.upper()
    return any(h in upper for h in RESTARTABLE_STATUS_HINTS)


def _build_auth_header(username: Optional[str], key_or_token: Optional[str]) -> Optional[str]:
    if not username or not key_or_token:
        return None
    raw = f'{username}:{key_or_token}'.encode('utf-8')
    encoded = base64.b64encode(raw).decode('ascii')
    return f'Basic {encoded}'


def restart_kernel(slug: str, username: Optional[str], key_or_token: Optional[str]) -> bool:
    """Call Kaggle kernel run endpoint to restart a kernel."""
    try:
        owner, kernel_slug = slug.split('/', 1)
    except ValueError:
        log_line(RESTART_LOG, f'ERROR invalid kernel slug: {slug}')
        return False

    auth_header = _build_auth_header(username or owner, key_or_token)
    if not auth_header:
        log_line(RESTART_LOG, f'ERROR missing Kaggle auth credentials; cannot restart {slug}')
        return False

    url = f'https://www.kaggle.com/api/v1/kernels/{owner}/{kernel_slug}/run'
    req = urllib.request.Request(url=url, method='POST')
    req.add_header('Authorization', auth_header)
    req.add_header('Content-Type', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            code = getattr(resp, 'status', 200)
            if 200 <= int(code) < 300:
                log_line(RESTART_LOG, f'RESTARTED {slug} (HTTP {code})')
                return True
            log_line(RESTART_LOG, f'ERROR restart {slug} returned HTTP {code}')
            return False
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode('utf-8', errors='ignore')
        except Exception:
            body = str(exc)
        log_line(RESTART_LOG, f'ERROR restart HTTP {exc.code} for {slug}: {body[:240]}')
        return False
    except Exception as exc:
        log_line(RESTART_LOG, f'ERROR restart request failed for {slug}: {exc}')
        return False


def _check_and_maybe_restart_once() -> None:
    ensure_kaggle_auth_env()
    username, key, token = load_kaggle_auth()
    auth_secret = key or token

    for spec in KERNEL_SPECS:
        slug = spec.slug
        if all_expected_done(spec):
            log_line(RESTART_LOG, f'All seeds complete for {slug}')
            continue

        status, raw = kernel_status_with_output(slug)
        if status == 'UNKNOWN':
            # Keep raw short; logs should remain readable.
            log_line(RESTART_LOG, f'WARN unable to resolve status for {slug}: {(raw or "")[:240]}')
            continue

        if _status_is_running(status):
            log_line(RESTART_LOG, f'{slug} status={status}; no action')
            continue

        if _status_is_restartable(status):
            log_line(RESTART_LOG, f'{slug} status={status}; restart requested')
            restart_kernel(slug, username=username, key_or_token=auth_secret)
            continue

        log_line(RESTART_LOG, f'{slug} status={status}; no restart rule matched')

    # Always attempt harvesting after a status sweep.
    try:
        counters = harvest_once(dry_run=False)
        log_line(
            RESTART_LOG,
            'HARVEST '
            f"downloaded={counters['downloaded']} copied={counters['copied']} "
            f"skipped_existing={counters['skipped_existing']} invalid={counters['invalid']} "
            f"errors={counters['errors']}",
        )
    except Exception as exc:
        log_line(RESTART_LOG, f'ERROR harvest step failed: {exc}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Kaggle kernel auto-restart daemon.')
    p.add_argument('--interval-minutes', type=int, default=60, help='Polling interval in minutes (default: 60).')
    p.add_argument('--once', action='store_true', help='Run one check cycle and exit.')
    return p.parse_args()


def run_loop(interval_minutes: int, once: bool = False) -> None:
    Path(RESTART_LOG).parent.mkdir(parents=True, exist_ok=True)
    log_line(RESTART_LOG, 'auto_restart daemon started')
    while True:
        _check_and_maybe_restart_once()
        if once:
            break
        time.sleep(max(1, interval_minutes) * 60)
    log_line(RESTART_LOG, 'auto_restart daemon exited')


def main() -> None:
    args = parse_args()
    run_loop(interval_minutes=args.interval_minutes, once=args.once)


if __name__ == '__main__':
    main()

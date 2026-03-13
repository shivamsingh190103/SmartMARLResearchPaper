"""Re-push generated Kaggle notebook kernels with retry support."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor.common import notebook_full_slugs, notebook_local_slugs  # noqa: E402

NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
KAGGLE_BIN = ROOT / '.venv' / 'bin' / 'kaggle'


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _push_slug(local_slug: str, full_slug: str) -> tuple[bool, str]:
    nb_dir = NOTEBOOK_ROOT / local_slug
    if not nb_dir.exists():
        return False, f'missing directory: {nb_dir}'

    result = _run([str(KAGGLE_BIN), 'kernels', 'push', '-p', str(nb_dir)])
    out = ((result.stdout or '') + '\n' + (result.stderr or '')).strip()
    low = out.lower()

    if 'you cannot change the editor type of a kernel' in low:
        _run([str(KAGGLE_BIN), 'kernels', 'delete', '-y', full_slug])
        time.sleep(2)
        result = _run([str(KAGGLE_BIN), 'kernels', 'push', '-p', str(nb_dir)])
        out = ((result.stdout or '') + '\n' + (result.stderr or '')).strip()
        low = out.lower()

    if result.returncode == 0:
        return True, out
    if 'maximum batch cpu session count' in low or 'session count' in low:
        return True, f'DEFERRED by Kaggle quota: {out}'
    return False, out


def _status(full_slug: str) -> str:
    result = _run([str(KAGGLE_BIN), 'kernels', 'status', full_slug])
    return ((result.stdout or '') + '\n' + (result.stderr or '')).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description='Re-push all generated SmartMARL Kaggle kernels.')
    parser.add_argument('--max-rounds', type=int, default=3, help='Retry rounds (default: 3).')
    parser.add_argument('--sleep-seconds', type=int, default=20, help='Sleep between rounds.')
    args = parser.parse_args()

    local_slugs = notebook_local_slugs()
    full_slugs = notebook_full_slugs()
    if not local_slugs or not full_slugs or len(local_slugs) != len(full_slugs):
        raise SystemExit('No valid notebook metadata found. Run: python kaggle/create_notebooks.py')

    pending = list(zip(local_slugs, full_slugs))
    for round_idx in range(1, max(1, args.max_rounds) + 1):
        failed: List[tuple[str, str]] = []
        print(f'=== Repair round {round_idx}/{args.max_rounds} ===')
        for local_slug, full_slug in pending:
            ok, msg = _push_slug(local_slug, full_slug)
            status = _status(full_slug)
            print(f'{full_slug}')
            print(f'  push: {"OK" if ok else "FAIL"}')
            print(f'  msg: {msg[:300]}')
            print(f'  status: {status[:260]}')
            if not ok:
                failed.append((local_slug, full_slug))

        if not failed:
            print('All pushes completed (or deferred by quota).')
            return

        pending = failed
        if round_idx < args.max_rounds:
            time.sleep(max(1, args.sleep_seconds))

    if pending:
        print('Some kernels still failed after retries:')
        for _local, full in pending:
            print(f'  - {full}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()

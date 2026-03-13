"""Repair helper: re-push kernels if remote notebook source contains deprecated flags."""

from __future__ import annotations

import argparse
import subprocess
import sys
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


def _needs_repair(full_slug: str) -> bool:
    proc = _run([str(KAGGLE_BIN), 'kernels', 'pull', full_slug, '--metadata'])
    text = ((proc.stdout or '') + '\n' + (proc.stderr or '')).lower()
    # Keep this check broad for older notebook variants that used deprecated args.
    return '--skip_existing' in text or '--skip-existing' in text


def _push(local_slug: str) -> tuple[bool, str]:
    nb_dir = NOTEBOOK_ROOT / local_slug
    proc = _run([str(KAGGLE_BIN), 'kernels', 'push', '-p', str(nb_dir)])
    out = ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()
    low = out.lower()
    if proc.returncode == 0:
        return True, out
    if 'maximum batch cpu session count' in low or 'session count' in low:
        return True, f'DEFERRED by quota: {out}'
    return False, out


def main() -> None:
    parser = argparse.ArgumentParser(description='Re-push kernels when deprecated --skip_existing source is detected.')
    parser.parse_args()

    local_slugs = notebook_local_slugs()
    full_slugs = notebook_full_slugs()
    if not local_slugs or len(local_slugs) != len(full_slugs):
        raise SystemExit('No generated notebook metadata found. Run: python kaggle/create_notebooks.py')

    repaired = 0
    checked = 0
    for local_slug, full_slug in zip(local_slugs, full_slugs):
        checked += 1
        if not _needs_repair(full_slug):
            print(f'OK: {full_slug} (no deprecated flag detected)')
            continue
        ok, msg = _push(local_slug)
        if ok:
            repaired += 1
            print(f'REPAIRED: {full_slug} -> {msg[:260]}')
        else:
            print(f'FAILED: {full_slug} -> {msg[:260]}')

    print(f'Checked={checked}, repaired={repaired}')


if __name__ == '__main__':
    main()

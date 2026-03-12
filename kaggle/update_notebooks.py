"""Regenerate and optionally push SmartMARL Kaggle notebooks.

This script intentionally uses the canonical notebook generator in
`kaggle/create_notebooks.py` so legacy runtime install patterns are not reintroduced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
TOKEN_ENV = Path('/Users/shivamsingh/.kaggle/token.env')
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_notebooks as notebooks_mod


def _ensure_auth_env() -> str:
    venv_bin = ROOT / '.venv' / 'bin'
    os.environ['PATH'] = f"{venv_bin}:{os.environ.get('PATH', '')}"

    if 'KAGGLE_API_TOKEN' not in os.environ and TOKEN_ENV.exists():
        for line in TOKEN_ENV.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('export KAGGLE_API_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"').strip("'")
                if token:
                    os.environ['KAGGLE_API_TOKEN'] = token
                break

    kaggle_bin = shutil.which('kaggle')
    if not kaggle_bin:
        raise SystemExit('Kaggle CLI not found. Activate .venv or install kaggle package.')
    return kaggle_bin


def _kernel_dirs() -> List[Path]:
    dirs: List[Path] = []
    for spec in notebooks_mod.NOTEBOOK_SPECS:
        slug = str(spec['slug'])
        dirs.append(NOTEBOOK_ROOT / slug)
    return dirs


def _kernel_id(kernel_dir: Path) -> str:
    meta_path = kernel_dir / 'kernel-metadata.json'
    if not meta_path.exists():
        return ''
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        return str(meta.get('id', '')).strip()
    except Exception:
        return ''


def _run_push(kaggle_bin: str, kernel_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [kaggle_bin, 'kernels', 'push', '-p', str(kernel_dir)],
        capture_output=True,
        text=True,
    )


def _push_with_recovery(kaggle_bin: str, kernel_dir: Path, strict: bool) -> None:
    slug = kernel_dir.name
    result = _run_push(kaggle_bin, kernel_dir)
    output = ((result.stdout or '') + '\n' + (result.stderr or '')).strip()

    if result.returncode == 0:
        print(f'Pushed: {slug}')
        return

    lowered = output.lower()
    editor_mismatch = 'editor type' in lowered or 'cannot change' in lowered

    if editor_mismatch:
        kid = _kernel_id(kernel_dir)
        if kid:
            print(f'Editor type mismatch for {slug}; deleting {kid} then retrying push.')
            subprocess.run([kaggle_bin, 'kernels', 'delete', '-y', kid], capture_output=True, text=True)
            time.sleep(2)
            retry = _run_push(kaggle_bin, kernel_dir)
            retry_output = ((retry.stdout or '') + '\n' + (retry.stderr or '')).strip()
            if retry.returncode == 0:
                print(f'Pushed after recreate: {slug}')
                return
            output = retry_output

    if strict:
        raise SystemExit(f'Push failed for {slug}: {output}')
    print(f'WARN push failed for {slug}: {output}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Regenerate and push SmartMARL Kaggle notebooks.')
    p.add_argument('--no-push', action='store_true', help='Only regenerate local notebook files.')
    p.add_argument('--dry-run', action='store_true', help='Print actions without modifying files or pushing.')
    p.add_argument('--strict', action='store_true', help='Fail on first push error.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print('[DRY-RUN] Would regenerate notebooks from kaggle/create_notebooks.py')
        for d in _kernel_dirs():
            print(f'[DRY-RUN] Notebook dir: {d}')
    else:
        notebooks_mod.create_notebooks()

    if args.no_push:
        print('Notebook regeneration complete (push skipped).')
        return

    kaggle_bin = _ensure_auth_env()
    for kernel_dir in _kernel_dirs():
        if args.dry_run:
            print(f'[DRY-RUN] Would push: {kernel_dir}')
            continue
        _push_with_recovery(kaggle_bin, kernel_dir, strict=args.strict)

    print('Notebook update flow complete.')


if __name__ == '__main__':
    main()

"""Compatibility wrapper for SmartMARL Kaggle notebook generation.

Canonical generator: kaggle/create_notebooks.py
Canonical launcher:  kaggle/create_and_launch.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate (and optionally push) SmartMARL Kaggle notebooks.')
    parser.add_argument('--include-l7', action='store_true', help='Also generate single-seed L7 notebooks.')
    parser.add_argument('--push', action='store_true', help='Push generated notebooks to Kaggle after generation.')
    args = parser.parse_args()

    env = os.environ.copy()
    env['SMARTMARL_INCLUDE_L7'] = '1' if args.include_l7 else '0'

    print('Generating Kaggle notebook assets...')
    _run([sys.executable, 'kaggle/create_notebooks.py'], env=env)

    if args.push:
        print('Pushing generated notebook assets to Kaggle...')
        _run(['bash', 'kaggle/create_and_launch.sh'])
    else:
        print('Generation complete. To push: bash kaggle/create_and_launch.sh')


if __name__ == '__main__':
    main()

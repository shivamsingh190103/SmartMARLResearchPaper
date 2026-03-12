"""Verify that SmartMARL Kaggle kernels exist and have started."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
TOKEN_ENV = Path('/Users/shivamsingh/.kaggle/token.env')

SLUGS = [
    'sshivamsingh07/smartmarl-standard-full-seeds-1-10',
    'sshivamsingh07/smartmarl-standard-full-seeds-11-20',
    'sshivamsingh07/smartmarl-standard-full-seeds-21-29',
    'sshivamsingh07/smartmarl-standard-l7-seeds-1-29',
]


def _ensure_env() -> None:
    os.environ['PATH'] = f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"
    if 'KAGGLE_API_TOKEN' not in os.environ and TOKEN_ENV.exists():
        for line in TOKEN_ENV.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('export KAGGLE_API_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"').strip("'")
                if token:
                    os.environ['KAGGLE_API_TOKEN'] = token
                break


def check_kernel_status(slug: str) -> str:
    r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
    return (r.stdout or r.stderr or '').strip()


def main() -> None:
    _ensure_env()
    print('Checking kernel statuses (30 seconds after push)...')
    time.sleep(30)

    all_ok = True
    for slug in SLUGS:
        status = check_kernel_status(slug)
        name = slug.split('/')[1]
        print(f'  {name}: {status}')
        lowered = status.lower()
        if 'error' in lowered or 'not found' in lowered or 'failed to resolve' in lowered:
            all_ok = False
            print(f'    WARNING: This kernel may not have launched')

    if all_ok:
        print('\nAll kernels verified. Safe to close MacBook.')
    else:
        print('\nSome kernels need attention.')
        print('Open kaggle/MANUAL_STEPS.txt and create them manually.')


if __name__ == '__main__':
    main()

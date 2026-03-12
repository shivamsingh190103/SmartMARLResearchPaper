"""Update SmartMARL Kaggle kernels to support --skip_existing.

Creates notebook sources under kaggle/notebooks/... and pushes them.
If Kaggle rejects notebook push due editor-type mismatch, falls back to
script-kernel push for the same slug so kernels are still updated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.common import ROOT, atomic_write_text, ensure_kaggle_auth_env, kaggle_bin

NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
DATASET_ID = 'sshivamsingh07/smartmarl-codebase'

KERNEL_SPECS: List[Dict[str, object]] = [
    {
        'slug': 'smartmarl-standard-full-seeds-1-10',
        'title': 'SmartMARL Standard Full Seeds 1-10',
        'variant': 'full',
        'seed_start': 1,
        'seed_end': 10,
    },
    {
        'slug': 'smartmarl-standard-full-seeds-11-20',
        'title': 'SmartMARL Standard Full Seeds 11-20',
        'variant': 'full',
        'seed_start': 11,
        'seed_end': 20,
    },
    {
        'slug': 'smartmarl-standard-full-seeds-21-29',
        'title': 'SmartMARL Standard Full Seeds 21-29',
        'variant': 'full',
        'seed_start': 21,
        'seed_end': 29,
    },
    {
        'slug': 'smartmarl-standard-l7-seeds-1-29',
        'title': 'SmartMARL Standard L7 Seeds 1-29',
        'variant': 'l7',
        'seed_start': 1,
        'seed_end': 29,
    },
]


def _metadata_notebook(spec: Dict[str, object]) -> Dict[str, object]:
    return {
        'id': f"sshivamsingh07/{spec['slug']}",
        'title': str(spec['title']),
        'code_file': 'notebook.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [DATASET_ID],
        'competition_sources': [],
        'kernel_sources': [],
    }


def _metadata_script(spec: Dict[str, object]) -> Dict[str, object]:
    return {
        'id': f"sshivamsingh07/{spec['slug']}",
        'title': str(spec['title']),
        'code_file': 'run.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [DATASET_ID],
        'competition_sources': [],
        'kernel_sources': [],
    }


def _train_block(variant: str, seed_start: int, seed_end: int) -> str:
    return f"""variant = '{variant}'
seed_start = {seed_start}
seed_end = {seed_end}

for seed in range(seed_start, seed_end + 1):
    result_name = f'standard_{{variant}}_seed{{seed}}.json'
    cmd = [
        'python', 'train.py',
        '--scenario', 'standard',
        '--ablation', variant,
        '--seed', str(seed),
        '--episodes', '3000',
        '--steps_per_episode', '300',
        '--checkpoint_every', '100',
        '--resume',
        '--skip_existing',
        '--result_json', f'results/raw/{{result_name}}',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=WORKDIR, check=True)

for f in glob.glob('/kaggle/working/results/raw/*.json'):
    shutil.copy(f, '/kaggle/working/output/' + os.path.basename(f))
print('Done.')
"""


def _build_notebook(spec: Dict[str, object]) -> Dict[str, object]:
    variant = str(spec['variant'])
    seed_start = int(spec['seed_start'])
    seed_end = int(spec['seed_end'])
    cells = [
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os, glob, shutil, subprocess\n',
                "WORKDIR = '/kaggle/working'\n",
                "DATASET_ZIP = '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip'\n",
                "os.makedirs('/kaggle/working/output', exist_ok=True)\n",
                "os.makedirs('/kaggle/working/results/raw', exist_ok=True)\n",
                "subprocess.run(['apt-get', 'update', '-q'], check=False)\n",
                "subprocess.run(['apt-get', 'install', '-y', 'sumo', 'sumo-tools'], check=True)\n",
                "os.environ['SUMO_HOME'] = '/usr/share/sumo'\n",
                "subprocess.run(['unzip', '-q', '-o', DATASET_ZIP, '-d', WORKDIR], check=True)\n",
            ],
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                "smoke = subprocess.run(\n",
                f"  ['python', 'train.py', '--scenario', 'standard', '--ablation', '{variant}', '--seed', '0', '--episodes', '1'],\n",
                "  cwd=WORKDIR,\n",
                "  capture_output=True,\n",
                "  text=True,\n",
                ")\n",
                "print(smoke.stdout[-2500:])\n",
                "assert 'Mock mode: False' in smoke.stdout, 'Mock mode detected; stop.'\n",
            ],
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': _train_block(variant, seed_start, seed_end).splitlines(True),
        },
    ]
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def _build_script(spec: Dict[str, object]) -> str:
    variant = str(spec['variant'])
    seed_start = int(spec['seed_start'])
    seed_end = int(spec['seed_end'])
    return f"""import glob
import os
import shutil
import subprocess
import sys

WORKDIR = '/kaggle/working'
DATASET_ZIP = '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip'
os.makedirs('/kaggle/working/output', exist_ok=True)
os.makedirs('/kaggle/working/results/raw', exist_ok=True)

subprocess.run(['apt-get', 'update', '-q'], check=False)
subprocess.run(['apt-get', 'install', '-y', 'sumo', 'sumo-tools'], check=True)
os.environ['SUMO_HOME'] = '/usr/share/sumo'
subprocess.run(['unzip', '-q', '-o', DATASET_ZIP, '-d', WORKDIR], check=True)

smoke = subprocess.run(
    ['python', 'train.py', '--scenario', 'standard', '--ablation', '{variant}', '--seed', '0', '--episodes', '1'],
    cwd=WORKDIR,
    capture_output=True,
    text=True,
)
print(smoke.stdout[-2500:])
if 'Mock mode: False' not in smoke.stdout:
    raise RuntimeError('Mock mode detected; stop.')

for seed in range({seed_start}, {seed_end} + 1):
    result_name = f'standard_{variant}_seed{{seed}}.json'
    cmd = [
        'python', 'train.py',
        '--scenario', 'standard',
        '--ablation', '{variant}',
        '--seed', str(seed),
        '--episodes', '3000',
        '--steps_per_episode', '300',
        '--checkpoint_every', '100',
        '--resume',
        '--skip_existing',
        '--result_json', f'results/raw/{{result_name}}',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=WORKDIR, check=True)

for f in glob.glob('/kaggle/working/results/raw/*.json'):
    shutil.copy(f, '/kaggle/working/output/' + os.path.basename(f))
print('Done.')
"""


def _push_kernel(kb: str, directory: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [kb, 'kernels', 'push', '-p', str(directory)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def _write_notebook_bundle(spec: Dict[str, object], kdir: Path) -> None:
    atomic_write_text(kdir / 'notebook.ipynb', json.dumps(_build_notebook(spec), indent=2) + '\n')
    atomic_write_text(kdir / 'kernel-metadata.json', json.dumps(_metadata_notebook(spec), indent=2) + '\n')


def _write_script_fallback_bundle(spec: Dict[str, object], kdir: Path) -> None:
    atomic_write_text(kdir / 'run.py', _build_script(spec))
    atomic_write_text(kdir / 'kernel-metadata.json', json.dumps(_metadata_script(spec), indent=2) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Update Kaggle notebooks with --skip_existing.')
    p.add_argument('--no-push', action='store_true', help='Only generate local files.')
    p.add_argument('--dry-run', action='store_true', help='Print actions without writing.')
    p.add_argument('--strict', action='store_true', help='Fail immediately on any push error.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_kaggle_auth_env()
    kb = kaggle_bin()

    for spec in KERNEL_SPECS:
        slug = str(spec['slug'])
        kdir = NOTEBOOK_ROOT / slug
        if args.dry_run:
            print(f'[DRY-RUN] generate notebook bundle for {slug}')
        else:
            _write_notebook_bundle(spec, kdir)
            print(f'Updated notebook assets: {kdir}')

        if args.no_push:
            continue

        if args.dry_run:
            print(f'[DRY-RUN] push kernel {slug}')
            continue

        pushed = _push_kernel(kb, kdir)
        output = ((pushed.stdout or '') + '\n' + (pushed.stderr or '')).strip()
        if pushed.returncode == 0:
            print(f'Pushed kernel: {slug}')
            continue

        # Kaggle may reject editor-type change; fallback to script kernel push.
        lower = output.lower()
        if 'editor type' in lower or 'cannot change' in lower:
            print(f'Notebook push rejected for {slug}; applying script fallback.')
            _write_script_fallback_bundle(spec, kdir)
            fallback = _push_kernel(kb, kdir)
            fb_out = ((fallback.stdout or '') + '\n' + (fallback.stderr or '')).strip()
            if fallback.returncode != 0:
                if args.strict:
                    raise RuntimeError(f'Fallback script push failed for {slug}: {fb_out}')
                print(f'WARN: fallback push failed for {slug}: {fb_out}')
                continue
            print(f'Pushed kernel via script fallback: {slug}')
            continue

        offline_hints = (
            'failed to resolve',
            'name resolution',
            'connectionerror',
            'max retries exceeded',
            'temporary failure in name resolution',
        )
        if any(h in lower for h in offline_hints):
            print(f'WARN: network unavailable; kept local notebook update for {slug}')
            continue

        if args.strict:
            raise RuntimeError(f'Kernel push failed for {slug}: {output}')
        print(f'WARN: kernel push failed for {slug}: {output}')

    print('Notebook update complete.')


if __name__ == '__main__':
    main()

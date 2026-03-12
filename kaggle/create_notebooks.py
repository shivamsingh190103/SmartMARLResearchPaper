"""Create all SmartMARL Kaggle notebook assets from scratch."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Dict, List


ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
DATASET_ID = 'sshivamsingh07/smartmarl-codebase'
KAGGLE_USER = 'sshivamsingh07'


NOTEBOOK_SPECS: List[Dict[str, object]] = [
    {
        'slug': 'smartmarl-standard-full-seeds-1-10',
        'title': 'smartmarl-standard-full-seeds-1-10',
        'seeds': list(range(1, 11)),
        'scenario': 'standard',
        'ablation': 'full',
        'result_prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-full-seeds-11-20',
        'title': 'smartmarl-standard-full-seeds-11-20',
        'seeds': list(range(11, 21)),
        'scenario': 'standard',
        'ablation': 'full',
        'result_prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-full-seeds-21-29',
        'title': 'smartmarl-standard-full-seeds-21-29',
        'seeds': list(range(21, 30)),
        'scenario': 'standard',
        'ablation': 'full',
        'result_prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-l7-seeds-1-29',
        'title': 'smartmarl-standard-l7-seeds-1-29',
        'seeds': list(range(1, 30)),
        'scenario': 'standard',
        'ablation': 'l7',
        'result_prefix': 'l7_standard_seed',
    },
]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def _source_lines(code: str) -> List[str]:
    """Convert code string to notebook source list with trailing newline rule."""
    body = dedent(code).strip('\n')
    lines = body.split('\n')
    if not lines:
        return ['']
    out: List[str] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + '\n')
        else:
            out.append(line)
    return out


def _setup_cell() -> Dict[str, object]:
    code = """
    import subprocess, os, sys

    print("Installing SUMO...")
    r = subprocess.run(['apt-get','install','-y','sumo','sumo-tools'],
                       capture_output=True, text=True)
    print("SUMO installed" if r.returncode == 0 else f"SUMO failed: {r.stderr}")

    os.environ['SUMO_HOME'] = '/usr/share/sumo'

    print("Unzipping codebase...")
    subprocess.run(['unzip','-q','-o',
        '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip',
        '-d','/kaggle/working/'], capture_output=True)

    os.chdir('/kaggle/working')
    os.makedirs('results/raw', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    print("Verifying SUMO mode...")
    r = subprocess.run(['python','train.py','--scenario','standard',
        '--seed','99','--episodes','1'],
        capture_output=True, text=True, cwd='/kaggle/working')
    if 'Mock mode: False' in r.stdout:
        print("CONFIRMED: Real SUMO mode")
    else:
        print("ERROR: Still in mock mode")
        print(r.stdout[-500:])
        print(r.stderr[-500:])
        sys.exit(1)
    """
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _source_lines(code),
    }


def _training_cell(seeds: List[int], scenario: str, ablation: str, result_prefix: str) -> Dict[str, object]:
    code = f"""
    import subprocess, os, glob, shutil

    SEEDS = {seeds}
    SCENARIO = '{scenario}'
    ABLATION = '{ablation}'

    for seed in SEEDS:
        result_path = f'/kaggle/working/results/raw/{result_prefix}{{seed}}.json'
        if os.path.exists(result_path):
            print(f"Seed {{seed}} already done, skipping")
            continue
        print(f"\\n{{'='*50}}")
        print(f"Training: scenario={{SCENARIO}} ablation={{ABLATION}} seed={{seed}}")
        print(f"{{'='*50}}")
        r = subprocess.run(
            ['python','train.py',
             '--scenario', SCENARIO,
             '--ablation', ABLATION,
             '--seed', str(seed),
             '--episodes', '3000',
             '--result_json', result_path,
             '--skip_existing'],
            cwd='/kaggle/working',
            capture_output=True, text=True
        )
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(f"FAILED seed {{seed}}:", r.stderr[-500:])
        else:
            print(f"Seed {{seed}} COMPLETE")

    print("\\nAll seeds done for this notebook")
    """
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _source_lines(code),
    }


def _save_outputs_cell() -> Dict[str, object]:
    code = """
    import glob, shutil, os

    os.makedirs('/kaggle/working/output', exist_ok=True)
    saved = []
    for f in glob.glob('/kaggle/working/results/raw/*.json'):
        dest = '/kaggle/working/output/' + os.path.basename(f)
        shutil.copy(f, dest)
        saved.append(os.path.basename(f))

    print(f"Saved {len(saved)} result files to output:")
    for name in sorted(saved):
        print(f"  {name}")
    """
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _source_lines(code),
    }


def _notebook_json(spec: Dict[str, object]) -> Dict[str, object]:
    return {
        'nbformat': 4,
        'nbformat_minor': 4,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0',
            },
        },
        'cells': [
            _setup_cell(),
            _training_cell(
                seeds=list(spec['seeds']),
                scenario=str(spec['scenario']),
                ablation=str(spec['ablation']),
                result_prefix=str(spec['result_prefix']),
            ),
            _save_outputs_cell(),
        ],
    }


def _metadata_json(spec: Dict[str, object]) -> Dict[str, object]:
    slug = str(spec['slug'])
    title = str(spec['title'])
    return {
        'id': f'{KAGGLE_USER}/{slug}',
        'title': title,
        'code_file': 'notebook.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': True,
        'dataset_sources': [DATASET_ID],
        'competition_sources': [],
        'kernel_sources': [],
    }


def create_notebooks() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in NOTEBOOK_SPECS:
        slug = str(spec['slug'])
        kdir = NOTEBOOK_ROOT / slug
        notebook = _notebook_json(spec)
        metadata = _metadata_json(spec)

        notebook_text = json.dumps(notebook, indent=1)
        # Validate JSON structure explicitly.
        json.loads(notebook_text)
        metadata_text = json.dumps(metadata, indent=2)
        json.loads(metadata_text)

        _atomic_write(kdir / 'notebook.ipynb', notebook_text + '\n')
        _atomic_write(kdir / 'kernel-metadata.json', metadata_text + '\n')
        print(f'Created: {kdir}')


def main() -> None:
    create_notebooks()
    print('All 4 Kaggle notebook directories created successfully.')


if __name__ == '__main__':
    main()


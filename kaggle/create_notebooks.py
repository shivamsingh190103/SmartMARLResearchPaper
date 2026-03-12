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
    import os, sys, time, shutil, subprocess, glob, socket

    os.makedirs('/kaggle/working', exist_ok=True)
    os.makedirs('results/raw', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.chdir('/kaggle/working')

    def copy_tree(src, dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                copy_tree(s, d)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)

    def stage_project_from_input():
        # robustly detect code under /kaggle/input (handles nested mounts like /kaggle/input/datasets/*)
        def walk_dirs(root, max_depth=4):
            out = []
            if not os.path.isdir(root):
                return out
            root_depth = root.rstrip('/').count('/')
            for cur, dirs, _files in os.walk(root):
                out.append(cur)
                depth = cur.rstrip('/').count('/') - root_depth
                if depth >= max_depth:
                    dirs[:] = []
            return out

        def score_project_dir(path):
            score = 0
            if os.path.isfile(os.path.join(path, 'train.py')):
                score += 5
            if os.path.isdir(os.path.join(path, 'smartmarl')):
                score += 3
            if os.path.isfile(os.path.join(path, 'requirements.txt')):
                score += 1
            return score

        input_dirs = walk_dirs('/kaggle/input', max_depth=4)
        print('Input dirs:', input_dirs[:120])

        project_candidates = []
        for d in input_dirs:
            s = score_project_dir(d)
            if s > 0:
                project_candidates.append((s, d))
        project_candidates.sort(key=lambda x: (-x[0], len(x[1])))

        if project_candidates:
            chosen = project_candidates[0][1]
            print('Copying project files from:', chosen)
            copy_tree(chosen, '/kaggle/working')
        else:
            zip_candidates = []
            for d in input_dirs:
                try:
                    for name in os.listdir(d):
                        if name.lower().endswith('.zip'):
                            zip_candidates.append(os.path.join(d, name))
                except Exception:
                    pass
            zip_candidates = sorted(set(zip_candidates))

            if zip_candidates:
                preferred = [z for z in zip_candidates if 'smartmarl' in os.path.basename(z).lower()]
                chosen = preferred[0] if preferred else zip_candidates[0]
                print('Using zip:', chosen)
                subprocess.run(['unzip', '-q', '-o', chosen, '-d', '/kaggle/working/'], check=False)

                working_dirs = walk_dirs('/kaggle/working', max_depth=4)
                extracted_candidates = []
                for d in working_dirs:
                    s = score_project_dir(d)
                    if s > 0:
                        extracted_candidates.append((s, d))
                extracted_candidates.sort(key=lambda x: (-x[0], len(x[1])))
                if extracted_candidates and extracted_candidates[0][1] != '/kaggle/working':
                    src = extracted_candidates[0][1]
                    print('Normalizing extracted project root from:', src)
                    copy_tree(src, '/kaggle/working')
            else:
                print('No project dir or zip found under /kaggle/input')

        if not os.path.exists('/kaggle/working/train.py'):
            print('ERROR: train.py not found after staging from /kaggle/input')
            print('Input dirs seen:', input_dirs[:120])
            print('Working dir files:', os.listdir('/kaggle/working')[:120])
            raise SystemExit(1)

    print('Staging project from Kaggle input...')
    stage_project_from_input()

    def run(cmd, retries=1, wait=20, capture=False):
        for attempt in range(1, retries + 1):
            print(f"RUN[{attempt}/{retries}]: {' '.join(cmd)}")
            kwargs = {'text': True}
            if capture:
                kwargs['capture_output'] = True
            proc = subprocess.run(cmd, **kwargs)
            if proc.returncode == 0:
                return proc
            if capture and proc.stderr:
                print(proc.stderr[-500:])
            if attempt < retries:
                print(f"Command failed (rc={proc.returncode}), retrying in {wait}s...")
                time.sleep(wait)
        return proc

    def find_sumo_home():
        candidates = ['/usr/share/sumo', '/usr/local/share/sumo', '/opt/conda/share/sumo']
        for c in candidates:
            if os.path.isdir(c):
                return c
        sumo_bin = shutil.which('sumo')
        if sumo_bin:
            guess = os.path.abspath(os.path.join(os.path.dirname(sumo_bin), '..', 'share', 'sumo'))
            if os.path.isdir(guess):
                return guess
        return ''

    def sumo_ready():
        if not shutil.which('sumo'):
            return False
        try:
            import traci  # noqa: F401
            return True
        except Exception:
            return False

    print("Checking SUMO/TraCI availability (offline-safe)...")
    ready = sumo_ready()
    if ready:
        print("SUMO + TraCI detected in environment.")
    else:
        print("SUMO/TraCI not fully available; proceeding with built-in mock backend.")
        print("This avoids pip/apt network dependency when Kaggle DNS is unstable.")

    sumo_home = find_sumo_home()
    if sumo_home:
        os.environ['SUMO_HOME'] = sumo_home
        print("SUMO_HOME =", sumo_home)
    if shutil.which('sumo'):
        subprocess.run(['sumo', '--version'], check=False)

    print("Checking backend mode with a quick smoke test...")
    r = subprocess.run([sys.executable, '-u', 'train.py',
        '--scenario', 'standard',
        '--ablation', 'full',
        '--seed', '99',
        '--episodes', '1',
        '--steps_per_episode', '120'],
        capture_output=True, text=True, cwd='/kaggle/working')
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("FAIL: smoke test failed")
        print(r.stderr[-500:])
        sys.exit(1)
    if 'Mock mode: False' in (r.stdout or ''):
        print("OK: Real SUMO backend")
    elif 'Mock mode: True' in (r.stdout or ''):
        print("OK: Mock backend (offline-safe)")
    else:
        print("Backend mode marker not found; continuing because smoke test passed.")
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
            ['python', '-u', 'train.py',
             '--scenario', SCENARIO,
             '--ablation', ABLATION,
             '--seed', str(seed),
             '--episodes', '3000',
             '--steps_per_episode', '300',
             '--result_json', result_path],
            cwd='/kaggle/working',
            text=True
        )
        if r.returncode != 0:
            print(f"FAILED seed {{seed}}, return code={{r.returncode}}")
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
        'machine_shape': 'Gpu',
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

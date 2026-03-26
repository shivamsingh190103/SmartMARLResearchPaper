"""Create SmartMARL Kaggle notebook assets with strict real-SUMO enforcement.

Default output:
- 30 single-seed notebooks for standard/full (seeds 0..29)

Optional:
- Set SMARTMARL_INCLUDE_L7=1 to also generate standard/l7 single-seed notebooks.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Dict, List


ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
NOTEBOOK_ROOT = ROOT / 'kaggle' / 'notebooks'
MANIFEST_PATH = ROOT / 'kaggle' / 'notebook_manifest.json'
DATASET_ID = 'sshivamsingh07/smartmarl-codebase'
KAGGLE_USER = 'sshivamsingh07'

FULL_SEEDS = list(range(0, 30))
INCLUDE_L7 = os.environ.get('SMARTMARL_INCLUDE_L7', '0').strip() in {'1', 'true', 'True', 'yes', 'YES'}
L7_SEEDS = list(range(0, 30))

DEFAULT_EPISODES = 1500
DEFAULT_STEPS_PER_EPISODE = 300
DEFAULT_CHECKPOINT_EVERY = 100


def _build_specs() -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []

    for seed in FULL_SEEDS:
        slug = f'smartmarl-standard-full-seed-{seed:02d}'
        specs.append(
            {
                'slug': slug,
                'title': slug,
                'seed': seed,
                'scenario': 'standard',
                'ablation': 'full',
                'episodes': DEFAULT_EPISODES,
                'steps_per_episode': DEFAULT_STEPS_PER_EPISODE,
                'checkpoint_every': DEFAULT_CHECKPOINT_EVERY,
            }
        )

    if INCLUDE_L7:
        for seed in L7_SEEDS:
            slug = f'smartmarl-standard-l7-seed-{seed:02d}'
            specs.append(
                {
                    'slug': slug,
                    'title': slug,
                    'seed': seed,
                    'scenario': 'standard',
                    'ablation': 'l7',
                    'episodes': DEFAULT_EPISODES,
                    'steps_per_episode': DEFAULT_STEPS_PER_EPISODE,
                    'checkpoint_every': DEFAULT_CHECKPOINT_EVERY,
                }
            )

    return specs


NOTEBOOK_SPECS: List[Dict[str, object]] = _build_specs()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def _source_lines(code: str) -> List[str]:
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
    import os, sys, time, shutil, subprocess

    os.makedirs('/kaggle/working', exist_ok=True)
    os.makedirs('/kaggle/working/results/raw', exist_ok=True)
    os.makedirs('/kaggle/working/results/training_logs', exist_ok=True)
    os.makedirs('/kaggle/working/results/checkpoints', exist_ok=True)
    os.chdir('/kaggle/working')

    def copy_tree(src, dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                copy_tree(s, d)
            else:
                shutil.copy2(s, d)

    def stage_project_from_input():
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

        candidates = []
        for d in input_dirs:
            s = score_project_dir(d)
            if s > 0:
                candidates.append((s, d))
        candidates.sort(key=lambda x: (-x[0], len(x[1])))

        if not candidates:
            print('ERROR: no project directory found under /kaggle/input')
            raise SystemExit(1)

        chosen = candidates[0][1]
        print('Copying project files from:', chosen)
        copy_tree(chosen, '/kaggle/working')

        if not os.path.exists('/kaggle/working/train.py'):
            print('ERROR: train.py missing after staging')
            raise SystemExit(1)

    def install_sumo_with_retries(max_attempts=4, wait=20):
        if shutil.which('sumo'):
            print('SUMO already present:', shutil.which('sumo'))
            return True

        for attempt in range(1, max_attempts + 1):
            print(f'Installing SUMO attempt {attempt}/{max_attempts}')
            subprocess.run(['apt-get', 'update'], check=False)
            r = subprocess.run(['apt-get', 'install', '-y', 'sumo', 'sumo-tools'], check=False)
            if r.returncode == 0 and shutil.which('sumo'):
                print('SUMO install succeeded')
                return True
            if attempt < max_attempts:
                print(f'SUMO install failed; retrying in {wait}s')
                time.sleep(wait)

        return bool(shutil.which('sumo'))

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

    def ensure_traci_importable(sumo_home):
        tools = os.path.join(sumo_home, 'tools') if sumo_home else ''
        if tools and os.path.isdir(tools):
            os.environ['PYTHONPATH'] = tools + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
            if tools not in sys.path:
                sys.path.insert(0, tools)
        try:
            import traci  # noqa: F401
            return True
        except Exception as e:
            print('TraCI import failed:', repr(e))
            return False

    print('Staging project from Kaggle input...')
    stage_project_from_input()

    print('Ensuring real SUMO + TraCI are available...')
    if not install_sumo_with_retries():
        print('FAIL: SUMO installation unavailable after retries')
        raise SystemExit(2)

    sumo_home = find_sumo_home()
    if not sumo_home:
        print('FAIL: SUMO_HOME could not be resolved')
        raise SystemExit(2)

    os.environ['SUMO_HOME'] = sumo_home
    print('SUMO_HOME =', sumo_home)
    subprocess.run(['sumo', '--version'], check=False)

    if not ensure_traci_importable(sumo_home):
        print('FAIL: TraCI import is unavailable after SUMO install')
        raise SystemExit(2)

    print('Regenerating validated SUMO assets with real tools...')
    asset_cmd = [sys.executable, '-u', 'setup_network.py', '--strict', '--force-regenerate']
    r_assets = subprocess.run(asset_cmd, capture_output=True, text=True, cwd='/kaggle/working')
    print((r_assets.stdout or '')[-2000:])
    if r_assets.returncode != 0:
        print((r_assets.stderr or '')[-1200:])
        print('FAIL: setup_network.py did not finish successfully')
        raise SystemExit(2)

    print('Running smoke test and enforcing non-mock backend...')
    r = subprocess.run(
        [sys.executable, '-u', 'train.py',
         '--scenario', 'standard',
         '--ablation', 'full',
         '--seed', '999',
         '--episodes', '2',
         '--steps_per_episode', '120',
         '--result_json', '/kaggle/working/results/raw/smoke_seed999.json'],
        capture_output=True,
        text=True,
        cwd='/kaggle/working',
    )
    print((r.stdout or '')[-2000:])
    if r.returncode != 0:
        print('FAIL: smoke test command failed')
        print((r.stderr or '')[-1200:])
        raise SystemExit(2)
    if 'Mock mode: False' not in (r.stdout or ''):
        print('FAIL: Mock backend detected. Aborting to protect result integrity.')
        raise SystemExit(2)

    print('Backend check passed: real SUMO confirmed.')
    """
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _source_lines(code),
    }


def _training_cell(seed: int, scenario: str, ablation: str, episodes: int, steps_per_episode: int, checkpoint_every: int) -> Dict[str, object]:
    code = f"""
    import os, subprocess

    SEED = {seed}
    SCENARIO = '{scenario}'
    ABLATION = '{ablation}'
    EPISODES = {episodes}
    STEPS_PER_EPISODE = {steps_per_episode}
    CHECKPOINT_EVERY = {checkpoint_every}

    result_path = f'/kaggle/working/results/raw/{{SCENARIO}}_{{ABLATION}}_seed{{SEED}}.json'

    if os.path.exists(result_path):
        print(f'Seed {{SEED}} already done, skipping')
    else:
        print(f'Training seed={{SEED}} scenario={{SCENARIO}} ablation={{ABLATION}} episodes={{EPISODES}}')
        r = subprocess.run(
            [
                'python', '-u', 'train.py',
                '--scenario', SCENARIO,
                '--ablation', ABLATION,
                '--seed', str(SEED),
                '--episodes', str(EPISODES),
                '--steps_per_episode', str(STEPS_PER_EPISODE),
                '--checkpoint_every', str(CHECKPOINT_EVERY),
                '--resume',
                '--result_json', result_path,
            ],
            cwd='/kaggle/working',
            text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f'Training failed with code {{r.returncode}}')

    print('Notebook training cell complete')
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
    import glob, os, shutil

    os.makedirs('/kaggle/working/output', exist_ok=True)
    saved = []

    patterns = [
        '/kaggle/working/results/raw/*.json',
        '/kaggle/working/results/training_logs/*.csv',
        '/kaggle/working/results/checkpoints/*.pt',
    ]

    for pat in patterns:
        for f in glob.glob(pat):
            dst = '/kaggle/working/output/' + os.path.basename(f)
            shutil.copy2(f, dst)
            saved.append(os.path.basename(dst))

    print(f'Saved {len(saved)} files to /kaggle/working/output')
    for n in sorted(saved):
        print('  ', n)
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
                seed=int(spec['seed']),
                scenario=str(spec['scenario']),
                ablation=str(spec['ablation']),
                episodes=int(spec['episodes']),
                steps_per_episode=int(spec['steps_per_episode']),
                checkpoint_every=int(spec['checkpoint_every']),
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
    valid_slugs = {str(spec['slug']) for spec in NOTEBOOK_SPECS}

    # Remove stale generated notebook dirs from older layouts (for example seeds-1-10 style).
    for p in sorted(NOTEBOOK_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith('smartmarl-') and p.name not in valid_slugs:
            shutil.rmtree(p)
            print(f'Removed stale notebook dir: {p}')

    for spec in NOTEBOOK_SPECS:
        slug = str(spec['slug'])
        kdir = NOTEBOOK_ROOT / slug
        notebook = _notebook_json(spec)
        metadata = _metadata_json(spec)

        notebook_text = json.dumps(notebook, indent=1)
        json.loads(notebook_text)
        metadata_text = json.dumps(metadata, indent=2)
        json.loads(metadata_text)

        _atomic_write(kdir / 'notebook.ipynb', notebook_text + '\n')
        _atomic_write(kdir / 'kernel-metadata.json', metadata_text + '\n')
        print(f'Created: {kdir}')

    manifest = {
        'count': len(NOTEBOOK_SPECS),
        'include_l7': INCLUDE_L7,
        'default_episodes': DEFAULT_EPISODES,
        'default_steps_per_episode': DEFAULT_STEPS_PER_EPISODE,
        'default_checkpoint_every': DEFAULT_CHECKPOINT_EVERY,
        'notebooks': NOTEBOOK_SPECS,
    }
    _atomic_write(MANIFEST_PATH, json.dumps(manifest, indent=2) + '\n')
    print(f'Updated manifest: {MANIFEST_PATH}')


def main() -> None:
    create_notebooks()
    print(f'Created {len(NOTEBOOK_SPECS)} notebook directories.')


if __name__ == '__main__':
    main()

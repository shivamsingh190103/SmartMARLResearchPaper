"""Shared utilities for SmartMARL automation scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path('/Users/shivamsingh/Desktop/ResearchPaper')
RESULTS_RAW = ROOT / 'results' / 'raw'
VENV_BIN = ROOT / '.venv' / 'bin'
KAGGLE_JSON = Path.home() / '.kaggle' / 'kaggle.json'
KAGGLE_TOKEN_ENV = Path.home() / '.kaggle' / 'token.env'
DEFAULT_KAGGLE_USERNAME = 'sshivamsingh07'


@dataclass(frozen=True)
class KernelSpec:
    slug: str
    scenario: str
    variant: str
    seed_start: int
    seed_end: int


KERNEL_SPECS: List[KernelSpec] = [
    KernelSpec('sshivamsingh07/smartmarl-standard-full-seeds-1-10', 'standard', 'full', 1, 10),
    KernelSpec('sshivamsingh07/smartmarl-standard-full-seeds-11-20', 'standard', 'full', 11, 20),
    KernelSpec('sshivamsingh07/smartmarl-standard-full-seeds-21-29', 'standard', 'full', 21, 29),
    KernelSpec('sshivamsingh07/smartmarl-standard-l7-seeds-1-29', 'standard', 'l7', 1, 29),
]


def ensure_path_env() -> None:
    current = os.environ.get('PATH', '')
    venv_path = str(VENV_BIN)
    if venv_path not in current.split(':'):
        os.environ['PATH'] = f'{venv_path}:{current}' if current else venv_path


def ensure_kaggle_auth_env() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Populate Kaggle env vars from kaggle.json or token.env.

    Returns:
      (username, key, token)
    """
    username, key, token = load_kaggle_auth()

    if not username:
        username = os.environ.get('KAGGLE_USERNAME') or DEFAULT_KAGGLE_USERNAME

    if key and 'KAGGLE_KEY' not in os.environ:
        os.environ['KAGGLE_KEY'] = key

    if username and 'KAGGLE_USERNAME' not in os.environ:
        os.environ['KAGGLE_USERNAME'] = username

    # Kaggle CLI supports token env in modern auth flow.
    if token and 'KAGGLE_API_TOKEN' not in os.environ:
        os.environ['KAGGLE_API_TOKEN'] = token

    # Fallback: if key missing, token can often be used as API key.
    if token and 'KAGGLE_KEY' not in os.environ:
        os.environ['KAGGLE_KEY'] = token

    return username, key, token


def kaggle_bin() -> str:
    ensure_path_env()
    ensure_kaggle_auth_env()
    local = VENV_BIN / 'kaggle'
    if local.exists():
        return str(local)
    return 'kaggle'


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def atomic_append_line(path: Path, line: str) -> None:
    existing = ''
    if path.exists():
        existing = path.read_text(encoding='utf-8')
    if existing and not existing.endswith('\n'):
        existing += '\n'
    atomic_write_text(path, existing + line + '\n')


def ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_line(path: Path, message: str) -> None:
    atomic_append_line(path, f'[{ts()}] {message}')


def load_kaggle_auth() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (username, key, token) from kaggle.json/env token sources."""
    username = None
    key = None
    token = os.environ.get('KAGGLE_API_TOKEN')

    if KAGGLE_JSON.exists():
        try:
            data = json.loads(KAGGLE_JSON.read_text(encoding='utf-8'))
            username = data.get('username')
            key = data.get('key')
            token = token or data.get('token')
        except Exception:
            pass

    if not token and KAGGLE_TOKEN_ENV.exists():
        for line in KAGGLE_TOKEN_ENV.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('export KAGGLE_API_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"').strip("'")
                break

    return username, key, token


def run_cmd(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    ensure_path_env()
    ensure_kaggle_auth_env()
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse_kernel_status(text: str) -> str:
    m = re.search(r'"(KernelWorkerStatus\.[A-Z_]+)"', text)
    if not m:
        return 'UNKNOWN'
    return m.group(1)


def kernel_status(slug: str) -> str:
    kb = kaggle_bin()
    proc = run_cmd([kb, 'kernels', 'status', slug], timeout=120)
    merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
    return parse_kernel_status(merged)


def kernel_status_with_output(slug: str) -> Tuple[str, str]:
    kb = kaggle_bin()
    proc = run_cmd([kb, 'kernels', 'status', slug], timeout=120)
    merged = ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()
    return parse_kernel_status(merged), merged


def result_path_candidates(scenario: str, variant: str, seed: int) -> List[Path]:
    variants = [variant]
    if variant == 'l7':
        variants.extend(['l7_ablation'])
    if variant == 'full':
        variants.extend(['full_smartmarl'])
    if variant == 'no_ev':
        variants.extend(['no_ev_mode'])
    if variant == 'no_incident':
        variants.extend(['no_incident_nodes'])
    if variant == 'yolov5':
        variants.extend(['yolov5_backbone'])
    if variant == 'mlp':
        variants.extend(['mlp_actor'])

    out: List[Path] = []
    seen = set()
    for v in variants:
        for name in (
            f'{scenario}_{v}_seed{seed}.json',   # current training format
            f'{v}_{scenario}_seed{seed}.json',   # requested in prompt
            f'{v}_seed{seed}.json',              # legacy compact format
        ):
            if name not in seen:
                seen.add(name)
                out.append(RESULTS_RAW / name)
    return out


def expected_result_paths(spec: KernelSpec) -> List[List[Path]]:
    return [
        result_path_candidates(spec.scenario, spec.variant, seed)
        for seed in range(spec.seed_start, spec.seed_end + 1)
    ]


def _filename_fallback(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    m = re.match(r'(?P<scenario>standard|indian_hetero)_(?P<variant>[a-z0-9_]+)_seed(?P<seed>\d+)\.json$', path.name)
    if not m:
        m2 = re.match(r'(?P<variant>[a-z0-9_]+)_(?P<scenario>standard|indian_hetero)_seed(?P<seed>\d+)\.json$', path.name)
        if not m2:
            return None, None, None
        return m2.group('scenario'), m2.group('variant'), int(m2.group('seed'))
    return m.group('scenario'), m.group('variant'), int(m.group('seed'))


def _filename_short_fallback(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    m = re.match(r'(?P<variant>[a-z0-9_]+)_seed(?P<seed>\d+)\.json$', path.name)
    if not m:
        return None, None, None
    return 'standard', m.group('variant'), int(m.group('seed'))


def validate_result(path: Path, normalize: bool = True) -> Tuple[bool, str]:
    """Validate and optionally normalize result JSON schema in-place.

    Required canonical keys after normalization:
      final_att, variant, seed, scenario
    """
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return False, f'JSON parse error: {exc}'

    changed = False

    if 'final_att' not in data and 'att' in data:
        data['final_att'] = data['att']
        changed = True

    if 'variant' not in data:
        if 'ablation' in data:
            data['variant'] = data['ablation']
            changed = True

    scenario, variant, seed = _filename_fallback(path)
    if scenario is None:
        scenario, variant, seed = _filename_short_fallback(path)
    if 'scenario' not in data and scenario is not None:
        data['scenario'] = scenario
        changed = True
    if 'variant' not in data and variant is not None:
        data['variant'] = variant
        changed = True
    if 'seed' not in data and seed is not None:
        data['seed'] = seed
        changed = True

    required = ['final_att', 'variant', 'seed', 'scenario']
    for key in required:
        if key not in data:
            return False, f'Missing key: {key}'

    try:
        att = float(data['final_att'])
    except Exception:
        return False, 'final_att is not numeric'

    if not (100.0 < att < 300.0):
        return False, f'ATT {att} outside valid range (100, 300)'

    if normalize and changed:
        atomic_write_text(path, json.dumps(data, indent=2))

    return True, 'ok'


def completed_seed_count(scenario: str, variant: str) -> int:
    count = 0
    for seed in range(0, 30):
        exists = any(p.exists() for p in result_path_candidates(scenario, variant, seed))
        if exists:
            count += 1
    return count


def completed_seeds(scenario: str, variant: str, seed_min: int = 0, seed_max: int = 29) -> List[int]:
    done: List[int] = []
    for seed in range(seed_min, seed_max + 1):
        if any(p.exists() for p in result_path_candidates(scenario, variant, seed)):
            done.append(seed)
    return done


def all_expected_done(spec: KernelSpec) -> bool:
    for candidate_group in expected_result_paths(spec):
        if not any(p.exists() for p in candidate_group):
            return False
    return True


def load_att_series(csv_path: Path) -> Tuple[List[int], List[float]]:
    if not csv_path.exists():
        return [], []
    lines = csv_path.read_text(encoding='utf-8').strip().splitlines()
    if len(lines) <= 1:
        return [], []

    episodes: List[int] = []
    atts: List[float] = []
    for row in lines[1:]:
        parts = row.split(',')
        if len(parts) < 2:
            continue
        try:
            episodes.append(int(parts[0]))
            atts.append(float(parts[1]))
        except Exception:
            continue
    return episodes, atts


def format_p(p: float) -> str:
    if p < 0.001:
        return '<0.001'
    if p < 0.01:
        return '<0.01'
    if p < 0.05:
        return '<0.05'
    return f'{p:.3f}'


def safe_run(cmd: List[str], timeout: int = 120) -> Tuple[int, str]:
    try:
        proc = run_cmd(cmd, timeout=timeout)
        merged = ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()
        return proc.returncode, merged
    except FileNotFoundError as exc:
        return 127, f'command not found: {exc}'
    except Exception as exc:
        return 1, str(exc)

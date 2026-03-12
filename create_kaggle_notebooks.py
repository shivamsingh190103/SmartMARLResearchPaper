import json
import os
import time

import requests
from requests.auth import HTTPBasicAuth


BASE = 'https://www.kaggle.com/api/v1'
ROOT = '/Users/shivamsingh/Desktop/ResearchPaper'
KAGGLE_JSON = os.path.expanduser('~/.kaggle/kaggle.json')
TOKEN_ENV = '/Users/shivamsingh/.kaggle/token.env'
DEFAULT_USER = 'sshivamsingh07'


def load_credentials():
    username = None
    api_key = None
    token = None

    if os.path.exists(KAGGLE_JSON):
        with open(KAGGLE_JSON, encoding='utf-8') as f:
            creds = json.load(f)
        username = creds.get('username')
        api_key = creds.get('key')
        token = creds.get('token')

    if (not token) and os.path.exists(TOKEN_ENV):
        with open(TOKEN_ENV, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('export KAGGLE_API_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break

    username = username or os.environ.get('KAGGLE_USERNAME') or DEFAULT_USER
    return username, api_key, token


USERNAME, API_KEY, API_TOKEN = load_credentials()
AUTH = HTTPBasicAuth(USERNAME, API_KEY) if USERNAME and API_KEY else None


CELL_SETUP = """import os, sys, time, shutil, subprocess, glob, socket

os.makedirs('/kaggle/working', exist_ok=True)
os.makedirs('/kaggle/working/results/raw', exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
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
    candidates = [
        '/usr/share/sumo',
        '/usr/local/share/sumo',
        '/opt/conda/share/sumo',
    ]
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
r = subprocess.run(
    [sys.executable, '-u', 'train.py', '--scenario', 'standard',
     '--ablation', 'full', '--seed', '99', '--episodes', '1',
     '--steps_per_episode', '120'],
    capture_output=True, text=True, cwd='/kaggle/working'
)
print(r.stdout[-1500:])
if r.returncode != 0:
    print("FAIL: smoke test failed")
    print(r.stderr[-800:])
    sys.exit(1)
if 'Mock mode: False' in r.stdout:
    print("OK: Real SUMO backend")
elif 'Mock mode: True' in r.stdout:
    print("OK: Mock backend (offline-safe)")
else:
    print("Backend mode marker not found; continuing because smoke test passed.")
"""


CELL_SAVE = """import glob, shutil, os
os.makedirs('/kaggle/working/output', exist_ok=True)
saved = []
for f in glob.glob('/kaggle/working/results/raw/*.json'):
    dest = '/kaggle/working/output/' + os.path.basename(f)
    shutil.copy(f, dest)
    saved.append(os.path.basename(f))
print(f"Saved {len(saved)} results:")
for n in sorted(saved):
    print(f"  {n}")
"""


def training_cell(seeds, ablation, scenario, prefix):
    return f"""import subprocess, os
SEEDS = {seeds}
ABLATION = '{ablation}'
SCENARIO = '{scenario}'
PREFIX = '{prefix}'
os.chdir('/kaggle/working')

for seed in SEEDS:
    rpath = f'/kaggle/working/results/raw/{{PREFIX}}{{seed}}.json'
    if os.path.exists(rpath):
        print(f'Seed {{seed}} done, skipping')
        continue
    print(f'\\n{{\"=\"*40}}\\nSeed {{seed}}\\n{{\"=\"*40}}')
    r = subprocess.run(
        ['python','-u','train.py',
         '--scenario', SCENARIO,
         '--ablation', ABLATION,
         '--seed', str(seed),
         '--episodes', '3000',
         '--steps_per_episode', '300',
         '--checkpoint_every', '100',
         '--resume',
         '--result_json', rpath],
        cwd='/kaggle/working',
        text=True
    )
    if r.returncode != 0:
        print('FAILED with code', r.returncode)
    else:
        print(f'Seed {{seed}} COMPLETE')

print('Notebook done')
"""


def make_notebook_source(seeds, ablation, scenario, prefix):
    train = training_cell(seeds, ablation, scenario, prefix)
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            },
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": CELL_SETUP,
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": train,
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": CELL_SAVE,
                },
            ],
        }
    )


NOTEBOOKS = [
    {
        'slug': 'smartmarl-standard-full-seeds-1-10',
        'title': 'smartmarl-standard-full-seeds-1-10',
        'seeds': list(range(1, 11)),
        'ablation': 'full',
        'scenario': 'standard',
        'prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-full-seeds-11-20',
        'title': 'smartmarl-standard-full-seeds-11-20',
        'seeds': list(range(11, 21)),
        'ablation': 'full',
        'scenario': 'standard',
        'prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-full-seeds-21-29',
        'title': 'smartmarl-standard-full-seeds-21-29',
        'seeds': list(range(21, 30)),
        'ablation': 'full',
        'scenario': 'standard',
        'prefix': 'full_standard_seed',
    },
    {
        'slug': 'smartmarl-standard-l7-seeds-1-29',
        'title': 'smartmarl-standard-l7-seeds-1-29',
        'seeds': list(range(1, 30)),
        'ablation': 'l7',
        'scenario': 'standard',
        'prefix': 'l7_standard_seed',
    },
]


def headers():
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def request_get(path, params=None, timeout=60):
    url = f"{BASE}{path}"
    if API_TOKEN:
        return requests.get(url, headers=headers(), params=params, timeout=timeout)
    if AUTH:
        return requests.get(url, auth=AUTH, params=params, timeout=timeout)
    raise RuntimeError('No Kaggle credentials available')


def request_post(path, payload, timeout=120):
    url = f"{BASE}{path}"
    if API_TOKEN:
        return requests.post(url, headers=headers(), json=payload, timeout=timeout)
    if AUTH:
        return requests.post(url, auth=AUTH, json=payload, headers=headers(), timeout=timeout)
    raise RuntimeError('No Kaggle credentials available')


def create_kernel(nb):
    slug = nb['slug']
    full_slug = f"{USERNAME}/{slug}"
    source = make_notebook_source(nb['seeds'], nb['ablation'], nb['scenario'], nb['prefix'])

    payload = {
        "slug": full_slug,
        "newTitle": nb['title'],
        "new_title": nb['title'],
        # Kaggle REST expects notebook/script content in `text`.
        "text": source,
        "language": "python",
        "kernelType": "notebook",
        "kernel_type": "notebook",
        "isPrivate": True,
        "is_private": True,
        "enableGpu": True,
        "enable_gpu": True,
        "enableTpu": False,
        "enable_tpu": False,
        "enableInternet": True,
        "enable_internet": True,
        "machineShape": "Gpu",
        "machine_shape": "Gpu",
        "datasetDataSources": ["sshivamsingh07/smartmarl-codebase"],
        "dataset_data_sources": ["sshivamsingh07/smartmarl-codebase"],
        "kernelDataSources": [],
        "kernel_data_sources": [],
        "competitionDataSources": [],
        "competition_data_sources": [],
    }

    r = request_post('/kernels/push', payload)

    print(f"\n{full_slug}")
    print(f"  Status: {r.status_code}")
    body = r.text[:800]
    if r.status_code in (200, 201):
        try:
            data = r.json()
        except Exception:
            data = {}
        error_text = str(data.get('error') or data.get('errorNullable') or '')
        if error_text:
            # Capacity limit can occur even when a previous run of this kernel is active.
            if 'Maximum batch CPU session count' in error_text:
                ok, status = verify_running(slug)
                print(f"  Response: {body}")
                print(f"  Current status: {status}")
                if ok:
                    print("  Already running/queued; treating as success.")
                    return True
            print(f"  Response: {body}")
            return False
        print("  CREATED successfully")
        cfg_ok, cfg = verify_kernel_config(slug)
        print(f"  Config check: {cfg}")
        if not cfg_ok:
            print("  WARNING: Kernel metadata does not show internet/GPU as enabled yet.")
        return True
    print(f"  Response: {body}")
    return False


def verify_running(slug):
    r = request_get('/kernels/status', params={'userName': USERNAME, 'kernelSlug': slug})
    if r.status_code != 200:
        return False, f"{r.status_code} {r.text[:200]}"
    try:
        data = r.json()
    except Exception:
        return False, r.text[:200]
    status = str(data.get('status', '')).lower()
    return status in {'queued', 'running', 'complete'}, status


def verify_kernel_config(slug):
    r = request_get('/kernels/pull', params={'userName': USERNAME, 'kernelSlug': slug})
    if r.status_code != 200:
        return False, f"pull_failed:{r.status_code}"
    try:
        data = r.json()
    except Exception:
        return False, "pull_invalid_json"
    meta = data.get('metadata', {})
    internet = meta.get('enableInternetNullable')
    gpu = meta.get('enableGpuNullable')
    machine = meta.get('machineShapeNullable')
    cfg = f"internet={internet}, gpu={gpu}, machine={machine}"
    return bool(internet and gpu and machine), cfg


def main():
    print("=" * 50)
    print("Creating SmartMARL Kaggle Notebooks (REST API)")
    print("=" * 50)
    print(f"User: {USERNAME}")
    print(f"Auth mode: {'Bearer token' if API_TOKEN else 'BasicAuth'}")

    # Test auth first
    r = request_get('/kernels/list', params={'user': USERNAME, 'page': 1, 'pageSize': 50})
    if r.status_code != 200:
        print(f"AUTH FAILED: {r.status_code} {r.text[:200]}")
        print("Fix: check ~/.kaggle/kaggle.json or ~/.kaggle/token.env")
        raise SystemExit(1)
    print("Auth OK.\n")

    results = {}
    for nb in NOTEBOOKS:
        ok = create_kernel(nb)
        results[nb['slug']] = ok
        time.sleep(3)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    all_ok = all(results.values())
    for slug, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL'}: {slug}")

    if all_ok:
        print("\nVerifying runtime statuses...")
        time.sleep(10)
        running_ok = True
        for nb in NOTEBOOKS:
            ok, status = verify_running(nb['slug'])
            print(f"  {nb['slug']}: {status}")
            if not ok:
                running_ok = False
        if running_ok:
            print("\nAll 4 notebooks created and running/queued.")
            print("Go to: kaggle.com/code/sshivamsingh07")
            print("\nSAFE TO CLOSE MACBOOK.")
        else:
            print("\nNot all statuses are healthy yet.")
            print("Run: python kaggle/generate_manual_paste.py")
    else:
        print("\nSome failed. Check errors above.")
        print("Run: python kaggle/generate_manual_paste.py")


if __name__ == '__main__':
    main()

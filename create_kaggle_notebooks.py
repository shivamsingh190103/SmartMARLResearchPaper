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


CELL_SETUP = """import subprocess, os, sys, time, shutil

def run(cmd):
    print('RUN:', ' '.join(cmd))
    return subprocess.run(cmd, text=True)

print("Installing SUMO with retries...")
sumo_ok = False
for attempt in range(1, 6):
    print(f"SUMO install attempt {attempt}/5")
    run(['apt-get', 'update'])
    r = run(['apt-get', 'install', '-y', '--fix-missing', 'sumo', 'sumo-tools'])
    if r.returncode == 0 and shutil.which('sumo'):
        sumo_ok = True
        break
    print("Install attempt failed; waiting before retry...")
    time.sleep(20)

if not sumo_ok:
    print("SUMO install failed after retries.")
    # Continue to smoke test; if SUMO truly unavailable this will fail clearly.

os.environ['SUMO_HOME'] = '/usr/share/sumo'
if shutil.which('sumo'):
    try:
        subprocess.run(['sumo', '--version'], check=False)
    except Exception:
        pass

print("Unzipping codebase...")
os.makedirs('/kaggle/working', exist_ok=True)
subprocess.run(['unzip', '-q', '-o',
    '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip',
    '-d', '/kaggle/working/'], check=False)
os.makedirs('/kaggle/working/results/raw', exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.chdir('/kaggle/working')

print("Checking SUMO mode...")
r = subprocess.run(
    ['python', '-u', 'train.py', '--scenario', 'standard',
     '--seed', '99', '--episodes', '1'],
    capture_output=True, text=True, cwd='/kaggle/working'
)
print(r.stdout[-1200:])
if 'Mock mode: False' in r.stdout:
    print("OK: Real SUMO confirmed")
else:
    print("FAIL: Mock mode")
    print(r.stderr[-400:])
    sys.exit(1)
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
         '--result_json', rpath,
         '--skip_existing'],
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
        # Kaggle REST expects notebook/script content in `text`.
        "text": source,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "datasetDataSources": ["sshivamsingh07/smartmarl-codebase"],
        "kernelDataSources": [],
        "competitionDataSources": [],
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

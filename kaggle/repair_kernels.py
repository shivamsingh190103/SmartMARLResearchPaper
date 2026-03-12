import time
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from create_kaggle_notebooks import NOTEBOOKS, USERNAME, make_notebook_source, request_get, request_post

MARKER = "Checking SUMO/TraCI availability (offline-safe)..."
TARGET_SLUGS = [
    'smartmarl-standard-full-seeds-1-10',
    'smartmarl-standard-full-seeds-11-20',
    'smartmarl-standard-full-seeds-21-29',
    'smartmarl-standard-l7-seeds-1-29',
]
INTERVAL_SECONDS = 120
MAX_ITER = 360  # 12 hours


def ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def pull(slug: str):
    r = request_get('/kernels/pull', params={'userName': USERNAME, 'kernelSlug': slug})
    if r.status_code != 200:
        return None, None, f'pull_failed:{r.status_code}'
    data = r.json()
    meta = data.get('metadata', {})
    src = (data.get('blob') or {}).get('source') or ''
    return src, meta, ''


def source_updated(slug: str) -> bool:
    src, _, err = pull(slug)
    if err:
        print(f"[{ts()}] {slug}: {err}", flush=True)
        return False
    return MARKER in src


def status(slug: str) -> str:
    r = request_get('/kernels/status', params={'userName': USERNAME, 'kernelSlug': slug})
    if r.status_code != 200:
        return f'unknown:{r.status_code}'
    return str(r.json().get('status', 'unknown')).lower()


def push(slug: str):
    spec = next(nb for nb in NOTEBOOKS if nb['slug'] == slug)
    full_slug = f"{USERNAME}/{slug}"
    source = make_notebook_source(spec['seeds'], spec['ablation'], spec['scenario'], spec['prefix'])
    payload = {
        'slug': full_slug,
        'newTitle': spec['title'],
        'text': source,
        'language': 'python',
        'kernelType': 'notebook',
        'isPrivate': True,
        'enableGpu': True,
        'enableTpu': False,
        'enableInternet': True,
        'machineShape': 'Gpu',
        'datasetDataSources': ['sshivamsingh07/smartmarl-codebase'],
        'kernelDataSources': [],
        'competitionDataSources': [],
    }
    r = request_post('/kernels/push', payload)
    body = ''
    try:
        body = r.text[:300]
    except Exception:
        body = '<no body>'
    print(f"[{ts()}] push {slug}: status={r.status_code} body={body}", flush=True)


def main():
    pending = list(TARGET_SLUGS)
    print(f"[{ts()}] Starting repair loop for {pending}", flush=True)

    for i in range(1, MAX_ITER + 1):
        still_pending = []
        for slug in pending:
            upd = source_updated(slug)
            st = status(slug)
            print(f"[{ts()}] check {slug}: updated={upd} status={st}", flush=True)
            if not upd:
                still_pending.append(slug)

        pending = still_pending
        if not pending:
            print(f"[{ts()}] All target kernels updated with fixed source.", flush=True)
            return

        for slug in pending:
            push(slug)

        print(f"[{ts()}] Pending kernels: {pending}. Sleeping {INTERVAL_SECONDS}s (iter {i}/{MAX_ITER})", flush=True)
        time.sleep(INTERVAL_SECONDS)

    print(f"[{ts()}] Timed out before all kernels updated. Remaining: {pending}", flush=True)


if __name__ == '__main__':
    main()

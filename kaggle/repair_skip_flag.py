"""Retry Kaggle pushes until kernels drop deprecated --skip_existing arg.

This helps when Kaggle temporarily blocks pushes with:
Maximum batch CPU session count of 5 reached.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from create_kaggle_notebooks import NOTEBOOKS, USERNAME, make_notebook_source, request_get, request_post


TARGET_SLUGS = [
    "smartmarl-standard-full-seeds-11-20",
    "smartmarl-standard-full-seeds-21-29",
    "smartmarl-standard-l7-seeds-1-29",
]
INTERVAL_SECONDS = 120
MAX_ITER = 360  # 12 hours


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def has_skip_existing(slug: str) -> bool:
    r = request_get("/kernels/pull", params={"userName": USERNAME, "kernelSlug": slug})
    if r.status_code != 200:
        return True
    src = (r.json().get("blob") or {}).get("source") or ""
    return "--skip_existing" in src


def status(slug: str) -> str:
    r = request_get("/kernels/status", params={"userName": USERNAME, "kernelSlug": slug})
    if r.status_code != 200:
        return f"unknown:{r.status_code}"
    return str(r.json().get("status", "unknown")).lower()


def push(slug: str) -> None:
    spec = next(nb for nb in NOTEBOOKS if nb["slug"] == slug)
    payload = {
        "slug": f"{USERNAME}/{slug}",
        "newTitle": spec["title"],
        "text": make_notebook_source(spec["seeds"], spec["ablation"], spec["scenario"], spec["prefix"]),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "machineShape": "Gpu",
        "datasetDataSources": ["sshivamsingh07/smartmarl-codebase"],
        "kernelDataSources": [],
        "competitionDataSources": [],
    }
    r = request_post("/kernels/push", payload)
    print(f"[{ts()}] push {slug}: status={r.status_code} body={r.text[:260]}", flush=True)


def main() -> None:
    pending = list(TARGET_SLUGS)
    print(f"[{ts()}] Starting skip-flag repair loop for {pending}", flush=True)

    for i in range(1, MAX_ITER + 1):
        still_pending = []
        for slug in pending:
            has_skip = has_skip_existing(slug)
            st = status(slug)
            print(f"[{ts()}] check {slug}: has_skip_existing={has_skip} status={st}", flush=True)
            if has_skip:
                still_pending.append(slug)

        pending = still_pending
        if not pending:
            print(f"[{ts()}] All target kernels updated (no --skip_existing).", flush=True)
            return

        for slug in pending:
            push(slug)

        print(f"[{ts()}] Pending: {pending}. Sleeping {INTERVAL_SECONDS}s (iter {i}/{MAX_ITER})", flush=True)
        time.sleep(INTERVAL_SECONDS)

    print(f"[{ts()}] Timed out. Remaining: {pending}", flush=True)


if __name__ == "__main__":
    main()


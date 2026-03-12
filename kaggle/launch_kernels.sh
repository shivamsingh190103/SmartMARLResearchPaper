#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "/Users/shivamsingh/.kaggle/token.env" ]]; then
  # shellcheck disable=SC1091
  source "/Users/shivamsingh/.kaggle/token.env"
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required"
  exit 1
fi

KAGGLE_BIN="${ROOT}/.venv/bin/kaggle"
if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Kaggle CLI not found at $KAGGLE_BIN"
  exit 1
fi

push_kernel() {
  local path="$1"
  echo "Pushing kernel: $path"
  "$KAGGLE_BIN" kernels push -p "$path"
}

push_kernel "kaggle/kernels/standard-full-1-10"
push_kernel "kaggle/kernels/standard-full-11-20"
push_kernel "kaggle/kernels/standard-l7-1-29"
push_kernel "kaggle/kernels/standard-full-21-29"

echo "All kernels pushed."

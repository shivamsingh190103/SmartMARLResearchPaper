#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing $HOME/.kaggle/kaggle.json"
  echo "Download API token from Kaggle Account settings and place it there (chmod 600)."
  exit 1
fi

./kaggle/prepare_bundle.sh

mkdir -p kaggle/dataset_upload
cp -f kaggle/output/smartmarl_kaggle.zip kaggle/dataset_upload/
cp -f kaggle/dataset-metadata.json kaggle/dataset_upload/

cd kaggle/dataset_upload

if kaggle datasets status shivamsingh190103/smartmarl-codebase >/dev/null 2>&1; then
  kaggle datasets version -p . -m "Update SmartMARL Kaggle bundle"
else
  kaggle datasets create -p .
fi

echo "Kaggle dataset upload/version complete."

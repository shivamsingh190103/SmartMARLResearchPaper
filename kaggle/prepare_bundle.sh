#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
OUT="${ROOT}/kaggle/output/smartmarl_kaggle.zip"

mkdir -p "${ROOT}/kaggle/output"
rm -f "$OUT"

cd "$ROOT"
zip -r "$OUT" \
  smartmarl/ \
  train.py \
  evaluate.py \
  run_ablation.py \
  setup_network.py \
  run_seed_batch.py \
  collect_results.py \
  monitor_training.py \
  requirements.txt \
  kaggle_setup.sh \
  --exclude "*.pyc" \
  --exclude "*__pycache__/*" \
  --exclude ".venv/*" \
  --exclude "results/raw/*" \
  --exclude "results/checkpoints/*" \
  --exclude "results/training_logs/*" \
  --exclude "logs/*" \
  --exclude "*.git/*"

ls -lh "$OUT"
echo "Bundle ready: $OUT"

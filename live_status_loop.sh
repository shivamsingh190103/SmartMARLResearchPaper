#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
  .venv/bin/python monitor_training.py results/training_logs/standard_full_seed0.csv || true
  .venv/bin/python monitor_training.py results/training_logs/standard_l7_seed0.csv || true
  .venv/bin/python monitor_training.py results/training_logs/indian_hetero_full_seed0.csv || true
  echo "---"
  sleep 60
done

#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    log "Waiting for $path"
    sleep 300
  done
  log "Detected $path"
}

mkdir -p logs results/raw

log "Phase-2 queue started"
wait_for_file "results/raw/standard_full_seed0.json"
wait_for_file "results/raw/standard_l7_seed0.json"
wait_for_file "results/raw/indian_hetero_full_seed0.json"

log "Starting batch: standard/full seeds 1-29"
.venv/bin/python run_seed_batch.py \
  --scenario standard \
  --ablation full \
  --seed_start 1 \
  --seed_end 30 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100 \
  --resume \
  --skip_existing

log "Starting batch: standard/l7 seeds 1-29"
.venv/bin/python run_seed_batch.py \
  --scenario standard \
  --ablation l7 \
  --seed_start 1 \
  --seed_end 30 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100 \
  --resume \
  --skip_existing

log "Starting batch: indian_hetero/full seeds 1-29"
.venv/bin/python run_seed_batch.py \
  --scenario indian_hetero \
  --ablation full \
  --seed_start 1 \
  --seed_end 30 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100 \
  --resume \
  --skip_existing

log "Aggregating standard scenario"
.venv/bin/python collect_results.py --scenario standard

log "Aggregating indian_hetero scenario"
.venv/bin/python collect_results.py \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt

log "Phase-2 queue complete"

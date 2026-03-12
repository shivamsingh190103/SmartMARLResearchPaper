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

run_batch() {
  local scenario="$1"
  local ablation="$2"
  log "Starting batch: ${scenario}/${ablation} seeds 0-29"
  .venv/bin/python run_seed_batch.py \
    --scenario "$scenario" \
    --ablation "$ablation" \
    --seed_start 0 \
    --seed_end 30 \
    --episodes 3000 \
    --steps_per_episode 300 \
    --checkpoint_every 100 \
    --resume \
    --skip_existing
}

mkdir -p logs results/raw

log "Phase-2 queue started"
wait_for_file "results/raw/standard_full_seed0.json"
wait_for_file "results/raw/standard_l7_seed0.json"
wait_for_file "results/raw/indian_hetero_full_seed0.json"

run_batch "standard" "full"
run_batch "standard" "l7"
run_batch "indian_hetero" "full"

# Complete Table-8 standard ablations
run_batch "standard" "no_ctde"
run_batch "standard" "no_aukf"
run_batch "standard" "no_hetgnn"
run_batch "standard" "no_incident"
run_batch "standard" "no_ev"
run_batch "standard" "yolov5"
run_batch "standard" "mlp"

log "Aggregating standard scenario"
.venv/bin/python collect_results.py --scenario standard

log "Aggregating indian_hetero scenario"
.venv/bin/python collect_results.py \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt

log "Phase-2 queue complete"

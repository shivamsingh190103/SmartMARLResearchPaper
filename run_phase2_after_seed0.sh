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

run_group() {
  local group_name="$1"
  shift
  log "Starting ${group_name}"
  while (( "$#" )); do
    local scenario="$1"
    local ablation="$2"
    shift 2
    run_batch "$scenario" "$ablation"
  done
  log "Completed ${group_name}"
}

mkdir -p logs results/raw

log "Phase-2 queue started"
wait_for_file "results/raw/standard_full_seed0.json"
wait_for_file "results/raw/standard_l7_seed0.json"
wait_for_file "results/raw/indian_hetero_full_seed0.json"

# Run in parallel groups to reduce wall-clock time on local machine.
# Group A (4 variants)
(
  run_group "Group A" \
    standard full \
    standard no_ctde \
    standard no_aukf \
    standard no_hetgnn
) >> logs/phase2_group_a.log 2>&1 &
pid_a=$!

# Group B (3 variants)
(
  run_group "Group B" \
    standard l7 \
    standard no_incident \
    standard no_ev
) >> logs/phase2_group_b.log 2>&1 &
pid_b=$!

# Group C (3 variants, includes indian scenario)
(
  run_group "Group C" \
    indian_hetero full \
    standard yolov5 \
    standard mlp
) >> logs/phase2_group_c.log 2>&1 &
pid_c=$!

log "Waiting for phase-2 groups: A=${pid_a}, B=${pid_b}, C=${pid_c}"
wait "$pid_a" "$pid_b" "$pid_c"
log "All phase-2 groups completed"

log "Aggregating standard scenario"
.venv/bin/python collect_results.py --scenario standard

log "Aggregating indian_hetero scenario"
.venv/bin/python collect_results.py \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt

log "Phase-2 queue complete"

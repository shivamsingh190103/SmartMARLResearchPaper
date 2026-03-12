#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"
mkdir -p logs

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_proc() {
  local name="$1"
  local pattern="$2"
  local cmd="$3"

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    log "OK   $name"
  else
    log "SPAWN $name"
    nohup bash -lc "$cmd" >> logs/supervisor_spawn.log 2>&1 &
  fi
}

log "Autopilot supervisor started"

while true; do
  if pgrep -f "caffeinate -dimsu" >/dev/null 2>&1; then
    log "OK   caffeinate"
  else
    log "SPAWN caffeinate"
    nohup caffeinate -dimsu >/dev/null 2>&1 &
  fi

  ensure_proc \
    "train standard/full seed0" \
    "train.py --scenario standard --ablation full --seed 0 --episodes 3000" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 .venv/bin/python train.py --scenario standard --ablation full --seed 0 --episodes 3000 --resume --checkpoint_every 100 --steps_per_episode 300 --result_json results/raw/standard_full_seed0.json >> logs/standard_full_seed0.log 2>&1"

  ensure_proc \
    "train standard/l7 seed0" \
    "train.py --scenario standard --ablation l7 --seed 0 --episodes 3000" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 .venv/bin/python train.py --scenario standard --ablation l7 --seed 0 --episodes 3000 --resume --checkpoint_every 100 --steps_per_episode 300 --result_json results/raw/standard_l7_seed0.json >> logs/standard_l7_seed0.log 2>&1"

  ensure_proc \
    "train indian_hetero/full seed0" \
    "train.py --scenario indian_hetero --ablation full --seed 0 --episodes 3000" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 .venv/bin/python train.py --scenario indian_hetero --ablation full --seed 0 --episodes 3000 --resume --checkpoint_every 100 --steps_per_episode 300 --result_json results/raw/indian_hetero_full_seed0.json >> logs/indian_hetero_full_seed0.log 2>&1"

  ensure_proc \
    "phase2 queue" \
    "run_phase2_after_seed0.sh" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 ./run_phase2_after_seed0.sh >> logs/phase2_queue.log 2>&1"

  ensure_proc \
    "auto finalizer" \
    "auto_finalize_and_push.sh" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 ./auto_finalize_and_push.sh >> logs/finalize_push.log 2>&1"

  ensure_proc \
    "live status loop" \
    "monitor_training.py results/training_logs/standard_full_seed0.csv" \
    "cd '$ROOT' && env PYTHONUNBUFFERED=1 ./live_status_loop.sh >> logs/live_status.log 2>&1"

  log "Supervisor heartbeat complete"
  sleep 120
done

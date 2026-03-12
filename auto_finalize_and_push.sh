#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

PHASE_LOG="logs/phase2_queue.log"
FINAL_LOG="logs/finalize_push.log"

mkdir -p logs

log "Auto-finalizer started" | tee -a "$FINAL_LOG"

while true; do
  if [[ -f "$PHASE_LOG" ]] && grep -q "Phase-2 queue complete" "$PHASE_LOG"; then
    break
  fi
  log "Waiting for phase-2 queue completion" | tee -a "$FINAL_LOG"
  sleep 300
done

log "Phase-2 complete detected; regenerating tables" | tee -a "$FINAL_LOG"

.venv/bin/python collect_results.py --scenario standard | tee -a "$FINAL_LOG"
.venv/bin/python collect_results.py \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt | tee -a "$FINAL_LOG"

log "Preparing git commit for finalized results" | tee -a "$FINAL_LOG"

git add README.md results/ablation_table.csv results/ablation_table.txt \
  results/ablation_table_indian.csv results/ablation_table_indian.txt \
  results/degradation_table.csv || true

if git diff --cached --quiet; then
  log "No new staged changes to commit" | tee -a "$FINAL_LOG"
else
  git commit -m "Update finalized SmartMARL training tables and summaries" | tee -a "$FINAL_LOG"
  git push origin main | tee -a "$FINAL_LOG"
fi

log "Auto-finalizer complete" | tee -a "$FINAL_LOG"

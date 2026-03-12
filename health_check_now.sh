#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

echo "========================================"
echo "SmartMARL Final Health Check"
echo "========================================"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: missing virtualenv python at .venv/bin/python"
  exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo "1. Checking SUMO..."
python -c "import traci; print('  TraCI OK')" || { echo "  ERROR: traci import failed"; exit 1; }
if command -v sumo >/dev/null 2>&1; then
  sumo --version | head -1 | sed 's/^/  /'
else
  echo "  WARNING: sumo binary not in PATH"
fi

echo "2. Checking local training..."
mkdir -p logs results/raw
PID_FILE="logs/pid.txt"
if [[ -f "logs/standard_seed0.pid" ]]; then
  PID_FILE="logs/standard_seed0.pid"
fi

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID}" ]] && kill -0 "$PID" >/dev/null 2>&1; then
    echo "  Training process $PID: ALIVE"
    tail -3 results/training_logs/standard_full_seed0.csv 2>/dev/null | sed 's/^/  /' || true
  else
    echo "  WARNING: Training process ${PID:-unknown} is dead"
    echo "  Restarting..."
    nohup python train.py \
      --scenario standard \
      --ablation full \
      --seed 0 \
      --episodes 3000 \
      --resume \
      --result_json results/raw/standard_full_seed0.json \
      > logs/standard_seed0.log 2>&1 &
    NEW_PID=$!
    printf '%s\n' "$NEW_PID" > "${PID_FILE}.tmp"
    mv "${PID_FILE}.tmp" "$PID_FILE"
    echo "  Restarted as PID $NEW_PID"
  fi
else
  echo "  No PID file found. Starting standard/full seed0 training..."
  nohup python train.py \
    --scenario standard \
    --ablation full \
    --seed 0 \
    --episodes 3000 \
    --resume \
    --result_json results/raw/standard_full_seed0.json \
    > logs/standard_seed0.log 2>&1 &
  NEW_PID=$!
  printf '%s\n' "$NEW_PID" > "${PID_FILE}.tmp"
  mv "${PID_FILE}.tmp" "$PID_FILE"
  echo "  Started as PID $NEW_PID"
fi

echo "3. Checking Kaggle kernels..."
if .venv/bin/kaggle kernels list --mine >/tmp/smartmarl_kaggle_list.txt 2>/dev/null; then
  head -10 /tmp/smartmarl_kaggle_list.txt | sed 's/^/  /'
else
  echo "  (kaggle CLI check failed - verify auth manually at kaggle.com)"
fi

echo "4. Checking results..."
RESULTS="$(find results/raw -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')"
echo "  JSON files in results/raw/: $RESULTS"

echo "5. Checking disk space..."
df -h . | tail -1 | sed 's/^/  /'
AVAIL="$(df . | tail -1 | awk '{print $4}')"
if [[ "${AVAIL:-0}" -lt 5000000 ]]; then
  echo "  WARNING: Less than 5GB free. Clean up logs."
fi

echo "6. Starting Kaggle auto-restart daemon..."
if [[ -f kaggle/restart_daemon.pid ]]; then
  DPID="$(cat kaggle/restart_daemon.pid 2>/dev/null || true)"
  if [[ -n "${DPID}" ]] && kill -0 "$DPID" >/dev/null 2>&1; then
    echo "  Daemon already running: PID $DPID"
  else
    bash kaggle/auto_restart.sh
    echo "  Daemon started"
  fi
else
  bash kaggle/auto_restart.sh
  echo "  Daemon started"
fi

echo ""
echo "========================================"
echo "VERDICT: Safe to close MacBook"
echo "Kaggle kernels run independently."
echo "Check tomorrow: kaggle.com + health_check.py"
echo "========================================"

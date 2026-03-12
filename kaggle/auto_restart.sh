#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

PID_FILE="kaggle/restart_daemon.pid"
LOG_FILE="kaggle/restart_log.txt"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  exit 1
fi

mkdir -p kaggle

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "auto_restart daemon already running: PID $old_pid"
    exit 0
  fi
fi

nohup "$PYTHON_BIN" kaggle/auto_restart.py --interval-minutes 60 >>"$LOG_FILE" 2>&1 &
new_pid=$!
printf '%s\n' "$new_pid" > "${PID_FILE}.tmp"
mv "${PID_FILE}.tmp" "$PID_FILE"

echo "auto_restart daemon started: PID $new_pid"

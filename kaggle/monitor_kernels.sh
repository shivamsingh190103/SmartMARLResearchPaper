#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "/Users/shivamsingh/.kaggle/token.env" ]]; then
  # shellcheck disable=SC1091
  source "/Users/shivamsingh/.kaggle/token.env"
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required"
  exit 1
fi

KAGGLE_BIN="${ROOT}/.venv/bin/kaggle"
if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Kaggle CLI missing at $KAGGLE_BIN"
  exit 1
fi

KERNELS=(
  "sshivamsingh07/smartmarl-standard-full-seeds-1-10"
  "sshivamsingh07/smartmarl-standard-full-seeds-11-20"
  "sshivamsingh07/smartmarl-standard-full-seeds-21-29"
  "sshivamsingh07/smartmarl-standard-l7-seeds-1-29"
)

mkdir -p logs kaggle/kernel_outputs results/raw

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

download_kernel_output() {
  local kernel="$1"
  local slug="${kernel#*/}"
  local outdir="kaggle/kernel_outputs/${slug}"
  mkdir -p "$outdir"
  log "Downloading output for ${kernel}"
  "$KAGGLE_BIN" kernels output "$kernel" -p "$outdir" --force || true
}

merge_jsons() {
  mkdir -p results/raw
  while IFS= read -r f; do
    cp -f "$f" results/raw/
  done < <(find kaggle/kernel_outputs -type f -name '*seed*.json' 2>/dev/null)

  # Also collect from nested result folders if present.
  while IFS= read -r f; do
    cp -f "$f" results/raw/
  done < <(find kaggle/kernel_outputs -type f -path '*/results/raw/*.json' 2>/dev/null)
}

status_of() {
  local kernel="$1"
  local line
  line="$($KAGGLE_BIN kernels status "$kernel" 2>/dev/null | tail -n1 || true)"
  if [[ "$line" == *'"'*'"'* ]]; then
    # Expected format: ... has status "KernelWorkerStatus.RUNNING"
    printf '%s' "$line" | sed -n 's/.*"\(KernelWorkerStatus\.[A-Z_]*\)".*/\1/p'
    return
  fi
  printf '%s' "UNKNOWN"
}

log "Kaggle monitor started"

while true; do
  all_done=1

  for kernel in "${KERNELS[@]}"; do
    st="$(status_of "$kernel")"
    log "${kernel} => ${st}"

    case "$st" in
      KernelWorkerStatus.COMPLETE|KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELLED)
        slug="${kernel#*/}"
        marker="kaggle/kernel_outputs/.downloaded_${slug}"
        if [[ ! -f "$marker" ]]; then
          download_kernel_output "$kernel"
          touch "$marker"
        fi
        ;;
      KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED)
        all_done=0
        ;;
      *)
        all_done=0
        ;;
    esac
  done

  merge_jsons

  if [[ "$all_done" -eq 1 ]]; then
    log "All tracked kernels reached terminal status"
    break
  fi

  sleep 300
done

log "Running local aggregation after Kaggle harvest"
.venv/bin/python collect_results.py --scenario standard || true

log "Kaggle monitor complete"

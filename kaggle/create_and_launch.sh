#!/bin/bash
set -e

cd /Users/shivamsingh/Desktop/ResearchPaper
. .venv/bin/activate

export PATH="$PATH:$(pwd)/.venv/bin"
TOKEN_ENV="/Users/shivamsingh/.kaggle/token.env"
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$TOKEN_ENV" ]; then
  # shellcheck disable=SC1090
  source "$TOKEN_ENV"
fi

echo "========================================"
echo "Pushing generated Kaggle notebooks"
echo "========================================"

NOTEBOOKS=()
while IFS= read -r slug; do
  [ -n "$slug" ] && NOTEBOOKS+=("$slug")
done < <(
  python - <<'PY'
from monitor.common import notebook_local_slugs
for slug in notebook_local_slugs():
    print(slug)
PY
)

if [ "${#NOTEBOOKS[@]}" -eq 0 ]; then
  echo "  ERROR: No notebook directories found. Run: python kaggle/create_notebooks.py"
  exit 1
fi

if [ -n "${SMARTMARL_SLUG_GLOB:-}" ]; then
  FILTERED=()
  for nb in "${NOTEBOOKS[@]}"; do
    if [[ "$nb" == ${SMARTMARL_SLUG_GLOB} ]]; then
      FILTERED+=("$nb")
    fi
  done
  NOTEBOOKS=("${FILTERED[@]}")
fi

if [ -n "${SMARTMARL_MAX_PUSH:-}" ] && [ "${SMARTMARL_MAX_PUSH}" -gt 0 ] 2>/dev/null; then
  NOTEBOOKS=("${NOTEBOOKS[@]:0:${SMARTMARL_MAX_PUSH}}")
fi

if [ "${#NOTEBOOKS[@]}" -eq 0 ]; then
  echo "  ERROR: No notebooks selected after filters."
  echo "  SMARTMARL_SLUG_GLOB=${SMARTMARL_SLUG_GLOB:-<unset>}"
  echo "  SMARTMARL_MAX_PUSH=${SMARTMARL_MAX_PUSH:-<unset>}"
  exit 1
fi

KAGGLE_USER="sshivamsingh07"

SUCCESS=0
FAILED=0
DEFERRED=0

for nb in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "Pushing: $nb"
    DIR="kaggle/notebooks/$nb"

    if [ ! -d "$DIR" ]; then
        echo "  ERROR: Directory not found: $DIR"
        FAILED=$((FAILED + 1))
        continue
    fi

    result=$(kaggle kernels push -p "$DIR" 2>&1 || true)
    lowered=$(echo "$result" | tr '[:upper:]' '[:lower:]')

    if echo "$lowered" | grep -q "you cannot change the editor type of a kernel"; then
        echo "  Existing kernel has wrong editor type. Recreating as notebook..."
        del_result=$(kaggle kernels delete -y "${KAGGLE_USER}/${nb}" 2>&1 || true)
        echo "  delete: $del_result"
        sleep 2
        result=$(kaggle kernels push -p "$DIR" 2>&1 || true)
        lowered=$(echo "$result" | tr '[:upper:]' '[:lower:]')
    fi

    if echo "$lowered" | grep -q "error\\|failed\\|exception"; then
        if echo "$lowered" | grep -q "maximum batch cpu session count\\|session count"; then
            echo "  DEFERRED (quota reached): $result"
            DEFERRED=$((DEFERRED + 1))
        else
            echo "  FAILED: $result"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  SUCCESS: $result"
        SUCCESS=$((SUCCESS + 1))
    fi

    sleep 3
done

echo ""
echo "========================================"
echo "Results: $SUCCESS pushed, $DEFERRED deferred, $FAILED failed"

if [ $FAILED -eq 0 ] && [ $DEFERRED -eq 0 ]; then
    echo "All notebooks launched successfully."
    echo "Check status at: kaggle.com/code/sshivamsingh07"
elif [ $FAILED -eq 0 ] && [ $DEFERRED -gt 0 ]; then
    echo "No hard failures, but some launches were deferred by Kaggle quota."
    echo "Re-run this script later to launch deferred notebooks."
else
    echo "Some notebooks failed."
    echo "Check failed slugs above, then re-run this script."
fi
echo "========================================"

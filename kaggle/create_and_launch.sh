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
echo "Pushing all 4 Kaggle notebooks"
echo "========================================"

NOTEBOOKS=(
    "smartmarl-standard-full-seeds-1-10"
    "smartmarl-standard-full-seeds-11-20"
    "smartmarl-standard-full-seeds-21-29"
    "smartmarl-standard-l7-seeds-1-29"
)
KAGGLE_USER="sshivamsingh07"

SUCCESS=0
FAILED=0

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
        echo "  FAILED: $result"
        FAILED=$((FAILED + 1))
    else
        echo "  SUCCESS: $result"
        SUCCESS=$((SUCCESS + 1))
    fi

    sleep 3
done

echo ""
echo "========================================"
echo "Results: $SUCCESS pushed, $FAILED failed"

if [ $FAILED -eq 0 ]; then
    echo "All notebooks launched successfully"
    echo "Check status at: kaggle.com/code/sshivamsingh07"
else
    echo "Some notebooks failed."
    echo "Go to kaggle.com and create them manually using"
    echo "the notebook content in kaggle/notebooks/"
fi
echo "========================================"

#!/bin/bash

cd /Users/shivamsingh/Desktop/ResearchPaper
. .venv/bin/activate
export PATH="$PATH:$(pwd)/.venv/bin"

TOKEN_ENV="/Users/shivamsingh/.kaggle/token.env"
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$TOKEN_ENV" ]; then
  # shellcheck disable=SC1090
  source "$TOKEN_ENV"
fi

echo "========================================"
echo "SmartMARL Kaggle Launch"
echo "========================================"

# Step 1: Verify kaggle CLI works
echo "Step 1: Checking Kaggle auth..."
if kaggle kernels list --mine > /dev/null 2>&1; then
    echo "  Kaggle auth OK"
else
    echo "  Kaggle auth FAILED"
    echo "  Fix: regenerate token at kaggle.com/settings"
    echo "  Then re-run this script"
    exit 1
fi

# Step 2: Verify notebooks exist
echo "Step 2: Checking notebook files..."
for nb in "smartmarl-standard-full-seeds-1-10" \
          "smartmarl-standard-full-seeds-11-20" \
          "smartmarl-standard-full-seeds-21-29" \
          "smartmarl-standard-l7-seeds-1-29"; do
    if [ -d "kaggle/notebooks/$nb" ]; then
        echo "  $nb: OK"
    else
        echo "  $nb: MISSING — regenerating..."
        python kaggle/create_notebooks.py
        break
    fi
done

# Step 3: Push all notebooks
echo "Step 3: Pushing notebooks to Kaggle..."
bash kaggle/create_and_launch.sh

# Step 4: Generate manual instructions as backup
echo "Step 4: Generating manual instructions backup..."
python kaggle/manual_instructions.py
echo "  Manual steps saved to: kaggle/MANUAL_STEPS.txt"

# Step 5: Verify
echo "Step 5: Verifying kernels..."
python kaggle/verify_notebooks.py

echo ""
echo "========================================"
echo "DONE. If all kernels show running:"
echo "  Close your MacBook."
echo ""
echo "If any kernel failed:"
echo "  Open kaggle/MANUAL_STEPS.txt"
echo "  Follow the manual steps (10 min)"
echo "========================================"

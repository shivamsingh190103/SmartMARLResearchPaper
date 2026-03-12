#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/shivamsingh/Desktop/ResearchPaper"
cd "$ROOT"

if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "/Users/shivamsingh/.kaggle/token.env" ]]; then
  # shellcheck disable=SC1091
  source "/Users/shivamsingh/.kaggle/token.env"
fi

if command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="kaggle"
elif [[ -x "${ROOT}/.venv/bin/kaggle" ]]; then
  KAGGLE_BIN="${ROOT}/.venv/bin/kaggle"
else
  echo "kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing Kaggle auth."
  echo "Either set KAGGLE_API_TOKEN or create $HOME/.kaggle/kaggle.json."
  exit 1
fi

resolve_username() {
  if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
    printf '%s' "$KAGGLE_USERNAME"
    return
  fi

  local cfg_view
  cfg_view="$("$KAGGLE_BIN" config view 2>/dev/null || true)"
  local u
  u="$(printf '%s\n' "$cfg_view" | sed -n 's/^- username: //p' | head -n1)"
  if [[ -n "$u" ]]; then
    printf '%s' "$u"
    return
  fi

  # Fallback if unable to detect.
  printf '%s' "sshivamsingh07"
}

./kaggle/prepare_bundle.sh

mkdir -p kaggle/dataset_upload
cp -f kaggle/output/smartmarl_kaggle.zip kaggle/dataset_upload/

USERNAME="$(resolve_username)"
DATASET_ID="${USERNAME}/smartmarl-codebase"

cat > kaggle/dataset_upload/dataset-metadata.json <<JSON
{
  "title": "smartmarl-codebase",
  "id": "${DATASET_ID}",
  "licenses": [
    {
      "name": "MIT"
    }
  ]
}
JSON

cd kaggle/dataset_upload

if "$KAGGLE_BIN" datasets status "$DATASET_ID" >/dev/null 2>&1; then
  "$KAGGLE_BIN" datasets version -p . -m "Update SmartMARL Kaggle bundle"
else
  "$KAGGLE_BIN" datasets create -p .
fi

echo "Kaggle dataset upload/version complete: $DATASET_ID"

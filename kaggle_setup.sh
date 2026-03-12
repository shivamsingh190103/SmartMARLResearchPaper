#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing SUMO ==="
apt-get update -q
apt-get install -y sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
sumo --version | head -n 1

echo "=== Installing Python dependencies ==="
pip install -q -r requirements.txt

echo "=== Verifying TraCI ==="
python - <<'PY'
import os
os.environ['SUMO_HOME']='/usr/share/sumo'
import traci
print('TraCI OK', traci.__version__ if hasattr(traci, '__version__') else 'installed')
PY

echo "=== Setup complete ==="

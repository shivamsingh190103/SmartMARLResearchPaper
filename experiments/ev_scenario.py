from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartmarl.experiments.ev_scenario import run_ev_experiment

if __name__ == "__main__":
    print(run_ev_experiment(use_ev_reward=True))
    print(run_ev_experiment(use_ev_reward=False))

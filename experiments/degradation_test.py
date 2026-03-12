from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartmarl.experiments.degradation_test import run_degradation_test

if __name__ == "__main__":
    df = run_degradation_test(output_dir="results")
    print(df.to_string(index=False))

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartmarl.experiments.ablation import run_all_ablations, format_table

if __name__ == "__main__":
    df = run_all_ablations()
    print(format_table(df.to_dict("records")))

# SmartMARL Research Pipeline

This repository contains the SmartMARL training, Kaggle orchestration, monitoring, and result-aggregation workflow.

## Current Status

As of March 12, 2026, the main failure modes reported in your Kaggle logs are addressed in this codebase:

1. `Temporary failure in name resolution` during `pip` or `apt` installs
2. `unrecognized arguments: --skip_existing` from stale notebook code
3. `cannot find ... smartmarl_kaggle.zip` from old dataset path assumptions

The generated notebooks are now offline-safe and stage code from nested Kaggle input mounts.

## Why Your Runs Were Failing

Your logs showed infrastructure-level DNS outages inside Kaggle sessions. During those windows, any runtime install (`pip install traci`, `apt-get install sumo`) fails and misleadingly looks like package errors.

At the same time, some notebooks were running older code that still passed `--skip_existing` before `train.py` accepted it, so seed jobs exited with code 2 and produced no per-seed JSON files.

## What Is Fixed

1. Notebook generation no longer depends on runtime `pip/apt` install loops.
2. Dataset staging handles nested Kaggle paths like `/kaggle/input/datasets/...`.
3. `train.py` supports `--skip_existing`.
4. `collect_results.py` now supports both JSON and CSV, multiple filename styles, and robust seed pairing.
5. Result tables are produced in both CSV and plain text formats.

## Local Setup

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Local Smoke Check

```bash
. .venv/bin/activate
python train.py --scenario standard --ablation full --seed 0 --episodes 1 --steps_per_episode 120
```

## Kaggle End-to-End Workflow

1. Build Kaggle notebook assets:

```bash
. .venv/bin/activate
python kaggle/create_notebooks.py
```

2. Build upload bundle:

```bash
./kaggle/prepare_bundle.sh
```

3. Upload dataset:

```bash
./kaggle/upload_dataset.sh
```

4. Push all notebook kernels:

```bash
./kaggle/create_and_launch.sh
```

5. Verify kernel status:

```bash
python kaggle/verify_notebooks.py
```

6. Monitor and harvest outputs:

```bash
./kaggle/monitor_kernels.sh
```

## Result Aggregation

Use the project virtual environment:

```bash
. .venv/bin/activate
python collect_results.py --raw_dir results/raw --scenario standard
```

Outputs:

1. `results/ablation_table.csv`
2. `results/ablation_table.txt`

For Indian scenario:

```bash
python collect_results.py \
  --raw_dir results/raw \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt
```

## Fast Troubleshooting

1. If you see `Temporary failure in name resolution`, do not trust package errors from that run. Re-run the notebook later.
2. If you see `unrecognized arguments: --skip_existing`, the kernel is still using stale code. Recreate notebooks and push again.
3. If you see `cannot find ... smartmarl_kaggle.zip`, the dataset mount changed; use current notebook generators in this repo.
4. If Kaggle API returns `Maximum batch CPU session count ... reached`, wait for active sessions to finish, then relaunch only pending kernels.

## Testing

```bash
. .venv/bin/activate
pytest tests -v
```

## Key Files

1. `/Users/shivamsingh/Desktop/ResearchPaper/train.py`
2. `/Users/shivamsingh/Desktop/ResearchPaper/collect_results.py`
3. `/Users/shivamsingh/Desktop/ResearchPaper/kaggle/create_notebooks.py`
4. `/Users/shivamsingh/Desktop/ResearchPaper/kaggle/create_and_launch.sh`
5. `/Users/shivamsingh/Desktop/ResearchPaper/kaggle/monitor_kernels.sh`

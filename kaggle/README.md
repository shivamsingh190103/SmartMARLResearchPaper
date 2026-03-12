# Kaggle Operations Guide

This folder contains the production Kaggle workflow for SmartMARL.

## Primary Goal

Run seeds in parallel on Kaggle and harvest per-seed JSON outputs into local `results/raw` for final ablation tables.

## Canonical Flow

1. Generate notebook assets:

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
. .venv/bin/activate
python kaggle/create_notebooks.py
```

2. Build dataset bundle:

```bash
./kaggle/prepare_bundle.sh
```

3. Upload dataset:

```bash
./kaggle/upload_dataset.sh
```

4. Push and launch notebooks:

```bash
./kaggle/create_and_launch.sh
```

5. Verify launch state:

```bash
python kaggle/verify_notebooks.py
```

6. Monitor and harvest outputs:

```bash
./kaggle/monitor_kernels.sh
```

## Notebook Set

1. `smartmarl-standard-full-seeds-1-10`
2. `smartmarl-standard-full-seeds-11-20`
3. `smartmarl-standard-full-seeds-21-29`
4. `smartmarl-standard-l7-seeds-1-29`

## Important Design Choice

Generated notebooks are offline-safe:

1. No required runtime `pip install traci` loop
2. No required runtime `apt-get install sumo` loop
3. Code is staged by scanning `/kaggle/input` recursively to handle nested mounts
4. Training proceeds even when Kaggle DNS is unstable

## Common Failure Messages

1. `Temporary failure in name resolution`
- Cause: Kaggle DNS/network outage during that session
- Action: rerun later; do not infer package incompatibility

2. `unrecognized arguments: --skip_existing`
- Cause: stale notebook source still running
- Action: regenerate notebooks and push again

3. `cannot find ... smartmarl_kaggle.zip`
- Cause: old hardcoded dataset path
- Action: use generated notebooks from this repo

4. `Maximum batch CPU session count ... reached`
- Cause: Kaggle concurrency quota reached
- Action: wait for running sessions to end, then relaunch only pending kernels

## Harvested Output Expectation

Each successful run should create files like:

1. `results/raw/full_standard_seed*.json`
2. `results/raw/l7_standard_seed*.json`

Then aggregate with:

```bash
. .venv/bin/activate
python collect_results.py --raw_dir results/raw --scenario standard
```

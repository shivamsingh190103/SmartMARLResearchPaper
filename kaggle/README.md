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

To launch in smaller batches (recommended for quota limits):

```bash
SMARTMARL_MAX_PUSH=4 ./kaggle/create_and_launch.sh
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

- Generated dynamically by `kaggle/create_notebooks.py`
- Default: 30 single-seed notebooks for `standard/full`:
  - `smartmarl-standard-full-seed-00` ... `smartmarl-standard-full-seed-29`
- Optional L7: set `SMARTMARL_INCLUDE_L7=1` before generation

## Important Design Choice

Generated notebooks are fail-fast for research integrity:

1. Code is staged by scanning `/kaggle/input` recursively to handle nested mounts
2. Runtime enforces SUMO availability (`apt-get install sumo sumo-tools` with retries)
3. TraCI import is validated after SUMO setup
4. Notebook aborts if backend resolves to mock mode (prevents invalid paper-scale runs)

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

5. `FAIL: Mock backend detected. Aborting to avoid invalid paper-scale results.`
- Cause: SUMO/TraCI unavailable in that Kaggle session
- Action: restart when backend is available; do not use mock outputs for paper tables

## Harvested Output Expectation

Each successful run should create files like:

1. `results/raw/standard_full_seed*.json`
2. `results/raw/standard_l7_seed*.json`

Then aggregate with:

```bash
. .venv/bin/activate
python collect_results.py --raw_dir results/raw --scenario standard
```

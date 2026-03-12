# SmartMARL Research Codebase

SmartMARL is an end-to-end multi-agent urban traffic signal control pipeline with explicit uncertainty coupling from perception to policy.

## Core Idea

SmartMARL combines three tightly linked stages:

1. Perception (P1)
- YOLO-like camera detections + radar measurements
- Adaptive UKF with online `R_k` adaptation
- Per-intersection uncertainty output `sigma2_r`

2. Graph Encoding (P2)
- Heterogeneous GNN with typed nodes (`Vint`, `Vlane`, `Vsens`, `Vinj`)
- Relation-specific weights (`W_spatial`, `W_flow`, `W_incident`) per layer
- `sigma2_r` is routed through `Vsens` into policy embeddings

3. Control (P3)
- GATv2-based actor (dynamic attention)
- Centralized critic for CTDE training
- Decentralized execution during inference

## Why L7 Matters

The critical L7 ablation keeps AUKF lane-state estimates but removes only `Vsens` coupling:

- `use_aukf=True`
- `use_vsens=False`

This isolates uncertainty propagation contribution from state-estimation contribution.

## Repository Layout

```text
smartmarl/
  env/            # SUMO/CityFlow env wrappers + graph builder
  perception/     # AUKF, detector/radar mocks, Hungarian association, noise injection
  models/         # HetGNN, GATv2 actor, centralized critic
  training/       # MA2C trainer, buffer, LR scheduler
  experiments/    # ablations, degradation tests, EV evaluation
  utils/          # metrics, stats, logger
tests/            # unit and integration tests
train.py          # training entry point
evaluate.py       # checkpoint evaluation
run_ablation.py   # table generation driver
```

## Quick Start

### 1) Environment

```bash
cd /Users/shivamsingh/Desktop/ResearchPaper
. .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate SUMO network

```bash
python setup_network.py
```

### 3) Smoke test in real mode

```bash
python train.py --scenario standard --seed 0 --episodes 3
```

Expected output should include:

```text
Mock mode: False
```

## Training Commands

### Standard full model

```bash
python train.py \
  --scenario standard \
  --ablation full \
  --seed 0 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100
```

### Resume interrupted run

```bash
python train.py \
  --scenario standard \
  --ablation full \
  --seed 0 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100 \
  --resume
```

### Critical L7 run

```bash
python train.py \
  --scenario standard \
  --ablation l7 \
  --seed 0 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100
```

### Indian heterogeneous scenario

```bash
python train.py \
  --scenario indian_hetero \
  --ablation full \
  --seed 0 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100
```

## Batch Orchestration

Run seed ranges safely with resume support:

```bash
python run_seed_batch.py \
  --scenario standard \
  --ablation full \
  --seed_start 0 \
  --seed_end 30 \
  --episodes 3000 \
  --steps_per_episode 300 \
  --checkpoint_every 100 \
  --resume \
  --skip_existing
```

Automated queue (waits for seed-0 outputs, then launches phase-2 batches):

```bash
./run_phase2_after_seed0.sh
```

## Hands-Free Autopilot

For fully unattended execution (recommended for long runs), use:

```bash
./autopilot_supervisor.sh
```

What it does automatically:

1. Starts and maintains `caffeinate -dimsu` (prevents sleep)
2. Keeps seed-0 training jobs alive with `--resume`
3. Keeps phase-2 queue alive (`run_phase2_after_seed0.sh`)
4. Keeps auto-finalizer alive (`auto_finalize_and_push.sh`)
5. Maintains live training status logging

Autopilot helper files:

- `autopilot_supervisor.sh`
- `live_status_loop.sh`
- `run_phase2_after_seed0.sh`
- `auto_finalize_and_push.sh`

## Kaggle Acceleration

Local exhaustive training can take many weeks. For parallel execution, use:

1. `./kaggle/prepare_bundle.sh`
2. Upload `kaggle/output/smartmarl_kaggle.zip` as a private Kaggle dataset
3. Run notebook runners in parallel from `kaggle/notebooks/`

See full guide:

- `kaggle/README.md`

## Monitoring

Live single-run monitor:

```bash
python monitor_training.py results/training_logs/standard_full_seed0.csv
```

Useful logs:

- `logs/live_status.log`
- `logs/phase2_queue.log`

## Automation Toolkit

These scripts are built for unattended multi-day Kaggle + local orchestration:

```bash
# 1) Update Kaggle kernels to include --skip_existing
python kaggle/update_notebooks.py

# 2) Full pre-close health check (safe to run any time)
chmod +x health_check_now.sh
bash health_check_now.sh

# 3) One-shot health report
python monitor/health_check.py

# 4) Continuous health watch (every 30 min)
python monitor/health_check.py --watch

# 5) Kaggle output harvesting
python kaggle/harvest_results.py --dry-run
python kaggle/harvest_results.py

# 6) Auto-restart daemon for timed-out kernels
bash kaggle/auto_restart.sh

# 7) Daily summary files
python monitor/daily_summary.py
cat monitor/today.txt

# 8) Finalizer once enough seeds arrive
python finalize_results.py
```

Finalizer outputs:

- `results/FINAL_TABLE8.txt`
- `results/FINAL_TABLE4.txt`
- `results/PAPER_NUMBERS.txt`

## Results Aggregation

Standard scenario table:

```bash
python collect_results.py --scenario standard
cat results/ablation_table.txt
```

Indian scenario table:

```bash
python collect_results.py \
  --scenario indian_hetero \
  --out_csv results/ablation_table_indian.csv \
  --out_txt results/ablation_table_indian.txt
```

## Tests

Run full test suite:

```bash
pytest tests/ -v
```

Run key claim tests only:

```bash
pytest tests/test_aukf.py tests/test_models.py tests/test_ablation_l7.py -v
```

## Reproducibility Notes

- Real publishable results require `Mock mode: False`
- SUMO + TraCI are required for final tables
- Keep `results/raw/*.json` per seed for statistical tests and CI computation
- Wilcoxon signed-rank is used for paired comparisons

## Citation

If you use this codebase, cite the SmartMARL manuscript and include the commit hash used for experiments.

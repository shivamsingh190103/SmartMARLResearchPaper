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

## Monitoring

Live single-run monitor:

```bash
python monitor_training.py results/training_logs/standard_full_seed0.csv
```

Useful logs:

- `logs/live_status.log`
- `logs/phase2_queue.log`

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

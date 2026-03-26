# 🚦 SmartMARL
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![SUMO](https://img.shields.io/badge/SUMO-1.18-orange.svg)](https://www.eclipse.org/sumo/) [![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-ee4c2c.svg)](https://pytorch.org/) [![Status](https://img.shields.io/badge/Status-Training%20in%20Progress-yellow.svg)](#project-status)

Uncertainty-aware HetGNN + MA2C traffic signal control system for adaptive, multi-intersection urban traffic optimization.

## Abstract
**SmartMARL: Uncertainty-Aware Heterogeneous Graph Neural Network Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control** proposes a three-stage perception-to-control pipeline for real-time traffic signal optimization under heterogeneous conditions. The method fuses YOLOv8n and 77GHz radar perception with an Adaptive UKF, propagates uncertainty through a heterogeneous graph encoder, and performs decentralized control with a GATv2 actor under CTDE training. Against the GPLight baseline, SmartMARL reports a **10.0% reduction in Average Travel Time (ATT) under Indian heterogeneous traffic** (**129.6s vs 143.8s**, Wilcoxon signed-rank, **p<0.001**, N=30 seeds). Under standard traffic, SmartMARL reports a **4.1% ATT reduction** over GPLight.

## Architecture Diagram
```text
[YOLOv8n + Radar] -> [AUKF beta=0.02] -> [HetGNN L=3] -> [GATv2 MA2C] -> [Traffic Lights]
     P1 Perception        P1->P2            P2 Encoding      P3 Control
```

- **P1 (Perception):** YOLOv8n + 77GHz radar with AUKF adaptive noise update (\(\beta=0.02\))
- **P2 (Encoding):** HetGNN with 4 node types (`Vint`, `Vlane`, `Vsens`, `Vinj`), ELU, 3 layers, 128-d hidden states
- **P3 (Control):** GATv2 actor (2 heads, 64-d) + MA2C CTDE (centralized critic only during training)
- **Deployment target:** NVIDIA Jetson Xavier NX (79ms avg inference, 87ms P99, 14.2W peak)

## Key Results Table
| Method | Standard ATT | Indian ATT | vs GPLight |
|---|---:|---:|---:|
| **SmartMARL** | 4.1% lower than GPLight | **129.6s** | **-4.1% (Standard), -10.0% (Indian)** |
| **GPLight (baseline)** | Baseline reference | **143.8s** | 0% |
| L1: No AUKF | TBD | +10.0% vs SmartMARL | Degrades vs GPLight margin |
| L2: Homogeneous GNN | TBD | +5.5% vs SmartMARL | Degrades vs GPLight margin |
| L3: No HetGNN | TBD | TBD | TBD |
| L4: No GATv2 | TBD | TBD | TBD |
| L5: Single agent | TBD | TBD | TBD |
| L6: No reward shaping | TBD | TBD | TBD |
| L7: No σ²r coupling (Vsens zeroed) | Training in progress | Result pending | Pending |

## Installation

### Prerequisites
- Python 3.10+
- SUMO 1.18 (or `eclipse-sumo` via pip)
- CUDA 11.8+ (optional; CPU is supported via mock fallback)

### Clone and setup
```bash
git clone https://github.com/shivamsingh190103/SmartMARLResearchPaper.git
cd SmartMARLResearchPaper
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Install SUMO (choose one)
```bash
# Option A (pip)
pip install eclipse-sumo
```

```bash
# Option B (system, macOS)
brew install sumo
```

```bash
# Option B (system, Ubuntu)
sudo apt install sumo sumo-tools
```

### Verify SUMO + TraCI
```bash
python -c "import traci; print('SUMO OK')"
```

## Quick Start
```bash
# Validate or regenerate the real SUMO grid + route assets
python setup_network.py --strict
```

```bash
# Train single seed (standard)
python train.py --scenario standard --seed 0 --episodes 3000
```

```bash
# Train Indian heterogeneous traffic
python train.py --scenario indian_hetero --seed 0 --episodes 3000
```

```bash
# Run L7 ablation
python train.py --scenario standard --ablation l7 --seed 0 --episodes 3000
```

```bash
# Run all ablations
python run_ablation.py
```

```bash
# Run the GPLight-style grouped baseline
python run_gplight_baseline.py --scenario standard --episodes 1500 --skip_existing
```

```bash
# Run classic rule baselines (FixedTime + MaxPressure)
python run_rule_baselines.py --scenario standard --seed_start 0 --seed_end 29 --skip_existing
```

```bash
# Collect and format ablation results
python collect_results.py
```

```bash
# Health check
python monitor/health_check.py
```

## Live Demo
```bash
# Interactive animated demo (preferred when a GUI backend is available)
python demo.py --episodes 60
```

```bash
# Headless demo (works in terminal-only environments)
python demo.py --no-gui --episodes 20
```

The headless demo writes a visual summary image to:

```bash
demo_results/smartmarl_demo_summary.png
```

The repository also ships a static demo snapshot:

![SmartMARL Demo Snapshot](smartmarl_demo.png)

## Research Utilities
```bash
# AUKF robustness sweep (camera/radar noise vs fused RMSE)
python -m smartmarl.experiments.aukf_noise_sweep --output results/aukf_noise_sweep.csv
```

```bash
# Complexity profiling across 4/9/16/25 intersections
# Uses fvcore symbolic FLOP tracing when available (fallback: torch profiler)
python analyze_complexity.py --device cpu --out results/complexity/complexity_summary.csv
```

```bash
# EV corridor comparison (fair protocol with checkpointed pretraining)
python -m smartmarl.experiments.ev_scenario
```

```bash
# Calibrate arrival-rate / speed priors from a public trajectory CSV
python -m smartmarl.calibration.demand_calibration --input data/ngsim.csv --output results/calibration/ngsim_profile.yaml
```

```bash
# Optional NGSIM pipeline (auto-normalize common NGSIM column names)
python -m smartmarl.calibration.ngsim_pipeline --input_csv data/ngsim.csv --output results/calibration/ngsim_profile.yaml
```

## Training at Scale (Kaggle)
SmartMARL now uses a single-seed Kaggle layout so every kernel stays inside the 12-hour limit with real SUMO enabled:

1. `smartmarl-standard-full-seed-00` ... `smartmarl-standard-full-seed-29`
2. Optional L7 single-seed notebooks can be generated with `SMARTMARL_INCLUDE_L7=1`

Each notebook:
- installs `sumo` + `sumo-tools`
- regenerates the grid and route files with `setup_network.py --strict --force-regenerate`
- runs a smoke test that aborts immediately if `Mock mode: False` is not confirmed
- trains one seed only and writes JSON/CSV/PT outputs to the Kaggle output bundle

```bash
# Generate Kaggle notebook assets
python create_kaggle_notebooks.py
```

```bash
# Launch a capped batch of notebooks
SMARTMARL_MAX_PUSH=4 bash kaggle/create_and_launch.sh
```

```bash
# Verify kernel status
python kaggle/verify_notebooks.py
```

- The generated notebooks default to `1500` episodes and `300` steps per episode.
- Use small launch batches (`SMARTMARL_MAX_PUSH=4`) to avoid wasting Kaggle quota if a notebook fails early.

## Configuration
| Parameter | Value | Description |
|---|---:|---|
| `lr` | `1e-4` | Optimizer learning rate (`learning_rate`) |
| `gamma` | `0.95` | Discount factor (`discount_gamma`) |
| `hidden_dim` | `128` | HetGNN embedding dimension (`embedding_dim`) |
| `L (layers)` | `3` | Number of HetGNN layers (`hetgnn_layers`) |
| `heads` | `2` | GATv2 attention heads (`actor_heads`) |
| `batch_size` | `N/A (on-policy)` | Training uses periodic updates (`batch_update_every=20`) |
| `beta_aukf` | `0.02` | Adaptive UKF noise update coefficient (`aukf_beta`) |
| `alpha_lj` | `0.2` | Non-lane-disciplined lateral behavior factor (`indian_hetero.yaml`) |
| `reward_weights` | `alpha=0.6, tev=0.85, penalty=0.15` | Reward composition weights |

## Project Status
- [x] SUMO environment + TraCI integration
- [x] AUKF perception module (β=0.02)
- [x] HetGNN encoder (4 node types, L=3)
- [x] GATv2 actor-critic (MA2C CTDE)
- [x] Indian heterogeneous traffic benchmark
- [x] Ablation framework (L1-L7)
- [x] Distributed Kaggle training pipeline
- [ ] L7 ablation results (training in progress)
- [ ] N=30 seeds collected (in progress)
- [ ] arXiv submission
- [ ] IEEE ITSC 2025 submission

## Citing This Work
```bibtex
@article{singh2025smartmarl,
  title={SmartMARL: Uncertainty-Aware Heterogeneous Graph Neural Network 
         Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control},
  author={Shivam Singh and Shivansh Tiwari,  Neeraj saroj
          and Sudhakar Dwivedi},
  journal={arXiv preprint},
  year={2026}
}
```

## License
MIT License

---

Made at AKGEC Ghaziabad

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
```python
import traci
print("SUMO OK")
```

## Quick Start
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
python run_ablation.py --skip_existing
```

```bash
# Collect and format results
python collect_results.py
```

```bash
# Health check
python monitor/health_check.py
```

## Training at Scale (Kaggle)
SmartMARL uses a 4-notebook distributed setup:

1. Notebook 1: `seeds 1-10` (`standard/full`)
2. Notebook 2: `seeds 11-20` (`standard/full`)
3. Notebook 3: `seeds 21-29` (`standard/full`)
4. Notebook 4: `seeds 1-29` (`l7` ablation)

```bash
# Generate Kaggle notebook assets
python create_kaggle_notebooks.py
```

```bash
# Verify kernel status
python kaggle/verify_notebooks.py
```

- Each notebook is designed to auto-resume with `--skip_existing`.
- Kaggle sessions are limited to 12 hours; practical throughput is roughly 3 seeds/session, so daily restarts are expected for full completion.

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
  author={Singh, Shivam and Tiwari, Shivansh and Saroj, Neeraj 
          and Dwivedi, Sudhakar},
  journal={arXiv preprint},
  year={2025}
}
```

## License
MIT License

---

Made at AKGEC Ghaziabad

"""Measure SmartMARL model complexity and inference latency."""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except Exception:
    FlopCountAnalysis = None

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer


VARIANTS = ("full", "gplight", "mlp")
GRID_SIZES = (2, 3, 4, 5)


def load_config() -> Dict:
    with open("smartmarl/configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parameter_count(trainer: MA2CTrainer) -> int:
    modules = [trainer.encoder, trainer.actor]
    if trainer.critic is not None:
        modules.append(trainer.critic)
    return int(sum(p.numel() for module in modules for p in module.parameters()))


def _make_trainer(variant: str, grid_size: int, device: str) -> tuple[MA2CTrainer, SumoTrafficEnv, Dict]:
    cfg = deepcopy(load_config())
    cfg["grid_size"] = grid_size
    cfg["num_intersections"] = grid_size * grid_size
    env = SumoTrafficEnv(
        scenario="standard",
        episode_length_seconds=120,
        num_intersections=cfg["num_intersections"],
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=0,
        use_traci=False,
    )
    trainer = MA2CTrainer(env=env, config=cfg, ablation=variant, seed=0, device=device)
    return trainer, env, cfg


def _latency_ms(trainer: MA2CTrainer, obs: Dict, repeats: int = 50) -> float:
    for _ in range(5):
        trainer.inference_policy(obs)
    start = time.perf_counter()
    for _ in range(repeats):
        trainer.inference_policy(obs)
    elapsed = time.perf_counter() - start
    return float(1000.0 * elapsed / repeats)


class _InferenceGraph(nn.Module):
    def __init__(self, trainer: MA2CTrainer, group_ids: torch.Tensor | None) -> None:
        super().__init__()
        self.trainer = trainer
        self.use_gplight = bool(trainer.variant.use_gplight)
        self.register_buffer("spatial", trainer.edge_index_dict["spatial"].detach().clone())
        self.register_buffer("flow_lane", trainer.edge_index_dict["flow_lane"].detach().clone())
        self.register_buffer("flow_sens", trainer.edge_index_dict["flow_sens"].detach().clone())
        self.register_buffer("incident", trainer.edge_index_dict["incident"].detach().clone())
        if group_ids is not None:
            self.register_buffer("group_ids", group_ids.detach().clone())
        else:
            self.group_ids = None

    def forward(self, int_feat: torch.Tensor, lane_feat: torch.Tensor, sens_feat: torch.Tensor, inj_feat: torch.Tensor):
        if self.use_gplight:
            fused = torch.cat([int_feat, lane_feat], dim=-1)
            h = self.trainer.encoder(fused, self.spatial)
            probs = self.trainer.actor(h, group_ids=self.group_ids)
        else:
            node_features = {
                "int": int_feat,
                "lane": lane_feat,
                "sens": sens_feat,
                "inj": inj_feat,
            }
            edge_dict = {
                "spatial": self.spatial,
                "flow_lane": self.flow_lane,
                "flow_sens": self.flow_sens,
                "incident": self.incident,
            }
            h = self.trainer.encoder(node_features, edge_dict)
            probs = self.trainer.actor(h, self.spatial)
        return probs


def _flops(trainer: MA2CTrainer, obs: Dict) -> float:
    node_features, _ = trainer.build_node_features(obs)
    group_ids = trainer._dynamic_group_ids(obs)
    graph = _InferenceGraph(trainer, group_ids).to(trainer.device)
    inputs = (
        node_features["int"],
        node_features["lane"],
        node_features["sens"],
        node_features["inj"],
    )

    if FlopCountAnalysis is not None:
        try:
            analysis = FlopCountAnalysis(graph, inputs)
            return float(analysis.total())
        except Exception:
            pass

    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            graph(*inputs)
        total = 0.0
        for event in prof.key_averages():
            total += float(getattr(event, "flops", 0.0) or 0.0)
        return total if total > 0 else float("nan")
    except Exception:
        return float("nan")


def run_analysis(device: str = "cpu") -> List[Dict]:
    rows: List[Dict] = []
    for grid_size in GRID_SIZES:
        for variant in VARIANTS:
            trainer, env, cfg = _make_trainer(variant, grid_size, device)
            obs, _ = env.reset(seed=0)
            latency_ms = _latency_ms(trainer, obs)
            rows.append(
                {
                    "variant": variant,
                    "grid_size": grid_size,
                    "num_intersections": int(cfg["num_intersections"]),
                    "device": device,
                    "parameter_count": parameter_count(trainer),
                    "latency_ms": latency_ms,
                    "flops": _flops(trainer, obs),
                }
            )
            env.close()
    return rows


def save_rows(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SmartMARL inference complexity.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="results/complexity/complexity_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    rows = run_analysis(device=device)
    save_rows(rows, Path(args.out))

    print("variant,grid_size,num_intersections,device,parameter_count,latency_ms,flops")
    for row in rows:
        flops_text = "nan" if not np.isfinite(row["flops"]) else f"{row['flops']:.0f}"
        print(
            f"{row['variant']},{row['grid_size']},{row['num_intersections']},"
            f"{row['device']},{row['parameter_count']},{row['latency_ms']:.3f},{flops_text}"
        )


if __name__ == "__main__":
    main()

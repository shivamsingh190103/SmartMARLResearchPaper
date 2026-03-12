"""Ablation runner for SmartMARL (Table 8 reproduction, including L7)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer
from smartmarl.utils.stats import format_mean_ci, wilcoxon_with_effect_size


@dataclass
class AblationVariant:
    key: str
    trainer_name: str
    table_label: str


ABLATION_VARIANTS = [
    AblationVariant("full_smartmarl", "full_smartmarl", "Full SmartMARL"),
    AblationVariant("no_ctde", "no_ctde", "-CTDE (->IndepQL)"),
    AblationVariant("no_aukf", "no_aukf", "-AUKF (->raw counts)"),
    AblationVariant("no_hetgnn", "no_hetgnn", "-HetGNN (->hom. GAT)"),
    AblationVariant("l7_ablation", "l7_ablation", "-Vsens only (L7)"),
    AblationVariant("no_incident_nodes", "no_incident_nodes", "-Incident nodes"),
    AblationVariant("no_ev_mode", "no_ev_mode", "-EV mode (normal)"),
    AblationVariant("yolov5_backbone", "yolov5_backbone", "YOLOv5->YOLOv8n"),
    AblationVariant("mlp_actor", "mlp_actor", "MLP->GATv2 actor"),
]


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _episodes_from_config(cfg: Dict, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return int(cfg["training_episodes_sumo"])


def _fast_mode_override(episodes: int, cfg: Dict) -> int:
    if os.getenv("SMARTMARL_FAST", "").strip().lower() in {"1", "true", "yes"}:
        return min(episodes, 20)
    file_override = Path.cwd().joinpath(".smartmarl_fast_episodes")
    if file_override.exists():
        try:
            return max(1, int(file_override.read_text().strip()))
        except Exception:
            pass
    return episodes


def run_variant(
    variant: AblationVariant,
    cfg: Dict,
    seeds: List[int],
    output_dir: str,
    scenario: str = "standard",
    episodes_override: Optional[int] = None,
    eval_episodes: int = 3,
) -> Dict:
    output = Path(output_dir)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    att_runs: List[float] = []
    awt_runs: List[float] = []
    tp_runs: List[float] = []

    train_episodes = _episodes_from_config(cfg, episodes_override)
    train_episodes = _fast_mode_override(train_episodes, cfg)

    for seed in seeds:
        config_path = (
            str(Path("smartmarl/configs/grid5x5/grid5x5_indian.sumocfg"))
            if scenario == "indian_hetero"
            else str(Path("smartmarl/configs/grid5x5/grid5x5.sumocfg"))
        )
        env = SumoTrafficEnv(
            config_path=config_path,
            scenario=scenario,
            episode_length_seconds=int(cfg["episode_length_seconds"]),
            num_intersections=int(cfg["num_intersections"]),
            num_phases=int(cfg["num_phases"]),
            min_green_time_seconds=int(cfg["min_green_time_seconds"]),
            seed=seed,
            use_traci=True,
        )
        trainer = MA2CTrainer(env=env, config=cfg, ablation=variant.trainer_name, seed=seed)

        trainer.train(num_episodes=train_episodes, progress=False)
        eval_metrics = trainer.evaluate(num_episodes=eval_episodes)

        att = float(eval_metrics["ATT"])
        awt = float(eval_metrics["AWT"])
        tp = float(eval_metrics["Throughput"])

        att_runs.append(att)
        awt_runs.append(awt)
        tp_runs.append(tp)

        pd.DataFrame(
            [{"seed": seed, "ATT": att, "AWT": awt, "Throughput": tp}]
        ).to_csv(raw_dir / f"{variant.key}_seed{seed}.csv", index=False)
        env.close()

    return {
        "variant": variant.key,
        "label": variant.table_label,
        "ATT_runs": att_runs,
        "AWT_runs": awt_runs,
        "Throughput_runs": tp_runs,
        "ATT_mean": float(np.mean(att_runs)) if att_runs else 0.0,
        "AWT_mean": float(np.mean(awt_runs)) if awt_runs else 0.0,
        "Throughput_mean": float(np.mean(tp_runs)) if tp_runs else 0.0,
    }


def _pvalue_str(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    if p >= 0.05:
        return "n.s."
    return f"{p:.3f}"


def format_table(rows: List[Dict]) -> str:
    lines = []
    lines.append("Variant              | ATT (s)      | ΔATT        | p-value")
    lines.append("---------------------|--------------|-------------|--------")
    for row in rows:
        lines.append(
            f"{row['label']:<21}| {row['ATT_fmt']:<12} | {row['delta_att']:<11} | {row['p_value_fmt']}"
        )
    return "\n".join(lines)


def run_all_ablations(
    config_path: str = "smartmarl/configs/default.yaml",
    output_dir: str = "results",
    scenario: str = "standard",
    episodes_override: Optional[int] = None,
    num_seeds_override: Optional[int] = None,
) -> pd.DataFrame:
    cfg = _load_config(config_path)
    seeds = list(cfg["seeds"])
    if num_seeds_override is not None:
        seeds = seeds[: max(1, int(num_seeds_override))]

    results = []
    for variant in ABLATION_VARIANTS:
        result = run_variant(
            variant=variant,
            cfg=cfg,
            seeds=seeds,
            output_dir=output_dir,
            scenario=scenario,
            episodes_override=episodes_override,
        )
        results.append(result)

    full = next(r for r in results if r["variant"] == "full_smartmarl")
    full_att = np.asarray(full["ATT_runs"], dtype=np.float64)

    table_rows = []
    for row in results:
        att_runs = np.asarray(row["ATT_runs"], dtype=np.float64)
        att_fmt = format_mean_ci(att_runs, confidence=0.95, seed=0)

        if row["variant"] == "full_smartmarl":
            delta_str = "-"
            p_fmt = "-"
            w = np.nan
            p = np.nan
            d = np.nan
        else:
            delta = float(np.mean(att_runs) - np.mean(full_att))
            delta_str = f"{delta:+.1f}s"
            if att_runs.size >= 2 and full_att.size >= 2:
                stats = wilcoxon_with_effect_size(att_runs, full_att)
                w = stats["W"]
                p = stats["p_value"]
                d = stats["cohens_d"]
                p_fmt = _pvalue_str(p)
            else:
                w = np.nan
                p = np.nan
                d = np.nan
                p_fmt = "n.s."

        table_rows.append(
            {
                "variant": row["variant"],
                "label": row["label"],
                "ATT_mean": float(np.mean(att_runs)) if att_runs.size else 0.0,
                "ATT_fmt": att_fmt,
                "delta_att": delta_str,
                "p_value": p,
                "p_value_fmt": p_fmt,
                "W": w,
                "cohens_d": d,
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(table_rows)
    df.to_csv(output / "ablation_table.csv", index=False)

    table_txt = format_table(table_rows)
    (output / "ablation_table.txt").write_text(table_txt, encoding="utf-8")

    return df

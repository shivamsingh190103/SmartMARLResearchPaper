"""Emergency-vehicle corridor experiment for SmartMARL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer, default_checkpoint_path
from smartmarl.utils.metrics import compute_metrics


def _scheduled_ev_active(env: SumoTrafficEnv) -> bool:
    return bool(env.ev_start_step <= env.current_step < env.ev_start_step + env.ev_duration_steps)


def _make_env(cfg: Dict, seed: int) -> SumoTrafficEnv:
    env = SumoTrafficEnv(
        config_path="smartmarl/configs/grid5x5/grid5x5.sumocfg",
        scenario="standard",
        episode_length_seconds=int(cfg["episode_length_seconds"]),
        num_intersections=int(cfg["num_intersections"]),
        num_phases=int(cfg["num_phases"]),
        min_green_time_seconds=int(cfg["min_green_time_seconds"]),
        seed=seed,
        use_traci=True,
    )
    env.configure_ev_corridor(start_step=250, duration_steps=90, preferred_phase=0)
    return env


def _evaluate_with_strategy(
    trainer: MA2CTrainer,
    env: SumoTrafficEnv,
    *,
    strategy: str,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    att = []
    awt = []
    tput = []

    for episode in range(num_episodes):
        trainer.reset_aukfs()
        obs, _ = env.reset(seed=trainer.seed + 20000 + episode)

        for _ in range(steps_per_episode):
            actions = trainer.inference_policy(obs)
            if strategy == "fixed_preemption" and _scheduled_ev_active(env):
                actions = env.recommended_ev_actions(actions)
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        metrics = compute_metrics(
            completed_vehicles=int(env.stats.completed_vehicles),
            total_waiting_time=float(env.stats.total_waiting_time),
            total_travel_time=float(env.stats.total_travel_time),
            sim_seconds=steps_per_episode,
        )
        att.append(metrics["ATT"])
        awt.append(metrics["AWT"])
        tput.append(metrics["Throughput"])

    return {
        "strategy": strategy,
        "ATT": float(sum(att) / len(att)) if att else 0.0,
        "AWT": float(sum(awt) / len(awt)) if awt else 0.0,
        "Throughput": float(sum(tput) / len(tput)) if tput else 0.0,
        "ATT_runs": att,
        "AWT_runs": awt,
        "Throughput_runs": tput,
    }


def _train_or_load(
    trainer: MA2CTrainer,
    *,
    checkpoint_path: str,
    episodes: int,
    force_retrain: bool = False,
) -> None:
    ckpt = Path(checkpoint_path)
    if ckpt.exists() and not force_retrain:
        trainer.load_checkpoint(str(ckpt))
        return
    trainer.train(num_episodes=episodes, progress=False)
    trainer.save_checkpoint(str(ckpt))


def run_ev_comparison(
    config_path: str = "smartmarl/configs/default.yaml",
    seed: int = 0,
    train_episodes: int = 3000,
    eval_episodes: int = 5,
    force_retrain: bool = False,
) -> Dict[str, Dict]:
    """Run fair EV comparison with shared pre-trained policies.

    Policies:
    - no_preemption: policy trained without EV-specific reward, no override
    - fixed_preemption: same no-EV policy + deterministic green-wave override
    - learned_adaptive: EV-aware policy trained with EV reward, no manual override
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    steps_per_episode = int(cfg["episode_length_seconds"])

    env_noev = _make_env(cfg, seed)
    env_noev.set_reward_mode("normal")
    trainer_noev = MA2CTrainer(env=env_noev, config=cfg, ablation="no_ev", seed=seed)
    ckpt_noev = default_checkpoint_path(
        results_dir=cfg.get("results_dir", "results"),
        variant="ev_no_preemption_policy",
        scenario="standard",
        seed=seed,
    )
    _train_or_load(trainer_noev, checkpoint_path=ckpt_noev, episodes=max(500, int(train_episodes)), force_retrain=force_retrain)

    results = {
        "no_preemption": _evaluate_with_strategy(
            trainer_noev,
            env_noev,
            strategy="no_preemption",
            num_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
        ),
        "fixed_preemption": _evaluate_with_strategy(
            trainer_noev,
            env_noev,
            strategy="fixed_preemption",
            num_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
        ),
    }
    env_noev.close()

    env_ev = _make_env(cfg, seed)
    env_ev.set_reward_mode("ev")
    trainer_ev = MA2CTrainer(env=env_ev, config=cfg, ablation="full", seed=seed)
    ckpt_ev = default_checkpoint_path(
        results_dir=cfg.get("results_dir", "results"),
        variant="ev_learned_adaptive_policy",
        scenario="standard",
        seed=seed,
    )
    _train_or_load(trainer_ev, checkpoint_path=ckpt_ev, episodes=max(500, int(train_episodes)), force_retrain=force_retrain)
    results["learned_adaptive"] = _evaluate_with_strategy(
        trainer_ev,
        env_ev,
        strategy="learned_adaptive",
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
    )
    env_ev.close()

    return results


def run_ev_experiment(
    config_path: str = "smartmarl/configs/default.yaml",
    seed: int = 0,
    strategy: str = "learned_adaptive",
    episodes: int = 3000,
    eval_episodes: int = 5,
    force_retrain: bool = False,
) -> Dict:
    if strategy not in {"no_preemption", "fixed_preemption", "learned_adaptive"}:
        raise ValueError("strategy must be one of: no_preemption, fixed_preemption, learned_adaptive")

    results = run_ev_comparison(
        config_path=config_path,
        seed=seed,
        train_episodes=max(500, int(episodes)),
        eval_episodes=eval_episodes,
        force_retrain=force_retrain,
    )
    return results[strategy]


if __name__ == "__main__":
    summary = run_ev_comparison()
    for strategy_name in ("no_preemption", "fixed_preemption", "learned_adaptive"):
        m = summary[strategy_name]
        print(strategy_name, round(m["ATT"], 3), round(m["AWT"], 3), round(m["Throughput"], 3))

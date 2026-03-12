import numpy as np
import torch

from smartmarl.env.sumo_env import SumoTrafficEnv
from smartmarl.training.ma2c import MA2CTrainer


CFG = {
    "training_episodes_sumo": 10,
    "finetuning_episodes_cityflow": 2,
    "episode_length_seconds": 100,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "lr_decay_factor": 0.99,
    "lr_decay_every_n_episodes": 100,
    "discount_gamma": 0.95,
    "hetgnn_layers": 3,
    "embedding_dim": 128,
    "actor_heads": 2,
    "actor_head_dim": 64,
    "reward_weight_alpha": 0.6,
    "ev_weight_tev": 0.85,
    "ev_weight_penalty": 0.15,
    "aukf_beta": 0.02,
    "aukf_state_dim": 4,
    "aukf_obs_dim": 4,
    "yolo_confidence_threshold": 0.45,
    "min_green_time_seconds": 5,
    "num_seeds": 30,
    "seeds": list(range(30)),
    "grid_size": 5,
    "num_intersections": 25,
    "num_phases": 4,
    "target_latency_ms": 100,
    "power_budget_watts": 20,
    "mock_training_steps": 20,
}


def test_l7_keeps_aukf_lane_state_but_zeroes_vsens():
    env = SumoTrafficEnv(use_traci=False, seed=0)
    obs, _ = env.reset(seed=0)

    trainer = MA2CTrainer(env=env, config=CFG, ablation="l7", seed=0, device="cpu")
    node_features, _ = trainer.build_node_features(obs)

    assert torch.allclose(node_features["sens"], torch.zeros_like(node_features["sens"]))
    assert torch.any(node_features["lane"] != 0.0), "AUKF lane state unexpectedly zeroed in L7"

    raw_lane = torch.tensor(obs["lane_features"], dtype=torch.float32)
    assert not torch.allclose(node_features["lane"].cpu(), raw_lane), "Expected AUKF-updated lane state, got raw lane features"

    env.close()

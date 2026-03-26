import numpy as np

from smartmarl.env.graph_builder import GraphBuilder
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
    "gplight_num_groups": 4,
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


def test_graph_builder_relations_are_distinct():
    builder = GraphBuilder(grid_size=5, num_intersections=25)
    edge_dict = builder.build_edge_index_dict()
    flow_lane = edge_dict["flow_lane"].t().tolist()
    flow_sens = edge_dict["flow_sens"].t().tolist()
    incident = edge_dict["incident"].t().tolist()

    # Lane flow: directed downstream (right/down only).
    assert [0, 1] in flow_lane
    assert [0, 5] in flow_lane
    assert [1, 0] not in flow_lane
    assert [5, 0] not in flow_lane

    # Sensor flow: self-loops only.
    assert [0, 0] in flow_sens
    assert [0, 1] not in flow_sens
    assert len(flow_sens) == 25

    # Incident flow: broader than 1-hop neighbors (2-hop propagation).
    assert [0, 2] in incident
    assert [0, 10] in incident
    assert [0, 1] in incident


def test_gplight_variant_inference_runs():
    env = SumoTrafficEnv(use_traci=False, seed=0)
    obs, _ = env.reset(seed=0)

    trainer = MA2CTrainer(env=env, config=CFG, ablation="gplight", seed=0, device="cpu")
    actions = trainer.inference_policy(obs)

    assert actions.shape == (CFG["num_intersections"],)
    assert np.all(actions >= 0)
    assert np.all(actions < CFG["num_phases"])

    env.close()

"""MA2C trainer with CTDE for SmartMARL."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from smartmarl.env.graph_builder import GraphBuilder
from smartmarl.models.actor import GATv2Actor, MLPActor
from smartmarl.models.critic import CentralizedCritic
from smartmarl.models.hetgnn import HetGNN
from smartmarl.perception.aukf import AdaptiveUKF
from smartmarl.perception.hungarian import associate_detections
from smartmarl.perception.radar_processor import RadarProcessor
from smartmarl.perception.yolo_detector import YOLODetector
from smartmarl.training.replay_buffer import TrajectoryBuffer
from smartmarl.training.scheduler import EpisodeLRScheduler
from smartmarl.utils.metrics import compute_metrics


@dataclass
class VariantConfig:
    use_aukf: bool = True
    use_vsens: bool = True
    use_hetgnn: bool = True
    use_ctde: bool = True
    use_incident_nodes: bool = True
    use_ev_mode: bool = True
    detector_backbone: str = "yolov8n"
    actor_type: str = "gatv2"


def variant_from_name(name: str) -> VariantConfig:
    variants = {
        "full": VariantConfig(),
        "full_smartmarl": VariantConfig(),
        "no_ctde": VariantConfig(use_ctde=False),
        "no_aukf": VariantConfig(use_aukf=False, use_vsens=False),
        "no_hetgnn": VariantConfig(use_hetgnn=False),
        "no_incident": VariantConfig(use_incident_nodes=False),
        "no_incident_nodes": VariantConfig(use_incident_nodes=False),
        "no_ev": VariantConfig(use_ev_mode=False),
        "no_ev_mode": VariantConfig(use_ev_mode=False),
        "yolov5": VariantConfig(detector_backbone="yolov5_tiny"),
        "yolov5_backbone": VariantConfig(detector_backbone="yolov5_tiny"),
        "mlp": VariantConfig(actor_type="mlp"),
        "mlp_actor": VariantConfig(actor_type="mlp"),
        "l7": VariantConfig(use_aukf=True, use_vsens=False, use_hetgnn=True, use_ctde=True),
        "l7_ablation": VariantConfig(use_aukf=True, use_vsens=False, use_hetgnn=True, use_ctde=True),
    }
    if name not in variants:
        raise ValueError(f"Unknown ablation variant: {name}")
    return variants[name]


class HomogeneousSharedEncoder(nn.Module):
    """Ablation encoder: shared relation weights (homogeneous GAT-style encoder)."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj_int = nn.LazyLinear(hidden_dim)
        self.input_proj_lane = nn.LazyLinear(hidden_dim)
        self.input_proj_sens = nn.LazyLinear(hidden_dim)
        self.input_proj_inj = nn.LazyLinear(hidden_dim)

        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.shared_relation = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

    @staticmethod
    def _aggregate(messages: torch.Tensor, target_idx: torch.Tensor, num_targets: int) -> torch.Tensor:
        out = torch.zeros((num_targets, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
        count = torch.zeros((num_targets, 1), device=messages.device, dtype=messages.dtype)
        out.index_add_(0, target_idx, messages)
        one = torch.ones((target_idx.shape[0], 1), device=messages.device, dtype=messages.dtype)
        count.index_add_(0, target_idx, one)
        return out / count.clamp_min(1.0)

    def _relation(
        self,
        source_feat: torch.Tensor,
        edge_index: torch.Tensor,
        num_targets: int,
        projector: nn.Linear,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((num_targets, self.hidden_dim), device=source_feat.device, dtype=source_feat.dtype)
        s_idx, t_idx = edge_index[0], edge_index[1]
        msg = projector(source_feat[s_idx])
        return self._aggregate(msg, t_idx, num_targets)

    def forward(self, node_features: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        h_int = self.input_proj_int(node_features["int"])
        h_lane = self.input_proj_lane(node_features["lane"])
        h_sens = self.input_proj_sens(node_features["sens"])
        h_inj = self.input_proj_inj(node_features["inj"])

        for layer in range(self.num_layers):
            W = self.shared_relation[layer]
            spatial = self._relation(h_int, edge_index_dict["spatial"], h_int.shape[0], W)
            flow_l = self._relation(h_lane, edge_index_dict["flow_lane"], h_int.shape[0], W)
            flow_s = self._relation(h_sens, edge_index_dict["flow_sens"], h_int.shape[0], W)
            inc = self._relation(h_inj, edge_index_dict["incident"], h_int.shape[0], W)
            h_int = F.elu(self.self_linears[layer](h_int) + spatial + flow_l + flow_s + inc)

        return h_int


class MA2CTrainer:
    def __init__(
        self,
        env,
        config: Dict,
        ablation: str = "full",
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.env = env
        self.config = dict(config)
        self.seed = seed
        self.variant_name = ablation
        self.variant = variant_from_name(ablation)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)

        self.num_intersections = int(self.config["num_intersections"])
        self.num_phases = int(self.config["num_phases"])
        self.embedding_dim = int(self.config["embedding_dim"])
        self.gamma = float(self.config["discount_gamma"])

        self.graph_builder = GraphBuilder(
            grid_size=int(self.config["grid_size"]),
            num_intersections=self.num_intersections,
        )
        self.edge_index_dict = {
            k: v.to(self.device)
            for k, v in self.graph_builder.build_edge_index_dict(
                include_incident_nodes=self.variant.use_incident_nodes
            ).items()
        }
        self.intersection_positions = self._build_intersection_positions()

        if self.variant.use_hetgnn:
            self.encoder: nn.Module = HetGNN(
                hidden_dim=self.embedding_dim,
                num_layers=int(self.config["hetgnn_layers"]),
            )
        else:
            self.encoder = HomogeneousSharedEncoder(
                hidden_dim=self.embedding_dim,
                num_layers=int(self.config["hetgnn_layers"]),
            )

        if self.variant.actor_type == "mlp":
            self.actor: nn.Module = MLPActor(
                input_dim=self.embedding_dim,
                hidden_dim=64,
                num_phases=self.num_phases,
            )
        else:
            self.actor = GATv2Actor(
                input_dim=self.embedding_dim,
                num_heads=int(self.config["actor_heads"]),
                head_dim=int(self.config["actor_head_dim"]),
                num_phases=self.num_phases,
            )

        self.critic: Optional[nn.Module] = None
        if self.variant.use_ctde:
            self.critic = CentralizedCritic(
                num_agents=self.num_intersections,
                embed_dim=self.embedding_dim,
            )

        self.aukfs = [
            AdaptiveUKF(
                state_dim=int(self.config["aukf_state_dim"]),
                obs_dim=int(self.config["aukf_obs_dim"]),
                beta=float(self.config["aukf_beta"]),
            )
            for _ in range(self.num_intersections)
        ]
        self.detector = YOLODetector(
            backbone=self.variant.detector_backbone,
            confidence_threshold=float(self.config["yolo_confidence_threshold"]),
            condition="clear",
            seed=seed,
        )
        self.radar_processor = RadarProcessor(condition="clear", seed=seed)

        params = list(self.encoder.parameters()) + list(self.actor.parameters())
        if self.critic is not None:
            params += list(self.critic.parameters())

        self.optimizer = optim.Adam(params, lr=float(self.config["learning_rate"]))
        self.scheduler = EpisodeLRScheduler(
            optimizer=self.optimizer,
            decay_factor=float(self.config["lr_decay_factor"]),
            decay_every_n_episodes=int(self.config["lr_decay_every_n_episodes"]),
        )
        self.buffer = TrajectoryBuffer()

        self.encoder.to(self.device)
        self.actor.to(self.device)
        if self.critic is not None:
            self.critic.to(self.device)

    def reset_aukfs(self) -> None:
        for f in self.aukfs:
            f.reset()

    def _build_intersection_positions(self) -> np.ndarray:
        grid = int(self.config["grid_size"])
        spacing = 400.0
        coords = []
        for r in range(grid):
            for c in range(grid):
                coords.append((c * spacing, r * spacing))
        return np.asarray(coords, dtype=np.float32)

    def _nearest_intersection_index(self, x: float, y: float) -> int:
        delta = self.intersection_positions - np.array([x, y], dtype=np.float32)
        dist2 = np.sum(delta * delta, axis=1)
        return int(np.argmin(dist2))

    def _build_mock_sensor_measurements(self, obs: Dict) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(obs.get("vehicle_positions", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if positions.size == 0:
            z_cam = np.asarray(obs["sensor_measurements"]["camera"], dtype=np.float32)
            z_rad = np.asarray(obs["sensor_measurements"]["radar"], dtype=np.float32)
            return z_cam, z_rad

        cam_dets = self.detector.detect(positions)
        rad_dets = self.radar_processor.process(positions)
        assoc = associate_detections(cam_dets, rad_dets, max_distance=2.0)

        cam_counts = np.zeros(self.num_intersections, dtype=np.float32)
        rad_counts = np.zeros(self.num_intersections, dtype=np.float32)
        rad_speed_sum = np.zeros(self.num_intersections, dtype=np.float32)
        rad_speed_num = np.zeros(self.num_intersections, dtype=np.float32)

        for i_cam, i_rad, _ in assoc["matches"]:
            c = cam_dets[i_cam]
            r = rad_dets[i_rad]
            idx = self._nearest_intersection_index(c["x"], c["y"])
            cam_counts[idx] += 1.0
            rad_counts[idx] += 1.0
            rad_speed_sum[idx] += abs(float(r.get("velocity", 0.0)))
            rad_speed_num[idx] += 1.0

        for i in assoc["unmatched_camera"]:
            c = cam_dets[i]
            idx = self._nearest_intersection_index(c["x"], c["y"])
            cam_counts[idx] += 1.0

        for i in assoc["unmatched_radar"]:
            r = rad_dets[i]
            idx = self._nearest_intersection_index(r["x"], r["y"])
            rad_counts[idx] += 1.0
            rad_speed_sum[idx] += abs(float(r.get("velocity", 0.0)))
            rad_speed_num[idx] += 1.0

        lane_feat = np.asarray(obs["lane_features"], dtype=np.float32)
        cam_speed = lane_feat[:, 2].copy()
        rad_speed = np.divide(rad_speed_sum, np.maximum(rad_speed_num, 1.0))
        rad_speed = np.where(rad_speed_num > 0, np.clip(8.0 + rad_speed, 0.0, 30.0), lane_feat[:, 2])

        cam_queue = np.clip(0.7 * cam_counts + 0.3 * lane_feat[:, 0], 0.0, None)
        rad_queue = np.clip(0.6 * rad_counts + 0.4 * lane_feat[:, 0], 0.0, None)

        cam_occ = np.clip(cam_counts / 20.0, 0.0, 1.0)
        rad_occ = np.clip(rad_counts / 20.0, 0.0, 1.0)

        z_cam = np.stack([cam_queue, cam_counts, cam_speed, cam_occ], axis=1).astype(np.float32)
        z_rad = np.stack([rad_queue, rad_counts, rad_speed, rad_occ], axis=1).astype(np.float32)
        return z_cam, z_rad

    def build_node_features(self, obs: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        int_feat = torch.tensor(obs["intersection_features"], dtype=torch.float32, device=self.device)
        lane_obs = np.asarray(obs["lane_features"], dtype=np.float32)

        if self.variant.use_aukf:
            z_cam, z_rad = self._build_mock_sensor_measurements(obs)

            lane_states = []
            sigma2 = []
            for i in range(self.num_intersections):
                state, sig = self.aukfs[i].update(z_cam[i], z_rad[i])
                lane_states.append(state)
                sigma2.append(sig)

            lane_feat = torch.tensor(np.stack(lane_states), dtype=torch.float32, device=self.device)
            sigma2_feat = torch.tensor(np.stack(sigma2), dtype=torch.float32, device=self.device)
        else:
            lane_feat = torch.tensor(lane_obs, dtype=torch.float32, device=self.device)
            sigma2_feat = torch.zeros_like(lane_feat)

        if self.variant.use_vsens:
            sens_feat = sigma2_feat
        else:
            sens_feat = torch.zeros_like(sigma2_feat)

        inj = np.asarray(obs["incidents"], dtype=np.float32)
        if not self.variant.use_incident_nodes:
            inj = np.zeros_like(inj)

        inj_feat = torch.tensor(inj, dtype=torch.float32, device=self.device)

        node_features = {
            "int": int_feat,
            "lane": lane_feat,
            "sens": sens_feat,
            "inj": inj_feat,
        }
        aux = {
            "sigma2_r": sigma2_feat,
            "lane_state": lane_feat,
        }
        return node_features, aux

    def encode(self, node_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(node_features, self.edge_index_dict)

    def select_actions(self, h_int: torch.Tensor, deterministic: bool = False):
        probs = self.actor(h_int, self.edge_index_dict["spatial"])
        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            actions = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(1, actions.unsqueeze(-1)).clamp_min(1e-8)).sum()
        else:
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum()
        entropy = dist.entropy().mean()
        return actions, log_prob, entropy, probs

    def compute_rewards(self, obs: Dict, info: Dict) -> np.ndarray:
        q = np.asarray(obs["queue_per_intersection"], dtype=np.float32)
        d = np.asarray(obs["delay_per_intersection"], dtype=np.float32)

        if self.variant.use_ev_mode and bool(info.get("ev_active", False)):
            t_ev = float(info.get("ev_travel_time", np.mean(q)))
            p = float(info.get("network_penalty", np.mean(d)))
            reward = 0.85 * (-t_ev) + 0.15 * (-p)
            return np.full(self.num_intersections, reward, dtype=np.float32)

        alpha = float(self.config["reward_weight_alpha"])
        return -(alpha * q + (1.0 - alpha) * d)

    def _update_policy(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        returns = self.buffer.discounted_returns(self.gamma, self.device)
        log_probs = torch.stack(self.buffer.log_probs)
        entropies = torch.stack(self.buffer.entropies)

        if self.critic is not None:
            values = torch.cat(self.buffer.values, dim=0).squeeze(-1)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))
            critic_loss = F.smooth_l1_loss(values, returns)
        else:
            values = torch.zeros_like(returns)
            advantages = returns
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))
            critic_loss = torch.tensor(0.0, device=self.device)

        actor_loss = -(log_probs * advantages).mean() - 0.001 * entropies.mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters())
        if self.critic is not None:
            all_params += list(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(self.config.get("grad_clip_norm", 1.0)))
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "entropy": float(entropies.mean().detach().cpu()),
            "value_mean": float(values.mean().detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
        }

    def _resolved_steps_per_episode(self, steps_per_episode: Optional[int] = None) -> int:
        if steps_per_episode is not None:
            return int(steps_per_episode)

        if bool(getattr(self.env, "mock_mode", False)):
            return int(self.config.get("mock_training_steps", self.config["episode_length_seconds"]))

        return int(self.config["episode_length_seconds"])

    @staticmethod
    def _append_metrics_row(csv_path: Optional[str], row: Dict[str, float]) -> None:
        if not csv_path:
            return
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def train(
        self,
        num_episodes: int,
        progress: bool = True,
        start_episode: int = 0,
        steps_per_episode: Optional[int] = None,
        checkpoint_every: int = 0,
        checkpoint_path: Optional[str] = None,
        metrics_csv_path: Optional[str] = None,
    ) -> Dict[str, float]:
        self.encoder.train()
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        episode_att = []
        episode_awt = []
        episode_tp = []

        iterator = trange(
            start_episode,
            start_episode + num_episodes,
            disable=not progress,
            desc=f"train:{self.variant_name}",
        )
        max_steps = self._resolved_steps_per_episode(steps_per_episode)

        for episode in iterator:
            self.reset_aukfs()
            self.buffer.clear()
            episode_rewards: list[float] = []

            if self.variant.use_ev_mode:
                self.env.set_reward_mode("ev")
            else:
                self.env.set_reward_mode("normal")

            obs, _ = self.env.reset(seed=self.seed + episode)

            for _ in range(max_steps):
                node_features, _ = self.build_node_features(obs)
                h_int = self.encode(node_features)

                actions, log_prob, entropy, _ = self.select_actions(h_int, deterministic=False)
                next_obs, _env_reward, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

                reward_vec = self.compute_rewards(next_obs, info)
                global_reward = float(np.mean(reward_vec))
                episode_rewards.append(global_reward)

                if self.critic is not None:
                    value = self.critic(h_int.reshape(1, -1))
                else:
                    value = torch.zeros((1, 1), device=self.device)

                self.buffer.add_step(log_prob, entropy, global_reward, value, h_int.detach())
                obs = next_obs
                if terminated or truncated:
                    break

            losses = self._update_policy()
            self.scheduler.step(episode + 1)

            ep_metrics = compute_metrics(
                completed_vehicles=int(getattr(self.env.stats, "completed_vehicles", 0)),
                total_waiting_time=float(getattr(self.env.stats, "total_waiting_time", 0.0)),
                total_travel_time=float(getattr(self.env.stats, "total_travel_time", 0.0)),
                sim_seconds=max_steps,
            )

            episode_att.append(ep_metrics["ATT"])
            episode_awt.append(ep_metrics["AWT"])
            episode_tp.append(ep_metrics["Throughput"])
            reward_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0

            self._append_metrics_row(
                metrics_csv_path,
                {
                    "episode": int(episode + 1),
                    "backend": "mock" if bool(getattr(self.env, "mock_mode", True)) else "traci",
                    "att": float(ep_metrics["ATT"]),
                    "awt": float(ep_metrics["AWT"]),
                    "throughput": float(ep_metrics["Throughput"]),
                    "reward_mean": reward_mean,
                    "actor_loss": float(losses["actor_loss"]),
                    "critic_loss": float(losses["critic_loss"]),
                    "entropy": float(losses["entropy"]),
                },
            )

            if checkpoint_every > 0 and checkpoint_path and (episode + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)

            iterator.set_postfix(
                att=f"{ep_metrics['ATT']:.1f}",
                awt=f"{ep_metrics['AWT']:.1f}",
                tp=f"{ep_metrics['Throughput']:.1f}",
                aloss=f"{losses['actor_loss']:.3f}",
            )

        return {
            "ATT": float(np.mean(episode_att)) if episode_att else 0.0,
            "AWT": float(np.mean(episode_awt)) if episode_awt else 0.0,
            "Throughput": float(np.mean(episode_tp)) if episode_tp else 0.0,
        }

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 5, steps_per_episode: Optional[int] = None) -> Dict[str, float]:
        self.encoder.eval()
        self.actor.eval()

        att = []
        awt = []
        tput = []

        max_steps = self._resolved_steps_per_episode(steps_per_episode)

        for episode in range(num_episodes):
            self.reset_aukfs()
            obs, _ = self.env.reset(seed=self.seed + 10000 + episode)

            for _ in range(max_steps):
                node_features, _ = self.build_node_features(obs)
                h_int = self.encode(node_features)
                actions, *_ = self.select_actions(h_int, deterministic=True)
                obs, _, terminated, truncated, _ = self.env.step(actions.detach().cpu().numpy())
                if terminated or truncated:
                    break

            metrics = compute_metrics(
                completed_vehicles=int(getattr(self.env.stats, "completed_vehicles", 0)),
                total_waiting_time=float(getattr(self.env.stats, "total_waiting_time", 0.0)),
                total_travel_time=float(getattr(self.env.stats, "total_travel_time", 0.0)),
                sim_seconds=max_steps,
            )
            att.append(metrics["ATT"])
            awt.append(metrics["AWT"])
            tput.append(metrics["Throughput"])

        return {
            "ATT": float(np.mean(att)) if att else 0.0,
            "AWT": float(np.mean(awt)) if awt else 0.0,
            "Throughput": float(np.mean(tput)) if tput else 0.0,
            "ATT_runs": att,
            "AWT_runs": awt,
            "Throughput_runs": tput,
        }

    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            "variant": self.variant_name,
            "config": self.config,
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.critic is not None:
            ckpt["critic"] = self.critic.state_dict()

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(path_obj))

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        if self.critic is not None and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

    def inference_policy(self, obs: Dict) -> np.ndarray:
        """Decentralized inference path: critic is not used by design."""
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            node_features, _ = self.build_node_features(obs)
            h_int = self.encode(node_features)
            actions, *_ = self.select_actions(h_int, deterministic=True)
        return actions.detach().cpu().numpy()


def default_checkpoint_path(results_dir: str, variant: str, scenario: str, seed: int) -> str:
    filename = f"{scenario}_{variant}_seed{seed}.pt"
    return str(Path(results_dir) / "checkpoints" / filename)

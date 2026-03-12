"""Heterogeneous GNN encoder with relation-specific weights for SmartMARL."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
    return x


class HetGNN(nn.Module):
    """
    3-layer HetGNN.

    Equation:
      h_i^{l+1} = ELU(W_self h_i^l + sum_r mean_{j in N_r(i)} W_r h_j^l)
    Relations are explicitly non-shared: spatial, flow, incident.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj_int = nn.LazyLinear(hidden_dim)
        self.input_proj_lane = nn.LazyLinear(hidden_dim)
        self.input_proj_sens = nn.LazyLinear(hidden_dim)
        self.input_proj_inj = nn.LazyLinear(hidden_dim)

        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_layers)])
        self.w_spatial = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.w_flow = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.w_incident = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

    @staticmethod
    def _aggregate(messages: torch.Tensor, target_idx: torch.Tensor, num_targets: int) -> torch.Tensor:
        out = torch.zeros((num_targets, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
        counts = torch.zeros((num_targets, 1), device=messages.device, dtype=messages.dtype)
        out.index_add_(0, target_idx, messages)
        ones = torch.ones((target_idx.shape[0], 1), device=messages.device, dtype=messages.dtype)
        counts.index_add_(0, target_idx, ones)
        return out / counts.clamp_min(1.0)

    def _relation_mean(
        self,
        src_feat: torch.Tensor,
        edge_index: torch.Tensor,
        num_targets: int,
        projector: nn.Linear,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((num_targets, self.hidden_dim), device=src_feat.device, dtype=src_feat.dtype)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        msg = projector(src_feat[src_idx])
        return self._aggregate(msg, dst_idx, num_targets)

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        h_int = self.input_proj_int(_ensure_2d_tensor(node_features["int"]))
        h_lane = self.input_proj_lane(_ensure_2d_tensor(node_features["lane"]))
        h_sens = self.input_proj_sens(_ensure_2d_tensor(node_features["sens"]))
        h_inj = self.input_proj_inj(_ensure_2d_tensor(node_features["inj"]))

        spatial_e = edge_index_dict["spatial"].long().to(h_int.device)
        flow_lane_e = edge_index_dict["flow_lane"].long().to(h_int.device)
        flow_sens_e = edge_index_dict["flow_sens"].long().to(h_int.device)
        incident_e = edge_index_dict["incident"].long().to(h_int.device)

        for layer in range(self.num_layers):
            base = self.self_linears[layer](h_int)

            spatial_msg = self._relation_mean(h_int, spatial_e, h_int.shape[0], self.w_spatial[layer])
            flow_lane_msg = self._relation_mean(h_lane, flow_lane_e, h_int.shape[0], self.w_flow[layer])
            flow_sens_msg = self._relation_mean(h_sens, flow_sens_e, h_int.shape[0], self.w_flow[layer])
            incident_msg = self._relation_mean(h_inj, incident_e, h_int.shape[0], self.w_incident[layer])

            h_int = F.elu(base + spatial_msg + flow_lane_msg + flow_sens_msg + incident_msg)

        return h_int

    def relation_weight_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "spatial": self.w_spatial[0].weight,
            "flow": self.w_flow[0].weight,
            "incident": self.w_incident[0].weight,
        }

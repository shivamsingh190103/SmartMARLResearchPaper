"""Graph builder for SmartMARL heterogeneous traffic graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class GraphBuilder:
    grid_size: int = 5
    num_intersections: int = 25

    def __post_init__(self) -> None:
        expected = self.grid_size * self.grid_size
        if self.num_intersections != expected:
            raise ValueError(
                f"num_intersections ({self.num_intersections}) must equal grid_size^2 ({expected})"
            )

    def _spatial_edges(self) -> torch.Tensor:
        edges = []
        g = self.grid_size
        for r in range(g):
            for c in range(g):
                i = r * g + c
                if c + 1 < g:
                    j = r * g + (c + 1)
                    edges.append((i, j))
                    edges.append((j, i))
                if r + 1 < g:
                    j = (r + 1) * g + c
                    edges.append((i, j))
                    edges.append((j, i))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _identity_flow_edges(self) -> torch.Tensor:
        idx = torch.arange(self.num_intersections, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    def build_edge_index_dict(self, include_incident_nodes: bool = True) -> Dict[str, torch.Tensor]:
        spatial = self._spatial_edges()
        flow_lane = self._identity_flow_edges()
        flow_sens = self._identity_flow_edges()
        incident = self._identity_flow_edges() if include_incident_nodes else torch.empty((2, 0), dtype=torch.long)

        return {
            "spatial": spatial,
            "flow_lane": flow_lane,
            "flow_sens": flow_sens,
            "incident": incident,
        }

    def node_counts(self) -> Dict[str, int]:
        n = self.num_intersections
        return {"int": n, "lane": n, "sens": n, "inj": n}

    def relation_summary(self) -> Dict[str, Tuple[int, int]]:
        edge_dict = self.build_edge_index_dict(include_incident_nodes=True)
        return {k: tuple(v.shape) for k, v in edge_dict.items()}

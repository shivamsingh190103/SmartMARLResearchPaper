"""Graph builder for SmartMARL heterogeneous traffic graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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

    def _neighbor_indices(self, index: int) -> List[int]:
        g = self.grid_size
        r, c = divmod(index, g)
        out: List[int] = []
        if c > 0:
            out.append(index - 1)
        if c + 1 < g:
            out.append(index + 1)
        if r > 0:
            out.append(index - g)
        if r + 1 < g:
            out.append(index + g)
        return out

    def _corridor_edges(self, include_self: bool = True) -> torch.Tensor:
        edges = []
        for idx in range(self.num_intersections):
            if include_self:
                edges.append((idx, idx))
            for nbr in self._neighbor_indices(idx):
                edges.append((idx, nbr))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _incident_edges(self) -> torch.Tensor:
        edges = []
        for idx in range(self.num_intersections):
            edges.append((idx, idx))
            for nbr in self._neighbor_indices(idx):
                edges.append((idx, nbr))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_edge_index_dict(self, include_incident_nodes: bool = True) -> Dict[str, torch.Tensor]:
        spatial = self._spatial_edges()
        flow_lane = self._corridor_edges(include_self=True)
        flow_sens = self._corridor_edges(include_self=True)
        incident = self._incident_edges() if include_incident_nodes else torch.empty((2, 0), dtype=torch.long)

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

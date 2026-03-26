"""Extended tests for GraphBuilder: node_counts, relation_summary, and error cases."""

from __future__ import annotations

import torch
import pytest

from smartmarl.env.graph_builder import GraphBuilder


class TestGraphBuilderExtended:
    def test_node_counts_correct(self):
        builder = GraphBuilder(grid_size=5, num_intersections=25)
        counts = builder.node_counts()
        assert counts == {"int": 25, "lane": 25, "sens": 25, "inj": 25}

    def test_node_counts_small_grid(self):
        builder = GraphBuilder(grid_size=2, num_intersections=4)
        counts = builder.node_counts()
        assert all(v == 4 for v in counts.values())

    def test_relation_summary_returns_shape_tuples(self):
        builder = GraphBuilder(grid_size=5, num_intersections=25)
        summary = builder.relation_summary()
        assert set(summary.keys()) == {"spatial", "flow_lane", "flow_sens", "incident"}
        for key, shape in summary.items():
            assert isinstance(shape, tuple)
            assert len(shape) == 2

    def test_relation_summary_spatial_shape(self):
        # A 5×5 grid has 2*(5*4 + 5*4)=80 bidirectional edges
        builder = GraphBuilder(grid_size=5, num_intersections=25)
        summary = builder.relation_summary()
        assert summary["spatial"][0] == 2  # shape[0] is always 2 for edge_index
        assert summary["spatial"][1] == 2 * (5 * 4 + 5 * 4)  # 80 directed edges

    def test_flow_lane_is_directed_downstream_only(self):
        builder = GraphBuilder(grid_size=3, num_intersections=9)
        edges = builder.build_edge_index_dict()["flow_lane"]
        pairs = edges.t().tolist()
        # In a 3x3 grid: 3*(3-1) + 3*(3-1) = 12 downstream edges
        assert len(pairs) == 12
        # Comprehensively verify no upstream (reversed) edges exist
        pair_set = set(map(tuple, pairs))
        for src, dst in list(pair_set):
            assert (dst, src) not in pair_set, (
                f"Found upstream edge ({dst}, {src}) in flow_lane which should be downstream-only"
            )

    def test_sensor_self_edges_count(self):
        for g in [2, 3, 5]:
            n = g * g
            builder = GraphBuilder(grid_size=g, num_intersections=n)
            edges = builder.build_edge_index_dict()["flow_sens"]
            pairs = edges.t().tolist()
            assert len(pairs) == n, f"Expected {n} self-loops for grid {g}x{g}"
            for i in range(n):
                assert [i, i] in pairs

    def test_incident_edges_include_self_loop(self):
        builder = GraphBuilder(grid_size=5, num_intersections=25)
        edges = builder.build_edge_index_dict()["incident"]
        pairs = edges.t().tolist()
        # Each node should link to itself (manhattan distance 0)
        for i in range(25):
            assert [i, i] in pairs

    def test_mismatched_grid_size_raises(self):
        with pytest.raises(ValueError, match="must equal grid_size"):
            GraphBuilder(grid_size=5, num_intersections=20)

    def test_no_incident_flag_returns_empty(self):
        builder = GraphBuilder(grid_size=3, num_intersections=9)
        edge_dict = builder.build_edge_index_dict(include_incident_nodes=False)
        assert edge_dict["incident"].numel() == 0

    def test_spatial_edges_are_bidirectional(self):
        builder = GraphBuilder(grid_size=3, num_intersections=9)
        edges = builder.build_edge_index_dict()["spatial"]
        pairs = set(map(tuple, edges.t().tolist()))
        for src, dst in list(pairs):
            assert (dst, src) in pairs, f"Missing reverse edge for ({src}, {dst})"

    def test_two_hop_incident_reaches_correct_nodes(self):
        builder = GraphBuilder(grid_size=5, num_intersections=25)
        edges = builder.build_edge_index_dict()["incident"]
        # Node 0 (top-left corner) should reach nodes within manhattan distance 2
        from_zero = [dst for src, dst in edges.t().tolist() if src == 0]
        # node 0 can reach: 0,1,2,5,6,10 (manhattan <=2)
        assert 0 in from_zero
        assert 1 in from_zero
        assert 2 in from_zero
        assert 5 in from_zero
        assert 10 in from_zero
        # node 3 is at (0,3), manhattan from (0,0) = 3, should NOT be reachable
        assert 3 not in from_zero

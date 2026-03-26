"""Extended model tests: CentralizedCritic, MLPActor, and GPLight components."""

from __future__ import annotations

import torch
import pytest
import numpy as np

from smartmarl.baselines.gplight import DynamicGroupAssigner, GPLightActor, GPLightEncoder
from smartmarl.models.actor import MLPActor
from smartmarl.models.critic import CentralizedCritic

# Threshold below which a softmax output is considered "essentially zero"
# (i.e., the phase was effectively masked out with -1e9 logit)
PROBABILITY_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# CentralizedCritic
# ---------------------------------------------------------------------------

class TestCentralizedCritic:
    def test_output_shape_2d_input(self):
        critic = CentralizedCritic(num_agents=25, embed_dim=128)
        h = torch.randn(25, 128)
        out = critic(h)
        assert out.shape == (1, 1)

    def test_output_shape_3d_input_batch(self):
        critic = CentralizedCritic(num_agents=4, embed_dim=64)
        h = torch.randn(8, 4, 64)  # batch of 8
        out = critic(h)
        assert out.shape == (8, 1)

    def test_output_is_scalar_per_sample(self):
        critic = CentralizedCritic(num_agents=5, embed_dim=32)
        h = torch.randn(5, 32)
        out = critic(h)
        assert out.numel() == 1

    def test_output_changes_with_different_input(self):
        critic = CentralizedCritic(num_agents=4, embed_dim=16)
        h1 = torch.zeros(4, 16)
        h2 = torch.ones(4, 16)
        out1 = critic(h1)
        out2 = critic(h2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_through_critic(self):
        critic = CentralizedCritic(num_agents=4, embed_dim=16)
        h = torch.randn(4, 16, requires_grad=True)
        out = critic(h)
        out.sum().backward()
        assert h.grad is not None


# ---------------------------------------------------------------------------
# MLPActor
# ---------------------------------------------------------------------------

class TestMLPActor:
    def test_output_shape_and_sums_to_one(self):
        actor = MLPActor(input_dim=128, hidden_dim=64, num_phases=4)
        h = torch.randn(5, 128)
        probs = actor(h)
        assert probs.shape == (5, 4)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_feasible_mask_zeroes_out_infeasible(self):
        actor = MLPActor(input_dim=64, hidden_dim=32, num_phases=4)
        h = torch.randn(3, 64)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
        probs = actor(h, feasible_mask=mask)
        # Masked-out phases should have essentially zero probability
        assert torch.all(probs[0, 2:] < PROBABILITY_EPSILON)
        assert torch.all(probs[1, 1] < PROBABILITY_EPSILON) and torch.all(probs[1, 3] < PROBABILITY_EPSILON)

    def test_extra_args_ignored(self):
        # MLPActor ignores extra positional args (edge_index) and kwargs
        actor = MLPActor(input_dim=64, hidden_dim=32, num_phases=4)
        h = torch.randn(2, 64)
        edge_index = torch.randint(0, 2, (2, 4))
        probs = actor(h, edge_index)
        assert probs.shape == (2, 4)

    def test_probabilities_non_negative(self):
        actor = MLPActor(input_dim=32, hidden_dim=16, num_phases=3)
        h = torch.randn(10, 32)
        probs = actor(h)
        assert torch.all(probs >= 0.0)

    def test_gradient_flows(self):
        actor = MLPActor(input_dim=32, hidden_dim=16, num_phases=4)
        h = torch.randn(4, 32, requires_grad=True)
        probs = actor(h)
        probs.sum().backward()
        assert h.grad is not None


# ---------------------------------------------------------------------------
# DynamicGroupAssigner
# ---------------------------------------------------------------------------

class TestDynamicGroupAssigner:
    def _default_positions(self, n=25):
        g = int(n ** 0.5)
        positions = []
        for r in range(g):
            for c in range(g):
                positions.append([float(c), float(r)])
        return np.array(positions, dtype=np.float32)

    def test_output_shape(self):
        pos = self._default_positions(25)
        assigner = DynamicGroupAssigner(positions=pos, num_groups=4)
        features = np.random.default_rng(0).random((25, 3)).astype(np.float32)
        labels = assigner.assign(features)
        assert labels.shape == (25,)

    def test_labels_in_valid_range(self):
        pos = self._default_positions(25)
        assigner = DynamicGroupAssigner(positions=pos, num_groups=4)
        features = np.random.default_rng(0).random((25, 3)).astype(np.float32)
        labels = assigner.assign(features)
        assert np.all(labels >= 0)
        assert np.all(labels < 4)

    def test_single_group(self):
        pos = self._default_positions(4)
        assigner = DynamicGroupAssigner(positions=pos, num_groups=1)
        features = np.random.default_rng(0).random((4, 2)).astype(np.float32)
        labels = assigner.assign(features)
        assert np.all(labels == 0)

    def test_1d_features_raises(self):
        pos = self._default_positions(4)
        assigner = DynamicGroupAssigner(positions=pos, num_groups=2)
        with pytest.raises(ValueError):
            assigner.assign(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_num_groups_capped_at_n(self):
        pos = self._default_positions(4)
        assigner = DynamicGroupAssigner(positions=pos, num_groups=100)
        features = np.random.default_rng(0).random((4, 2)).astype(np.float32)
        labels = assigner.assign(features)
        assert labels.shape == (4,)


# ---------------------------------------------------------------------------
# GPLightEncoder
# ---------------------------------------------------------------------------

class TestGPLightEncoder:
    def test_output_shape(self):
        encoder = GPLightEncoder(hidden_dim=64, num_layers=2)
        x = torch.randn(10, 8)
        edge_index = torch.randint(0, 10, (2, 20))
        out = encoder(x, edge_index)
        assert out.shape == (10, 64)

    def test_empty_edge_index(self):
        encoder = GPLightEncoder(hidden_dim=32, num_layers=2)
        x = torch.randn(5, 4)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = encoder(x, edge_index)
        assert out.shape == (5, 32)

    def test_output_changes_with_different_input(self):
        encoder = GPLightEncoder(hidden_dim=32, num_layers=2)
        edge_index = torch.randint(0, 4, (2, 8))
        x1 = torch.zeros(4, 8)
        x2 = torch.ones(4, 8)
        out1 = encoder(x1, edge_index)
        out2 = encoder(x2, edge_index)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# GPLightActor
# ---------------------------------------------------------------------------

class TestGPLightActor:
    def test_output_shape_and_probs_sum_to_one(self):
        actor = GPLightActor(input_dim=64, num_phases=4, num_groups=3)
        h = torch.randn(10, 64)
        group_ids = torch.randint(0, 3, (10,))
        probs = actor(h, group_ids=group_ids)
        assert probs.shape == (10, 4)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_feasible_mask_applied(self):
        actor = GPLightActor(input_dim=64, num_phases=4, num_groups=2)
        h = torch.randn(4, 64)
        group_ids = torch.zeros(4, dtype=torch.long)
        mask = torch.tensor([[1, 1, 0, 0]] * 4, dtype=torch.bool)
        probs = actor(h, group_ids=group_ids, feasible_mask=mask)
        assert torch.all(probs[:, 2:] < PROBABILITY_EPSILON)

    def test_probabilities_non_negative(self):
        actor = GPLightActor(input_dim=32, num_phases=3, num_groups=2)
        h = torch.randn(6, 32)
        group_ids = torch.randint(0, 2, (6,))
        probs = actor(h, group_ids=group_ids)
        assert torch.all(probs >= 0.0)

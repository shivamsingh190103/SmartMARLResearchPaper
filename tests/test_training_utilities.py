"""Tests for training utilities: TrajectoryBuffer and EpisodeLRScheduler."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch
import pytest

from smartmarl.training.replay_buffer import TrajectoryBuffer
from smartmarl.training.scheduler import EpisodeLRScheduler


# ---------------------------------------------------------------------------
# TrajectoryBuffer
# ---------------------------------------------------------------------------

class TestTrajectoryBuffer:
    def _make_step(self, reward: float = 1.0):
        log_prob = torch.tensor(-0.5)
        entropy = torch.tensor(0.3)
        value = torch.tensor(0.8)
        embedding = torch.randn(128)
        return log_prob, entropy, reward, value, embedding

    def test_starts_empty(self):
        buf = TrajectoryBuffer()
        assert len(buf) == 0

    def test_add_step_increments_length(self):
        buf = TrajectoryBuffer()
        buf.add_step(*self._make_step())
        assert len(buf) == 1
        buf.add_step(*self._make_step())
        assert len(buf) == 2

    def test_rewards_stored_correctly(self):
        buf = TrajectoryBuffer()
        buf.add_step(*self._make_step(reward=3.14))
        assert buf.rewards[0] == pytest.approx(3.14)

    def test_clear_empties_buffer(self):
        buf = TrajectoryBuffer()
        for _ in range(5):
            buf.add_step(*self._make_step())
        buf.clear()
        assert len(buf) == 0
        assert buf.log_probs == []
        assert buf.entropies == []
        assert buf.rewards == []
        assert buf.values == []
        assert buf.embeddings == []

    def test_discounted_returns_zero_gamma(self):
        buf = TrajectoryBuffer()
        rewards = [1.0, 2.0, 3.0]
        for r in rewards:
            buf.add_step(*self._make_step(reward=r))
        returns = buf.discounted_returns(gamma=0.0, device=torch.device("cpu"))
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(returns, expected)

    def test_discounted_returns_one_gamma(self):
        buf = TrajectoryBuffer()
        rewards = [1.0, 1.0, 1.0]
        for r in rewards:
            buf.add_step(*self._make_step(reward=r))
        returns = buf.discounted_returns(gamma=1.0, device=torch.device("cpu"))
        # With gamma=1: G_0 = 3, G_1 = 2, G_2 = 1
        expected = torch.tensor([3.0, 2.0, 1.0])
        torch.testing.assert_close(returns, expected)

    def test_discounted_returns_typical_gamma(self):
        buf = TrajectoryBuffer()
        gamma = 0.9
        rewards = [1.0, 0.0, 2.0]
        for r in rewards:
            buf.add_step(*self._make_step(reward=r))
        returns = buf.discounted_returns(gamma=gamma, device=torch.device("cpu"))
        g2 = 2.0
        g1 = 0.0 + gamma * g2
        g0 = 1.0 + gamma * g1
        expected = torch.tensor([g0, g1, g2], dtype=torch.float32)
        torch.testing.assert_close(returns, expected, atol=1e-5, rtol=1e-5)

    def test_discounted_returns_single_step(self):
        buf = TrajectoryBuffer()
        buf.add_step(*self._make_step(reward=5.0))
        returns = buf.discounted_returns(gamma=0.99, device=torch.device("cpu"))
        assert returns.shape == (1,)
        assert float(returns[0]) == pytest.approx(5.0)

    def test_embeddings_are_stored(self):
        buf = TrajectoryBuffer()
        emb = torch.randn(128)
        buf.add_step(torch.tensor(-1.0), torch.tensor(0.5), 2.0, torch.tensor(1.0), emb)
        torch.testing.assert_close(buf.embeddings[0], emb)


# ---------------------------------------------------------------------------
# EpisodeLRScheduler
# ---------------------------------------------------------------------------

class TestEpisodeLRScheduler:
    def _make_optimizer(self, lr: float = 1e-3) -> MagicMock:
        group = {"lr": lr}
        opt = MagicMock()
        opt.param_groups = [group]
        return opt

    def test_step_at_zero_does_nothing(self):
        opt = self._make_optimizer(1e-3)
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.99, decay_every_n_episodes=100)
        sched.step(0)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_step_at_non_multiple_does_nothing(self):
        opt = self._make_optimizer(1e-3)
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.5, decay_every_n_episodes=100)
        sched.step(50)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_step_at_multiple_decays_lr(self):
        lr = 1e-3
        opt = self._make_optimizer(lr)
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.9, decay_every_n_episodes=100)
        sched.step(100)
        assert opt.param_groups[0]["lr"] == pytest.approx(lr * 0.9)

    def test_multiple_decays_compound(self):
        lr = 1e-3
        opt = self._make_optimizer(lr)
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.9, decay_every_n_episodes=100)
        sched.step(100)
        sched.step(200)
        assert opt.param_groups[0]["lr"] == pytest.approx(lr * 0.9 * 0.9)

    def test_multiple_param_groups_all_decayed(self):
        opt = MagicMock()
        opt.param_groups = [{"lr": 1e-3}, {"lr": 2e-3}]
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.5, decay_every_n_episodes=10)
        sched.step(10)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
        assert opt.param_groups[1]["lr"] == pytest.approx(1e-3)

    def test_negative_episode_does_nothing(self):
        opt = self._make_optimizer(1e-3)
        sched = EpisodeLRScheduler(optimizer=opt, decay_factor=0.5, decay_every_n_episodes=100)
        sched.step(-100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

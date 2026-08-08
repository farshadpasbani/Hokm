"""
Tests for Monte-Carlo return storage (config.NFSPConfig.mc_returns).
"""

import math
import random

import pytest
import torch

from config import HokmConfig
from enhanced_player import EnhancedPlayer
from game_constants import STATE_DIM


def _mk_player(**kw):
    p = EnhancedPlayer("MC", mc_returns=True, **kw)
    p.learning_enabled = True
    return p


def _push(p, reward, done):
    s = torch.zeros(STATE_DIM)
    p.store_experience(s, 3, reward, s, done, rl_eligible=True)


class TestMcReturnMath:
    def test_returns_to_go_and_terminal_done(self):
        p = _mk_player()
        gamma = p.gamma
        _push(p, 0.0, False)
        _push(p, 0.0, False)
        assert len(p.memory) == 0, "transitions must be buffered until done"
        _push(p, 1.0, True)
        assert len(p.memory) == 3
        stored = [(r, d) for (_, _, r, _, d, _) in p.memory.memory]
        expected = [gamma**2, gamma, 1.0]
        for (r, d), e in zip(stored, expected):
            assert d is True
            assert math.isclose(r, e, rel_tol=1e-6)

    def test_interleaved_rewards(self):
        p = _mk_player()
        g = p.gamma
        rewards = [0.5, -0.25, 2.0]
        for r in rewards[:-1]:
            _push(p, r, False)
        _push(p, rewards[-1], True)
        # G_t computed backward: G_2 = 2.0, G_1 = -.25 + g*2, G_0 = .5 + g*G_1
        g2 = 2.0
        g1 = -0.25 + g * g2
        g0 = 0.5 + g * g1
        stored = [r for (_, _, r, _, _, _) in p.memory.memory]
        for got, exp in zip(stored, [g0, g1, g2]):
            assert math.isclose(got, exp, rel_tol=1e-6)

    def test_reset_drops_pending(self):
        p = _mk_player()
        _push(p, 1.0, False)
        p.reset()
        _push(p, 1.0, True)
        assert len(p.memory) == 1, "aborted-hand transitions must not leak"

    def test_off_by_default(self):
        q = EnhancedPlayer("plain")
        q.learning_enabled = True
        _push(q, 0.0, False)
        assert len(q.memory) == 1, "default path must store immediately"


class TestMcIntegration:
    def test_full_hand_stores_only_terminal_targets(self):
        from train_backend import TrainBackend

        cfg = HokmConfig(seed=99, num_games=3, model_save_interval=10_000)
        cfg.nfsp.mc_returns = True
        tb = TrainBackend(num_games=3, model_save_interval=10_000, config=cfg)
        tb.train(log_fn=lambda m: None)
        mem = tb.shared_learner.memory.memory
        assert len(mem) > 0
        assert all(entry[4] for entry in mem), "all MC transitions must be done=True"
        # Returns must be bounded by |terminal| + shaping ~ O(2.3) with defaults
        assert all(abs(entry[2]) < 5.0 for entry in mem)

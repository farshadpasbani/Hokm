"""
Tests for the Deep Monte-Carlo stack (dmc.py / dmc_train.py).
"""

import random

import torch

from baselines import RandomAgent
from dmc import CentralCritic, DMCNet, DMCPlayer, history_features
from dmc_train import (
    DMCTrainer,
    _aux_targets,
    _full_info_vec,
    play_episode,
    train_single,
)
from game_constants import STATE_DIM, card_to_index
from hokm import Hokm


class TestNetShapes:
    def test_forward_shapes(self):
        net = DMCNet()
        b, t = 5, 9
        q, v, aux = net(
            torch.zeros(b, STATE_DIM),
            torch.zeros(b, t, dtype=torch.long),
            torch.zeros(b, t, dtype=torch.long),
            torch.tensor([0, 3, 9, 1, 9]),
            torch.tensor([0, 5, 51, 13, 26]),
        )
        assert q.shape == (b,)
        assert v.shape == (b,)
        assert aux.shape == (b, 156)

    def test_q_values_single_decision(self):
        net = DMCNet()
        q = net.q_values(torch.zeros(STATE_DIM), [1, 2, 3], [0, 1, 2], [7, 8])
        assert q.shape == (2,)
        q_empty = net.q_values(torch.zeros(STATE_DIM), [], [], [7, 8, 9])
        assert q_empty.shape == (3,)

    def test_critic_shape(self):
        c = CentralCritic()
        out = c(torch.zeros(4, CentralCritic.IN_DIM))
        assert out.shape == (4,)


class TestFeatureExtraction:
    def _mid_game(self, seed=3):
        players = [RandomAgent(f"P{i}", rng=random.Random(i)) for i in range(4)]
        g = Hokm(players, minimal_logging=True, rng=random.Random(seed))
        g.start_game()
        g.choose_trump_suit()
        # play two full tricks + one card
        plays = 0
        while plays < 9:
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            assert g.apply_play(nxt, card) is None
            plays += 1
        return g

    def test_history_matches_play_log(self):
        g = self._mid_game()
        hist_c, hist_s = history_features(g, 0)
        assert len(hist_c) == 9
        assert hist_c == [card_to_index(c) for _, c in g.play_log_this_hand]
        assert hist_s == [s % 4 for s, _ in g.play_log_this_hand]
        # relative to seat 2, seats shift by -2 mod 4
        _, hist_s2 = history_features(g, 2)
        assert hist_s2 == [(s - 2) % 4 for s, _ in g.play_log_this_hand]

    def test_full_info_contains_all_hands(self):
        g = self._mid_game()
        x = _full_info_vec(g, 0)
        total_cards = sum(len(p.hand) for p in g.players)
        assert int(x[: 4 * 52].sum().item()) == total_cards
        # my block matches my hand exactly
        mine = {card_to_index(c) for c in g.players[0].hand}
        got = {i for i in range(52) if x[i] > 0}
        assert got == mine

    def test_aux_targets_label_hidden_cards(self):
        g = self._mid_game()
        target, mask = _aux_targets(g, 0)
        # every hidden card is held by exactly one opponent in the labels
        hidden = mask[:52].bool()
        for idx in range(52):
            if hidden[idx]:
                holders = sum(int(target[j * 52 + idx].item()) for j in range(3))
                assert holders == 1, f"card {idx} held by {holders} opponents"
        # nothing in my own hand is inside the mask
        for c in g.players[0].hand:
            assert not hidden[card_to_index(c)]


class TestEpisodeAndTraining:
    def test_play_episode_returns_and_decisions(self):
        net = DMCNet()
        rng = random.Random(0)
        opps = [RandomAgent("E", rng=random.Random(1)),
                RandomAgent("W", rng=random.Random(2))]
        decisions, won, diff = play_episode(net, opps, rng, epsilon=0.5)
        assert decisions, "learner seats must have made decisions"
        rets = {d.ret for d in decisions}
        assert len(rets) == 1, "all decisions in a hand share the return"
        expected_sign = 1.0 if won else -1.0
        assert next(iter(rets)) * expected_sign > 0

    def test_smoke_train_improves_nothing_but_runs(self):
        tr = train_single(30, seed=1, out=None, log_every=0, steps_per_game=1)
        assert tr.games == 30
        assert len(tr.buffer) > 100
        assert tr.q_losses, "gradient steps must have happened"

    def test_checkpoint_roundtrip_and_player(self, tmp_path):
        tr = train_single(5, seed=2, out=None, log_every=0, steps_per_game=1)
        path = str(tmp_path / "dmc_test.pt")
        tr.save(path)
        p = DMCPlayer("D", checkpoint=path)
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm([p] + others, minimal_logging=True, rng=random.Random(4))
        g.start_game()
        g.choose_trump_suit()
        for _ in range(220):
            if g.scores[1] >= 7 or g.scores[2] >= 7:
                break
            if all(len(pl.hand) == 0 for pl in g.players):
                break
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            assert g.apply_play(nxt, card) is None
        assert g.scores[1] + g.scores[2] > 0

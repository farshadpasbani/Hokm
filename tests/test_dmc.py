"""
Tests for the Deep Monte-Carlo stack (dmc.py / dmc_train.py).
"""

import random
from types import SimpleNamespace

import pytest
import torch

from baselines import RandomAgent
from dmc import (
    MAX_HISTORY,
    STATE_ENC,
    CentralCritic,
    DMCNet,
    DMCPlayer,
    history_features,
)
from dmc_train import (
    DMCTrainer,
    _aux_targets,
    _full_info_vec,
    play_episode,
    train_single,
)
from game_constants import Card, STATE_DIM, card_to_index
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


def _fixed_net():
    """A net with pinned weights, so every assertion below is on values."""
    torch.manual_seed(20260830)
    return DMCNet()


class TestValueComposition:
    """The dueling head composes as Q(s,a) = V(s) + A(s,a).

    Shape and dtype assertions cannot see the sign: Q = V - A produces
    exactly the same tensor shapes. These compare against terms rebuilt
    from the net's own sub-modules.
    """

    STATIC = torch.linspace(-1.0, 1.0, STATE_DIM)
    HIST = ([3, 17], [1, 2])

    def _encoded(self, net):
        cards, seats = self.HIST
        return net.encode(
            self.STATIC.unsqueeze(0), torch.tensor([cards]),
            torch.tensor([seats]), torch.tensor([len(cards)]),
        )

    def test_q_is_value_plus_advantage(self):
        net = _fixed_net()
        enc = self._encoded(net)
        action = torch.tensor([41])

        v = net.v_head(enc).squeeze(-1)
        a = net.adv_trunk(
            torch.cat([enc, net.card_emb(action)], dim=-1)
        ).squeeze(-1)
        q = net.q_from_encoding(enc, action)

        assert torch.allclose(q, v + a, atol=1e-6)
        # Guard the guard: if A were ~0 then V+A and V-A would agree and the
        # assertion above would hold for either sign.
        assert a.abs().item() > 1e-3
        assert not torch.allclose(q, v - a, atol=1e-4)
        # forward() (the training path) must report that same Q, and Q must
        # not collapse onto V.
        cards, seats = self.HIST
        fq, fv, _ = net(
            self.STATIC.unsqueeze(0), torch.tensor([cards]),
            torch.tensor([seats]), torch.tensor([len(cards)]), action,
        )
        assert torch.allclose(fq, q) and torch.allclose(fv, v)
        assert not torch.allclose(fq, fv, atol=1e-4)

    def test_q_values_ranks_candidates_by_advantage(self):
        """`q_values` (the inference path) shares one state encoding across
        candidates, so what separates them is the action term alone."""
        net = _fixed_net()
        cands = [0, 13, 26, 39]
        q = net.q_values(self.STATIC, *self.HIST, cands)
        enc = self._encoded(net)
        expected = torch.stack([
            net.q_from_encoding(enc, torch.tensor([c]))[0] for c in cands
        ])
        assert torch.allclose(q, expected, atol=1e-6)
        assert q.std().item() > 1e-4, "candidates must be distinguishable"


class TestHistoryEncoding:
    """The GRU branch of `DMCNet.encode`, pinned by value."""

    def test_gru_consumes_the_time_axis(self):
        """`nn.GRU(..., batch_first=True)` means the input is [B, T, F].

        Dropping `batch_first` leaves every shape in `encode` valid for a
        single-row batch, so only a value check catches it. The reference
        feeds the identical events one step at a time: a [1, 1, F] slice
        means the same thing under either axis convention.
        """
        net = _fixed_net()
        hc = torch.tensor([[7, 19, 33]])
        hs = torch.tensor([[1, 2, 3]])
        enc = net.encode(
            torch.zeros(1, STATE_DIM), hc, hs, torch.tensor([3])
        )
        got = enc[:, STATE_ENC:]

        ev = torch.cat([net.card_emb(hc), net.seat_emb(hs)], dim=-1)
        state = None
        for t in range(hc.shape[1]):
            step, state = net.gru(ev[:, t:t + 1, :], state)
        assert torch.allclose(got, step[:, 0], atol=1e-6)

    def test_event_order_changes_the_encoding(self):
        """A sequence encoder that ignored order would still pass the
        shape tests; reversing the history must move the hidden state."""
        net = _fixed_net()
        static = torch.zeros(1, STATE_DIM)
        fwd = net.encode(
            static, torch.tensor([[7, 19, 33]]), torch.tensor([[1, 2, 3]]),
            torch.tensor([3]),
        )
        rev = net.encode(
            static, torch.tensor([[33, 19, 7]]), torch.tensor([[3, 2, 1]]),
            torch.tensor([3]),
        )
        assert not torch.allclose(fwd, rev, atol=1e-4)

    def test_length_one_history_reads_the_first_step(self):
        """Boundary: `(hist_lens - 1).clamp(min=0)` must select index 0."""
        net = _fixed_net()
        hc = torch.tensor([[11]])
        hs = torch.tensor([[2]])
        enc = net.encode(
            torch.zeros(1, STATE_DIM), hc, hs, torch.tensor([1])
        )
        got = enc[:, STATE_ENC:]
        ev = torch.cat([net.card_emb(hc), net.seat_emb(hs)], dim=-1)
        out, _ = net.gru(ev)
        assert torch.allclose(got, out[:, 0], atol=1e-6)
        assert got.abs().sum().item() > 0.0

    def test_empty_history_contributes_nothing(self):
        """Two ways to say "no history", both of which must zero the GRU
        half: a padded row whose length is 0 (its padding still decodes to
        a real embedding), and a tensor with no time steps at all."""
        net = _fixed_net()
        padded = net.encode(
            torch.zeros(2, STATE_DIM),
            torch.tensor([[11], [11]]),
            torch.tensor([[2], [2]]),
            torch.tensor([1, 0]),
        )[:, STATE_ENC:]
        assert padded[1].abs().sum().item() == 0.0
        assert padded[0].abs().sum().item() > 0.0

        empty = net.encode(
            torch.zeros(1, STATE_DIM),
            torch.zeros(1, 0, dtype=torch.long),
            torch.zeros(1, 0, dtype=torch.long),
            torch.tensor([0]),
        )[:, STATE_ENC:]
        assert empty.abs().sum().item() == 0.0


class TestRelativeSeats:
    """`history_features` encodes seats as `(seat - my_seat) % 4`.

    At an even `my_seat` a `+` implementation is indistinguishable
    (-2 ≡ +2 mod 4, -0 ≡ +0), so the odd seats are what pins the sign.
    """

    CARDS = [Card("Hearts", "Ace"), Card("Spades", "2"),
             Card("Clubs", "King"), Card("Diamonds", "7")]

    def _game(self, n=4):
        log = [(i % 4, self.CARDS[i % 4]) for i in range(n)]
        return SimpleNamespace(play_log_this_hand=log), log

    def test_seats_are_relative_to_the_observer(self):
        game, log = self._game()
        for my_seat in range(4):
            cards, seats = history_features(game, my_seat)
            assert cards == [card_to_index(c) for _, c in log]
            assert seats == [(s - my_seat) % 4 for s, _ in log]
            if my_seat % 2:  # only odd seats separate minus from plus
                assert seats != [(s + my_seat) % 4 for s, _ in log]

    def test_history_is_truncated_to_the_most_recent_events(self):
        game, log = self._game(MAX_HISTORY + 8)
        cards, seats = history_features(game, 1)
        assert len(cards) == MAX_HISTORY == len(seats)
        assert cards == [card_to_index(c) for _, c in log][-MAX_HISTORY:]
        assert seats == [(s - 1) % 4 for s, _ in log][-MAX_HISTORY:]

    def test_no_game_yields_empty_history(self):
        assert history_features(None, 0) == ([], [])
        assert history_features(SimpleNamespace(play_log_this_hand=[]), 2) == ([], [])


class TestPlayerConsultsTheNetwork:
    """`DMCPlayer.play_card` must actually score its candidates.

    The single-legal-card shortcut is the only path allowed to skip the
    net; on a multi-card decision the net's argmax is the answer.
    """

    def test_multi_card_decision_takes_the_network_argmax(self):
        calls = []

        def scoring_q(static, hist_c, hist_s, candidates):
            calls.append(list(candidates))
            # Rank strictly ascending, so the last candidate must win.
            return torch.arange(len(candidates), dtype=torch.float32)

        net = _fixed_net()
        net.q_values = scoring_q
        p = DMCPlayer("D", net=net)
        p.hand = [Card("Hearts", "2"), Card("Hearts", "King"),
                  Card("Hearts", "9")]

        card, idx = p.play_card("Hearts")
        assert calls == [[card_to_index(c) for c in p.hand]]
        assert card is p.hand[-1], "the argmax candidate must be played"
        assert idx == card_to_index(card)

    def test_single_legal_card_needs_no_network(self):
        def explode(*_a, **_k):
            raise AssertionError("no search is needed for a forced play")

        net = _fixed_net()
        net.q_values = explode
        p = DMCPlayer("D", net=net)
        p.hand = [Card("Hearts", "2"), Card("Spades", "9")]
        card, idx = p.play_card("Hearts")
        assert str(card) == "2 of Hearts"
        assert idx == card_to_index(card)

    def test_no_valid_cards_raises(self):
        p = DMCPlayer("D", net=_fixed_net())
        p.hand = []
        with pytest.raises(ValueError, match="No valid cards"):
            p.play_card(None)

    def test_live_history_reaches_the_network(self):
        """The seated player hands the net the hand's real play history,
        expressed relative to its own (odd) seat."""
        seen = []
        net = _fixed_net()
        p = DMCPlayer("D", net=net)
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm(
            [others[0], p, others[1], others[2]],
            minimal_logging=True,
            rng=random.Random(5),
        )

        def recording_q(static, hist_c, hist_s, candidates):
            log = list(g.play_log_this_hand)
            seen.append((
                list(hist_c),
                list(hist_s),
                [card_to_index(c) for _, c in log],
                [(s - 1) % 4 for s, _ in log],  # p sits at seat 1
            ))
            return torch.zeros(len(candidates))

        net.q_values = recording_q
        g.start_game()
        g.choose_trump_suit()
        assert g.players.index(p) == 1 and p._seat == 1
        for _ in range(60):
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            assert g.apply_play(nxt, card) is None
            if seen and seen[-1][2]:
                break
        else:
            raise AssertionError("the DMC seat never decided with history")

        got_cards, got_seats, exp_cards, exp_seats = seen[-1]
        assert exp_cards, "the pin is only meaningful with history present"
        assert got_cards == exp_cards
        assert got_seats == exp_seats


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

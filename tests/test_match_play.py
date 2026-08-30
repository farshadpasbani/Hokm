"""
Multi-hand MATCH play for the Mini App backend (`game_service.GameSession`).

One `Hokm` instance models exactly one hand; the match layer lives in
`GameSession`. These tests pin:

  * hand results folding into `match_score` (Kot = 2 points),
  * Hakem rotation between hands matching the engine's `rotate_hakem()`,
  * `next_hand()` dealing a full 13-card hand to every seat,
  * `match_over` / `match_winner` firing at `match_target`,
  * `new_game()` resetting the match,
  * the Flask surface for a full hand → `/api/next_hand` transition.

Everything runs against `HeuristicAgent` seats: `AI_KIND=heuristic`, so
`game_service._build_ai_seat` skips the (default) PIMC search — these tests
simulate several full hands and must stay cheap next to a training run.
"""

import os
import random

import pytest

# `game_service` reads AI_KIND once, at import; set it before importing in
# case this module is imported first. The autouse fixture below is what
# actually guarantees it, since another test module may import first.
os.environ.setdefault("AI_KIND", "heuristic")

import game_service  # noqa: E402
from game_service import GameServiceError, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _heuristic_seats(monkeypatch):
    """Guarantee cheap rule-based AI seats regardless of the ambient env."""
    monkeypatch.setenv("AI_KIND", "heuristic")
    monkeypatch.setattr(game_service, "AI_KIND", "heuristic")
    monkeypatch.setattr(game_service, "MODEL_PATH", "")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _play_hand(session, data=None):
    """Drive the session through one hand, always playing `legal_cards[0]`."""
    if data is None:
        data = session.new_game()
    if data["phase"] == "choose_trump":
        data = session.set_trump(data["trump_options"][0])
    for _ in range(60):  # 13 tricks max; generous bound
        if data.get("game_over"):
            return data
        assert data["your_turn"], f"expected human turn, got {data['phase']}"
        data = session.play_card(data["legal_cards"][0])
    raise AssertionError("hand did not finish within bound")


def _expected_next_hakem(game):
    """Replay `Hokm.rotate_hakem()`'s rule by hand, without mutating state."""
    winners = game.team1 if game.scores[1] > game.scores[2] else game.team2
    hakem_team = game.team1 if game.hakem in game.team1 else game.team2
    if winners is not hakem_team:
        return winners[0]
    return hakem_team[(hakem_team.index(game.hakem) + 1) % 2]


def _force_hand_result(session, team1_tricks, team2_tricks):
    """Plant a finished hand on the engine and record it (no card play)."""
    g = session.game
    g.scores[1], g.scores[2] = team1_tricks, team2_tricks
    for p in g.players:
        p.hand = []
        g.tricks_won[p] = 0
    g.tricks_won[g.team1[0]] = team1_tricks
    g.tricks_won[g.team2[0]] = team2_tricks
    session.hand_result = None
    session._finish_hand_if_over()


def _deal_fingerprint(session):
    """Everything the shuffle decides: who is Hakem and who holds what."""
    g = session.game
    return (
        g.hakem.name,
        [sorted(str(c) for c in p.hand) for p in g.players],
    )


# --------------------------------------------------------------------------
# seeded deals
#
# `GameSession(..., rng=...)` feeds the engine's existing `Hokm(rng=...)`
# hook, so a test can pin the deal. Without it the engine stays on
# module-level `random`, which is what production does.
# --------------------------------------------------------------------------

class TestSeededDeals:
    SEED = 20260830

    def _seeded(self, user="guest:seed"):
        return GameSession(user, "Tester", rng=random.Random(self.SEED))

    def test_same_seed_deals_the_same_hands(self):
        a, b = self._seeded("guest:s1"), self._seeded("guest:s2")
        a.new_game()
        b.new_game()
        assert _deal_fingerprint(a) == _deal_fingerprint(b)
        # A deal is only pinned if it is also a real deal: hands still held
        # plus whatever the AI seats have already played is the whole deck.
        cards = [c for h in _deal_fingerprint(a)[1] for c in h]
        cards += [str(c) for c in a.game.cards_played_this_hand]
        assert len(cards) == 52 and len(set(cards)) == 52

    def test_same_seed_plays_the_same_hands_end_to_end(self):
        """Deterministic seats + a pinned deal => the whole match replays.

        Two hands, because the engine keeps the seeded Random across hands:
        the second deal must come from the same stream, not a fresh one.
        """
        results = []
        for user in ("guest:s3", "guest:s4"):
            sess = self._seeded(user)
            trace = []
            data = None
            for _ in range(2):
                final = _play_hand(sess, data)
                trace.append((
                    final["hakem"],
                    [[s, str(c)] for s, c in sess.game.play_log_this_hand],
                    final["scores"],
                    final["result"],
                ))
                data = sess.next_hand()
            results.append(trace)
        assert results[0] == results[1]
        assert len(results[0][0][1]) >= 7 * 4 - 3, "a full hand must be played"
        assert results[0][0] != results[0][1], "two different hands, not a repeat"

    def test_different_seeds_deal_differently(self):
        one = GameSession("guest:s5", "Tester", rng=random.Random(1))
        two = GameSession("guest:s6", "Tester", rng=random.Random(2))
        one.new_game()
        two.new_game()
        assert _deal_fingerprint(one) != _deal_fingerprint(two)

    def test_unseeded_session_leaves_the_engine_on_module_random(self):
        """Default construction must not change production behaviour."""
        sess = GameSession("guest:s7", "Tester")
        sess.new_game()
        assert sess.game.rng is None


# --------------------------------------------------------------------------
# match bookkeeping
# --------------------------------------------------------------------------

class TestMatchScoring:
    def test_new_game_starts_a_fresh_match(self):
        sess = GameSession("guest:m1", "Tester")
        data = sess.new_game()
        assert sess.match_score == {"Team 1": 0, "Team 2": 0}
        assert sess.match_target == 7
        assert data["match_score"] == {"Team 1": 0, "Team 2": 0}
        assert data["match_over"] is False

    def test_hand_result_recorded_into_match_score(self):
        sess = GameSession("guest:m2", "Tester")
        final = _play_hand(sess)

        assert final["phase"] == "ended"
        winner = final["hand_result"]
        assert winner in ("Team 1", "Team 2")
        assert winner == ("Team 1" if final["you_won"] else "Team 2")
        points = 2 if final["kot"] else 1
        assert sess.match_score[winner] == points
        loser = "Team 2" if winner == "Team 1" else "Team 1"
        assert sess.match_score[loser] == 0
        assert final["match_score"] == sess.match_score
        assert final["match_over"] is False
        assert final["match_winner"] is None
        # the hand is over but the session keeps the match alive
        assert sess.game is not None

    def test_kot_counts_double(self):
        sess = GameSession("guest:m3", "Tester")
        sess.new_game()
        _force_hand_result(sess, 7, 0)
        assert sess.hand_result["hand_result"] == "Team 1"
        assert sess.hand_result["kot"] is True
        assert sess.match_score["Team 1"] == 2
        assert sess.match_score["Team 2"] == 0

    def test_non_sweep_is_one_point(self):
        sess = GameSession("guest:m4", "Tester")
        sess.new_game()
        _force_hand_result(sess, 6, 7)
        assert sess.hand_result["kot"] is False
        assert sess.match_score == {"Team 1": 0, "Team 2": 1}

    def test_recording_a_hand_is_idempotent(self):
        sess = GameSession("guest:m5", "Tester")
        sess.new_game()
        _force_hand_result(sess, 7, 2)
        for _ in range(3):
            sess._finish_hand_if_over()
        assert sess.match_score["Team 1"] == 1

    def test_match_over_triggers_at_target(self):
        sess = GameSession("guest:m6", "Tester")
        sess.new_game()
        sess.match_target = 3

        _force_hand_result(sess, 7, 1)
        assert sess.match_over is False
        sess.next_hand()

        _force_hand_result(sess, 7, 0)  # kot: 1 + 2 = 3 == target
        assert sess.match_score["Team 1"] == 3
        assert sess.match_over is True
        assert sess.match_winner == "Team 1"
        with pytest.raises(GameServiceError, match="match is over"):
            sess.next_hand()

    def test_match_target_env_override(self, monkeypatch):
        monkeypatch.setenv("MATCH_TARGET", "3")
        sess = GameSession("guest:m7", "Tester")
        assert sess.match_target == 3
        assert sess.new_game()["match_target"] == 3

    def test_new_game_resets_the_match(self):
        sess = GameSession("guest:m8", "Tester")
        sess.new_game()
        _force_hand_result(sess, 7, 0)
        assert sess.match_score["Team 1"] == 2

        data = sess.new_game()
        assert sess.match_score == {"Team 1": 0, "Team 2": 0}
        assert sess.match_over is False
        assert sess.match_winner is None
        assert sess.hand_result is None
        assert data["scores"] == {"Team 1": 0, "Team 2": 0}


# --------------------------------------------------------------------------
# hand-to-hand transitions
# --------------------------------------------------------------------------

class TestNextHand:
    def test_hakem_rotates_per_engine_logic(self):
        sess = GameSession("guest:n1", "Tester")
        sess.new_game()
        for _ in range(3):
            _play_hand(sess, sess.state())
            # `_finish_hand_if_over` already rotated, so recompute the
            # expectation from the pre-rotation Hakem we recorded.
            g = sess.game
            hand_hakem = next(
                p for p in g.players if p.name == sess.hand_result["hakem"]
            )
            winners = g.team1 if g.scores[1] > g.scores[2] else g.team2
            hakem_team = g.team1 if hand_hakem in g.team1 else g.team2
            if winners is not hakem_team:
                expected = winners[0]
            else:
                expected = hakem_team[(hakem_team.index(hand_hakem) + 1) % 2]
            assert g.hakem is expected
            sess.next_hand()

    def test_rotation_helper_matches_engine(self):
        """Sanity-check the manual expectation against `Hokm.rotate_hakem`."""
        sess = GameSession("guest:n2", "Tester")
        sess.new_game()
        g = sess.game
        g.scores[1], g.scores[2] = 7, 3
        for p in g.players:
            g.tricks_won[p] = 0
        g.tricks_won[g.team1[0]] = 7
        g.tricks_won[g.team2[0]] = 3
        expected = _expected_next_hakem(g)
        g.update_last_winning_team()
        g.rotate_hakem()
        assert g.hakem is expected

    def test_next_hand_deals_thirteen_cards_each(self):
        """Human forced as next Hakem so nobody has played when we count."""
        sess = GameSession("guest:n3", "Tester")
        _play_hand(sess)
        sess.game.hakem = sess.human

        data = sess.next_hand()
        assert data["phase"] == "choose_trump"
        assert len(data["hakem_cards"]) == 5
        g = sess.game
        assert g.trump_suit is None
        assert [len(p.hand) for p in g.players if p is not sess.human] == [0, 0, 0]

        data = sess.set_trump(data["trump_options"][0])
        assert [len(p.hand) for p in g.players] == [13, 13, 13, 13]
        assert g.scores == {1: 0, 2: 0}
        assert data["scores"] == {"Team 1": 0, "Team 2": 0}
        assert data["your_turn"] is True  # Hakem leads the first trick

    def test_next_hand_conserves_the_deck(self):
        sess = GameSession("guest:n4", "Tester")
        _play_hand(sess)
        data = sess.next_hand()
        if data["phase"] == "choose_trump":
            data = sess.set_trump(data["trump_options"][0])
        g = sess.game
        assert sum(len(p.hand) for p in g.players) + len(g.current_trick) == 52
        assert all(len(p.hand) in (12, 13) for p in g.players)
        assert g.trump_suit is not None

    def test_next_hand_rejected_mid_hand(self):
        sess = GameSession("guest:n5", "Tester")
        data = sess.new_game()
        if data["phase"] == "choose_trump":
            sess.set_trump(data["trump_options"][0])
        with pytest.raises(GameServiceError, match="still in progress"):
            sess.next_hand()

    def test_playing_after_hand_end_is_rejected(self):
        sess = GameSession("guest:n6", "Tester")
        final = _play_hand(sess)
        card = final["hand"][0] if final["hand"] else "Ace of Spades"
        with pytest.raises(GameServiceError, match="hand is over"):
            sess.play_card(card)

    def test_state_after_hand_end_keeps_match_context(self):
        sess = GameSession("guest:n7", "Tester")
        _play_hand(sess)
        body = sess.state()
        assert body["phase"] == "ended"
        assert body["game_over"] is True
        assert body["hand_result"] in ("Team 1", "Team 2")
        assert body["match_score"] == sess.match_score
        assert sum(sess.match_score.values()) in (1, 2)  # not double-counted

    def test_two_hands_accumulate(self):
        sess = GameSession("guest:n8", "Tester")
        _play_hand(sess)
        first = dict(sess.match_score)
        _play_hand(sess, sess.next_hand())
        assert sum(sess.match_score.values()) > sum(first.values())


# --------------------------------------------------------------------------
# the 13th trick must be scored (regression, both AI-turn loops)
# --------------------------------------------------------------------------

class TestFinalTrickIsScored:
    """
    Both AI loops used to test "all hands empty" *before* resolving a
    completed trick, so the 13th trick of a hand that went the distance was
    dropped: a 6-6 hand reported a draw and only 12 tricks were counted.
    """

    @staticmethod
    def _hand_at_final_trick(game):
        """Empty every hand and leave a complete, unresolved 13th trick."""
        players = game.players
        game.scores = {1: 6, 2: 6}
        game.round_count = 12
        cards = [p.hand.pop(0) for p in players]
        for p in players:
            p.hand = []
        game.current_trick = list(zip(players, cards))
        game.lead_suit = cards[0].suit
        return game

    def test_service_loop_scores_the_last_trick(self):
        sess = GameSession("guest:f1", "Tester")
        data = sess.new_game()
        if data["phase"] == "choose_trump":
            sess.set_trump(data["trump_options"][0])
        self._hand_at_final_trick(sess.game)

        body = sess.state()
        assert sess.game.scores[1] + sess.game.scores[2] == 13
        assert body["result"] in ("Team 1 wins", "Team 2 wins")
        assert body["hand_result"] in ("Team 1", "Team 2")
        assert sum(sess.match_score.values()) == 1

    def test_app_loop_scores_the_last_trick(self):
        import app

        game = app.create_game("Tester")
        game.start_game()
        game.set_trump_suit(game.hakem.hand[0].suit)
        self._hand_at_final_trick(game)

        events = app.run_ai_turns(game, game.players[0])
        assert game.scores[1] + game.scores[2] == 13
        assert [e["type"] for e in events] == ["trick"]
        assert app.game_over(game)  # 7-6, not a 6-6 "draw"


# --------------------------------------------------------------------------
# Flask surface
# --------------------------------------------------------------------------

@pytest.fixture()
def guest_client(monkeypatch):
    import server

    monkeypatch.setattr(server, "BOT_TOKEN", "")
    monkeypatch.setattr(server, "ALLOW_GUESTS", True)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


class TestNextHandApi:
    HEADERS = {"X-Guest-Id": "matchguest1", "X-Guest-Name": "Tester"}

    def _post(self, client, path, body=None):
        resp = client.post(path, json=body or {}, headers=self.HEADERS)
        return resp, resp.get_json()

    def test_guest_match_flow_hand_then_next_hand(self, guest_client):
        resp, data = self._post(guest_client, "/api/new_game")
        assert resp.status_code == 200
        assert data["match_score"] == {"Team 1": 0, "Team 2": 0}

        if data["phase"] == "choose_trump":
            _, data = self._post(
                guest_client,
                "/api/set_trump",
                {"trump_suit": data["trump_options"][0]},
            )

        for _ in range(60):
            if data.get("game_over"):
                break
            assert data["your_turn"], data["phase"]
            _, data = self._post(
                guest_client, "/api/play_card", {"card": data["legal_cards"][0]}
            )
        else:
            raise AssertionError("hand did not finish within bound")

        assert data["phase"] == "ended"
        assert data["hand_result"] in ("Team 1", "Team 2")
        assert isinstance(data["kot"], bool)
        assert data["match_over"] is False
        won = sum(data["match_score"].values())
        assert won in (1, 2)

        resp, nxt = self._post(guest_client, "/api/next_hand")
        assert resp.status_code == 200
        assert nxt["status"] == "success"
        assert nxt["phase"] in ("choose_trump", "playing")
        assert nxt["scores"] == {"Team 1": 0, "Team 2": 0}
        assert sum(nxt["match_score"].values()) == won

    def test_next_hand_mid_hand_is_400(self, guest_client):
        self._post(guest_client, "/api/new_game")
        resp, body = self._post(guest_client, "/api/next_hand")
        assert resp.status_code == 400
        assert body["status"] == "error"

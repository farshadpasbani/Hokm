"""
Tests for the general matchup harness: agents.make_team, evaluate.py's
spec-based path (including the legacy-flag translation), and DMC
eval-in-the-loop.

Game counts are deliberately tiny — this suite checks plumbing, not strength.
"""

import random

import pytest
import torch

from agents import SpecError, describe, make_team, parse_spec, spec_checkpoint
from baselines import HeuristicAgent, RandomAgent
from dmc import DMCNet, DMCPlayer
from dmc_train import DMCTrainer, best_checkpoint_path
from enhanced_player import EnhancedPlayer
from evaluate import (
    MatchupResult,
    build_parser,
    resolve_matchups,
    run_match,
    wilson_ci,
)
from pimc import PIMCPlayer

TINY_PIMC = 2  # keep search cost negligible; a heavy trainer may share this box


@pytest.fixture(scope="module")
def nfsp_ckpt(tmp_path_factory):
    """A real (random-init) NFSP checkpoint on disk, in the shape
    load_policy_state expects."""
    path = tmp_path_factory.mktemp("ckpts") / "nfsp_tiny.pth"
    p = EnhancedPlayer("ckpt src")
    torch.save(p.export_state_dict(), str(path))
    return str(path)


@pytest.fixture(scope="module")
def dmc_ckpt(tmp_path_factory):
    """A fresh DMCNet saved in the {"dmc_net": state_dict} format."""
    path = tmp_path_factory.mktemp("ckpts") / "dmc_tiny.pt"
    torch.save({"dmc_net": DMCNet().state_dict()}, str(path))
    return str(path)


class TestParseSpec:
    @pytest.mark.parametrize(
        "spec,family,arg",
        [
            ("random", "random", None),
            ("heuristic", "heuristic", None),
            ("untrained", "untrained", None),
            ("  HEURISTIC ", "heuristic", None),
            ("pimc", "pimc", None),
            ("pimc:32", "pimc", 32),
            ("nfsp:models/a.pth", "nfsp", "models/a.pth"),
            ("dmc:models/b.pt", "dmc", "models/b.pt"),
        ],
    )
    def test_valid(self, spec, family, arg):
        assert parse_spec(spec) == (family, arg)

    @pytest.mark.parametrize(
        "spec",
        [
            "",
            "   ",
            "bogus",
            "nfsp",           # missing path
            "nfsp:",          # empty path
            "dmc",
            "dmc:",
            "pimc:",          # empty count
            "pimc:abc",       # non-integer
            "pimc:0",         # must be >= 1
            "pimc:-4",
            "random:5",       # takes no argument
            "heuristic:x",
            None,
        ],
    )
    def test_bad_specs_raise(self, spec):
        with pytest.raises(ValueError):
            parse_spec(spec)

    def test_spec_error_is_a_value_error(self):
        assert issubclass(SpecError, ValueError)

    def test_error_message_is_actionable(self):
        with pytest.raises(ValueError) as e:
            parse_spec("nfsp")
        assert "checkpoint path" in str(e.value)

    def test_spec_checkpoint(self, nfsp_ckpt):
        assert spec_checkpoint(f"nfsp:{nfsp_ckpt}") == nfsp_ckpt
        assert spec_checkpoint("heuristic") is None
        assert spec_checkpoint("pimc:8") is None

    def test_describe_shortens_paths(self):
        assert describe("dmc:some/deep/dir/net.pt") == "dmc:net.pt"
        assert describe("heuristic") == "heuristic"


class TestMakeTeam:
    NAMES = ("Seat A", "Seat B")

    def _check_pair(self, team, cls):
        assert len(team) == 2
        assert all(isinstance(p, cls) for p in team)
        assert [p.name for p in team] == list(self.NAMES)
        assert team[0] is not team[1], "seats must be independent objects"
        for p in team:
            assert p.learning_enabled is False
            assert p.epsilon == 0.0
            assert p.eta == 0.0

    def test_random(self):
        self._check_pair(make_team("random", seed=1, names=self.NAMES), RandomAgent)

    def test_heuristic(self):
        self._check_pair(
            make_team("heuristic", seed=1, names=self.NAMES), HeuristicAgent
        )

    def test_untrained(self):
        team = make_team("untrained", seed=1, names=self.NAMES)
        self._check_pair(team, EnhancedPlayer)

    def test_nfsp(self, nfsp_ckpt):
        team = make_team(f"nfsp:{nfsp_ckpt}", seed=1, names=self.NAMES)
        self._check_pair(team, EnhancedPlayer)
        # Both seats loaded the same weights.
        a = team[0].q_net.state_dict()
        b = team[1].q_net.state_dict()
        for k in a:
            assert torch.equal(a[k], b[k])

    def test_dmc(self, dmc_ckpt):
        team = make_team(f"dmc:{dmc_ckpt}", seed=1, names=self.NAMES)
        self._check_pair(team, DMCPlayer)
        saved = torch.load(dmc_ckpt, map_location="cpu", weights_only=True)["dmc_net"]
        got = team[0].net.state_dict()
        for k in saved:
            assert torch.equal(saved[k], got[k])

    def test_pimc_default(self):
        team = make_team("pimc", seed=1, names=self.NAMES)
        self._check_pair(team, PIMCPlayer)
        assert team[0].determinizations == PIMCPlayer("x").determinizations

    def test_pimc_with_count(self):
        team = make_team(f"pimc:{TINY_PIMC}", seed=1, names=self.NAMES)
        self._check_pair(team, PIMCPlayer)
        assert all(p.determinizations == TINY_PIMC for p in team)

    def test_seats_get_independent_rngs(self):
        team = make_team("random", seed=7, names=self.NAMES)
        assert team[0]._rng is not team[1]._rng
        a = [team[0]._rng.random() for _ in range(5)]
        b = [team[1]._rng.random() for _ in range(5)]
        assert a != b

    def test_same_seed_reproduces(self):
        a = make_team("random", seed=7, names=self.NAMES)
        b = make_team("random", seed=7, names=self.NAMES)
        assert [a[0]._rng.random() for _ in range(5)] == [
            b[0]._rng.random() for _ in range(5)
        ]

    def test_bad_spec_raises(self):
        with pytest.raises(ValueError):
            make_team("nope", seed=1, names=self.NAMES)

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            make_team(
                f"nfsp:{tmp_path / 'nope.pth'}", seed=1, names=self.NAMES
            )

    def test_wrong_number_of_names(self):
        with pytest.raises(ValueError):
            make_team("random", seed=1, names=("only one",))


class TestRunMatch:
    def test_random_vs_heuristic(self):
        r = run_match("random", "heuristic", games=3, seed=11)
        assert isinstance(r, MatchupResult)
        assert r.team1 == "random"
        assert r.team2 == "heuristic"
        assert r.opponent == "heuristic"
        assert r.games == 3
        assert r.wins + r.losses + r.ties == 3
        assert r.win_rate == pytest.approx(r.wins / 3)
        assert 0.0 <= r.ci95_low <= r.win_rate <= r.ci95_high <= 1.0
        assert r.sum_team1_tricks > 0 or r.sum_team2_tricks > 0
        assert r.mean_trick_diff == pytest.approx(
            (r.sum_team1_tricks - r.sum_team2_tricks) / 3
        )
        assert r.seed == 11
        # Serializable, and both new fields survive.
        row = r.to_row()
        assert row["team1"] == "random" and row["team2"] == "heuristic"

    def test_deterministic_given_seed(self):
        a = run_match("random", "heuristic", games=3, seed=5)
        b = run_match("random", "heuristic", games=3, seed=5)
        assert a.to_row() == b.to_row()

    def test_label_overrides_display_name(self, nfsp_ckpt):
        spec = f"nfsp:{nfsp_ckpt}"
        r = run_match(spec, spec, games=1, seed=3, label="self")
        assert r.opponent == "self"
        assert r.team1 == r.team2 == spec

    def test_wilson_ci_edges(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)
        low, high = wilson_ci(10, 10)
        assert high == 1.0 and 0.0 < low < 1.0


class TestArgTranslation:
    """The legacy CLI must land on the new spec-based path."""

    def _resolve(self, argv):
        return resolve_matchups(build_parser().parse_args(argv))

    def test_legacy_single_opponent(self):
        t1, ms = self._resolve(["--checkpoint", "models/x.pth",
                                "--opponent", "heuristic"])
        assert t1 == "nfsp:models/x.pth"
        assert ms == [("heuristic", "heuristic")]

    def test_legacy_defaults_to_all(self):
        t1, ms = self._resolve(["--checkpoint", "models/x.pth"])
        assert t1 == "nfsp:models/x.pth"
        assert [label for label, _ in ms] == [
            "random", "heuristic", "self", "untrained"
        ]

    def test_legacy_self_maps_to_team1_spec(self):
        t1, ms = self._resolve(["--checkpoint", "models/x.pth",
                                "--opponent", "self"])
        assert ms == [("self", "nfsp:models/x.pth")]

    def test_legacy_all_includes_self(self):
        t1, ms = self._resolve(["--checkpoint", "models/x.pth",
                                "--opponent", "all"])
        assert dict(ms)["self"] == t1

    def test_new_style_pairs_specs(self):
        t1, ms = self._resolve(["--team1", "dmc:models/dmc_latest.pt",
                                "--team2", "pimc:32"])
        assert t1 == "dmc:models/dmc_latest.pt"
        assert ms == [("pimc:32", "pimc:32")]

    def test_new_style_all_is_checkpoint_free_baselines(self):
        _, ms = self._resolve(["--team1", "pimc:8", "--team2", "all"])
        assert [label for label, _ in ms] == ["random", "heuristic", "untrained"]

    def test_team1_wins_over_checkpoint_alias(self):
        t1, _ = self._resolve(["--team1", "heuristic",
                               "--checkpoint", "models/x.pth"])
        assert t1 == "heuristic"

    def test_missing_team1_raises(self):
        with pytest.raises(ValueError):
            self._resolve(["--games", "3"])

    def test_bad_team1_spec_raises(self):
        with pytest.raises(ValueError):
            self._resolve(["--team1", "nonsense"])

    def test_games_and_seed_defaults_unchanged(self):
        args = build_parser().parse_args(["--checkpoint", "models/x.pth"])
        assert args.games == 1000 and args.seed == 42


class TestQuickEval:
    def test_returns_probability(self):
        tr = DMCTrainer(seed=0)
        wr = tr.quick_eval(games=4, opponent="heuristic")
        assert isinstance(wr, float)
        assert 0.0 <= wr <= 1.0

    def test_does_not_disturb_trainer_rng(self):
        tr = DMCTrainer(seed=0)
        before = tr.rng.getstate()
        tr.quick_eval(games=2, opponent="random")
        assert tr.rng.getstate() == before

    def test_repeatable(self):
        tr = DMCTrainer(seed=0)
        assert tr.quick_eval(games=3) == tr.quick_eval(games=3)

    def test_zero_games_is_zero(self):
        assert DMCTrainer(seed=0).quick_eval(games=0) == 0.0

    def test_leaves_net_in_eval_mode(self):
        tr = DMCTrainer(seed=0)
        tr.net.eval()
        tr.quick_eval(games=1)
        assert not tr.net.training

    def test_best_checkpoint_path(self, tmp_path):
        out = str(tmp_path / "sub" / "dmc_latest.pt")
        assert best_checkpoint_path(out) == str(tmp_path / "sub" / "dmc_best.pt")
        assert best_checkpoint_path(None) is None

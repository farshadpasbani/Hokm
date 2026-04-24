"""
Evaluation CLI — the canonical "how strong is this checkpoint" script.

What it does
------------
Given a single `.pth` checkpoint, it seats the trained agent in positions
0 and 2 (Team 1) and fills positions 1 and 3 with one of several baselines:

    - "random"     : RandomAgent
    - "heuristic"  : HeuristicAgent
    - "self"       : Another copy of the same checkpoint (sanity: ~50% win)
    - "untrained"  : Fresh EnhancedPlayer with random-init weights
    - "all"        : Runs each of the above in turn

For each matchup it reports win-rate with a Wilson 95% confidence interval,
tie rate, and mean trick differential. Results are serialized to JSON and
(optionally) CSV.

Design decisions
----------------
* Evaluation is **deterministic** given `--seed`. Each matchup uses a distinct
  derived seed so they can be compared against each other consistently.
* Exploration is off: ε = 0, η = 0, `learning_enabled = False` on the
  trained seats — see `dev_eval.run_evaluation`'s docstring for why.
* We quantify uncertainty with **Wilson intervals** rather than Normal
  approximations because win rates near 0 / 1 on small N are badly skewed
  under Normal.

Example
-------
    python evaluate.py --checkpoint models/nfsp_shared_2025..._game_1000.pth \\
        --opponent all --games 1000 --seed 42 \\
        --out dev_cache/eval_suite.json --csv dev_cache/eval_suite.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from baselines import HeuristicAgent, RandomAgent
from enhanced_player import EnhancedPlayer
from game_constants import ACTION_DIM, STATE_DIM
from hokm import Hokm
from seed_utils import seed_all


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (handles edge cases)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


@dataclass
class MatchupResult:
    opponent: str
    games: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    ci95_low: float
    ci95_high: float
    mean_trick_diff: float           # mean of (team1_tricks - team2_tricks)
    sum_team1_tricks: int
    sum_team2_tricks: int
    seed: Optional[int]

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Player construction
# ------------------------------------------------------------------

def _load_trained_seat(name: str, checkpoint_path: str) -> EnhancedPlayer:
    """Trained agent: load weights, disable learning & exploration."""
    p = EnhancedPlayer(name, STATE_DIM, ACTION_DIM, epsilon=0.0, eta=0.0)
    p.learning_enabled = False
    if checkpoint_path:
        p.load_policy_state(checkpoint_path)
    return p


def _build_match(
    checkpoint: str,
    opponent: str,
    seed: int,
) -> Hokm:
    """
    Construct a 4-player Hokm game. Team 1 (seats 0, 2) = trained checkpoint,
    Team 2 (seats 1, 3) = the opponent family.

    Deterministic given `seed`.
    """
    rng = random.Random(seed)

    # Team 1: two independent player objects, same loaded weights.
    t1_south = _load_trained_seat("Player 1 (trained S)", checkpoint)
    t1_north = _load_trained_seat("Player 3 (trained N)", checkpoint)

    if opponent == "random":
        t2_east = RandomAgent("Player 2 (random E)", rng=random.Random(seed + 1))
        t2_west = RandomAgent("Player 4 (random W)", rng=random.Random(seed + 2))
    elif opponent == "heuristic":
        t2_east = HeuristicAgent("Player 2 (heur E)", rng=random.Random(seed + 1))
        t2_west = HeuristicAgent("Player 4 (heur W)", rng=random.Random(seed + 2))
    elif opponent == "self":
        t2_east = _load_trained_seat("Player 2 (self E)", checkpoint)
        t2_west = _load_trained_seat("Player 4 (self W)", checkpoint)
    elif opponent == "untrained":
        t2_east = _load_trained_seat("Player 2 (untr. E)", "")
        t2_west = _load_trained_seat("Player 4 (untr. W)", "")
    else:
        raise ValueError(f"Unknown opponent: {opponent!r}")

    players = [t1_south, t2_east, t1_north, t2_west]
    return Hokm(players, minimal_logging=True, rng=rng)


# ------------------------------------------------------------------
# Match loop
# ------------------------------------------------------------------

def run_matchup(
    checkpoint: str,
    opponent: str,
    games: int,
    seed: int,
) -> MatchupResult:
    """Play `games` hands; Team 1 = trained, Team 2 = baseline. Wins = Team 1 wins."""
    seed_all(seed)
    game = _build_match(checkpoint, opponent, seed)

    wins = losses = ties = 0
    sum_t1 = sum_t2 = 0
    for _ in range(games):
        game.play_game(save_excel_log=False)
        s1, s2 = game.scores[1], game.scores[2]
        sum_t1 += s1
        sum_t2 += s2
        if s1 > s2:
            wins += 1
        elif s2 > s1:
            losses += 1
        else:
            ties += 1

    win_rate = wins / games if games else 0.0
    low, high = wilson_ci(wins, games)
    mean_diff = (sum_t1 - sum_t2) / games if games else 0.0
    return MatchupResult(
        opponent=opponent,
        games=games,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        ci95_low=low,
        ci95_high=high,
        mean_trick_diff=mean_diff,
        sum_team1_tricks=sum_t1,
        sum_team2_tricks=sum_t2,
        seed=seed,
    )


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def _print_table(results: List[MatchupResult]) -> None:
    header = f"{'opponent':<12} {'games':>6} {'wins':>5} {'ties':>5} {'win%':>7} {'95% CI':>18} {'Δ tricks':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"[{r.ci95_low*100:5.1f}, {r.ci95_high*100:5.1f}]"
        print(
            f"{r.opponent:<12} {r.games:>6d} {r.wins:>5d} {r.ties:>5d} "
            f"{r.win_rate*100:>6.2f}% {ci:>18} {r.mean_trick_diff:>+9.2f}"
        )


def _write_csv(path: str, results: List[MatchupResult]) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].to_row().keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

OPPONENTS = ("random", "heuristic", "self", "untrained")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a Hokm checkpoint against baseline opponents."
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pth checkpoint produced by train_backend / train_hokm.",
    )
    ap.add_argument(
        "--opponent",
        default="all",
        choices=OPPONENTS + ("all",),
        help="Opponent family (or 'all' to run every one sequentially).",
    )
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path.",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path.",
    )
    args = ap.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    targets = OPPONENTS if args.opponent == "all" else (args.opponent,)
    # Derive a distinct seed per matchup so matchups aren't all identical deals.
    results = [
        run_matchup(
            args.checkpoint, opp, args.games, seed=args.seed + i
        )
        for i, opp in enumerate(targets)
    ]

    _print_table(results)

    if args.out:
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "generated_at": datetime.now().isoformat(),
            "games_per_matchup": args.games,
            "base_seed": args.seed,
            "results": [r.to_row() for r in results],
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON: {args.out}")

    if args.csv:
        _write_csv(args.csv, results)
        print(f"Wrote CSV : {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

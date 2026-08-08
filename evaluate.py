"""
Evaluation CLI — the canonical "how strong is this agent" script.

Any two agent families can be matched head-to-head. Team 1 takes seats 0 and
2, Team 2 takes seats 1 and 3; both sides are described by the spec grammar
in `agents.py`:

    random | heuristic | untrained | nfsp:<path.pth> | dmc:<path.pt>
    pimc | pimc:<determinizations>

For each matchup it reports win-rate with a Wilson 95% confidence interval,
tie rate, and mean trick differential. Results are serialized to JSON and
(optionally) CSV.

Design decisions
----------------
* Evaluation is **deterministic** given `--seed`. Each matchup uses a distinct
  derived seed so they can be compared against each other consistently.
* Exploration is off: ε = 0, η = 0, `learning_enabled = False` on every
  learned seat — see `dev_eval.run_evaluation`'s docstring for why. That is
  enforced centrally in `agents.make_team`, not here.
* We quantify uncertainty with **Wilson intervals** rather than Normal
  approximations because win rates near 0 / 1 on small N are badly skewed
  under Normal.

Examples
--------
    # New, general form
    python evaluate.py --team1 dmc:models/dmc_latest.pt --team2 pimc:32 \\
        --games 500 --seed 42

    # Sweep a DMC net against every checkpoint-free baseline
    python evaluate.py --team1 dmc:models/dmc_latest.pt --team2 all --games 500

    # Legacy form (still supported): NFSP checkpoint vs the baseline suite
    python evaluate.py --checkpoint models/nfsp_..._game_1000.pth \\
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

from agents import BASELINE_SPECS, SpecError, make_team, parse_spec, spec_checkpoint
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
    """One head-to-head result. `opponent` is the display label for Team 2 and
    is kept as the first field for backward compatibility with existing JSON /
    CSV consumers (dev_blueprint's eval endpoint, saved eval suites); `team1`
    and `team2` carry the full specs actually played."""

    opponent: str
    team1: str
    team2: str
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
# Match construction
# ------------------------------------------------------------------

# Offset so Team 1's seat RNGs never collide with Team 2's for the same seed.
_TEAM1_SEED_OFFSET = 500_000


def build_match(team1: str, team2: str, seed: int) -> Hokm:
    """
    Construct a 4-player Hokm game: Team 1 (seats 0, 2) vs Team 2 (seats 1, 3).

    Deterministic given `seed`.
    """
    rng = random.Random(seed)

    t1 = make_team(
        team1,
        seed=seed + _TEAM1_SEED_OFFSET,
        names=("Player 1 (T1 S)", "Player 3 (T1 N)"),
    )
    t2 = make_team(team2, seed=seed, names=("Player 2 (T2 E)", "Player 4 (T2 W)"))

    players = [t1[0], t2[0], t1[1], t2[1]]
    return Hokm(players, minimal_logging=True, rng=rng)


# ------------------------------------------------------------------
# Match loop
# ------------------------------------------------------------------

def run_match(
    team1: str,
    team2: str,
    games: int,
    seed: int,
    label: Optional[str] = None,
) -> MatchupResult:
    """Play `games` hands of `team1` (seats 0+2) vs `team2` (seats 1+3).

    Wins are counted for Team 1. `label` overrides the display name for Team 2
    (used so a legacy `--opponent self` run still prints as "self")."""
    seed_all(seed)
    game = build_match(team1, team2, seed)

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
        opponent=label or team2,
        team1=team1,
        team2=team2,
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


def run_matchup(
    checkpoint: str,
    opponent: str,
    games: int,
    seed: int,
) -> MatchupResult:
    """Legacy entry point: NFSP `checkpoint` (Team 1) vs a named baseline.

    Kept because `dev_blueprint`'s eval endpoint calls it positionally.
    Delegates to `run_match`."""
    team1 = f"nfsp:{checkpoint}"
    team2 = team1 if opponent == "self" else opponent
    if opponent not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent!r}")
    return run_match(team1, team2, games, seed, label=opponent)


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def _print_table(results: List[MatchupResult]) -> None:
    from agents import describe

    if not results:
        return
    t1_label = describe(results[0].team1)
    width = max(12, max(len(describe(r.team2)) for r in results))
    print(f"Team 1 (seats 0+2): {t1_label}")
    header = (
        f"{'opponent':<{width}} {'games':>6} {'wins':>5} {'ties':>5} "
        f"{'win%':>7} {'95% CI':>18} {'Δ tricks':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"[{r.ci95_low*100:5.1f}, {r.ci95_high*100:5.1f}]"
        name = r.opponent if r.opponent != r.team2 else describe(r.team2)
        print(
            f"{name:<{width}} {r.games:>6d} {r.wins:>5d} {r.ties:>5d} "
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

# Legacy `--opponent` choices. "self" means "a second copy of Team 1".
OPPONENTS = ("random", "heuristic", "self", "untrained")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate two Hokm agent families head-to-head.",
        epilog=(
            "Agent specs: random | heuristic | untrained | nfsp:<path.pth> | "
            "dmc:<path.pt> | pimc | pimc:<determinizations>"
        ),
    )
    ap.add_argument(
        "--team1",
        default=None,
        help="Spec for the team in seats 0 and 2 (the one whose win-rate is reported).",
    )
    ap.add_argument(
        "--team2",
        default=None,
        help=(
            "Spec for the team in seats 1 and 3, or 'all' for each of "
            + "/".join(BASELINE_SPECS)
            + " in turn."
        ),
    )
    # --- backward-compatible aliases -------------------------------------
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Deprecated alias: NFSP .pth for Team 1 (same as --team1 nfsp:<path>).",
    )
    ap.add_argument(
        "--opponent",
        default=None,
        choices=OPPONENTS + ("all",),
        help="Deprecated alias for --team2 (adds 'self' = another copy of Team 1).",
    )
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    ap.add_argument("--csv", default=None, help="Optional CSV output path.")
    return ap


def resolve_matchups(args: argparse.Namespace) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Translate parsed args (new *or* legacy flags) into the new path.

    Returns `(team1_spec, [(label, team2_spec), ...])`. Pure and side-effect
    free so the legacy-flag mapping is directly unit-testable.
    """
    legacy = args.team1 is None and args.checkpoint is not None

    if args.team1 is not None:
        team1 = args.team1
    elif args.checkpoint is not None:
        team1 = f"nfsp:{args.checkpoint}"
    else:
        raise SpecError(
            "Team 1 is required: pass --team1 <spec> (or the legacy "
            "--checkpoint <path.pth>)."
        )
    parse_spec(team1)  # fail fast on a malformed spec

    requested = args.team2 if args.team2 is not None else args.opponent
    if requested is None:
        requested = "all"

    if requested == "all":
        # The legacy suite includes the self-play sanity check; the general
        # form expands to the checkpoint-free baselines only, since "all"
        # must be meaningful for any Team 1.
        labels = list(OPPONENTS) if legacy else list(BASELINE_SPECS)
    else:
        labels = [requested]

    matchups: List[Tuple[str, str]] = []
    for label in labels:
        spec = team1 if label == "self" else label
        parse_spec(spec)
        matchups.append((label, spec))
    return team1, matchups


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        team1, matchups = resolve_matchups(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    # Fail fast on missing checkpoints instead of dying mid-suite.
    for spec in [team1] + [s for _, s in matchups]:
        ckpt = spec_checkpoint(spec)
        if ckpt and not os.path.isfile(ckpt):
            print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
            return 2

    # Derive a distinct seed per matchup so matchups aren't all identical deals.
    results = [
        run_match(team1, spec, args.games, seed=args.seed + i, label=label)
        for i, (label, spec) in enumerate(matchups)
    ]

    _print_table(results)

    if args.out:
        payload = {
            "team1": team1,
            # Retained for consumers that keyed off the old field.
            "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
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

#!/usr/bin/env python3
"""
A/B harness for PIMC search changes.

Seats a PIMC team (seats 0 and 2 = engine Team 1) against a HeuristicAgent
team (seats 1 and 3 = Team 2) and plays N seeded hands, reporting wins with
a Wilson 95% interval and mean trick differential.

Two variants can be run over the *same* seeds so the comparison is paired:

  * ``new`` — the current PIMC: hakem trump-bias in determinization plus
    last-seat candidate pruning.
  * ``old`` — the pre-change behaviour: no bias (multiplier 1.0, which is
    numerically a no-op) and no pruning. The rollout-speed work is not
    toggled here because it is decision-neutral by construction and pinned
    as such in ``tests/test_pimc.py::TestDecisionPins``.

Reproducibility
---------------
`--seed` alone is *not* enough: `baselines.HeuristicAgent` picks its lead
suit with ``max()`` over a ``set`` of suit *names*, so ties are broken by
string-hash order and the same seed gives different results between
processes (measured: 10 vs 7 wins out of 20). This script therefore
re-execs itself once with ``PYTHONHASHSEED=0`` unless it is already set.

Example
-------
    python scripts/pimc_ab.py --games 60 --determinizations 24 --seed 42 \\
        --variant both
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Tuple

if os.environ.get("PYTHONHASHSEED") is None:
    # Must be set before the interpreter starts, hence the re-exec.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pimc  # noqa: E402
from baselines import HeuristicAgent  # noqa: E402
from hokm import Hokm  # noqa: E402
from pimc import PIMCPlayer  # noqa: E402


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def run(variant: str, games: int, determinizations: int, seed: int) -> dict:
    prune = variant == "new"
    # `old` disables the hakem inference by flattening the multiplier.
    saved_bias = pimc.HAKEM_TRUMP_BIAS
    if variant == "old":
        pimc.HAKEM_TRUMP_BIAS = 1.0
    try:
        p0 = PIMCPlayer(
            "PIMC S", determinizations=determinizations,
            rng=random.Random(seed + 11), prune_last_seat=prune,
        )
        p2 = PIMCPlayer(
            "PIMC N", determinizations=determinizations,
            rng=random.Random(seed + 22), prune_last_seat=prune,
        )
        o1 = HeuristicAgent("Heur E", rng=random.Random(seed + 33))
        o3 = HeuristicAgent("Heur W", rng=random.Random(seed + 44))
        game = Hokm([p0, o1, p2, o3], minimal_logging=True, rng=random.Random(seed))

        wins = losses = ties = 0
        sum_t1 = sum_t2 = 0
        t_wall = time.perf_counter()
        t_cpu = time.process_time()
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
        wall = time.perf_counter() - t_wall
        cpu = time.process_time() - t_cpu
    finally:
        pimc.HAKEM_TRUMP_BIAS = saved_bias

    low, high = wilson_ci(wins, games)
    return {
        "variant": variant,
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / games if games else 0.0,
        "ci95": (low, high),
        "mean_trick_diff": (sum_t1 - sum_t2) / games if games else 0.0,
        "wall_s": wall,
        "cpu_s": cpu,
        "s_per_game": wall / games if games else 0.0,
        "aborted": game.aborted_games,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--determinizations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=("new", "old", "both"), default="both")
    args = ap.parse_args()

    variants = ("old", "new") if args.variant == "both" else (args.variant,)
    header = (
        f"{'variant':<8} {'games':>6} {'wins':>5} {'ties':>5} {'win%':>7} "
        f"{'95% CI':>16} {'Δtricks':>8} {'s/game':>8}"
    )
    print(f"PIMC vs HeuristicAgent  d={args.determinizations}  seed={args.seed}")
    print(header)
    print("-" * len(header))
    for v in variants:
        r = run(v, args.games, args.determinizations, args.seed)
        ci = f"[{r['ci95'][0] * 100:4.1f},{r['ci95'][1] * 100:5.1f}]"
        print(
            f"{r['variant']:<8} {r['games']:>6d} {r['wins']:>5d} {r['ties']:>5d} "
            f"{r['win_rate'] * 100:>6.1f}% {ci:>16} "
            f"{r['mean_trick_diff']:>+8.2f} {r['s_per_game']:>8.3f}"
        )
        if r["aborted"]:
            print(f"  WARNING: {r['aborted']} aborted hand(s)")


if __name__ == "__main__":
    main()

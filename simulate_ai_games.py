#!/usr/bin/env python3
"""
Run N complete all-AI Hokm games and write a CSV of every trick for manual review.

Teams (fixed seats): Team 1 = seat0 + seat2, Team 2 = seat1 + seat3.

Usage:
  python simulate_ai_games.py --games 100
  python simulate_ai_games.py --games 100 -o game_logs/my_review.csv
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

from enhanced_player import EnhancedPlayer
from game_constants import ACTION_DIM, STATE_DIM
from hokm import Hokm


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate all-AI Hokm games and log each trick to CSV."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of full games (first team to 7 tricks wins each game).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="CSV output path (default: game_logs/trick_review_<timestamp>.csv)",
    )
    args = parser.parse_args(argv)

    if args.games < 1:
        print("--games must be at least 1", file=sys.stderr)
        return 2

    out_path = args.output or os.path.join(
        "game_logs",
        f"trick_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    players = [
        EnhancedPlayer(f"Seat{i}", STATE_DIM, ACTION_DIM) for i in range(4)
    ]
    game = Hokm(players, trick_csv_path=out_path)

    for i in range(args.games):
        print(f"Game {i + 1}/{args.games}...", flush=True)
        game.play_game(save_excel_log=False)

    game.close_trick_csv()
    print(f"Wrote {os.path.abspath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

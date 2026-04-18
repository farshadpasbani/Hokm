"""
Standalone NFSP training script (no Flask).

Writes checkpoints to `<project>/models/nfsp_shared_<session>_game_<N>.pth`
so they land in the same directory the dev console and web app look in.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch

from enhanced_player import EnhancedPlayer, SharedNFSPLearner
from game_constants import ACTION_DIM, STATE_DIM
from hokm import Hokm
from seed_utils import seed_all

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_ROOT, "models")


def train_ai_players(
    num_games: int = 1000,
    save_interval: int = 100,
    seed: "int | None" = None,
) -> str:
    """Run self-play training and return the path of the final checkpoint."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    seed_all(seed)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    shared = SharedNFSPLearner()
    training_players = [
        EnhancedPlayer(f"Player {i + 1}", STATE_DIM, ACTION_DIM, shared_learner=shared)
        for i in range(4)
    ]

    game = Hokm(training_players, minimal_logging=True)
    last_path = ""

    for i in range(num_games):
        game.play_game(save_excel_log=False)
        if (i + 1) % save_interval == 0:
            ck = os.path.join(
                MODELS_DIR, f"nfsp_shared_{session_id}_game_{i + 1}.pth"
            )
            torch.save(shared.export_state_dict(), ck)
            last_path = ck

    final = os.path.join(MODELS_DIR, f"nfsp_shared_{session_id}_final.pth")
    torch.save(shared.export_state_dict(), final)
    return final


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Train Hokm NFSP agents (self-play).")
    ap.add_argument("--num-games", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None, help="Reproducibility seed")
    args = ap.parse_args()
    final = train_ai_players(
        num_games=args.num_games,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    print(f"Final checkpoint: {final}")


if __name__ == "__main__":
    _cli()

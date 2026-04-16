from game_constants import STATE_DIM, ACTION_DIM
from hokm import Hokm
from enhanced_player import EnhancedPlayer
import torch
import pandas as pd
import os
from datetime import datetime


def train_ai_players(num_games=1000):
    training_players = [
        EnhancedPlayer("Training AI 1", STATE_DIM, ACTION_DIM),
        EnhancedPlayer("Training AI 2", STATE_DIM, ACTION_DIM),
        EnhancedPlayer("Training AI 3", STATE_DIM, ACTION_DIM),
        EnhancedPlayer("Training AI 4", STATE_DIM, ACTION_DIM),
    ]

    # Create game with training players
    game = Hokm(training_players)

    # Train for specified number of games
    for i in range(num_games):
        game.play_game()

        # Save models periodically
        if (i + 1) % 100 == 0:
            for j, player in enumerate(training_players):
                torch.save(
                    player.policy_net.state_dict(), f"training_ai_{j+1}_policy_net.pth"
                )
            print(f"Saved models after {i + 1} games")

    # Save final models
    for i, player in enumerate(training_players):
        torch.save(
            player.policy_net.state_dict(), f"final_training_ai_{i+1}_policy_net.pth"
        )

    # Save game log
    game.save_game_log()


if __name__ == "__main__":
    train_ai_players()

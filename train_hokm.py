from game_constants import STATE_DIM, ACTION_DIM
from hokm import Hokm
from enhanced_player import EnhancedPlayer, SharedNFSPLearner
import torch


def train_ai_players(num_games=1000):
    shared = SharedNFSPLearner()
    training_players = [
        EnhancedPlayer(f"Training AI {i + 1}", STATE_DIM, ACTION_DIM, shared_learner=shared)
        for i in range(4)
    ]

    game = Hokm(training_players, minimal_logging=True)

    for i in range(num_games):
        game.play_game(save_excel_log=False)

        if (i + 1) % 100 == 0:
            torch.save(
                shared.export_state_dict(),
                f"training_ai_nfsp_shared_{i + 1}.pth",
            )

    torch.save(shared.export_state_dict(), "final_training_ai_nfsp_shared.pth")

    game.save_game_log()


if __name__ == "__main__":
    train_ai_players()

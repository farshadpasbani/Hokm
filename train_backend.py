# train_backend.py
import os
import time

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; training runs off the main thread (e.g. Flask)
import matplotlib.pyplot as plt
import pandas as pd
import torch
import traceback
from hokm import Hokm
from enhanced_player import EnhancedPlayer, SharedNFSPLearner
from datetime import datetime
from typing import Callable, Optional


class TrainBackend:
    def __init__(self, num_games=1000, model_save_interval=100):
        self.num_games = num_games
        self.model_save_interval = model_save_interval
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.shared_learner = SharedNFSPLearner()
        self.players = []
        for i in range(4):
            try:
                player = EnhancedPlayer(
                    f"Player {i + 1}", shared_learner=self.shared_learner
                )
                self.players.append(player)
            except Exception as e:
                raise
        for player in self.players:
            if not hasattr(player, "model"):
                raise ValueError(f"Player {player.name} has no model attribute")
        self.game = Hokm(self.players, minimal_logging=True)
        self.summary_data = []
        self.metrics = {
            "game_number": [],
            "team1_win_rate": [],
            "team2_win_rate": [],
            "player1_avg_reward": [],
            "player2_avg_reward": [],
            "player3_avg_reward": [],
            "player4_avg_reward": [],
            "player1_trick_wins": [],
            "player2_trick_wins": [],
            "player3_trick_wins": [],
            "player4_trick_wins": [],
        }
        self._time_play_games = 0.0
        self._time_post_game = 0.0
        self._time_checkpoints = 0.0

    def train(
        self,
        stop_event=None,
        on_progress=None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Run self-play training for num_games.

        stop_event: optional threading.Event; checked between games.
        on_progress: optional callback(completed_game_index_1based, metrics_dict)
                       metrics_dict has keys game_number, team1_win_rate, ... (lists).
        log_fn: optional one-line logger (e.g. dev console) for summaries and stop reason.
        """
        os.makedirs("models", exist_ok=True)
        os.makedirs("summaries", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("game_logs", exist_ok=True)
        successful_games = 0

        for game_idx in range(self.num_games):
            if stop_event is not None and stop_event.is_set():
                if log_fn:
                    log_fn(
                        f"Stop requested before game {game_idx + 1}; "
                        f"ending after {game_idx} game(s) finished ({self.num_games} planned)."
                    )
                break
            try:
                self.game.game_log = pd.DataFrame()
                for player in self.players:
                    if not hasattr(player, "model"):
                        raise ValueError(
                            f"Player {player.name} lost model attribute before game {game_idx + 1}"
                        )
                t0 = time.perf_counter()
                self.game.play_game(save_excel_log=False)
                self._time_play_games += time.perf_counter() - t0

                t1 = time.perf_counter()
                summary = self.game._create_summary_statistics()
                if not summary.empty:
                    self.summary_data.append(summary)
                    successful_games += 1

                    self.metrics["game_number"].append(game_idx + 1)
                    self.metrics["team1_win_rate"].append(
                        summary["Team 1 Win Rate"].iloc[0]
                    )
                    self.metrics["team2_win_rate"].append(
                        summary["Team 2 Win Rate"].iloc[0]
                    )
                    self.metrics["player1_avg_reward"].append(
                        summary["Player 1 Avg Reward"].iloc[0]
                    )
                    self.metrics["player2_avg_reward"].append(
                        summary["Player 2 Avg Reward"].iloc[0]
                    )
                    self.metrics["player3_avg_reward"].append(
                        summary["Player 3 Avg Reward"].iloc[0]
                    )
                    self.metrics["player4_avg_reward"].append(
                        summary["Player 4 Avg Reward"].iloc[0]
                    )
                    self.metrics["player1_trick_wins"].append(
                        summary["Player 1 Trick Wins"].iloc[0]
                    )
                    self.metrics["player2_trick_wins"].append(
                        summary["Player 2 Trick Wins"].iloc[0]
                    )
                    self.metrics["player3_trick_wins"].append(
                        summary["Player 3 Trick Wins"].iloc[0]
                    )
                    self.metrics["player4_trick_wins"].append(
                        summary["Player 4 Trick Wins"].iloc[0]
                    )
                self._time_post_game += time.perf_counter() - t1

                if (
                    game_idx + 1
                ) % self.model_save_interval == 0 or game_idx == self.num_games - 1:
                    tc0 = time.perf_counter()
                    try:
                        ck_path = os.path.join(
                            "models",
                            f"nfsp_shared_{self.session_id}_game_{game_idx + 1}.pth",
                        )
                        torch.save(
                            self.shared_learner.export_state_dict(),
                            ck_path,
                        )
                    except Exception:
                        traceback.print_exc()
                    self._time_checkpoints += time.perf_counter() - tc0

                if on_progress is not None:
                    snap = {k: list(v) for k, v in self.metrics.items()}
                    on_progress(game_idx + 1, snap)

            except Exception:
                traceback.print_exc()
                continue

        if log_fn:
            log_fn(
                f"Training loop done: {successful_games} successful game(s) with metrics "
                f"out of {self.num_games} planned (each planned game was attempted unless stopped early)."
            )
            log_fn(
                f"Profile (s): play_games={self._time_play_games:.2f}, "
                f"post_game_metrics={self._time_post_game:.2f}, "
                f"checkpoints={self._time_checkpoints:.2f}"
            )
        if successful_games == 0:
            if log_fn:
                log_fn(
                    "No successful games — skipping summary CSV and plots (check console for per-game errors)."
                )
            return

        self.save_summaries()
        self.generate_visualizations()

    def save_summaries(self):
        if not self.summary_data:
            return
        summary_df = pd.concat(self.summary_data, ignore_index=True)
        summary_path = f"summaries/summary_{self.session_id}.csv"
        summary_df.to_csv(summary_path, index=False)

    def generate_visualizations(self):
        if not self.metrics["game_number"]:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.metrics["game_number"],
            self.metrics["team1_win_rate"],
            label="Team 1 Win Rate",
        )
        plt.plot(
            self.metrics["game_number"],
            self.metrics["team2_win_rate"],
            label="Team 2 Win Rate",
        )
        plt.xlabel("Game Number")
        plt.ylabel("Win Rate")
        plt.title("Team Win Rates Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/team_win_rates_{self.session_id}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for i in range(1, 5):
            plt.plot(
                self.metrics["game_number"],
                self.metrics[f"player{i}_avg_reward"],
                label=f"Player {i} Avg Reward",
            )
        plt.xlabel("Game Number")
        plt.ylabel("Average Reward")
        plt.title("Player Average Rewards Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/player_avg_rewards_{self.session_id}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for i in range(1, 5):
            plt.plot(
                self.metrics["game_number"],
                self.metrics[f"player{i}_trick_wins"],
                label=f"Player {i} Trick Wins",
            )
        plt.xlabel("Game Number")
        plt.ylabel("Trick Wins")
        plt.title("Player Trick Wins Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/player_trick_wins_{self.session_id}.png")
        plt.close()


if __name__ == "__main__":
    trainer = TrainBackend(num_games=1000, model_save_interval=100)
    trainer.train()

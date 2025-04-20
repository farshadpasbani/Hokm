# train_backend.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import traceback
from hokm import Hokm
from enhanced_player import EnhancedPlayer
from datetime import datetime


class TrainBackend:
    def __init__(self, num_games=1000, model_save_interval=100):
        self.num_games = num_games
        self.model_save_interval = model_save_interval
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.players = []
        for i in range(4):
            try:
                player = EnhancedPlayer(f"Player {i+1}")
                self.players.append(player)
                print(f"Initialized {player.name}")
            except Exception as e:
                print(f"Failed to initialize Player {i+1}: {e}")
                raise
        for player in self.players:
            if not hasattr(player, "model"):
                raise ValueError(f"Player {player.name} has no model attribute")
        self.game = Hokm(self.players)
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

    def train(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("summaries", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("game_logs", exist_ok=True)
        successful_games = 0

        for game_idx in range(self.num_games):
            print(f"\nStarting game {game_idx + 1}")
            try:
                self.game.game_log = pd.DataFrame()
                for player in self.players:
                    if not hasattr(player, "model"):
                        raise ValueError(
                            f"Player {player.name} lost model attribute before game {game_idx + 1}"
                        )
                self.game.play_game()

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
                    print(f"Collected summary for game {game_idx + 1}")
                else:
                    print(f"Warning: Empty summary for game {game_idx + 1}")

                save_full_log = game_idx == self.num_games - 1
                self.game.save_game_log(
                    file_name=f"game_log_{self.session_id}_last_game.xlsx",
                    save_full_log=save_full_log,
                )

                if (
                    game_idx + 1
                ) % self.model_save_interval == 0 or game_idx == self.num_games - 1:
                    for player in self.players:
                        try:
                            if hasattr(player, "model"):
                                model_path = (
                                    f"models/{player.name}_game_{game_idx + 1}.pth"
                                )
                                torch.save(player.model.state_dict(), model_path)
                                print(f"Saved model for {player.name} to {model_path}")
                            else:
                                print(
                                    f"Warning: {player.name} has no model attribute, skipping model save"
                                )
                        except Exception as e:
                            print(f"Error saving model for {player.name}: {e}")

            except Exception as e:
                print(f"Error in game {game_idx + 1}: {e}")
                print("Stack trace:")
                traceback.print_exc()
                for player in self.players:
                    print(
                        f"Player {player.name} model exists: {hasattr(player, 'model')}"
                    )
                print(f"Game log columns: {list(self.game.game_log.columns)}")
                print(f"Game log size: {len(self.game.game_log)}")
                continue

        print(f"Completed {successful_games} successful games out of {self.num_games}")
        if successful_games == 0:
            print(
                "Warning: No successful games, skipping summary and visualization generation"
            )
            return

        self.save_summaries()
        self.generate_visualizations()

    def save_summaries(self):
        if not self.summary_data:
            print("No summary data to save")
            return
        summary_df = pd.concat(self.summary_data, ignore_index=True)
        summary_path = f"summaries/summary_{self.session_id}.xlsx"
        try:
            summary_df.to_excel(summary_path, index=False)
            print(f"Saved summary metrics to {summary_path}")
        except Exception as e:
            print(f"Error saving summary Excel: {e}")
            summary_df.to_csv(summary_path.replace(".xlsx", ".csv"), index=False)
            print(f"Saved summary as CSV to {summary_path.replace('.xlsx', '.csv')}")

    def generate_visualizations(self):
        if not self.metrics["game_number"]:
            print("No metrics data for visualizations")
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

        print(f"Saved visualizations to plots/ directory")


if __name__ == "__main__":
    trainer = TrainBackend(num_games=1000, model_save_interval=100)
    trainer.train()

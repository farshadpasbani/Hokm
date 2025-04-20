# run_game.py

from hokm import Hokm
from enhanced_player import EnhancedPlayer, TeamStrategy
import pandas as pd
import os
import traceback

try:
    # Initialize shared team strategy
    team_strategy = TeamStrategy()
    players = [
        EnhancedPlayer(
            f"Player {i+1}",
            state_dim=114,
            action_dim=13,
            team_strategy=team_strategy,
            epsilon=0.1,
        )
        for i in range(4)
    ]

    # Initialize game
    game = Hokm(players)
    game.hakem = players[0]

    # Play multiple games
    num_games = 6  # Test for error in game 6
    for i in range(num_games):
        print(f"\nStarting Game {game.game_count + 1}")
        try:
            print(f"Deck size at start: {len(game.deck.cards)}")
            print(
                f"Player hands: {[p.name + ': ' + str([str(c) for c in p.hand]) for p in game.players]}"
            )
            game.play_game()
            print(
                f"Game {game.game_count} completed. Team 1 Score: {game.scores[1]}, Team 2 Score: {game.scores[2]}"
            )
            for player in players:
                print(f"{player.name} Total Reward: {player.total_reward:.2f}")
        except Exception as e:
            print(f"Error in game {game.game_count + 1}: {e}")
            traceback.print_exc()
            game.save_game_log()  # Save partial log
            continue

    # Verify all games are logged
    print("\nGame Log Summary:")
    log_file = os.path.join(
        "game_logs", f"game_log_{game.session_id}_game_{game.game_count}.xlsx"
    )
    try:
        log_df = pd.read_excel(log_file, sheet_name="All Games")
        unique_games = log_df["Game"].unique()
        print(f"Total games logged: {len(unique_games)}")
        print(f"Games: {unique_games}")
        summary = pd.read_excel(log_file, sheet_name="Summary")
        print("\nSummary Statistics:")
        print(summary)
    except FileNotFoundError:
        print(f"Game log file {log_file} not found. Check if games were logged.")
    except Exception as e:
        print(f"Error reading game log: {e}")

except Exception as e:
    print(f"Error running game: {e}")
    traceback.print_exc()

import matplotlib.pyplot as plt
from hokm_game import Hokm, DQNPlayer
import torch

# Create players
players = [
    DQNPlayer("Player 1", 52, 13, epsilon=0.1),
    DQNPlayer("Player 2", 52, 13, epsilon=0.1),
    DQNPlayer("Player 3", 52, 13, epsilon=0.1),
    DQNPlayer("Player 4", 52, 13, epsilon=0.1),
]

# Training settings
num_games = 1000
save_frequency = 100  # Save model weights every 100 games

# Tracking metrics
reward_tracker = {player.name: [] for player in players}
win_tracker = {player.name: [] for player in players}
q_value_history = {player.name: [] for player in players}
action_counts = {
    player.name: [0] * 13 for player in players
}  # 13 possible card actions

# Training loop
for game_num in range(num_games):
    print(f"\nStarting Game {game_num + 1}/{num_games}")
    game = Hokm(players)
    game.play_game()  # This will update rewards, Q-values, and actions for each player

    # Track cumulative rewards and wins
    for player in players:
        # Assume `total_reward` is accumulated in play_game for each player
        reward_tracker[player.name].append(player.total_reward)

        # Track wins (1 if the player's team wins, otherwise 0)
        win = 1 if game.tricks_won[player] >= 7 else 0
        win_tracker[player.name].append(win)

        # Track max Q-value for convergence analysis
        if player.policy_net is not None:  # In case policy_net is defined elsewhere
            state = torch.tensor(player.get_state(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = player.policy_net(state)
                max_q_value = q_values.max().item()
                q_value_history[player.name].append(max_q_value)

        # Track action counts
        # Track action counts
        for action in player.actions_taken:
            if 0 <= action < len(action_counts[player.name]):  # Check if within bounds
                action_counts[player.name][action] += 1
            else:
                print(f"Warning: Action {action} is out of range for {player.name}")

    # Save models periodically
    if (game_num + 1) % save_frequency == 0:
        for player in players:
            torch.save(player.policy_net.state_dict(), f"{player.name}_policy_net.pth")

game.save_game_log("hokm_game_log.xlsx")
print("Training complete.")

# Plot Cumulative Rewards
plt.figure(figsize=(10, 6))
for player_name, rewards in reward_tracker.items():
    plt.plot(rewards, label=player_name)
plt.xlabel("Games")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards over Training")
plt.legend()
plt.show()

# Plot Win Rates
plt.figure(figsize=(10, 6))
for player_name, wins in win_tracker.items():
    win_rate = [sum(wins[: i + 1]) / (i + 1) for i in range(len(wins))]
    plt.plot(win_rate, label=player_name)
plt.xlabel("Games")
plt.ylabel("Win Rate")
plt.title("Win Rate over Time")
plt.legend()
plt.show()

# Plot Q-Value Convergence
plt.figure(figsize=(10, 6))
for player_name, q_values in q_value_history.items():
    plt.plot(q_values, label=player_name)
plt.xlabel("Training Steps")
plt.ylabel("Max Q-Value")
plt.title("Q-Value Convergence")
plt.legend()
plt.show()

# Plot Action Distribution
plt.figure(figsize=(10, 6))
for player_name, counts in action_counts.items():
    plt.bar(range(13), counts, alpha=0.7, label=player_name)
plt.xlabel("Action (Card Index)")
plt.ylabel("Frequency")
plt.title("Action Distribution over Training")
plt.legend()
plt.show()

# hokm.py

import random
import torch
import pandas as pd
import os
from datetime import datetime
from game_constants import Card, suits, ranks, rank_values
from enhanced_player import EnhancedPlayer, TeamStrategy
import time
import csv


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        if len(self.cards) != 52:
            raise ValueError(
                f"Deck initialized with {len(self.cards)} cards, expected 52"
            )

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        print(f"Deck size before dealing: {len(self.cards)}")
        if len(self.cards) < num_cards:
            raise ValueError(f"Not enough cards in deck to deal {num_cards} cards")
        dealt_cards = [self.cards.pop() for _ in range(num_cards)]
        print(f"Deck size after dealing: {len(self.cards)}")
        for card in dealt_cards:
            if not isinstance(card, Card):
                raise ValueError(f"Invalid card dealt: {card}")
        return dealt_cards


class Hokm:
    def __init__(self, players):
        self.players = players
        self.deck = Deck()
        self.current_trick = []
        self.lead_suit = None
        self.trump_suit = None
        self.hakem = None
        self.scores = {1: 0, 2: 0}
        self.game_log = pd.DataFrame()
        self.difficulty_level = 1
        self.hakem_cards = None
        self.game_count = 0
        self.round_count = 0
        self.trick_count = 0
        self.team1 = [self.players[0], self.players[2]]
        self.team2 = [self.players[1], self.players[3]]
        self.team_strategy = TeamStrategy()
        self.tricks_won = {player: 0 for player in self.players}
        self.last_winning_team = self.team1
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        for player in self.players:
            player.team = self.team1 if player in self.team1 else self.team2
            player.tricks_won = self.tricks_won
            player.team_strategy = self.team_strategy

    def reset_players(self):
        for player in self.players:
            player.reset()
            if player.hand:
                print(f"Clearing {player.name}'s hand: {[str(c) for c in player.hand]}")
                player.hand = []  # Force clear the hand
        self.tricks_won = {player: 0 for player in self.players}
        self.scores = {1: 0, 2: 0}
        self.current_trick = []
        self.lead_suit = None
        self.round_count = 0
        self.trick_count = 0

    def start_game(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.reset_players()
        if not self.hakem:
            self.hakem = random.choice(self.players)
            print(f"{self.hakem.name} is the Hakem for this game")
        self.hakem_cards = self.deck.deal(5)
        self.hakem.hand = self.hakem_cards.copy()
        print(
            f"Hakem {self.hakem.name} received cards: {[str(c) for c in self.hakem.hand]}"
        )
        self.log_game_state("Game initialized", player_hands=True)
        return self.hakem_cards

    def set_trump_suit(self, trump_suit):
        if trump_suit not in suits:
            raise ValueError(f"Invalid trump suit: {trump_suit}")
        self.trump_suit = trump_suit
        for player in self.players:
            player.update_trump_suit(trump_suit)
        print(f"Trump suit set to: {self.trump_suit}")
        cards_per_player = 8 if self.hakem else 13
        for player in self.players:
            num_cards = 8 if player == self.hakem else 13
            player.draw(self.deck, num_cards)
            print(f"{player.name} hand after draw: {[str(c) for c in player.hand]}")
        self.log_game_state("Game started", player_hands=True)

    def choose_trump_suit(self):
        suit_counts = {suit: 0 for suit in suits}
        suit_values = {suit: 0 for suit in suits}
        for card in self.hakem.hand:
            suit_counts[card.suit] += 1
            suit_values[card.suit] += card.value * (2 if card.value >= 10 else 1)
        best_suit = max(suits, key=lambda s: suit_counts[s] * 10 + suit_values[s])
        self.set_trump_suit(best_suit)

    def play_round(self):
        self.current_trick = []
        self.lead_suit = None
        hakem_index = self.players.index(self.hakem)

        # For the first trick, Hakem plays first; otherwise, use the trick winner or next player
        if self.round_count == 0:
            starting_player_index = hakem_index
            print(f"First trick: Hakem ({self.hakem.name}) plays first")
        else:
            starting_player_index = (
                hakem_index + 1
            ) % 4  # Fallback or adjust based on trick winner
            print(
                f"Trick {self.round_count + 1}: Starting with player {self.players[starting_player_index].name}"
            )

        current_player = self.players[starting_player_index]

        player_rewards = {}
        player_action_indices = {}
        player_valid_cards = {}
        play_order = []  # Track order of players in this trick

        for _ in range(4):
            try:
                print(
                    f"{current_player.name} hand before play: {[str(c) for c in current_player.hand]}"
                )
                state = current_player.get_state()
                valid_cards = (
                    current_player.hand
                    if self.lead_suit is None
                    else [
                        card
                        for card in current_player.hand
                        if card.suit == self.lead_suit
                    ]
                    or current_player.hand
                )
                player_valid_cards[current_player.name] = [str(c) for c in valid_cards]

                result = current_player.play_card(self.lead_suit)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        f"Invalid return from play_card for {current_player.name}: {result}"
                    )
                card, action_index = result
                if not isinstance(card, Card):
                    raise ValueError(
                        f"Invalid card returned by {current_player.name}: {card}"
                    )

                if card not in current_player.hand:
                    error_msg = f"{current_player.name} attempted to play {card}, not in hand: {[str(c) for c in current_player.hand]}"
                    print(error_msg)
                    self.log_game_state(error_msg, player_hands=True)
                    raise ValueError(error_msg)

                current_player.hand.remove(card)
                print(f"{current_player.name} played {card}")
                print(f"Hand after removal: {[str(c) for c in current_player.hand]}")

                self.current_trick.append((current_player, card))
                play_order.append(current_player.name)
                current_player.current_trick = self.current_trick
                if not self.lead_suit:
                    self.lead_suit = card.suit

                current_player.actions_taken.append(action_index)
                current_player.played_cards_memory.add(str(card))
                current_player.team_strategy.update_card_count(card)
                current_player.played_suit_counts[suits.index(card.suit)] += 1

                reward = self.evaluate_play(
                    current_player, card, self.lead_suit, self.round_count
                )
                player_rewards[current_player.name] = reward
                player_action_indices[current_player.name] = action_index
                next_state = current_player.get_state()
                done = len(current_player.hand) == 0
                current_player.store_experience(
                    state, action_index, reward, next_state, done
                )
                current_player.optimize_model()
            except Exception as e:
                error_msg = f"Error in play_round for {current_player.name}: {str(e)}"
                print(error_msg)
                self.log_game_state(error_msg, player_hands=True)
                raise
            current_player_index = (self.players.index(current_player) + 1) % 4
            current_player = self.players[current_player_index]

        winner = self.determine_trick_winner()
        print(f"{winner.name} won the trick")
        team = 1 if winner in self.team1 else 2
        self.scores[team] += 1
        self.log_round(
            self.round_count,
            self.lead_suit,
            play_order,
            self.current_trick,
            winner,
            player_rewards,
            player_action_indices,
            player_valid_cards,
        )
        self.log_game_state("Trick completed")
        self.round_count += 1  # Increment round_count after each trick
        return winner

    def play_game(self):
        self.game_count += 1
        print(f"Starting game {self.game_count}")
        self.start_game()
        self.choose_trump_suit()
        self.round_count = 0  # Reset round_count at start of game
        current_player = self.hakem  # Start with Hakem for first trick
        while any(len(player.hand) > 0 for player in self.players):
            try:
                winner = self.play_round()
                current_player = winner  # Winner of the trick starts the next one
                if self.scores[1] >= 7 or self.scores[2] >= 7:
                    break
            except Exception as e:
                print(f"Error in round {self.round_count}: {e}")
                self.log_game_state(f"Round error: {str(e)}", player_hands=True)
                break
        self.update_last_winning_team()
        self.rotate_hakem()
        self.adjust_difficulty()
        self.log_game_state("Game ended", player_hands=True)
        self.save_game_log()

    def determine_trick_winner(self):
        winning_card = self.current_trick[0][1]
        winner = self.current_trick[0][0]
        has_trump = any(card.suit == self.trump_suit for _, card in self.current_trick)
        for player, card in self.current_trick[1:]:
            if has_trump:
                if card.suit == self.trump_suit:
                    if (
                        winning_card.suit != self.trump_suit
                        or card.value > winning_card.value
                    ):
                        winning_card = card
                        winner = player
            else:
                if card.suit == self.lead_suit:
                    if (
                        winning_card.suit != self.lead_suit
                        or card.value > winning_card.value
                    ):
                        winning_card = card
                        winner = player
        self.tricks_won[winner] += 1
        return winner

    def evaluate_play(self, player, card, lead_suit, round_num):
        return player.evaluate_play(card, lead_suit, round_num)

    def update_last_winning_team(self):
        team1_tricks = sum(self.tricks_won[player] for player in self.team1)
        team2_tricks = sum(self.tricks_won[player] for player in self.team2)
        self.last_winning_team = self.team1 if team1_tricks >= 7 else self.team2

    def rotate_hakem(self):
        current_team = self.team1 if self.hakem in self.team1 else self.team2
        if self.last_winning_team != current_team:
            self.hakem = self.last_winning_team[0]
        else:
            current_idx = current_team.index(self.hakem)
            next_idx = (current_idx + 1) % 2
            self.hakem = current_team[next_idx]

    def log_round(
        self,
        round_num,
        lead_suit,
        play_order,
        current_trick,
        winner,
        player_rewards,
        player_action_indices,
        player_valid_cards,
    ):
        played_cards = {player.name: "None" for player in self.players}
        for player, card in current_trick:
            played_cards[player.name] = str(card)

        hand_sizes = {player.name: len(player.hand) for player in self.players}
        hands = {
            player.name: [str(card) for card in player.hand] for player in self.players
        }
        team1_score = sum(self.tricks_won[player] for player in self.team1)
        team2_score = sum(self.tricks_won[player] for player in self.team2)
        game_winner = (
            "Team 1" if team1_score >= 7 else "Team 2" if team2_score >= 7 else None
        )

        trick_contents = ", ".join([f"{p.name}: {str(c)}" for p, c in current_trick])

        row = {
            "Game": self.game_count,
            "Round": round_num,
            "Trick Number": self.trick_count,
            "Hakem": self.hakem.name,
            "Trump Suit": self.trump_suit,
            "Lead Suit": lead_suit,
            "Play Order": ", ".join(play_order),
            "Trick Contents": trick_contents,
            "Player 1 Card": played_cards.get("Player 1", "None"),
            "Player 2 Card": played_cards.get("Player 2", "None"),
            "Player 3 Card": played_cards.get("Player 3", "None"),
            "Player 4 Card": played_cards.get("Player 4", "None"),
            "Trick Winner": winner.name,
            "Winning Team": "Team 1" if winner in self.team1 else "Team 2",
            "Player 1 Hand": ", ".join(hands.get("Player 1", [])),
            "Player 2 Hand": ", ".join(hands.get("Player 2", [])),
            "Player 3 Hand": ", ".join(hands.get("Player 3", [])),
            "Player 4 Hand": ", ".join(hands.get("Player 4", [])),
            "Player 1 Hand Size": hand_sizes.get("Player 1", 0),
            "Player 2 Hand Size": hand_sizes.get("Player 2", 0),
            "Player 3 Hand Size": hand_sizes.get("Player 3", 0),
            "Player 4 Hand Size": hand_sizes.get("Player 4", 0),
            "Player 1 Reward": player_rewards.get("Player 1", 0.0),
            "Player 2 Reward": player_rewards.get("Player 2", 0.0),
            "Player 3 Reward": player_rewards.get("Player 3", 0.0),
            "Player 4 Reward": player_rewards.get("Player 4", 0.0),
            "Player 1 Action Index": player_action_indices.get("Player 1", -1),
            "Player 2 Action Index": player_action_indices.get("Player 2", -1),
            "Player 3 Action Index": player_action_indices.get("Player 3", -1),
            "Player 4 Action Index": player_action_indices.get("Player 4", -1),
            "Player 1 Valid Cards": ", ".join(player_valid_cards.get("Player 1", [])),
            "Player 2 Valid Cards": ", ".join(player_valid_cards.get("Player 2", [])),
            "Player 3 Valid Cards": ", ".join(player_valid_cards.get("Player 3", [])),
            "Player 4 Valid Cards": ", ".join(player_valid_cards.get("Player 4", [])),
            "Game Winner": game_winner,
            "Team 1 Score": team1_score,
            "Team 2 Score": team2_score,
            "Difficulty Level": self.difficulty_level,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.game_log = pd.concat(
            [self.game_log, pd.DataFrame([row])], ignore_index=True
        )
        self.trick_count += 1

    def log_game_state(self, event, player_hands=False):
        hand_sizes = {player.name: len(player.hand) for player in self.players}
        hands = {
            player.name: [str(card) for card in player.hand] for player in self.players
        }
        team1_score = sum(self.tricks_won[player] for player in self.team1)
        team2_score = sum(self.tricks_won[player] for player in self.team2)
        row = {
            "Game": self.game_count,
            "Round": self.round_count,
            "Trick Number": self.trick_count,
            "Hakem": self.hakem.name if self.hakem else "None",
            "Trump Suit": self.trump_suit,
            "Lead Suit": self.lead_suit,
            "Event": event,
            "Player 1 Hand Size": hand_sizes.get("Player 1", 0),
            "Player 2 Hand Size": hand_sizes.get("Player 2", 0),
            "Player 3 Hand Size": hand_sizes.get("Player 3", 0),
            "Player 4 Hand Size": hand_sizes.get("Player 4", 0),
            "Player 1 Hand": (
                ", ".join(hands.get("Player 1", [])) if player_hands else ""
            ),
            "Player 2 Hand": (
                ", ".join(hands.get("Player 2", [])) if player_hands else ""
            ),
            "Player 3 Hand": (
                ", ".join(hands.get("Player 3", [])) if player_hands else ""
            ),
            "Player 4 Hand": (
                ", ".join(hands.get("Player 4", [])) if player_hands else ""
            ),
            "Team 1 Score": team1_score,
            "Team 2 Score": team2_score,
            "Difficulty Level": self.difficulty_level,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.game_log = pd.concat(
            [self.game_log, pd.DataFrame([row])], ignore_index=True
        )

    def save_game_log(self, file_name=None):
        if file_name is None:
            file_name = f"game_log_{self.session_id}_game_{self.game_count}.xlsx"
        os.makedirs("game_logs", exist_ok=True)
        file_path = os.path.join("game_logs", file_name)
        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                self.game_log.to_excel(writer, sheet_name="All Games", index=False)
                summary = self._create_summary_statistics()
                summary.to_excel(writer, sheet_name="Summary", index=False)
                team_stats = self._create_team_statistics()
                team_stats.to_excel(writer, sheet_name="Team Stats", index=False)
            print(f"Game log saved to {file_path}")
        except Exception as e:
            print(f"Error saving Excel game log: {e}")
            csv_path = file_path.replace(".xlsx", ".csv")
            try:
                self.game_log.to_csv(csv_path, index=False)
                print(f"Game log saved as CSV to {csv_path}")
            except Exception as csv_e:
                print(f"Error saving CSV game log: {csv_e}")

    def _create_summary_statistics(self):
        if self.game_log.empty:
            return pd.DataFrame()
        unique_games = self.game_log["Game"].unique()
        total_games = len(unique_games)
        total_tricks = len(self.game_log[self.game_log["Event"] == "Trick completed"])
        trump_counts = (
            self.game_log.groupby("Game")["Trump Suit"].first().value_counts()
        )
        most_common_trump = trump_counts.idxmax() if not trump_counts.empty else "N/A"
        game_winners = (
            self.game_log.groupby("Game")["Game Winner"].last().value_counts()
        )
        most_winning_team = game_winners.idxmax() if not game_winners.empty else "N/A"

        # Calculate team win rates
        team1_wins = len(self.game_log[self.game_log["Game Winner"] == "Team 1"])
        team2_wins = len(self.game_log[self.game_log["Game Winner"] == "Team 2"])
        team1_win_rate = team1_wins / total_games if total_games > 0 else 0
        team2_win_rate = team2_wins / total_games if total_games > 0 else 0

        # Calculate player rewards and trick wins
        player_stats = {}
        for i in range(1, 5):
            player_name = f"Player {i}"
            reward_col = f"{player_name} Reward"
            if reward_col in self.game_log.columns:
                player_stats[f"{player_name} Avg Reward"] = self.game_log[
                    reward_col
                ].mean()
            else:
                player_stats[f"{player_name} Avg Reward"] = 0.0
            player_stats[f"{player_name} Trick Wins"] = self.tricks_won[player_name]

        return pd.DataFrame(
            {
                "Total Games Played": [total_games],
                "Total Tricks Played": [total_tricks],
                "Average Tricks per Game": [
                    total_tricks / total_games if total_games > 0 else 0
                ],
                "Most Common Trump Suit": [most_common_trump],
                "Most Winning Team": [most_winning_team],
                "Team 1 Win Rate": [team1_win_rate],
                "Team 2 Win Rate": [team2_win_rate],
                **player_stats,
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )

    def _create_team_statistics(self):
        if self.game_log.empty:
            return pd.DataFrame()
        game_stats = (
            self.game_log.groupby("Game")
            .agg(
                {
                    "Game Winner": "last",
                    "Winning Team": "count",
                    "Team 1 Score": "last",
                    "Team 2 Score": "last",
                }
            )
            .reset_index()
        )
        team_stats = pd.DataFrame(
            {
                "Team": ["Team 1", "Team 2"],
                "Total Wins": [
                    len(game_stats[game_stats["Game Winner"] == "Team 1"]),
                    len(game_stats[game_stats["Game Winner"] == "Team 2"]),
                ],
                "Total Tricks Won": [
                    len(self.game_log[self.game_log["Winning Team"] == "Team 1"]),
                    len(self.game_log[self.game_log["Winning Team"] == "Team 2"]),
                ],
                "Average Tricks per Game": [
                    game_stats[game_stats["Game Winner"] == "Team 1"][
                        "Winning Team"
                    ].mean()
                    or 0,
                    game_stats[game_stats["Game Winner"] == "Team 2"][
                        "Winning Team"
                    ].mean()
                    or 0,
                ],
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )
        return team_stats

    def adjust_difficulty(self):
        team1_wins = sum(
            1
            for _, row in self.game_log.iterrows()
            if row["Game Winner"] == "Team 1"
            and row["Difficulty Level"] == self.difficulty_level
        )
        team2_wins = sum(
            1
            for _, row in self.game_log.iterrows()
            if row["Game Winner"] == "Team 2"
            and row["Difficulty Level"] == self.difficulty_level
        )
        total_games = team1_wins + team2_wins
        if total_games >= 10:
            win_rate = team1_wins / total_games if total_games > 0 else 0
            if win_rate > 0.7 and self.difficulty_level < 3:
                self.difficulty_level += 1
                print(f"Difficulty increased to level {self.difficulty_level}")
            elif win_rate < 0.3 and self.difficulty_level > 1:
                self.difficulty_level -= 1
                print(f"Difficulty decreased to level {self.difficulty_level}")

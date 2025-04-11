import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd

# Define suits and ranks
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
rank_values = {rank: i for i, rank in enumerate(ranks, start=2)}


# Define Replay Memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DQN neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Adjusted network architecture to match 52-dimensional input
        self.fc1 = nn.Linear(52, 256)  # Input is 52-dimensional one-hot encoding
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# Define the Card class
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = rank_values[rank]

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


# Define the Deck class
class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        return [self.cards.pop() for _ in range(num_cards)]


# Define the DQN-enabled Player class
class DQNPlayer:
    def __init__(self, name, state_dim, action_dim, epsilon=0.1):
        self.name = name
        self.hand = []
        self.epsilon = epsilon
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.target_update_frequency = 10
        self.steps_done = 0
        self.total_reward = 0.0
        self.actions_taken = []
        self.played_cards_memory = set()

        # Initialize DQN and target networks with correct dimensions
        self.policy_net = DQN(
            52, action_dim
        )  # Input dimension is 52 (one-hot encoding)
        self.target_net = DQN(52, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.update_target_net()

    def reset(self):
        """Reset player-specific metrics at the start of each game."""
        self.total_reward = 0.0
        self.actions_taken = []
        self.played_cards_memory.clear()

    def draw(self, deck, num_cards):
        """Draw cards from the deck to the player's hand."""
        self.hand += deck.deal(num_cards)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state(self):
        """Simplified state representation."""
        # Convert hand to state representation (13 values for each suit)
        state = [0] * 52  # One-hot encoding for all cards
        for card in self.hand:
            card_idx = suits.index(card.suit) * 13 + ranks.index(card.rank)
            state[card_idx] = 1
        return state

    def select_action(self, padded_card_values, num_valid_actions):
        if random.random() < self.epsilon:
            return random.randint(0, num_valid_actions - 1)
        else:
            with torch.no_grad():
                # Convert state to tensor
                state_tensor = torch.tensor(
                    self.get_state(), dtype=torch.float32
                ).unsqueeze(0)
                action_values = self.policy_net(state_tensor)

                # Create a mask for valid actions
                mask = torch.zeros(13, dtype=torch.float32)
                mask[:num_valid_actions] = 1

                # Apply mask to action values
                masked_action_values = action_values * mask - (1 - mask) * 1e10
                return masked_action_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.total_reward += reward  # Accumulate reward
        self.actions_taken.append(action)  # Track the action taken

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = list(zip(*experiences))
        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_net()

    def update_played_cards_memory(self, card):
        """Track played cards."""
        self.played_cards_memory.add(card)

    def play_card(self, lead_suit=None):
        """Play a card from the player's hand."""
        if not self.hand:
            raise ValueError("No cards in hand to play")

        # Filter cards to see if the player has cards in the lead suit
        if lead_suit:
            valid_cards = [card for card in self.hand if card.suit == lead_suit]
        else:
            valid_cards = self.hand

        # If no cards match the lead suit, they can play any card
        if not valid_cards:
            valid_cards = self.hand

        # Get action index within the valid range
        action_index = self.select_action(None, len(valid_cards))
        chosen_card = valid_cards[action_index]

        # Remove the chosen card from the player's hand
        self.hand.remove(chosen_card)
        self.actions_taken.append(action_index)
        return chosen_card, action_index

    def __repr__(self):
        return self.name


# Define the Hokm Game class
class Hokm:
    def __init__(self, players):
        self.players = players
        self.deck = Deck()
        self.trump_suit = None
        self.tricks_won = {player: 0 for player in players}
        self.team1 = [players[0], players[2]]
        self.team2 = [players[1], players[3]]
        self.current_hakem = self.team1[0]
        self.last_winning_team = self.team1
        self.last_trick_winner = None
        self.current_bid = 0
        self.bid_winner = None
        self.game_log = pd.DataFrame(
            columns=[
                "Game",
                "Round",
                "Hakem",
                "Trump Suit",
                "Lead Suit",
                "Player 1 Card",
                "Player 2 Card",
                "Player 3 Card",
                "Player 4 Card",
                "Trick Winner",
                "Winning Team",
                "Bid",
                "Bid Winner",
            ]
        )
        self.game_count = 0
        self.round_count = 0

    def reset_players(self):
        """Reset all players for a new game."""
        for player in self.players:
            player.hand = []
            player.played_cards_memory.clear()
            player.actions_taken = []
            player.total_reward = 0.0

    def deal_cards(self):
        """Deal all cards to players."""
        self.deck = Deck()  # Create a new deck
        self.deck.shuffle()

        # Deal initial 5 cards to each player
        for player in self.players:
            player.draw(self.deck, 5)

        # Choose trump suit
        self.choose_trump_suit()

        # Deal remaining 8 cards to each player
        for player in self.players:
            player.draw(self.deck, 8)

    def play_round(self, round_num):
        """Play a single round of the game."""
        lead_suit = None
        trick = []
        cards_played = []

        # Determine play order starting with current hakem
        start_idx = self.players.index(self.current_hakem)
        play_order = self.players[start_idx:] + self.players[:start_idx]

        # Each player plays a card
        for player in play_order:
            card_played, action_index = player.play_card(lead_suit)

            if not lead_suit:
                lead_suit = card_played.suit

            # Update played cards memory
            for p in self.players:
                if isinstance(p, DQNPlayer):
                    p.update_played_cards_memory(card_played)

            # Calculate reward
            reward = self.evaluate_play(player, card_played, lead_suit)

            # Store experience
            if isinstance(player, DQNPlayer):
                player.store_experience(
                    player.get_state(),
                    action_index,
                    reward,
                    player.get_state(),
                    done=False,
                )

            trick.append((player, card_played))
            cards_played.append(card_played)

        # Determine trick winner
        winner = self.determine_trick_winner(trick, lead_suit)
        self.tricks_won[winner] += 1
        self.last_trick_winner = winner

        # Log the round
        self.log_round(round_num, lead_suit, cards_played, winner)

        # Check if game should end
        team1_tricks = sum(self.tricks_won[player] for player in self.team1)
        team2_tricks = sum(self.tricks_won[player] for player in self.team2)

        if team1_tricks >= 7 or team2_tricks >= 7:
            # Apply final rewards
            winning_team = self.team1 if team1_tricks >= 7 else self.team2
            for player in winning_team:
                if isinstance(player, DQNPlayer):
                    player.total_reward += 10.0
            return True

        return False

    def play_game(self):
        """Play a complete game of Hokm."""
        self.reset_players()
        self.deal_cards()
        self.round_count = 0

        # Initialize tricks won for new game
        self.tricks_won = {player: 0 for player in self.players}

        # Play rounds until game ends
        while True:
            self.round_count += 1
            game_ended = self.play_round(self.round_count)
            if game_ended:
                break

        # Update game state
        self.update_last_winning_team()
        self.rotate_hakem()
        self.game_count += 1

        # Save game log
        self.save_game_log()

    def update_last_winning_team(self):
        """Update the last winning team based on current game results."""
        team1_tricks = sum(self.tricks_won[player] for player in self.team1)
        team2_tricks = sum(self.tricks_won[player] for player in self.team2)

        if team1_tricks >= 7:
            self.last_winning_team = self.team1
        else:
            self.last_winning_team = self.team2

    def rotate_hakem(self):
        """Rotate the hakem within the last winning team."""
        # First determine which team the current hakem is in
        if self.current_hakem in self.team1:
            current_team = self.team1
            other_team = self.team2
        else:
            current_team = self.team2
            other_team = self.team1

        # If the last winning team is different from current hakem's team
        if self.last_winning_team != current_team:
            # Switch to the other team and start with their first player
            self.current_hakem = self.last_winning_team[0]
        else:
            # Stay in the same team, rotate to next player
            current_idx = current_team.index(self.current_hakem)
            next_idx = (current_idx + 1) % 2
            self.current_hakem = current_team[next_idx]

    def log_round(self, round_num, lead_suit, cards_played, winner):
        """Log details of the current round."""
        row = {
            "Game": self.game_count,
            "Round": round_num,
            "Hakem": self.current_hakem.name,
            "Trump Suit": self.trump_suit,
            "Lead Suit": lead_suit,
            "Player 1 Card": cards_played[0],
            "Player 2 Card": cards_played[1],
            "Player 3 Card": cards_played[2],
            "Player 4 Card": cards_played[3],
            "Trick Winner": winner.name,
            "Winning Team": "Team 1" if winner in self.team1 else "Team 2",
            "Bid": self.current_bid,
            "Bid Winner": self.bid_winner.name if self.bid_winner else "None",
        }
        self.game_log = pd.concat(
            [self.game_log, pd.DataFrame([row])], ignore_index=True
        )

    def evaluate_play(self, player, card, lead_suit=None):
        reward = 0

        # Base rewards from previous implementation
        if lead_suit and card.suit != lead_suit:
            if any(c.suit == lead_suit for c in player.hand):
                return -5.0

        if card.value >= 10:
            reward += 0.5

        if card.suit == self.trump_suit:
            reward += 1.0

        if not any(c.suit == card.suit and c.value > card.value for c in player.hand):
            reward += 0.5

        # Add team-based rewards
        team_reward = self._calculate_team_reward(player)
        reward += team_reward

        return reward

    def _calculate_team_reward(self, player):
        """Calculate team-based rewards."""
        team = self.team1 if player in self.team1 else self.team2
        teammate = team[1] if team[0] == player else team[0]

        team_tricks = self.tricks_won[player] + self.tricks_won[teammate]
        opponent_tricks = 13 - team_tricks

        # Reward for team progress towards winning
        if team_tricks >= 7:
            return 3.0  # Big reward for winning
        elif team_tricks > opponent_tricks:
            return 1.0  # Smaller reward for leading
        elif team_tricks == opponent_tricks:
            return 0.5  # Small reward for being tied
        else:
            return -0.5  # Penalty for falling behind

    def determine_trick_winner(self, trick, lead_suit):
        winning_card = trick[0][1]
        winner = trick[0][0]
        has_trump = any(card.suit == self.trump_suit for _, card in trick)

        for player, card in trick[1:]:
            # If trump cards are played, only compare trump cards
            if has_trump:
                if card.suit == self.trump_suit:
                    if (
                        winning_card.suit != self.trump_suit
                        or card.value > winning_card.value
                    ):
                        winning_card = card
                        winner = player
            # If no trump cards, follow lead suit
            else:
                if card.suit == lead_suit and card.value > winning_card.value:
                    winning_card = card
                    winner = player

        return winner

    def choose_trump_suit(self):
        """Hakem chooses the trump suit based on initial 5 cards."""
        # Count cards of each suit in Hakem's hand
        suit_counts = {suit: 0 for suit in suits}
        suit_values = {suit: 0 for suit in suits}

        for card in self.current_hakem.hand:
            suit_counts[card.suit] += 1
            # Give more weight to high-value cards
            if card.value >= 10:  # 10, J, Q, K, A
                suit_values[card.suit] += card.value * 2
            else:
                suit_values[card.suit] += card.value

        # Choose the suit with the highest combined score of count and values
        best_suit = max(suits, key=lambda s: (suit_counts[s] * 10 + suit_values[s]))
        self.trump_suit = best_suit
        # print(
        #    f"{self.current_hakem.name} has chosen {self.trump_suit} as the trump suit."
        # )

    def save_game_log(self, file_name="game_log.xlsx"):
        """Save the game log to an Excel file."""
        self.game_log.to_excel(file_name, index=False)
        # print(f"Game log saved to {file_name}.")

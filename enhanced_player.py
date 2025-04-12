import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from game_constants import Card, suits, ranks, rank_values


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, experience, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return None

        priorities = torch.tensor(self.priorities[: len(self.memory)])
        probs = priorities**self.alpha
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)
        experiences = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class TeamStrategy:
    def __init__(self):
        self.team_memory = {}
        self.card_count = {suit: 13 for suit in suits}

    def update_card_count(self, card):
        self.card_count[card.suit] -= 1

    def should_play_high(self, player, lead_suit, current_trick):
        # Get teammate
        teammate = player._get_teammate()
        if not teammate:
            return False

        # If teammate has played a high card
        if current_trick and current_trick[-1][0] == teammate:
            played_card = current_trick[-1][1]
            if played_card.value >= 10:
                return True

        # If we're close to winning
        team_tricks = sum(player.tricks_won[p] for p in player.team)
        if team_tricks >= 6:
            return True

        # If we have a strong hand in the lead suit
        if lead_suit:
            high_cards = [
                c for c in player.hand if c.suit == lead_suit and c.value >= 10
            ]
            if len(high_cards) >= 2:
                return True

        return False

    def should_conserve_trump(self, player):
        remaining_trump = sum(1 for c in player.hand if c.suit == player.trump_suit)
        played_trump = 13 - self.card_count[player.trump_suit]
        return remaining_trump < 3 and played_trump < 6

    def get_optimal_card(self, player, lead_suit, current_trick):
        valid_cards = player.hand
        if lead_suit:
            suit_cards = [c for c in player.hand if c.suit == lead_suit]
            if suit_cards:
                valid_cards = suit_cards

        # Apply team strategy
        if self.should_play_high(player, lead_suit, current_trick):
            return max(valid_cards, key=lambda c: c.value)

        # Conserve trump if needed
        if self.should_conserve_trump(player):
            non_trump = [c for c in valid_cards if c.suit != player.trump_suit]
            if non_trump:
                return min(non_trump, key=lambda c: c.value)

        return None


class EnhancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)


class EnhancedPlayer:
    def __init__(self, name, state_dim, action_dim, epsilon=0.1):
        self.name = name
        self.hand = []
        self.epsilon = epsilon
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.memory = PrioritizedReplayMemory(10000)
        self.batch_size = 32
        self.target_update_frequency = 10
        self.steps_done = 0
        self.total_reward = 0.0
        self.actions_taken = []
        self.played_cards_memory = set()
        self.team_strategy = TeamStrategy()
        self.current_trick = []
        self.trump_suit = None
        self.tricks_won = {}
        self.team = None

        # Initialize networks
        self.policy_net = EnhancedDQN(state_dim, action_dim)
        self.target_net = EnhancedDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.update_target_net()

    def draw(self, deck, num_cards):
        """Draw cards from the deck to the player's hand."""
        self.hand += deck.deal(num_cards)

    def reset(self):
        """Reset player-specific metrics at the start of each game."""
        self.total_reward = 0.0
        self.actions_taken = []
        self.played_cards_memory.clear()
        self.hand = []

    def _get_teammate(self):
        if not self.team:
            return None
        return self.team[1] if self.team[0] == self else self.team[0]

    def get_state(self):
        state = []

        # Current hand (52 dim)
        hand_state = [0] * 52
        for card in self.hand:
            card_idx = suits.index(card.suit) * 13 + ranks.index(card.rank)
            hand_state[card_idx] = 1
        state.extend(hand_state)

        # Played cards (52 dim)
        played_state = [0] * 52
        for card in self.played_cards_memory:
            card_idx = suits.index(card.suit) * 13 + ranks.index(card.rank)
            played_state[card_idx] = 1
        state.extend(played_state)

        # Current trick (4 * 52 dim)
        trick_state = [0] * (4 * 52)
        for i, (_, card) in enumerate(self.current_trick):
            card_idx = suits.index(card.suit) * 13 + ranks.index(card.rank)
            trick_state[i * 52 + card_idx] = 1
        state.extend(trick_state)

        # Team scores (2 dim)
        team_scores = [
            sum(self.tricks_won[p] for p in self.team),
            sum(
                self.tricks_won[p]
                for p in (
                    self.team[0].team if self.team[0] != self else self.team[1].team
                )
            ),
        ]
        state.extend(team_scores)

        # Trump suit (4 dim)
        trump_state = [0] * 4
        trump_state[suits.index(self.trump_suit)] = 1
        state.extend(trump_state)

        return torch.FloatTensor(state)

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

        try:
            # Get action index within the valid range
            action_index = self.select_action(None, len(valid_cards))
            if action_index < 0 or action_index >= len(valid_cards):
                # Fallback to random selection if action index is invalid
                action_index = random.randint(0, len(valid_cards) - 1)

            chosen_card = valid_cards[action_index]

            # Remove the chosen card from the player's hand
            self.hand.remove(chosen_card)
            self.actions_taken.append(action_index)
            return chosen_card, action_index
        except Exception as e:
            # Fallback to random selection if there's any error
            chosen_card = random.choice(valid_cards)
            self.hand.remove(chosen_card)
            return chosen_card, 0

    def evaluate_play(self, card, lead_suit=None):
        reward = 0

        # Base rewards
        if card.suit == self.trump_suit:
            reward += 1.0

        # Suit following
        if lead_suit:
            if card.suit == lead_suit:
                reward += 0.5
            elif any(c.suit == lead_suit for c in self.hand):
                reward -= 2.0

        # Trick winning potential
        if self._can_win_trick(card, lead_suit):
            reward += 1.0

        # Team strategy
        teammate = self._get_teammate()
        if teammate and self._can_help_teammate(card):
            reward += 1.5

        # Card value
        if card.value >= 10:
            reward += 0.3

        # Game progress
        team_tricks = sum(self.tricks_won[p] for p in self.team)
        if team_tricks >= 7:
            reward += 5.0

        return reward

    def _can_win_trick(self, card, lead_suit):
        if not self.current_trick:
            return True

        highest_card = max(self.current_trick, key=lambda x: x[1].value)[1]
        if card.suit == self.trump_suit:
            return (
                highest_card.suit != self.trump_suit or card.value > highest_card.value
            )
        return card.suit == lead_suit and card.value > highest_card.value

    def _can_help_teammate(self, card):
        if not self.current_trick:
            return False

        teammate = self._get_teammate()
        if not teammate:
            return False

        teammate_card = next((c for p, c in self.current_trick if p == teammate), None)
        if not teammate_card:
            return False

        return card.suit == teammate_card.suit and card.value > teammate_card.value

    def optimize_model(self, beta=0.4):
        if len(self.memory.memory) < self.batch_size:
            return

        experiences, indices, weights = self.memory.sample(self.batch_size, beta)
        if not experiences:
            return

        batch = list(zip(*experiences))
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        td_errors = (current_q_values - target_q_values).abs()
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().numpy())

        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def store_experience(self, state, action, reward, next_state, done):
        priority = abs(reward) + 1e-6  # Ensure non-zero priority
        self.memory.push((state, action, reward, next_state, done), priority)

    def __repr__(self):
        return self.name

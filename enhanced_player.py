# enhanced_player.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from game_constants import Card, suits, ranks, rank_values

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        print(f"Replay memory size: {len(self.memory)}")

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return None
        try:
            priorities = [
                float(p.item() if isinstance(p, torch.Tensor) else p)
                for p in self.priorities[: len(self.memory)]
            ]
            priorities = np.array(priorities, dtype=np.float32)
            priorities_tensor = torch.from_numpy(priorities).to(device)
            probs = priorities_tensor**self.alpha
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, batch_size, replacement=True)
            experiences = [self.memory[idx] for idx in indices]
            weights = (len(self.memory) * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            return experiences, indices, weights
        except Exception as e:
            print(f"Error in sample: {e}")
            return None

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)

    def __len__(self):
        return len(self.memory)


class TeamStrategy:
    def __init__(self):
        self.card_count = {suit: 13 for suit in suits}
        self.team_memory = {}

    def update_card_count(self, card):
        if not isinstance(card, Card):
            card = Card.from_string(card)
        self.card_count[card.suit] = max(0, self.card_count[card.suit] - 1)

    def should_play_high(self, player, lead_suit, current_trick):
        teammate = player._get_teammate()
        if not teammate:
            return False
        if current_trick and current_trick[-1][0] == teammate:
            played_card = current_trick[-1][1]
            if played_card.value >= 10:
                return True
        team_tricks = sum(player.tricks_won.get(p, 0) for p in player.team)
        if team_tricks >= 6:
            return True
        if lead_suit:
            high_cards = [
                card
                for card in player.hand
                if card.suit == lead_suit and card.value >= 10
            ]
            if len(high_cards) >= 2:
                return True
        return False

    def should_conserve_trump(self, player):
        remaining_trump = sum(
            1 for card in player.hand if card.suit == player.trump_suit
        )
        played_trump = 13 - self.card_count[player.trump_suit]
        return remaining_trump < 3 and played_trump < 6

    def get_optimal_card(self, player, lead_suit, current_trick):
        valid_cards = (
            player.hand
            if lead_suit is None
            else [card for card in player.hand if card.suit == lead_suit] or player.hand
        )
        if not valid_cards:
            print(f"{player.name} TeamStrategy: No valid cards")
            return None
        if self.should_play_high(player, lead_suit, current_trick):
            optimal = max(valid_cards, key=lambda card: card.value, default=None)
            if optimal:
                print(f"{player.name} TeamStrategy: High card {optimal}")
                return optimal
        if self.should_conserve_trump(player):
            non_trump = [card for card in valid_cards if card.suit != player.trump_suit]
            if non_trump:
                optimal = min(non_trump, key=lambda card: card.value, default=None)
                if optimal:
                    print(f"{player.name} TeamStrategy: Conserve trump, play {optimal}")
                    return optimal
        print(f"{player.name} TeamStrategy: No optimal card")
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
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)


class EnhancedPlayer:
    def __init__(
        self, name, state_dim=114, action_dim=13, team_strategy=None, epsilon=0.1
    ):
        self.name = name
        self.hand = []  # Stores Card objects
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
        self.team_strategy = team_strategy or TeamStrategy()
        self.current_trick = []
        self.trump_suit = None
        self.team = None
        self.tricks_won = {}
        self.lead_suit = None
        self.trump_state = [0] * 4
        self.played_suit_counts = [0] * 4
        self.policy_net = EnhancedDQN(state_dim, action_dim)
        self.target_net = EnhancedDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.update_target_net()
        self.model = self.policy_net  # Set the model attribute to policy_net

    def draw(self, deck, num_cards):
        new_cards = deck.deal(num_cards)
        self.hand.extend(new_cards)  # Store Card objects
        for card in new_cards:
            if not isinstance(card, Card):
                raise ValueError(f"Invalid card drawn: {card}")

    def reset(self):
        self.total_reward = 0.0
        self.actions_taken = []
        self.played_cards_memory.clear()
        self.hand = []
        self.current_trick = []
        self.lead_suit = None
        self.played_suit_counts = [0] * 4

    def _get_teammate(self):
        if not self.team or len(self.team) < 2:
            return None
        return self.team[1] if self.team[0] == self else self.team[0]

    def get_state(self):
        state = []
        hand_state = [0] * 52
        for card in self.hand:
            card_idx = suits.index(card.suit) * 13 + ranks.index(card.rank)
            hand_state[card_idx] = 1
        state.extend(hand_state)
        state.extend(self.played_suit_counts)
        trick_state = [0] * 52
        if self.current_trick:
            last_card = self.current_trick[-1][1]
            card_idx = suits.index(last_card.suit) * 13 + ranks.index(last_card.rank)
            trick_state[card_idx] = 1
        state.extend(trick_state)
        team_tricks = (
            sum(self.tricks_won.get(p, 0) for p in self.team) if self.team else 0
        )
        opponent_tricks = (
            sum(self.tricks_won.get(p, 0) for p in self.tricks_won) - team_tricks
        )
        state.extend([team_tricks, opponent_tricks])
        state.extend(self.trump_state)
        return torch.FloatTensor(state).to(device)

    def update_trump_suit(self, trump_suit):
        self.trump_suit = trump_suit
        self.trump_state = [0] * 4
        if trump_suit:
            self.trump_state[suits.index(trump_suit)] = 1

    def select_action(self, valid_cards):
        num_valid_actions = len(valid_cards)
        if num_valid_actions <= 0:
            print(
                f"Warning: No valid actions for {self.name}, hand: {[str(c) for c in self.hand]}"
            )
            return 0
        optimal_card = self.team_strategy.get_optimal_card(
            self, self.lead_suit, self.current_trick
        )
        if optimal_card and optimal_card in valid_cards:
            action_index = valid_cards.index(optimal_card)
            print(
                f"{self.name} selected optimal card {optimal_card} at index {action_index}"
            )
            return action_index
        if random.random() < self.epsilon:
            action_index = random.randint(0, num_valid_actions - 1)
            print(f"{self.name} random action index: {action_index}")
            return action_index
        try:
            with torch.no_grad():
                self.policy_net.eval()
                state_tensor = self.get_state().unsqueeze(0)
                action_values = self.policy_net(state_tensor)  # Shape: (1, 13)
                self.policy_net.train()
                mask = torch.zeros(action_values.shape[1], dtype=torch.float32).to(
                    device
                )
                mask[:num_valid_actions] = 1
                masked_values = action_values * mask
                action_index = masked_values.argmax().item()
                if action_index >= num_valid_actions:
                    print(
                        f"Warning: DQN selected invalid action {action_index}, choosing random"
                    )
                    action_index = random.randint(0, num_valid_actions - 1)
                print(f"{self.name} DQN action index: {action_index}")
                return action_index
        except Exception as e:
            print(f"Error in select_action for {self.name}: {e}")
            return random.randint(0, num_valid_actions - 1)

    def play_card(self, lead_suit=None):
        if not self.hand:
            raise ValueError(f"No cards in hand to play for {self.name}")
        self.lead_suit = lead_suit
        valid_cards = (
            self.hand
            if lead_suit is None
            else [card for card in self.hand if card.suit == lead_suit] or self.hand
        )
        if not valid_cards:
            raise ValueError(f"No valid cards to play for {self.name}")
        print(f"{self.name} hand before play: {[str(c) for c in self.hand]}")
        print(f"{self.name} valid cards: {[str(c) for c in valid_cards]}")
        try:
            action_index = self.select_action(valid_cards)
            if not isinstance(action_index, int) or action_index >= len(valid_cards):
                raise ValueError(f"Invalid action_index: {action_index}")
            chosen_card = valid_cards[action_index]
            print(f"Chosen card: {chosen_card}")
            return chosen_card, action_index
        except Exception as e:
            print(f"Error in play_card for {self.name}: {e}")
            raise ValueError(f"Failed to play card for {self.name}: {str(e)}")

    def evaluate_play(self, card, lead_suit=None, round_num=1):
        reward = 0
        if card.suit == self.trump_suit:
            reward += 1.0
        if lead_suit:
            if card.suit == lead_suit:
                reward += 0.5
            elif (
                any(c.suit == lead_suit for c in self.hand)
                and card.suit != self.trump_suit
            ):
                reward -= 2.0
        if self._can_win_trick(card, lead_suit):
            reward += 1.0
        if (
            self.team_strategy.should_conserve_trump(self)
            and card.suit != self.trump_suit
        ):
            reward += 0.5
        teammate = self._get_teammate()
        if teammate and self._can_help_teammate(card):
            reward += 1.5
        if card.value >= 10:
            reward += 0.3
        team_tricks = (
            sum(self.tricks_won.get(p, 0) for p in self.team) if self.team else 0
        )
        if team_tricks >= 7:
            reward += 5.0
        if round_num < 5 and card.value >= 12:
            reward -= 1.0
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

    def store_experience(self, state, action, reward, next_state, done):
        try:
            with torch.no_grad():
                self.policy_net.eval()
                current_q = self.policy_net(state.unsqueeze(0)).gather(
                    1, torch.tensor([[action]], dtype=torch.int64).to(device)
                )
                self.policy_net.train()
                next_q = self.target_net(next_state.unsqueeze(0)).max(1)[0].detach()
                target_q = reward + self.gamma * next_q * (1 - done)
                priority = float(abs(current_q - target_q).item()) + 1e-6
            self.memory.push((state, action, reward, next_state, done), priority)
            self.total_reward += reward
        except Exception as e:
            print(f"Error in store_experience for {self.name}: {e}")
            priority = 1e-6
            self.memory.push((state, action, reward, next_state, done), priority)

    def optimize_model(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return
        result = self.memory.sample(self.batch_size, beta)
        if result is None:
            return
        experiences, indices, weights = result
        batch = list(zip(*experiences))
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        current_q_values = self.policy_net(states).gather(1, actions)
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        td_errors = (current_q_values - target_q_values).abs()
        loss = (td_errors.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_net()
        self.update_epsilon()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def __repr__(self):
        return self.name

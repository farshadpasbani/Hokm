# enhanced_player.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from game_constants import (
    Card,
    suits,
    ranks,
    rank_values,
    card_to_index,
    index_to_card,
    STATE_DIM,
    ACTION_DIM,
)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_policy(path: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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
        self,
        name,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        team_strategy=None,
        epsilon=0.1,
        is_human=False,
    ):
        self.name = name
        self.is_human = is_human
        self.learning_enabled = True
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

    def load_policy_state(self, path: str) -> None:
        """Load policy (and mirror to target) from a .pth state dict."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(path)
        blob = _torch_load_policy(path)
        self.policy_net.load_state_dict(blob)
        self.target_net.load_state_dict(blob)
        self.update_target_net()

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
        """Return a global card index in0..51 for a legal card from valid_cards."""
        if not valid_cards:
            print(
                f"Warning: No valid actions for {self.name}, hand: {[str(c) for c in self.hand]}"
            )
            return 0
        optimal_card = self.team_strategy.get_optimal_card(
            self, self.lead_suit, self.current_trick
        )
        if optimal_card and optimal_card in valid_cards:
            idx = card_to_index(optimal_card)
            print(f"{self.name} selected optimal card {optimal_card} (idx {idx})")
            return idx
        if random.random() < self.epsilon:
            choice = random.choice(valid_cards)
            idx = card_to_index(choice)
            print(f"{self.name} random play: {choice} (idx {idx})")
            return idx
        try:
            global_indices = [card_to_index(c) for c in valid_cards]
            with torch.no_grad():
                self.policy_net.eval()
                q = self.policy_net(self.get_state().unsqueeze(0)).squeeze(0)
                self.policy_net.train()
            best_idx = global_indices[0]
            best_val = q[best_idx].item()
            for gi in global_indices[1:]:
                v = q[gi].item()
                if v > best_val:
                    best_val = v
                    best_idx = gi
            print(f"{self.name} DQN chose index {best_idx} ({index_to_card(best_idx)})")
            return int(best_idx)
        except Exception as e:
            print(f"Error in select_action for {self.name}: {e}")
            return card_to_index(random.choice(valid_cards))

    def play_card(self, lead_suit, selected_card=None):
        self.lead_suit = lead_suit
        if self.is_human:
            if selected_card is None:
                raise ValueError("Human player must provide a selected card")
            valid_cards = (
                self.hand
                if lead_suit is None
                else [card for card in self.hand if card.suit == lead_suit] or self.hand
            )
            if selected_card not in valid_cards:
                raise ValueError(
                    f"Invalid card {selected_card} for lead suit {lead_suit}"
                )
            return selected_card, -1  # No RL action index for humans

        valid_cards = (
            self.hand
            if lead_suit is None
            else [card for card in self.hand if card.suit == lead_suit] or self.hand
        )
        if not valid_cards:
            raise ValueError(f"No valid cards to play for {self.name}")

        action_index = self.select_action(valid_cards)
        card = index_to_card(action_index)
        if card not in valid_cards:
            card = random.choice(valid_cards)
            action_index = card_to_index(card)
        return card, action_index

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
        if not self.learning_enabled:
            return
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
        if not self.learning_enabled:
            return
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

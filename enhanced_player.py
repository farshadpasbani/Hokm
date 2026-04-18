# enhanced_player.py — Neural Fictitious Self-Play (NFSP) for Hokm agents.
# See: Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game_constants import (
    ACTION_DIM,
    STATE_DIM,
    Card,
    card_to_index,
    index_to_card,
    ranks,
    suits,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_policy(path: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class PrioritizedReplayMemory:
    """RL transitions for the best-response (Q) learner only."""

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
            print(f"Error in replay sample: {e}")
            return None

    def update_priorities(self, indices, priorities):
        idx_flat = (
            indices.cpu().numpy().reshape(-1)
            if isinstance(indices, torch.Tensor)
            else np.asarray(indices, dtype=np.int64).reshape(-1)
        )
        pri_flat = np.asarray(priorities, dtype=np.float64).reshape(-1)
        for idx, priority in zip(idx_flat, pri_flat):
            self.priorities[int(idx)] = float(priority)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    """Best-response value network Q(s, a) with |A|=52 (masked argmax at play time)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class AveragePolicyNetwork(nn.Module):
    """Sluggish average policy π_σ(s) — logits over 52 cards (masked at sampling)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TeamStrategy:
    """Card-counting bookkeeping only (used by game logging / rewards)."""

    def __init__(self):
        self.card_count = {suit: 13 for suit in suits}
        self.team_memory = {}

    def update_card_count(self, card):
        if not isinstance(card, Card):
            card = Card.from_string(card)
        self.card_count[card.suit] = max(0, self.card_count[card.suit] - 1)

    def should_conserve_trump(self, player):
        remaining_trump = sum(
            1 for card in player.hand if card.suit == player.trump_suit
        )
        played_trump = 13 - self.card_count[player.trump_suit]
        return remaining_trump < 3 and played_trump < 6


class EnhancedPlayer:
    """
    NFSP agent:
    - With probability η: action ~ softmax(π_σ(s)) over legal cards.
    - With probability 1−η: ε-greedy Q (best response); these steps produce RL transitions.
    - Supervised updates on π_σ from a reservoir of all self-play (s, a).
    """

    def __init__(
        self,
        name,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        team_strategy=None,
        epsilon=0.15,
        is_human=False,
        eta=0.25,
        sl_reservoir_size=150000,
    ):
        self.name = name
        self.is_human = is_human
        self.learning_enabled = True
        self.hand = []
        self.epsilon = epsilon
        self.eta = eta  # anticipatory parameter: P(play average policy)
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.sl_lr = 0.001
        self.memory = PrioritizedReplayMemory(100000)
        self.sl_buffer = deque(maxlen=sl_reservoir_size)
        self.sl_reservoir_size = sl_reservoir_size
        self.batch_size = 32
        self.target_update_frequency = 200
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
        self.last_rl_eligible = True

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.avg_policy_net = AveragePolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.sl_optimizer = optim.Adam(self.avg_policy_net.parameters(), lr=self.sl_lr)
        self.update_target_net()
        self.model = self.q_net  # train_backend / legacy

    def load_policy_state(self, path: str) -> None:
        """Load NFSP checkpoint {q_net, avg_policy_net} or legacy flat state_dict."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(path)
        blob = _torch_load_policy(path)
        if isinstance(blob, dict) and "q_net" in blob:
            self.q_net.load_state_dict(blob["q_net"])
            if "avg_policy_net" in blob:
                self.avg_policy_net.load_state_dict(blob["avg_policy_net"])
        else:
            self.q_net.load_state_dict(blob)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def export_state_dict(self):
        """Checkpoint format for training saves."""
        return {
            "q_net": self.q_net.state_dict(),
            "avg_policy_net": self.avg_policy_net.state_dict(),
        }

    def draw(self, deck, num_cards):
        new_cards = deck.deal(num_cards)
        self.hand.extend(new_cards)
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
        """NFSP mixture: η → average policy sample; else ε-greedy Q."""
        if not valid_cards:
            print(f"Warning: No valid actions for {self.name}")
            return 0
        global_indices = [card_to_index(c) for c in valid_cards]
        state_t = self.get_state()

        if random.random() < self.eta:
            self.last_rl_eligible = False
            with torch.no_grad():
                self.avg_policy_net.eval()
                logits = self.avg_policy_net(state_t.unsqueeze(0)).squeeze(0)
                self.avg_policy_net.train()
            sub = logits[global_indices]
            probs = F.softmax(sub, dim=0)
            pick = torch.multinomial(probs, 1).item()
            return int(global_indices[pick])

        self.last_rl_eligible = True
        if random.random() < self.epsilon:
            return card_to_index(random.choice(valid_cards))

        with torch.no_grad():
            self.q_net.eval()
            q = self.q_net(state_t.unsqueeze(0)).squeeze(0)
            self.q_net.train()
        best_idx = global_indices[0]
        best_val = q[best_idx].item()
        for gi in global_indices[1:]:
            v = q[gi].item()
            if v > best_val:
                best_val = v
                best_idx = gi
        return int(best_idx)

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
            self.last_rl_eligible = False
            return selected_card, -1

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
        reward = 0.0
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

    def store_experience(self, state, action, reward, next_state, done, rl_eligible=True):
        if not self.learning_enabled or self.is_human:
            return
        if action is None or action < 0:
            return

        self.sl_buffer.append((state.detach().cpu(), int(action)))

        if not rl_eligible:
            return

        try:
            with torch.no_grad():
                self.q_net.eval()
                current_q = self.q_net(state.unsqueeze(0)).gather(
                    1, torch.tensor([[action]], dtype=torch.int64).to(device)
                )
                self.q_net.train()
                next_q = self.target_q_net(next_state.unsqueeze(0)).max(1)[0].detach()
                target_q = reward + self.gamma * next_q * (1 - float(done))
                priority = float(abs(current_q - target_q).item()) + 1e-6
            self.memory.push((state, action, reward, next_state, done), priority)
            self.total_reward += reward
        except Exception as e:
            print(f"Error in store_experience for {self.name}: {e}")
            self.memory.push((state, action, reward, next_state, done), 1e-6)

    def optimize_model(self, beta=0.4):
        if not self.learning_enabled:
            return
        self._optimize_q(beta)
        self._optimize_sl()
        self.update_epsilon()

    def _optimize_q(self, beta=0.4):
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
        if isinstance(weights, torch.Tensor):
            w = weights.detach().to(dtype=torch.float32, device=device)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        weights_t = w.unsqueeze(1) if w.dim() == 1 else w

        current_q_values = self.q_net(states).gather(1, actions)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        td_errors = (current_q_values - target_q_values).abs()
        loss = (td_errors.pow(2) * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_net()

    def _optimize_sl(self):
        if len(self.sl_buffer) < self.batch_size:
            return
        batch = random.sample(self.sl_buffer, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        logits = self.avg_policy_net(states)
        loss = F.cross_entropy(logits, actions)
        self.sl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_policy_net.parameters(), max_norm=1.0)
        self.sl_optimizer.step()

    def update_target_net(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

    def update_epsilon(self):
        self.epsilon = max(0.02, self.epsilon * 0.9995)

    def __repr__(self):
        return self.name

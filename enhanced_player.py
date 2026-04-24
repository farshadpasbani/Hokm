# enhanced_player.py — Neural Fictitious Self-Play (NFSP) for Hokm agents.
# See: Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"

from __future__ import annotations

import os
import random
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple

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

# Reward modes — kept as module-level strings (not a config import) so this
# module has no dependency on `config.py`. See config.HokmConfig for defaults.
REWARD_HEURISTIC = "heuristic"
REWARD_OUTCOME = "outcome"
REWARD_MIXED = "mixed"

if TYPE_CHECKING:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_policy(path: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_state_dict_compat(module: nn.Module, state_dict: dict) -> None:
    """
    Load weights with tolerance for legacy checkpoints:

    - `strict=False` so BatchNorm (bn*) → LayerNorm (ln*) name changes from a
      previous refactor are ignored (norm layers keep default init).
    - **Shape-mismatched tensors are dropped** rather than raising. This is
      critical after observation-space changes (e.g. STATE_DIM 114 → 194): the
      first Linear's weight shape changes, so its checkpoint entry is skipped;
      every deeper layer still loads because its shape is unchanged.

    A one-line summary of what was skipped is printed so training logs stay
    honest about the partial load.
    """
    own = module.state_dict()
    filtered: dict = {}
    skipped: list = []
    for k, v in state_dict.items():
        if k in own and hasattr(v, "shape") and hasattr(own[k], "shape"):
            if tuple(v.shape) == tuple(own[k].shape):
                filtered[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            # Name missing in current module (e.g. legacy bn* keys) — let
            # strict=False handle these silently.
            filtered[k] = v
    module.load_state_dict(filtered, strict=False)
    if skipped:
        head = skipped[0]
        print(
            f"[load_policy_state] skipped {len(skipped)} shape-mismatched "
            f"tensors (e.g. {head[0]}: ckpt={head[1]} vs current={head[2]}); "
            f"those layers reset to fresh init."
        )


class UniformReplayMemory:
    """O(1) amortized uniform sampling over a circular buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: list = []
        self.position = 0

    def push(self, experience: Tuple) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.memory)
        if n < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    """Q(s,·): LayerNorm MLP; training uses full 52-d head; play uses legal-action subset."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, output_dim)
        self.to(device)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc4(self._embed(x))

    def q_values_at_indices(self, x: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """Compute Q(s,a) only for legal actions (single state)."""
        if not indices:
            return torch.zeros(0, device=device)
        h = self._embed(x).squeeze(0)
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        w = self.fc4.weight.index_select(0, idx)
        b = self.fc4.bias.index_select(0, idx)
        return h @ w.t() + b


class AveragePolicyNetwork(nn.Module):
    """Average policy π_σ; inference uses legal-action logits only."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, output_dim)
        self.to(device)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self._embed(x))

    def logits_at_indices(self, x: torch.Tensor, indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.zeros(0, device=device)
        h = self._embed(x).squeeze(0)
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        w = self.fc3.weight.index_select(0, idx)
        b = self.fc3.bias.index_select(0, idx)
        return h @ w.t() + b


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


class SharedNFSPLearner:
    """
    One NFSP parameter set + replay + optimizers shared by all seats (self-play).
    Used by TrainBackend / train_hokm for fast training.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        learn_every: int = 4,
        sl_reservoir_size: int = 150000,
        *,
        gamma: float = 0.99,
        q_lr: float = 1e-3,
        pi_lr: float = 1e-3,
        batch_size: int = 32,
        target_update_frequency: int = 200,
        replay_size: int = 100_000,
        epsilon_start: float = 0.15,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        grad_clip: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learn_every = learn_every
        self.gamma = gamma
        self.learning_rate = q_lr
        self.sl_lr = pi_lr
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.steps_done = 0
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.grad_clip = grad_clip
        self._ply_since_learn = 0

        # Loss telemetry — populated each time _optimize_q / _optimize_sl runs
        # a real gradient step. Drained by snapshot_losses() once per game by
        # TrainBackend and rendered on the metrics dashboard. Both buffers are
        # short, so unbounded growth between snapshots is fine in practice.
        self._q_losses: List[float] = []
        self._sl_losses: List[float] = []
        self._grad_steps_q = 0
        self._grad_steps_sl = 0

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.avg_policy_net = AveragePolicyNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.memory = UniformReplayMemory(replay_size)
        self.sl_buffer: deque = deque(maxlen=sl_reservoir_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.sl_optimizer = optim.Adam(self.avg_policy_net.parameters(), lr=self.sl_lr)

    def push_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        rl_eligible: bool,
    ) -> None:
        self.sl_buffer.append((state.detach().cpu(), int(action)))
        if rl_eligible and action >= 0:
            self.memory.push((state, action, reward, next_state, done))

    def maybe_optimize(self) -> None:
        self._ply_since_learn += 1
        if self._ply_since_learn < self.learn_every:
            return
        self._ply_since_learn = 0
        self._optimize_q()
        self._optimize_sl()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _optimize_q(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        experiences = self.memory.sample(self.batch_size)
        if not experiences:
            return
        batch = list(zip(*experiences))
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

        current_q_values = self.q_net(states).gather(1, actions)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        self._q_losses.append(float(loss.detach().item()))
        self._grad_steps_q += 1
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_q_net.eval()

    def _optimize_sl(self) -> None:
        if len(self.sl_buffer) < self.batch_size:
            return
        batch = random.sample(self.sl_buffer, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        logits = self.avg_policy_net(states)
        loss = F.cross_entropy(logits, actions)
        self.sl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_policy_net.parameters(), max_norm=self.grad_clip)
        self.sl_optimizer.step()
        self._sl_losses.append(float(loss.detach().item()))
        self._grad_steps_sl += 1

    def snapshot_losses(self) -> dict:
        """Return mean loss + grad-step counts since the last snapshot, then
        clear the buffers. Called once per game by TrainBackend.

        Returns NaN for either loss when no gradient steps occurred this
        interval (e.g. early games before the replay buffer fills, or when
        learning is disabled). NaN is preferable to 0.0 because the chart
        downstream just skips NaNs instead of drawing a misleading dip.
        """
        import math

        q = (sum(self._q_losses) / len(self._q_losses)) if self._q_losses else math.nan
        s = (sum(self._sl_losses) / len(self._sl_losses)) if self._sl_losses else math.nan
        out = {
            "q_loss": q,
            "sl_loss": s,
            "q_steps": len(self._q_losses),
            "sl_steps": len(self._sl_losses),
        }
        self._q_losses.clear()
        self._sl_losses.clear()
        return out

    @classmethod
    def from_config(cls, cfg) -> "SharedNFSPLearner":
        """Build a learner from a HokmConfig (only fields we support)."""
        n = cfg.nfsp
        net = cfg.network
        return cls(
            state_dim=net.state_dim,
            action_dim=net.action_dim,
            learn_every=n.learn_every,
            sl_reservoir_size=n.sl_reservoir_size,
            gamma=n.gamma,
            q_lr=net.q_lr,
            pi_lr=net.pi_lr,
            batch_size=n.batch_size,
            target_update_frequency=n.target_update_frequency,
            replay_size=n.replay_size,
            epsilon_start=n.epsilon_start,
            epsilon_min=n.epsilon_min,
            epsilon_decay=n.epsilon_decay,
            grad_clip=net.grad_clip,
        )

    def export_state_dict(self):
        return {
            "q_net": self.q_net.state_dict(),
            "avg_policy_net": self.avg_policy_net.state_dict(),
        }

    def load_policy_state(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(path)
        blob = _torch_load_policy(path)
        if isinstance(blob, dict) and "q_net" in blob:
            _load_state_dict_compat(self.q_net, blob["q_net"])
            if "avg_policy_net" in blob:
                _load_state_dict_compat(self.avg_policy_net, blob["avg_policy_net"])
        else:
            _load_state_dict_compat(self.q_net, blob)
        self.target_q_net.load_state_dict(self.q_net.state_dict())


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
        shared_learner: Optional[SharedNFSPLearner] = None,
        learn_every: int = 4,
        reward_mode: str = REWARD_HEURISTIC,
        shaping_weight: float = 0.1,
        win_bonus: float = 5.0,
    ):
        self.name = name
        self.is_human = is_human
        self.learning_enabled = True
        self.hand = []
        self.eta = eta
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.sl_lr = 0.001
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
        # Kept for compatibility with TeamStrategy.should_conserve_trump; the
        # modern state representation in get_state() uses the finer-grained
        # `cards_played_this_hand` tracked on the Hokm instance instead.
        self.played_suit_counts = [0] * 4
        # Seat-relative pointers, populated by Hokm.start_game() → _sync_seats().
        # When None (e.g. before a game starts, or in a unit test), get_state()
        # gracefully degrades to a zero-filled slice rather than crashing.
        self._game = None
        self._seat = None
        self._partner = None
        self._lho = None
        self._rho = None
        self.last_rl_eligible = True
        self.learn_every = learn_every
        self._ply_since_learn = 0
        self._shared: Optional[SharedNFSPLearner] = shared_learner
        # Reward shaping / alignment. See config.NFSPConfig.reward_mode.
        self.reward_mode = reward_mode
        self.shaping_weight = float(shaping_weight)
        self.win_bonus = float(win_bonus)

        if self._shared is not None:
            self.epsilon = self._shared.epsilon
            self.q_net = self._shared.q_net
            self.target_q_net = self._shared.target_q_net
            self.avg_policy_net = self._shared.avg_policy_net
            self.memory = self._shared.memory
            self.sl_buffer = self._shared.sl_buffer
            self.optimizer = self._shared.optimizer
            self.sl_optimizer = self._shared.sl_optimizer
        else:
            self.epsilon = epsilon
            self.memory = UniformReplayMemory(100000)
            self.sl_buffer = deque(maxlen=sl_reservoir_size)
            self.q_net = QNetwork(state_dim, action_dim)
            self.target_q_net = QNetwork(state_dim, action_dim)
            self.avg_policy_net = AveragePolicyNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
            self.sl_optimizer = optim.Adam(
                self.avg_policy_net.parameters(), lr=self.sl_lr
            )
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.model = self.q_net

    def load_policy_state(self, path: str) -> None:
        if self._shared is not None:
            self._shared.load_policy_state(path)
            return
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(path)
        blob = _torch_load_policy(path)
        if isinstance(blob, dict) and "q_net" in blob:
            _load_state_dict_compat(self.q_net, blob["q_net"])
            if "avg_policy_net" in blob:
                _load_state_dict_compat(self.avg_policy_net, blob["avg_policy_net"])
        else:
            _load_state_dict_compat(self.q_net, blob)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def export_state_dict(self):
        if self._shared is not None:
            return self._shared.export_state_dict()
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

    # ------------------------------------------------------------------
    # Seat-relative bookkeeping. Hokm.start_game() calls _sync_seats()
    # on each player so the network can read Hakem-awareness, partner
    # identity, and per-opponent voids in a stable frame of reference.
    # ------------------------------------------------------------------

    def _sync_seats(self, game) -> None:
        """Populate _game / _seat / _partner / _lho / _rho for this hand."""
        self._game = game
        try:
            seat = game.players.index(self)
        except (ValueError, AttributeError):
            # Not seated in a game (e.g. a unit test). Leave pointers None.
            self._game = None
            self._seat = None
            self._partner = None
            self._lho = None
            self._rho = None
            return
        self._seat = seat
        self._lho = game.players[(seat + 1) % 4]  # plays immediately after me
        self._partner = game.players[(seat + 2) % 4]
        self._rho = game.players[(seat + 3) % 4]  # plays immediately before me

    def _current_trick_winner(self):
        """
        Return the (player, card) pair currently winning the in-progress trick,
        or (None, None) if the trick is empty. Applies Hokm's trump-beats-lead
        rule. Kept generic so baselines can reuse it via the base class.
        """
        if not self.current_trick:
            return None, None
        trump = self.trump_suit
        trumps = [(p, c) for p, c in self.current_trick if c.suit == trump]
        if trumps:
            return max(trumps, key=lambda pc: pc[1].value)
        lead = self.current_trick[0][1].suit
        following = [(p, c) for p, c in self.current_trick if c.suit == lead]
        if following:
            return max(following, key=lambda pc: pc[1].value)
        # Shouldn't happen (first card always matches its own suit), but be safe.
        return self.current_trick[0]

    def get_state(self):
        """
        Build the 194-dim observation. See game_constants.STATE_LAYOUT for
        the canonical byte map and the rationale for each block.

        Defensive invariants:
          * Tolerates missing `_game` / `_partner` etc. by zero-filling the
            corresponding slices. Lets unit tests construct a bare player
            without a Hokm instance.
          * All normalisations (scores, winner value, suit counts) are scaled
            into [0, 1] so LayerNorm has a well-conditioned input range.
        """
        state: List[float] = []

        # ---------- 1. My hand (52) ----------
        hand_state = [0] * 52
        for card in self.hand:
            hand_state[card_to_index(card)] = 1
        state.extend(hand_state)

        # ---------- 2. Cards already played this hand (52) ----------
        # Public information shared across all seats. Sourced from the Hokm
        # game so every player sees the same history.
        played = [0] * 52
        game = self._game
        if game is not None and getattr(game, "cards_played_this_hand", None):
            for c in game.cards_played_this_hand:
                played[card_to_index(c)] = 1
        state.extend(played)

        # ---------- 3. Per-opponent voids (3×4 = 12) ----------
        # For each of [RHO, partner, LHO] (seat-relative), a 4-dim mask of
        # suits they've *proved* they are void in (failed to follow suit).
        void_flags = [0] * 12
        if game is not None:
            vm = getattr(game, "void_map", None) or {}
            others = [self._rho, self._partner, self._lho]
            for rel_idx, opp in enumerate(others):
                if opp is None:
                    continue
                suits_void = vm.get(opp, ())
                for s_idx, s in enumerate(suits):
                    if s in suits_void:
                        void_flags[rel_idx * 4 + s_idx] = 1
        state.extend(void_flags)

        # ---------- 4. Cards currently on the table this trick (52) ----------
        trick_state = [0] * 52
        for _, c in self.current_trick:
            trick_state[card_to_index(c)] = 1
        state.extend(trick_state)

        # ---------- 5. Lead suit (4) ----------
        lead_state = [0] * 4
        if self.current_trick:
            lead_state[suits.index(self.current_trick[0][1].suit)] = 1
        state.extend(lead_state)

        # ---------- 6. Trick position (4): 1st/2nd/3rd/4th to play ----------
        # get_state() is called *before* I play, so len(current_trick) ∈ {0,1,2,3}
        # maps to positions 1..4 respectively.
        pos = [0] * 4
        pos_idx = min(len(self.current_trick), 3)
        pos[pos_idx] = 1
        state.extend(pos)

        # ---------- 7. Current winner seat (5) + 8. winning card value (1) ----------
        # Relative layout so the network learns one invariant policy rather
        # than four seat-specific ones.
        winner_rel = [0] * 5  # [empty, me, partner, LHO, RHO]
        winner_val = 0.0
        if self.current_trick:
            wp, wc = self._current_trick_winner()
            if wc is not None:
                winner_val = wc.value / 14.0
            if wp is self:
                winner_rel[1] = 1
            elif wp is self._partner:
                winner_rel[2] = 1
            elif wp is self._lho:
                winner_rel[3] = 1
            elif wp is self._rho:
                winner_rel[4] = 1
            else:
                # Unknown seat (e.g. seats not synced): fall back to "empty".
                winner_rel[0] = 1
        else:
            winner_rel[0] = 1
        state.extend(winner_rel)
        state.append(winner_val)

        # ---------- 9. Hakem flags (2) ----------
        hakem_is_me = 0
        hakem_is_partner = 0
        if game is not None:
            hakem = getattr(game, "hakem", None)
            if hakem is not None:
                if hakem is self:
                    hakem_is_me = 1
                elif hakem is self._partner:
                    hakem_is_partner = 1
        state.extend([hakem_is_me, hakem_is_partner])

        # ---------- 10. Score (2) ----------
        team_tricks = (
            sum(self.tricks_won.get(p, 0) for p in self.team) if self.team else 0
        )
        opp_tricks = (
            sum(self.tricks_won.get(p, 0) for p in self.tricks_won) - team_tricks
        )
        state.extend([team_tricks / 7.0, opp_tricks / 7.0])

        # ---------- 11. Trump one-hot (4) ----------
        state.extend(self.trump_state)

        # ---------- 12. Hand per-suit counts (4), normalised /13 ----------
        # Redundant with the 52-dim hand block but provides an easier inductive
        # bias for "long suit" reasoning (lemma #6).
        counts = [0, 0, 0, 0]
        for c in self.hand:
            counts[suits.index(c.suit)] += 1
        state.extend([x / 13.0 for x in counts])

        return torch.FloatTensor(state).to(device)

    def update_trump_suit(self, trump_suit):
        self.trump_suit = trump_suit
        self.trump_state = [0] * 4
        if trump_suit:
            self.trump_state[suits.index(trump_suit)] = 1

    def _epsilon_value(self) -> float:
        if self._shared is not None:
            return self._shared.epsilon
        return self.epsilon

    def select_action(self, valid_cards):
        """NFSP mixture: η → average policy sample; else ε-greedy Q (legal actions only)."""
        if not valid_cards:
            return 0
        global_indices = [card_to_index(c) for c in valid_cards]
        state_t = self.get_state()

        if random.random() < self.eta:
            self.last_rl_eligible = False
            with torch.no_grad():
                self.avg_policy_net.eval()
                logits = self.avg_policy_net.logits_at_indices(state_t, global_indices)
                self.avg_policy_net.train()
            probs = F.softmax(logits, dim=0)
            pick = torch.multinomial(probs, 1).item()
            return int(global_indices[pick])

        self.last_rl_eligible = True
        if random.random() < self._epsilon_value():
            return card_to_index(random.choice(valid_cards))

        with torch.no_grad():
            self.q_net.eval()
            q = self.q_net.q_values_at_indices(state_t, global_indices)
            self.q_net.train()
        j = int(q.argmax().item())
        return int(global_indices[j])

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

    def _heuristic_reward(self, card, lead_suit, round_num):
        """Original dense shaping. See RULES.md and evaluate_play()."""
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

    def evaluate_play(self, card, lead_suit=None, round_num=1):
        """
        Per-play shaping reward. Behavior depends on `self.reward_mode`:

          - "heuristic" : full dense shaping (legacy default, pre-overhaul).
          - "outcome"   : zero per-play reward — only the terminal bonus
                          (see compute_terminal_reward) matters. Aligns RL
                          objective with actually winning the hand.
          - "mixed"     : shaping_weight * heuristic_reward, plus terminal.
                          Use when pure-outcome is too sparse to learn from
                          with small compute budgets.

        Callers (Hokm.evaluate_play) are unchanged; this method is the single
        source of truth for per-play reward.
        """
        if self.reward_mode == REWARD_OUTCOME:
            return 0.0
        dense = self._heuristic_reward(card, lead_suit, round_num)
        if self.reward_mode == REWARD_MIXED:
            return self.shaping_weight * dense
        return dense

    def compute_terminal_reward(self) -> float:
        """
        Reward delivered on the final transition of the hand (done=True).

        Returns 0 in pure heuristic mode (backward compatible). In outcome /
        mixed modes, returns ±win_bonus for the binary hand outcome plus a
        small trick-differential signal (bounded by ±0.1 * 13 = ±1.3).
        """
        if self.reward_mode == REWARD_HEURISTIC or not self.team:
            return 0.0
        my_tricks = sum(self.tricks_won.get(p, 0) for p in self.team)
        total = sum(self.tricks_won.values())
        opp_tricks = total - my_tricks
        won = my_tricks >= 7
        diff = my_tricks - opp_tricks
        return (self.win_bonus if won else -self.win_bonus) + 0.1 * diff

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

        if self._shared is not None:
            self._shared.push_transition(
                state, action, reward, next_state, done, rl_eligible
            )
            if rl_eligible:
                self.total_reward += reward
            return

        self.sl_buffer.append((state.detach().cpu(), int(action)))
        if not rl_eligible:
            return
        self.memory.push((state, action, reward, next_state, done))
        self.total_reward += reward

    def optimize_model(self, beta=0.4):
        del beta  # uniform replay — no IS weights
        if not self.learning_enabled:
            return
        if self._shared is not None:
            self._shared.maybe_optimize()
            self.epsilon = self._shared.epsilon
            return

        self._ply_since_learn += 1
        if self._ply_since_learn < self.learn_every:
            return
        self._ply_since_learn = 0
        self._optimize_q()
        self._optimize_sl()
        self.update_epsilon()

    def _optimize_q(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        experiences = self.memory.sample(self.batch_size)
        if not experiences:
            return
        batch = list(zip(*experiences))
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

        current_q_values = self.q_net(states).gather(1, actions)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.update_target_net()

    def _optimize_sl(self) -> None:
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

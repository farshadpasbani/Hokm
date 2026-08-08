"""
Deep Monte-Carlo (DMC) network and inference player for Hokm.

This is the redesigned learning stack (see TRAINING_REPORT.md §redesign):

  * **Action-as-input**: the net scores Q(s, a) with the candidate card
    embedded as an input, so per-play patterns ("any trump beats any
    off-suit ace") generalize across all 52 cards instead of being
    relearned per output head.
  * **Sequence encoding**: a GRU consumes the hand's play history as
    (card, relative-seat) events, alongside the existing 194-dim static
    observation (reused from `EnhancedPlayer.get_state`, so the tested
    feature pipeline stays the single source of truth).
  * **Centralized training, decentralized execution**: `CentralCritic`
    sees all four hands during self-play (the trainer deals them, so the
    information is free) and regresses the hand's return; the actor's
    V-head is distilled toward the critic's value, and an auxiliary head
    predicts which opponent holds each unseen card (labels also free).
    At play time only the actor runs — no hidden information touched.

The dueling composition Q(s,a) = V(s) + A(s,a) ties the distilled value
into action scoring.

Training lives in `dmc_train.py`. This module is inference + model.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enhanced_player import EnhancedPlayer
from game_constants import Card, STATE_DIM, card_to_index

CARD_EMB = 32
SEAT_EMB = 8
GRU_HIDDEN = 64
STATE_ENC = 256
TRUNK = 256
MAX_HISTORY = 52


def history_features(game, my_seat: int) -> Tuple[List[int], List[int]]:
    """(card_idx, relative_seat) events for the current hand in play order,
    from the engine's exact seat-attributed `play_log_this_hand`. This is
    THE feature path for both training and inference — keep them identical.
    """
    cards: List[int] = []
    seats: List[int] = []
    if game is not None:
        for seat, c in getattr(game, "play_log_this_hand", []) or []:
            cards.append(card_to_index(c))
            seats.append((seat - my_seat) % 4)
    return cards[-MAX_HISTORY:], seats[-MAX_HISTORY:]


class DMCNet(nn.Module):
    """Actor: Q(s, a) with sequence encoder + aux heads.

    forward() returns per-candidate Q values plus the value/aux outputs
    needed by the trainer; `q_values()` is the inference-only path.
    """

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.card_emb = nn.Embedding(52, CARD_EMB)
        self.seat_emb = nn.Embedding(4, SEAT_EMB)
        self.gru = nn.GRU(CARD_EMB + SEAT_EMB, GRU_HIDDEN, batch_first=True)
        self.static_enc = nn.Sequential(
            nn.Linear(state_dim, STATE_ENC), nn.LayerNorm(STATE_ENC), nn.ReLU()
        )
        enc_dim = STATE_ENC + GRU_HIDDEN
        self.v_head = nn.Sequential(
            nn.Linear(enc_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.aux_head = nn.Linear(enc_dim, 3 * 52)
        self.adv_trunk = nn.Sequential(
            nn.Linear(enc_dim + CARD_EMB, TRUNK), nn.LayerNorm(TRUNK), nn.ReLU(),
            nn.Linear(TRUNK, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode(
        self,
        static: torch.Tensor,           # [B, state_dim]
        hist_cards: torch.Tensor,       # [B, T] long (padded with 0)
        hist_seats: torch.Tensor,       # [B, T] long
        hist_lens: torch.Tensor,        # [B] long (0 = empty history)
    ) -> torch.Tensor:
        """Return the state encoding E_s of shape [B, STATE_ENC+GRU_HIDDEN]."""
        b = static.shape[0]
        s_enc = self.static_enc(static)
        if hist_cards.shape[1] == 0:
            h = torch.zeros(b, GRU_HIDDEN, device=static.device)
        else:
            ev = torch.cat(
                [self.card_emb(hist_cards), self.seat_emb(hist_seats)], dim=-1
            )
            out, _ = self.gru(ev)  # [B, T, H]
            idx = (hist_lens - 1).clamp(min=0)
            h = out[torch.arange(b), idx]
            h = torch.where(hist_lens.unsqueeze(1) > 0, h, torch.zeros_like(h))
        return torch.cat([s_enc, h], dim=-1)

    def q_from_encoding(
        self, enc: torch.Tensor, action_idx: torch.Tensor
    ) -> torch.Tensor:
        """Q(s,a) = V(s) + A(s,a). enc: [B, E]; action_idx: [B] long."""
        v = self.v_head(enc)                              # [B, 1]
        a = self.adv_trunk(
            torch.cat([enc, self.card_emb(action_idx)], dim=-1)
        )                                                 # [B, 1]
        return (v + a).squeeze(-1)                        # [B]

    def forward(self, static, hist_cards, hist_seats, hist_lens, action_idx):
        enc = self.encode(static, hist_cards, hist_seats, hist_lens)
        q = self.q_from_encoding(enc, action_idx)
        v = self.v_head(enc).squeeze(-1)
        aux = self.aux_head(enc)  # [B, 156] logits
        return q, v, aux

    @torch.no_grad()
    def q_values(
        self,
        static: torch.Tensor,           # [state_dim]
        hist_cards: Sequence[int],
        hist_seats: Sequence[int],
        candidates: Sequence[int],
    ) -> torch.Tensor:
        """Q for each candidate action at a single decision point."""
        n = len(candidates)
        t = len(hist_cards)
        static_b = static.unsqueeze(0)
        hc = torch.tensor([list(hist_cards)], dtype=torch.long) if t else torch.zeros(1, 0, dtype=torch.long)
        hs = torch.tensor([list(hist_seats)], dtype=torch.long) if t else torch.zeros(1, 0, dtype=torch.long)
        hl = torch.tensor([t], dtype=torch.long)
        enc = self.encode(static_b, hc, hs, hl)           # [1, E]
        enc_n = enc.expand(n, -1)
        act = torch.tensor(list(candidates), dtype=torch.long)
        return self.q_from_encoding(enc_n, act)           # [n]


class CentralCritic(nn.Module):
    """Training-only value net over FULL information (all four hands).

    Input: 4×52 hand one-hots (seat-relative to the mover: me, LHO,
    partner, RHO) + trump one-hot (4) + team/opp trick counts (2) = 214.
    """

    IN_DIM = 4 * 52 + 4 + 2

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.IN_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference seat
# ---------------------------------------------------------------------------

class DMCPlayer(EnhancedPlayer):
    """Greedy DMC seat for evaluation and app play.

    Maintains its own play-history sequence via the engine's public
    `cards_played_this_hand` plus the live trick (both synced by the
    engine on every play in both the play_round and apply_play paths).
    """

    def __init__(
        self,
        name: str,
        net: Optional[DMCNet] = None,
        checkpoint: Optional[str] = None,
        epsilon: float = 0.0,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(name)
        self.learning_enabled = False
        self.eta = 0.0
        self.epsilon = epsilon
        self._rng = rng or random.Random()
        self.net = net or DMCNet()
        if checkpoint:
            blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.net.load_state_dict(blob["dmc_net"])
        self.net.eval()

    def store_experience(self, *_a, **_k) -> None:
        return None

    def optimize_model(self, *_a, **_k) -> None:
        return None

    def _history(self) -> Tuple[List[int], List[int]]:
        my_seat = self._seat if self._seat is not None else 0
        return history_features(self._game, my_seat)

    def play_card(self, lead_suit, selected_card=None):
        self.lead_suit = lead_suit
        valid = (
            self.hand
            if lead_suit is None
            else [c for c in self.hand if c.suit == lead_suit] or self.hand
        )
        if not valid:
            raise ValueError(f"No valid cards to play for {self.name}")
        if len(valid) == 1 or (self.epsilon and self._rng.random() < self.epsilon):
            card = valid[0] if len(valid) == 1 else self._rng.choice(valid)
            return card, card_to_index(card)
        static = self.get_state().cpu()
        hist_c, hist_s = self._history()
        cand = [card_to_index(c) for c in valid]
        q = self.net.q_values(static, hist_c, hist_s, cand)
        card = valid[int(q.argmax().item())]
        return card, card_to_index(card)

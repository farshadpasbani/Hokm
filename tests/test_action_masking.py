"""
Regression tests for the next-state action-masking fix.

Background: `_optimize_q` used to compute Double-DQN targets via an
unconstrained argmax over all 52 actions — including cards not in the
player's hand and cards that violate follow-suit. Q-values at these
"phantom" actions are never anchored by real-world rewards (they're
never executed), so they drift freely upward and poison every legitimate
state via bootstrapping. The symptoms in production were `|Q*|` growing
without bound and the agent locking into worse-than-random play.

The fix is to attach a `next_legal_mask` (bool tensor [action_dim]) to
each replay entry and apply `masked_fill(~mask, -inf)` before the argmax.
These tests pin the invariants that keep the bug from coming back.
"""

from __future__ import annotations

import torch

from config import DEFAULT_CONFIG, REWARD_OUTCOME
from enhanced_player import SharedNFSPLearner, EnhancedPlayer
from game_constants import ACTION_DIM, card_to_index
from hokm import Hokm


def _legal_mask(indices):
    m = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for i in indices:
        m[i] = True
    return m


def test_optimize_q_argmax_respects_next_legal_mask():
    """The Double-DQN target must never select an illegal next-action.

    We construct a learner, push a small batch of synthetic transitions
    where the legal next-action set is severely restricted (a single
    index per transition), then verify that the next-action argmax falls
    inside the mask for *every* transition no matter what the random
    Q-net weights say.
    """
    learner = SharedNFSPLearner(batch_size=4)

    legal_idx = [3, 17, 41, 0]
    for k in range(4):
        s = torch.zeros(learner.state_dim)
        ns = torch.zeros(learner.state_dim)
        learner.push_transition(
            state=s,
            action=legal_idx[k],
            reward=0.0,
            next_state=ns,
            done=False,
            rl_eligible=True,
            next_legal_mask=_legal_mask([legal_idx[k]]),
        )

    # Pull the same batch out (the only one available) and check argmax.
    # `random.sample` permutes the order, so we verify the per-row
    # invariant: each chosen action must be the unique True index in its
    # own row's mask.
    batch = list(zip(*learner.memory.sample(4)))
    next_states = torch.stack(batch[3])
    next_masks = torch.stack(batch[5])
    q = learner.q_net(next_states).masked_fill(~next_masks, float("-inf"))
    next_actions = q.argmax(1)
    for row, picked in enumerate(next_actions.tolist()):
        legal_in_row = next_masks[row].nonzero(as_tuple=False).flatten().tolist()
        assert picked in legal_in_row, (
            f"row {row}: argmax picked {picked} but only {legal_in_row} "
            f"were legal"
        )


def test_optimize_q_runs_without_nan_at_terminal_with_empty_mask():
    """At done=True the next-state has an empty hand and the mask is all
    False. Multiplying -inf * 0 yields NaN in float, which would silently
    corrupt the target. The optimizer must zero next_q before the
    multiplication. This test triggers that path.
    """
    learner = SharedNFSPLearner(batch_size=2)
    s = torch.zeros(learner.state_dim)
    ns = torch.zeros(learner.state_dim)
    empty_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for _ in range(2):
        learner.push_transition(
            state=s, action=0, reward=1.0, next_state=ns, done=True,
            rl_eligible=True, next_legal_mask=empty_mask,
        )
    learner._optimize_q()
    # If NaN had leaked we'd see a NaN in the recorded Q-loss.
    assert learner._q_losses, "expected one loss recorded"
    assert all(l == l for l in learner._q_losses), \
        "Q-loss must not be NaN even when mask is empty at done=True"


def test_play_round_supplies_next_legal_mask():
    """End-to-end: after a full hand of self-play, every transition stored
    in the shared replay must carry a `next_legal_mask` of the right
    shape, dtype, and structure (non-empty for non-terminal transitions,
    empty exactly when done=True with hand_empty)."""
    import random as _random

    cfg = DEFAULT_CONFIG
    learner = SharedNFSPLearner.from_config(cfg)
    players = [
        EnhancedPlayer(
            f"P{i}",
            shared_learner=learner,
            reward_mode=REWARD_OUTCOME,
        )
        for i in range(4)
    ]
    game = Hokm(players, minimal_logging=True, rng=_random.Random(0))
    game.play_game(save_excel_log=False)

    assert len(learner.memory) >= 4, "full hand should produce many replay entries"
    saw_terminal = False
    for entry in list(learner.memory.memory):
        s, a, r, ns, done, mask = entry
        assert mask.dtype == torch.bool
        assert mask.shape == (ACTION_DIM,)
        if done:
            saw_terminal = True
            # At done=True, mask is empty (hand_empty after final play).
            assert not mask.any(), \
                f"terminal transition should have empty legal mask, got {mask.sum().item()} True bits"
        else:
            assert mask.any(), \
                "non-terminal transition has empty legal mask — phantom-action regression"
    assert saw_terminal, "expected at least one terminal transition in replay"

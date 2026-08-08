# Hokm AI — Architecture

One page summarizing how the pieces fit. See `RULES.md` for the game rules
themselves and `README.md` for how to run everything.

## 1. Data flow

```
  game_constants.py          ─┐
      (Card, ranks,           │
       STATE_DIM=194,         │
       STATE_LAYOUT, etc.)    │
                              ▼
  hokm.py (Hokm, Deck) ──► play_round() ──► evaluate_play() ──► store_experience()
     │   ▲                        │              │                   │
     │   └─ baselines.py ─────────┤              │                   ▼
     │   └─ enhanced_player.py ───┘              │           SharedNFSPLearner
     │                                           │           (Q, target Q, π̄)
     │                                           ▼
     │                                  compute_terminal_reward()  (opt-in)
     │
     ▼
  train_backend.TrainBackend  (curriculum / opponent mix)
     ├─► torch.save → models/nfsp_shared_*.pth
     └─► summaries/, plots/

  evaluate.py  ──► loads checkpoint ──► runs N games vs each baseline
                                      ──► Wilson 95% CIs to stdout / JSON / CSV

  dev_blueprint.py (/dev/*) ──► web UI for training, listing models, and
                                 running the evaluator (ε = η = 0)

  app.py (Flask) ──► human vs 3 AIs; AIs load the same checkpoint format
                      as TrainBackend produces.
```

## 2. What is trained vs what is frozen

**Trained** (weights updated each `maybe_optimize()` tick):
- `SharedNFSPLearner.q_net` — the best-response Q-network.
- `SharedNFSPLearner.avg_policy_net` — the average policy π̄ (NFSP's σ).

**Frozen/derived**:
- `SharedNFSPLearner.target_q_net` — copied from `q_net` every
  `target_update_frequency` optimization steps.
- Per-seat `EnhancedPlayer` instances share those networks when constructed
  with `shared_learner=...`, so there are **no** per-seat weights in
  self-play training.

## 3. Losses

- **Q-loss** (`_optimize_q`): double-DQN MSE against
  `r + γ · Q_target(s', argmax_a' Q(s', a'))` over a **uniform** replay
  batch of size `batch_size`. Grad-clipped at `grad_clip` on `q_net`.
- **SL-loss** (`_optimize_sl`): cross-entropy of π̄(a | s) against the
  greedy-Q action drawn from a **reservoir** of all training transitions
  (size `sl_reservoir_size`). This is NFSP's approximation of the
  time-averaged best response.

Both losses run together every `learn_every` plies (not every play,
reducing overhead).

## 4. Reward

Configurable via `HokmConfig.nfsp.reward_mode`:

| mode       | per-play                        | terminal                          |
|------------|---------------------------------|-----------------------------------|
| heuristic  | `_heuristic_reward(...)`        | 0                                 |
| outcome    | 0                               | ±`win_bonus` + 0.1·Δtricks       |
| mixed      | `shaping_weight · heuristic`    | ±`win_bonus` + 0.1·Δtricks       |

`compute_terminal_reward()` is added to the **last** transition of the hand
(game over OR the agent's hand empty). Heuristic mode is the original
pre-overhaul default; `outcome` is the scientifically correct starting
point — "optimize for winning", then dial in shaping if outcome-only is
too sparse to learn under your compute budget.

## 5. Opponent curricula

`HokmConfig.opponents` governs per-game opponent swapping:

- `trainable_seats` — which seats use the shared learner.
- Non-trainable seats sample **per game** from
  `{self_play, random, heuristic, frozen_pool}` weights.
- `frozen_pool_dir` — a directory of past `.pth` checkpoints to draw from
  (fictitious-play style). `None` → skipped.

Only self-play seats push RL transitions to the shared buffer (baselines
are `learning_enabled = False`).

## 6. Inference / deployment (web app)

- `app.create_game` builds one `EnhancedPlayer` per AI seat and calls
  `load_policy_state(path)` on each.
- Checkpoints produced by `TrainBackend` are shared-learner state dicts
  (`{"q_net": ..., "avg_policy_net": ...}`). `load_policy_state` handles
  both that format and raw `q_net` state dicts.
- The web app's game loop uses `apply_play` / `resolve_trick_if_complete`,
  **not** `play_round`, so no `store_experience` / `optimize_model` runs in
  production — weights are read-only at play time.
- **Recommended runtime settings** for human-facing play: `ε = 0`, `η = 0`,
  `learning_enabled = False`. (The dev console / `evaluate.py` do this
  already; `app.create_game` does not yet and should be tightened if you
  notice the AI making surprising random moves.)

## 7. Evaluation

`evaluate.py` is the canonical benchmark script. For every matchup it:

- Seeds Python, NumPy, and Torch.
- Seats the trained policy on Team 1 (seats 0, 2) and the opponent family
  on Team 2 (seats 1, 3).
- Plays N hands (default 1000) and reports win rate with a **Wilson 95%
  CI**, tie rate, and mean trick differential.
- Serializes results to JSON and/or CSV.

`dev_blueprint.py` also exposes a simpler 4-seat evaluation via
`/dev/api/evaluate` — it now sets `ε = 0` **and** `η = 0`, fixing the
previous train/eval mismatch where the avg-policy branch still sampled 25%
of the time in "evaluation".

## 8. Observation space

The 194-dim feature vector built by `EnhancedPlayer.get_state()` is the
contract between the game and the networks. It was expanded from the
original 114 dims specifically to make the following "human lemmas"
*information-theoretically* learnable:

| Block | Dims | Unlocks |
|---|---|---|
| hand one-hot | 52 | baseline |
| **cards played this hand (rank-level)** | 52 | high-card promotion, suit establishment |
| **per-opponent voids (3 × 4)** | 12 | void inference from failure-to-follow |
| full current trick (not just last card) | 52 | Markov state of in-progress trick |
| lead suit | 4 | explicit lead signal |
| **trick position (1st / 2nd / 3rd / 4th to play)** | 4 | seat-strategy (second-hand-low / third-hand-high) |
| **current-winner seat (empty / me / partner / LHO / RHO)** | 5 | partner-ducking |
| current-winner card value (normalised) | 1 | risk evaluation |
| **Hakem is me / Hakem is partner** | 2 | asymmetric Hakem-aware play |
| team & opp trick counts (normalised /7) | 2 | score-adaptive aggressiveness |
| trump one-hot | 4 | baseline |
| hand per-suit counts (normalised /13) | 4 | long-suit inductive bias |

Total 194. Named slices live in `game_constants.STATE_LAYOUT`; tests in
`tests/test_state.py` pin each slice to its semantics.

`cards_played_this_hand` and `void_map` are maintained on the `Hokm`
instance and read by every seat, so all four players share a coherent
view of public information. Voids are inferred automatically: when a
non-leader plays a card whose suit differs from the lead suit, they are
marked void in the led suit for the rest of the hand.

Legacy checkpoints with a different input dim (e.g. the 114-dim format
from earlier runs) are loaded with a one-line warning from
`_load_state_dict_compat`: the first Linear resets to fresh init, all
deeper layers continue to load normally.

## 9. Honest limitations

* `choose_trump_suit()` is a heuristic on Hakem's first 5 cards and is not
  learned. Training a Hakem-specific policy would require an extra head.
* "One game" = one hand **in the engine and in training/evaluation**. Match
  scoring (hands to 7, Kot = 2 points) exists only in
  `game_service.GameSession`, which drives the engine hand by hand for the
  Mini App; the reward signal the agents learn from is still per-hand.
* Compute in this repo is modest. Achieving robust human-level strength
  likely needs 10⁵–10⁶ self-play hands plus the baseline curriculum, and
  evaluation against strong heuristic baselines at each milestone.
* Baselines in `baselines.py` are intentionally simple — a competent
  Hokm player is stronger than `HeuristicAgent` once they count trumps
  and model partners.

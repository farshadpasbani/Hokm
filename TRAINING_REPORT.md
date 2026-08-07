# NFSP Training Report — 2026-08-07 session

Everything below was run on the fixed engine (see "the observation bug").
CPU-only container, 4 cores, ~12–15 games/s training throughput.

## TL;DR

1. A severe **observation bug** was found and fixed before training: in the
   training path, every seat about to act saw the *previous* trick's four
   cards instead of the live trick (~66 of 194 observation dims corrupted at
   ~35 of 48 decisions per hand). All prior training runs were affected.
2. Four training configurations were run on the fixed engine (~105k games
   total). All of them **plateau at ≈14–18% win rate vs `HeuristicAgent`**
   (and ≈54–59% vs random) under greedy-Q evaluation.
3. Monte-Carlo returns (`NFSPConfig.mc_returns`, added this session) reach
   the plateau **~4–5× faster** than 1-step TD but stop at the same level —
   the ceiling is *not* credit-assignment speed. Cutting NFSP's η-noise
   (0.25 → 0.05) changed nothing either — the ceiling is *architectural*
   (see Diagnosis).
4. The rule-based `HeuristicAgent` remains the strongest available opponent
   and stays the Mini App default. The best NFSP checkpoint is committed
   under `checkpoints/` for warm-starting future runs.

## Definitive evaluation of the best checkpoint

`checkpoints/nfsp_td_outcome_20k.pth` (Run 1 @ 20k), 1000 games per
matchup, seed 42 (`dev_cache/final_eval.json`):

| opponent  | win rate | 95% CI        | Δ tricks |
|-----------|---------:|---------------|---------:|
| random    | 59.0%    | [55.9, 62.0]  | +0.59 |
| heuristic | 16.4%    | [14.2, 18.8]  | −2.77 |
| self      | 52.1%    | [49.0, 55.2]  | +0.06 (sanity ✓) |
| untrained | 56.1%    | [53.0, 59.1]  | +0.31 |

Learning is statistically real (beats random and untrained with
non-overlapping CIs) but far below the rule-based baseline.

## The observation bug (fixed in this branch)

`Hokm.play_round()` rebound `self.current_trick` to a new list each trick
but only re-pointed a player's reference *after that player played*. Seats
that had not yet played in the current trick observed the previous trick's
final state: wrong cards-on-table, wrong lead suit, wrong trick position,
wrong current-winner. Instrumentation showed 35 of ~48 decision points per
hand carried a wrong current-trick block. The fix syncs all players to the
new list at trick start; `tests/test_observation_freshness.py` guards it.

Consequence: **all pre-fix training results (including the 50k/100k runs
referenced in code comments) should be considered invalid.**

## Runs

Common setup: outcome-aligned terminal reward (±1.0 win bonus + 0.1·trick
diff), Double-DQN with legal-action masking, Polyak targets (τ=0.002),
γ=0.97, η=0.25, shared learner for seats 0+2, evaluation = pure greedy Q
(ε=η=0) with Wilson 95% CIs, 300 games per matchup unless noted.

### Run 1 — 1-step TD, outcome reward (seed 42, stopped at 40k of 60k)

Opponent mix: self 0.5 / heuristic 0.4 / random 0.1.

| games | vs random | vs heuristic | win-when-able* | duck-ok* |
|------:|----------:|-------------:|---------------:|---------:|
| 0 (untrained) | 56.3% | 12.0% | 55.6% (random line) | 54.5% |
| 4k | 54.7% | 11.7% | — | — |
| 8k | 53.0% | 12.0% | 63.8% (10k) | 72.3% (10k) |
| 20k | **58.7%** | **17.7%** | **75.0%** | **79.1%** |
| 40k | 57.0% | 16.7% | 64.0% | 76.1% |

\* Behavioral probes, seat 0 last-to-play: "played a winning card when one
existed and partner wasn't winning" / "dumped cheap when partner was
already winning". `HeuristicAgent` scores 100% / 95.5%; random ≈55%.

Reading: real learning 0→20k, then plateau/oscillation. Stopped at 40k.

### Run 2 — mixed reward fine-tune (warm-start from Run 1 @40k, stopped at 12.5k)

`reward_mode="mixed"` (dense heuristic shaping ×0.1), heuristic share
raised to 0.5, ε=0.05. **No lift**: 55.7%/14.3% at 10k, probes flat.
Dense shaping did not crack the plateau.

### Run 3 — Monte-Carlo returns (fresh, seed 7)

`mc_returns=True` (added this session): transitions stored with full
discounted return-to-go, no bootstrapping. Same opponent mix as Run 2.

| games | vs random | vs heuristic |
|------:|----------:|-------------:|
| 4k | 57.0% | 16.3% |
| 10k | 53.7% | 16.0% |
| 20k | 54.0% | 13.7% |

Reading: reaches Run 1's 20k-game level within ~4k games (~5× faster
per-sample credit assignment), then flattens at the same ceiling.
Stopped at 20k.

### Run 4 — MC returns + η=0.05 (fresh, seed 8, 15k games)

Tests the "η-noise corrupts MC returns" hypothesis: identical to Run 3
but with only 5% of training moves sampled from the average policy.

| games | vs random | vs heuristic |
|------:|----------:|-------------:|
| 4k | 55.3% | 15.3% |
| 14k | 55.3% | 15.7% |

Reading: within noise of Run 3 — **η-noise is ruled out** as the
binding constraint at this scale.

## Diagnosis

The plateau is consistent across four configurations spanning three
learning signals and two exploration levels. That rules out reward
sparsity, credit-propagation speed, and trajectory noise as the binding
constraints. What remains is architectural:

1. **No hidden-information modeling.** The observation gives proven voids
   only. Beating the heuristic consistently requires inference about
   unseen hands (finesse/promotion reasoning) that a reactive MLP policy
   on this observation struggles to represent.
2. **Representation.** A flat 52-way output head must learn each card's
   value independently; per-play patterns ("any trump beats any
   off-suit") don't generalize across cards. Fixed-summary features
   can't encode the play-order patterns sequence models capture.
3. **No variance control for the team game.** The hand's outcome depends
   heavily on the partner's play; nothing in the pipeline exploits the
   fact that the trainer *knows all four hands* during self-play.
4. **Sample scale.** ~15 games/s on CPU. Modern card-game results
   (DouZero, Suphx) use orders of magnitude more samples and/or search
   at decision time.

## Recommended redesign (in order of strength-per-effort)

1. **Determinized search at play time (PIMC).** Sample opponent hands
   consistent with voids/played cards, roll out each with the heuristic
   or Q-net as policy, pick the action with the best average outcome.
   The classic trick-taking-AI approach (Bridge/Skat engines); would
   likely beat the heuristic immediately, with today's net or none.
   The engine already tracks everything a determinizer needs
   (`void_map`, `cards_played_this_hand`, `legal_cards_for_player`).
2. **Replace NFSP with Deep Monte-Carlo** (DouZero-style — `mc_returns`
   is the first step and is now available). Hokm rewards strength and
   coordination, not equilibrium unexploitability; the average-policy
   machinery costs more than it buys (measured: the avg-policy head
   plays *worse* than greedy Q, 49% vs 57% against random).
3. **Action-as-input Q(s, a) + sequence encoding** of the play history
   (small LSTM/transformer) so per-play patterns generalize across cards.
4. **Centralized training, decentralized execution**: a critic that sees
   all four hands during self-play as a baseline for advantage
   estimation, plus an auxiliary head predicting opponents' hands
   (labels are free at training time). Directly attacks diagnoses 1 & 3.
5. **League self-play** (frozen-checkpoint pool via the existing
   `frozen_pool` config + heuristic as a permanent member) instead of a
   fixed mix.
6. **Throughput**: parallel actor processes + GPU learner; strip pandas
   from the hot path. These methods want 10–100M games, not 100k.

## What ships today

- App default remains `HeuristicAgent` (crisp fundamentals: 100%
  win-when-able, 95.5% duck-when-partner-winning), which is clearly
  stronger than every NFSP checkpoint produced so far.
- Best NFSP checkpoint (`Run 1 @ 20k`) is committed under `checkpoints/`
  for warm-starting; set `MODEL_PATH=checkpoints/<file>` to play against
  it in the Mini App.
- `models_release/` stays empty — dropping any `.pth` there makes the app
  use it automatically (see `game_service._default_model_path`).

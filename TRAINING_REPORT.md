# NFSP Training Report — 2026-08-07 session

Everything below was run on the fixed engine (see "the observation bug").
CPU-only container, 4 cores, ~12–15 games/s training throughput.

## TL;DR

1. A severe **observation bug** was found and fixed before training: in the
   training path, every seat about to act saw the *previous* trick's four
   cards instead of the live trick (~66 of 194 observation dims corrupted at
   ~35 of 48 decisions per hand). All prior training runs were affected.
2. Three training configurations were run on the fixed engine (~90k games
   total). All of them **plateau at ≈15–18% win rate vs `HeuristicAgent`**
   (and ≈55–59% vs random) under greedy-Q evaluation.
3. Monte-Carlo returns (`NFSPConfig.mc_returns`, added this session) reach
   the plateau **~4–5× faster** than 1-step TD but stop at the same level —
   strong evidence the ceiling is *not* credit-assignment speed but
   model/approach capacity.
4. The rule-based `HeuristicAgent` remains the strongest available opponent
   and stays the Mini App default. The best NFSP checkpoint is committed
   under `checkpoints/` for warm-starting future runs.

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

Reading: reaches Run 1's 20k-game level within ~4k games (~5× faster
per-sample credit assignment), then flattens at the same ceiling.

## Diagnosis

The plateau is consistent across three different learning signals, which
rules out the usual first-order suspects (reward sparsity, propagation
speed). The remaining explanations, in likely order of impact:

1. **η-noise in trajectories.** With η=0.25, a quarter of all training
   moves are sampled from the average-policy net, which stays near-uniform
   for a long time. Team outcomes — and especially undiscounted MC
   returns — are heavily corrupted by these noise moves (your partner
   throwing an ace away randomly changes the label on *your* good plays).
2. **No hidden-information modeling.** The observation gives proven voids
   only. The heuristic effectively "knows" the trick mechanics perfectly;
   beating it consistently requires inference about unseen hands
   (finesse/promotion reasoning) that a reactive MLP policy on this
   observation struggles to represent.
3. **Capacity/optimization.** 194→256→128→64→52 MLP, uniform replay,
   batch 32, CPU. Modern card-game results (DouZero, Suphx, ReBeL) use
   orders of magnitude more samples and/or search at decision time.

## Recommended roadmap (in order)

1. **Drop η to 0.05–0.1** during training (standard anticipatory-NFSP
   values) and re-run MC for ≥100k games. Cheapest experiment, directly
   attacks the top suspect.
2. **Larger Q-net + prioritized replay + n-step (already available via
   `mc_returns`)**, batch 128–256 on GPU. This is "the same algorithm,
   properly fed".
3. **Determinized search at play time** (PIMC): sample opponent hands
   consistent with voids/played cards, roll out with the Q-net as a
   policy prior, pick the action with the best average outcome. This is
   the classic trick-taking-AI approach and would likely beat the
   heuristic even with today's net as the prior.
4. **Belief features**: add per-card "probability opponent i holds it"
   estimates (even simple count-based ones) to the observation.

## What ships today

- App default remains `HeuristicAgent` (crisp fundamentals: 100%
  win-when-able, 95.5% duck-when-partner-winning), which is clearly
  stronger than every NFSP checkpoint produced so far.
- Best NFSP checkpoint (`Run 1 @ 20k`) is committed under `checkpoints/`
  for warm-starting; set `MODEL_PATH=checkpoints/<file>` to play against
  it in the Mini App.
- `models_release/` stays empty — dropping any `.pth` there makes the app
  use it automatically (see `game_service._default_model_path`).

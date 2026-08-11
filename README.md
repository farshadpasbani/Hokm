# Hokm — Persian Trick-Taking Card Game with NFSP AI

A Python implementation of **Hokm (حکم)** with Neural Fictitious Self-Play
(NFSP) agents, a Flask web UI for human-vs-AI play, and a developer console
for training and evaluation.

The rules enforced by the engine are documented in [`RULES.md`](./RULES.md).
The training / inference architecture is documented in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Requirements

Python 3.10+ (tested on 3.11). Install dependencies:

```bash
pip install -r requirements.txt
```

Optional for tests:

```bash
pip install pytest
```

## Quickstart

### 1. Train a model

```bash
python train_hokm.py --num-games 1000 --save-interval 100 --seed 42
```

or via the dev console (with live charts):

```bash
python app.py
# open http://localhost:5000/dev/console
```

Checkpoints land in `models/nfsp_shared_<timestamp>_game_<N>.pth`.

### 2. Evaluate a checkpoint against baselines

```bash
python evaluate.py \
    --checkpoint models/nfsp_shared_<...>_game_<N>.pth \
    --opponent all \
    --games 1000 \
    --seed 42 \
    --out dev_cache/eval_suite.json \
    --csv dev_cache/eval_suite.csv
```

Output: win rate vs each of `{random, heuristic, self, untrained}` with
Wilson 95% confidence intervals.

### 3. Play against three AIs

```bash
python app.py
# open http://localhost:5000
```

Pick the three AI checkpoints from the dev console's "Models & play"
section; each page load then seats those three opponents against you.

## Telegram Mini App (production)

The repo also ships a production service that runs Hokm as a playable
Telegram Mini App: `server.py` serves a mobile UI, a per-user **match**
API (first to 7 hand wins, Kot counts double, Hakem rotates between
hands) authenticated with signed Telegram `initData`, and the bot
webhook — all from one container. The default opponent is **PIMC**
(`pimc.py`) — determinized Monte-Carlo search that beats the rule-based
heuristic 62% [57.4, 66.9] and every trained net produced so far (see
[`TRAINING_REPORT.md`](./TRAINING_REPORT.md)). See
[`DEPLOYMENT.md`](./DEPLOYMENT.md) for the BotFather + hosting
walkthrough, or smoke-test it locally:

```bash
pip install -r requirements-prod.txt
python server.py
# open http://localhost:8080 (guest mode when BOT_TOKEN is unset)
```

## AI agents & evaluation

Four agent families, all evaluable head-to-head with the general harness:

```bash
python evaluate.py --team1 pimc:48 --team2 heuristic --games 400
python evaluate.py --team1 dmc:models/dmc_latest.pt --team2 all --games 300
python evaluate.py --team1 nfsp:checkpoints/nfsp_td_outcome_20k.pth --team2 pimc:32
```

| family | spec | what it is |
|---|---|---|
| PIMC | `pimc[:N]` | Determinized MC search (`pimc.py`); strongest. |
| Heuristic | `heuristic` | Rule-based baseline (`baselines.py`). |
| NFSP | `nfsp:<path>` | Greedy Q from `train_hokm.py`/`train_backend.py`. |
| DMC | `dmc:<path>` | Deep MC net (`dmc.py`); train with `dmc_train.py --games N --actors 3` (league self-play, parallel actors, `--eval-every` best-checkpoint tracking). |

## Project layout

| File                   | Purpose |
|------------------------|---------|
| `hokm.py`              | Core game engine (deck, tricks, scoring, rotation). |
| `enhanced_player.py`   | NFSP agent + `SharedNFSPLearner` (shared weights). |
| `baselines.py`         | `RandomAgent`, `HeuristicAgent` non-learning baselines. |
| `game_constants.py`    | Card / suit constants, state/action dims. |
| `config.py`            | `HokmConfig` dataclass — single source of truth for hyperparams. |
| `seed_utils.py`        | One-call seeding for `random` / `numpy` / `torch`. |
| `train_backend.py`     | `TrainBackend` — self-play and opponent-curriculum training. |
| `train_hokm.py`        | Standalone CLI wrapper around `TrainBackend`-ish loop. |
| `evaluate.py`          | Evaluation CLI; win rate + 95% CI vs each baseline. |
| `dev_eval.py`          | Programmatic greedy evaluation (used by the dev console). |
| `dev_blueprint.py`     | Flask blueprint for the dev console API (`/dev/*`). |
| `app.py`               | Flask app for human-vs-AI play. |
| `tests/`               | Pytest suite for rules and reward-mode invariants. |

## Design notes

1. **Train/eval parity bug fixed.** Evaluation now sets both ε = 0 and
   η = 0, so the agent is pure-greedy at test time instead of still
   sampling from the NFSP average policy 25% of the time.
2. **Reward alignment.** The default reward mode (`config.NFSPConfig.reward_mode`)
   is now `"outcome"`: per-play shaping = 0, and a terminal reward of
   `±win_bonus + 0.1·Δtricks` is injected on the last transition of the
   hand. Set to `"heuristic"` for pre-overhaul behavior, or `"mixed"` for
   `shaping_weight · heuristic + terminal`.
3. **Seeds.** `HokmConfig.seed` → `seed_utils.seed_all` seeds Python,
   NumPy, and Torch. `Hokm(..., rng=random.Random(seed))` seeds deck
   shuffles and Hakem choice deterministically. `evaluate.py` uses a
   distinct derived seed per matchup so comparisons are apples-to-apples.
4. **Legal-action inference.** `QNetwork.q_values_at_indices` /
   `AveragePolicyNetwork.logits_at_indices` compute outputs only for the
   valid-card subset at each decision, avoiding the cost of masking a
   full 52-dim head at every ply.
5. **Single checkpoint format.** All four training seats share one
   `SharedNFSPLearner`; checkpoints are a `{"q_net": ..., "avg_policy_net": ...}`
   dict. `load_policy_state` also accepts raw `q_net` state dicts for
   backwards compatibility.

## Testing

```bash
pytest
```

Rule correctness, reward-mode semantics, terminal-reward sign correctness,
and full-game deterministic replay are covered.

## License

MIT — see `LICENSE`.

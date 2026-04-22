# train_backend.py
"""
Training orchestration for Hokm NFSP agents.

`TrainBackend` can run either:
  * pure self-play (default — all 4 seats share one NFSP learner), or
  * a mixed opponent curriculum (self-play + uniform-random + heuristic
    + frozen past checkpoints), configured via `HokmConfig.opponents`.

In either case, checkpoints are the **shared learner's** weights and land
in <project>/models/ so both the dev console and `app.py` can load them.
"""

from __future__ import annotations

import os
import random
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))

import matplotlib

matplotlib.use("Agg")  # non-interactive; training may run in a Flask worker thread
import matplotlib.pyplot as plt
import pandas as pd
import torch

from baselines import HeuristicAgent, RandomAgent
from config import DEFAULT_CONFIG, HokmConfig
from enhanced_player import EnhancedPlayer, SharedNFSPLearner
from hokm import Hokm
from seed_utils import seed_all


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def _make_learner_player(
    seat: int,
    shared: SharedNFSPLearner,
    cfg: HokmConfig,
) -> EnhancedPlayer:
    n = cfg.nfsp
    return EnhancedPlayer(
        f"Player {seat + 1}",
        shared_learner=shared,
        learn_every=n.learn_every,
        eta=n.eta,
        epsilon=n.epsilon_start,
        reward_mode=n.reward_mode,
        shaping_weight=n.shaping_weight,
        win_bonus=n.win_bonus,
    )


def _make_baseline(
    kind: str,
    seat: int,
    rng: random.Random,
    frozen_pool: List[str],
) -> EnhancedPlayer:
    """Construct a non-learner agent of the requested kind."""
    name = f"Player {seat + 1} ({kind})"
    if kind == "random":
        return RandomAgent(name, rng=rng)
    if kind == "heuristic":
        return HeuristicAgent(name, rng=rng)
    if kind == "frozen":
        if not frozen_pool:
            # Fall back gracefully to heuristic if the pool is empty.
            return HeuristicAgent(name, rng=rng)
        pick = rng.choice(frozen_pool)
        p = EnhancedPlayer(name)
        p.learning_enabled = False
        p.eta = 0.0
        p.epsilon = 0.0
        p.load_policy_state(pick)
        return p
    raise ValueError(f"Unknown baseline kind: {kind!r}")


def _opponent_kind_sampler(cfg: HokmConfig, rng: random.Random) -> Callable[[], str]:
    """Return a function that samples one of {self, random, heuristic, frozen}."""
    w = cfg.opponents
    weights = [
        ("self", max(0.0, float(w.self_play))),
        ("random", max(0.0, float(w.random))),
        ("heuristic", max(0.0, float(w.heuristic))),
        ("frozen", max(0.0, float(w.frozen_pool))),
    ]
    total = sum(x for _, x in weights)
    if total <= 0:
        return lambda: "self"

    norm = [(k, v / total) for k, v in weights]

    def _sample() -> str:
        r = rng.random()
        acc = 0.0
        for k, p in norm:
            acc += p
            if r <= acc:
                return k
        return norm[-1][0]

    return _sample


def _frozen_pool_paths(cfg: HokmConfig) -> List[str]:
    d = cfg.opponents.frozen_pool_dir
    if not d or not os.path.isdir(d):
        return []
    return [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith(".pth")
    ]


# ------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------

class TrainBackend:
    """
    Parameters
    ----------
    num_games : int
        Games to play (each call to `Hokm.play_game()` is one hand).
    model_save_interval : int
        Save a checkpoint every N games.
    config : HokmConfig, optional
        Central config bundle. If omitted, `DEFAULT_CONFIG` is used.
    """

    def __init__(
        self,
        num_games: int = 1000,
        model_save_interval: int = 100,
        config: Optional[HokmConfig] = None,
    ):
        self.config: HokmConfig = config or DEFAULT_CONFIG
        # Allow callers (or the dev console) to override the two knobs
        # without constructing a full config.
        self.num_games = num_games
        self.model_save_interval = model_save_interval
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        seed_all(self.config.seed)
        self._rng = random.Random(self.config.seed) if self.config.seed is not None else random.Random()

        self.shared_learner = SharedNFSPLearner.from_config(self.config)
        self._self_test_shared_learner()

        # Keep a set of 4 *learner* seat players around; these are stable
        # references used in pure self-play. In mixed-opponent mode we
        # swap them per game based on `config.opponents.trainable_seats`.
        self._learner_players = [
            _make_learner_player(i, self.shared_learner, self.config) for i in range(4)
        ]

        self.players: List[EnhancedPlayer] = list(self._learner_players)
        self.game = Hokm(
            self.players,
            minimal_logging=self.config.minimal_logging,
            rng=self._rng if self.config.seed is not None else None,
        )
        self.summary_data: List[pd.DataFrame] = []
        self.metrics = {
            "game_number": [],
            "team1_win_rate": [],
            "team2_win_rate": [],
            "player1_avg_reward": [],
            "player2_avg_reward": [],
            "player3_avg_reward": [],
            "player4_avg_reward": [],
            "player1_trick_wins": [],
            "player2_trick_wins": [],
            "player3_trick_wins": [],
            "player4_trick_wins": [],
        }
        self._time_play_games = 0.0
        self._time_post_game = 0.0
        self._time_checkpoints = 0.0

    # -------------------------------------------------------------
    # Startup self-test
    # -------------------------------------------------------------

    def _self_test_shared_learner(self) -> None:
        """Fail loudly *before* the training loop if the network input dim
        disagrees with what `EnhancedPlayer.get_state()` produces.

        Motivating bug: `NetworkConfig.state_dim` was left at 114 after
        `STATE_DIM` was bumped to 194, which meant every forward pass
        through the policy net threw `RuntimeError` — but the exception
        was silently swallowed inside `Hokm.play_game`, so 100k games
        completed with all-zero metrics and a checkpoint that had never
        received a single gradient. One forward pass here catches that
        class of bug in ~1 ms.
        """
        from game_constants import STATE_DIM as _STATE_DIM

        try:
            probe = torch.zeros(1, _STATE_DIM)
            self.shared_learner.q_net(probe)
            self.shared_learner.avg_policy_net(probe)
        except Exception as e:
            net_state_dim = getattr(self.shared_learner, "state_dim", "?")
            raise RuntimeError(
                "Shared learner self-test failed: the network expects "
                f"input_dim={net_state_dim} but EnhancedPlayer.get_state() "
                f"produces {_STATE_DIM}-dim vectors. Check "
                "`config.NetworkConfig.state_dim` vs "
                "`game_constants.STATE_DIM`. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    # -------------------------------------------------------------
    # Opponent rotation
    # -------------------------------------------------------------

    def _maybe_swap_opponents(self) -> None:
        """
        Before each game, rebuild `self.players` from the opponent mix.
        Seats in `config.opponents.trainable_seats` get the shared learner;
        other seats get a baseline sampled from the opponent pool.
        """
        mix = self.config.opponents
        trainable = set(int(i) for i in (mix.trainable_seats or [0, 1, 2, 3]))
        # Short-circuit: if the mix is pure self-play and all seats are trainable,
        # nothing to do (saves baseline allocations on the hot path).
        if mix.random == 0 and mix.heuristic == 0 and mix.frozen_pool == 0 and trainable == {0, 1, 2, 3}:
            return

        sampler = _opponent_kind_sampler(self.config, self._rng)
        pool = _frozen_pool_paths(self.config)

        new_players: List[EnhancedPlayer] = []
        for seat in range(4):
            if seat in trainable:
                new_players.append(self._learner_players[seat])
            else:
                kind = sampler()
                if kind == "self":
                    new_players.append(self._learner_players[seat])
                else:
                    new_players.append(_make_baseline(kind, seat, self._rng, pool))

        self.players = new_players
        # Rewire Hokm to the new list (teams / tricks_won / team_strategy pointers).
        self.game.players = new_players
        self.game.team1 = [new_players[0], new_players[2]]
        self.game.team2 = [new_players[1], new_players[3]]
        self.game.tricks_won = {p: 0 for p in new_players}
        for p in new_players:
            p.team = self.game.team1 if p in self.game.team1 else self.game.team2
            p.tricks_won = self.game.tricks_won
            p.team_strategy = self.game.team_strategy
        self.game.hakem = None  # force re-selection by start_game

    # -------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------

    def train(
        self,
        stop_event=None,
        on_progress=None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """Run self-play training for `num_games`; call callbacks as we go."""
        models_dir = os.path.join(_ROOT, "models")
        summaries_dir = os.path.join(_ROOT, "summaries")
        plots_dir = os.path.join(_ROOT, "plots")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(os.path.join(_ROOT, "game_logs"), exist_ok=True)
        self._models_dir = models_dir
        self._summaries_dir = summaries_dir
        self._plots_dir = plots_dir
        if log_fn:
            log_fn(f"Config seed: {self.config.seed}")
            log_fn(f"Reward mode: {self.config.nfsp.reward_mode}")
            mix = self.config.opponents
            if mix.random or mix.heuristic or mix.frozen_pool:
                log_fn(
                    "Opponent mix: self=%.2f random=%.2f heuristic=%.2f frozen=%.2f"
                    % (mix.self_play, mix.random, mix.heuristic, mix.frozen_pool)
                )
            log_fn(f"Checkpoints directory (absolute): {models_dir}")

        successful_games = 0

        for game_idx in range(self.num_games):
            if stop_event is not None and stop_event.is_set():
                if log_fn:
                    log_fn(
                        f"Stop requested before game {game_idx + 1}; "
                        f"ending after {game_idx} game(s) finished ({self.num_games} planned)."
                    )
                break
            try:
                self._maybe_swap_opponents()
                self.game.game_log = pd.DataFrame()

                t0 = time.perf_counter()
                self.game.play_game(save_excel_log=False)
                self._time_play_games += time.perf_counter() - t0

                if (
                    (game_idx + 1) % self.model_save_interval == 0
                    or game_idx == self.num_games - 1
                ):
                    tc0 = time.perf_counter()
                    ck_name = (
                        f"nfsp_shared_{self.session_id}_game_{game_idx + 1}.pth"
                    )
                    ck_path = os.path.join(models_dir, ck_name)
                    try:
                        torch.save(
                            self.shared_learner.export_state_dict(),
                            ck_path,
                        )
                        if log_fn and (
                            (game_idx + 1) % max(self.model_save_interval * 5, 500)
                            == 0
                            or game_idx == self.num_games - 1
                        ):
                            log_fn(f"Saved checkpoint: {ck_path}")
                    except Exception:
                        traceback.print_exc()
                        if log_fn:
                            log_fn(f"ERROR saving checkpoint to {ck_path}")
                    self._time_checkpoints += time.perf_counter() - tc0

                t1 = time.perf_counter()
                summary = self.game._create_summary_statistics()
                if not summary.empty:
                    self.summary_data.append(summary)
                    successful_games += 1

                    self.metrics["game_number"].append(game_idx + 1)
                    self.metrics["team1_win_rate"].append(
                        summary["Team 1 Win Rate"].iloc[0]
                    )
                    self.metrics["team2_win_rate"].append(
                        summary["Team 2 Win Rate"].iloc[0]
                    )
                    for i in range(1, 5):
                        self.metrics[f"player{i}_avg_reward"].append(
                            summary[f"Player {i} Avg Reward"].iloc[0]
                        )
                        self.metrics[f"player{i}_trick_wins"].append(
                            summary[f"Player {i} Trick Wins"].iloc[0]
                        )
                self._time_post_game += time.perf_counter() - t1

                if on_progress is not None:
                    snap = {k: list(v) for k, v in self.metrics.items()}
                    on_progress(game_idx + 1, snap)

            except Exception:
                traceback.print_exc()
                continue

        aborted = int(getattr(self.game, "aborted_games", 0) or 0)
        last_error = getattr(self.game, "last_error", None)
        if log_fn:
            log_fn(
                f"Training loop done: {successful_games} successful game(s) with metrics "
                f"out of {self.num_games} planned."
            )
            if aborted:
                err_name = type(last_error).__name__ if last_error else "unknown"
                err_msg = str(last_error) if last_error else ""
                log_fn(
                    f"WARNING: {aborted}/{self.num_games} game(s) aborted in play_round. "
                    f"Last error: {err_name}: {err_msg}"
                )
            log_fn(
                f"Profile (s): play_games={self._time_play_games:.2f}, "
                f"post_game_metrics={self._time_post_game:.2f}, "
                f"checkpoints={self._time_checkpoints:.2f}"
            )
            log_fn(
                f"Latest weights pattern: nfsp_shared_{self.session_id}_game_<N>.pth under {self._models_dir}"
            )
        if successful_games == 0:
            if log_fn:
                log_fn(
                    "No successful games — skipping summary CSV and plots "
                    "(check console for per-game errors)."
                )
            return

        self.save_summaries()
        self.generate_visualizations()

    # -------------------------------------------------------------
    # Artifact emission
    # -------------------------------------------------------------

    def save_summaries(self):
        if not self.summary_data:
            return
        summary_df = pd.concat(self.summary_data, ignore_index=True)
        summary_path = os.path.join(
            self._summaries_dir, f"summary_{self.session_id}.csv"
        )
        summary_df.to_csv(summary_path, index=False)

    def generate_visualizations(self):
        if not self.metrics["game_number"]:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.metrics["game_number"],
            self.metrics["team1_win_rate"],
            label="Team 1 Win Rate",
        )
        plt.plot(
            self.metrics["game_number"],
            self.metrics["team2_win_rate"],
            label="Team 2 Win Rate",
        )
        plt.xlabel("Game Number")
        plt.ylabel("Win Rate")
        plt.title("Team Win Rates Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self._plots_dir, f"team_win_rates_{self.session_id}.png")
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        for i in range(1, 5):
            plt.plot(
                self.metrics["game_number"],
                self.metrics[f"player{i}_avg_reward"],
                label=f"Player {i} Avg Reward",
            )
        plt.xlabel("Game Number")
        plt.ylabel("Average Reward")
        plt.title("Player Average Rewards Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(
                self._plots_dir, f"player_avg_rewards_{self.session_id}.png"
            )
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        for i in range(1, 5):
            plt.plot(
                self.metrics["game_number"],
                self.metrics[f"player{i}_trick_wins"],
                label=f"Player {i} Trick Wins",
            )
        plt.xlabel("Game Number")
        plt.ylabel("Trick Wins")
        plt.title("Player Trick Wins Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self._plots_dir, f"player_trick_wins_{self.session_id}.png")
        )
        plt.close()


if __name__ == "__main__":
    trainer = TrainBackend(num_games=1000, model_save_interval=100)
    trainer.train()

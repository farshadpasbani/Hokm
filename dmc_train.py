"""
Deep Monte-Carlo trainer for Hokm with league self-play and parallel actors.

Design (see dmc.py for the model):
  * Seats 0 and 2 are the learner; seats 1 and 3 are drawn per game from a
    league: the current policy, frozen past snapshots (PFSP: prefer
    opponents we lose to), the rule-based heuristic, and a little random.
  * Episodes are generated with the engine's step API (apply_play /
    resolve_trick_if_complete) — no pandas, no learning hooks in the hot
    loop.
  * Every learner decision records: the 194-d static observation, the
    exact play-history sequence, the chosen action, the legal candidates,
    a full-information snapshot (all four hands, seat-relative) for the
    centralized critic, and auxiliary labels (which opponent holds each
    unseen card).
  * At hand end each decision gets the undiscounted return
    G = ±1 (win/loss) + 0.1 · (team tricks − opp tricks).
  * Losses:  huber(Q(s,a), G)
           + 0.5 · MSE(V(s), V_central(s_full).detach())   (distillation)
           + 0.3 · BCE(aux, opponent-hand labels)          (belief shaping)
           and for the critic: MSE(V_central(s_full), G).

Run:
    python dmc_train.py --games 30000 --actors 3
    python dmc_train.py --games 500 --actors 0   # single-process (tests)
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from baselines import HeuristicAgent, RandomAgent
from dmc import CentralCritic, DMCNet, MAX_HISTORY, history_features
from enhanced_player import EnhancedPlayer
from game_constants import Card, card_to_index, suits
from hokm import Hokm

_ROOT = os.path.dirname(os.path.abspath(__file__))

WIN_BONUS = 1.0
TRICK_DIFF_W = 0.1
LEARNER_SEATS = (0, 2)


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    static: torch.Tensor          # [194]
    hist_cards: List[int]
    hist_seats: List[int]
    action: int                   # 0..51
    full_info: torch.Tensor       # [CentralCritic.IN_DIM]
    aux_target: torch.Tensor      # [3*52] float 0/1
    aux_mask: torch.Tensor        # [3*52] float; 1 where label is informative
    ret: float = 0.0              # filled at hand end


class _Seat(EnhancedPlayer):
    """Learner seat shell: engine bookkeeping only; decisions are made by
    the trainer loop, not by this object."""

    def __init__(self, name):
        super().__init__(name)
        self.learning_enabled = False
        self.eta = 0.0
        self.epsilon = 0.0

    def store_experience(self, *_a, **_k):
        return None

    def optimize_model(self, *_a, **_k):
        return None


def _full_info_vec(game: Hokm, my_seat: int) -> torch.Tensor:
    """All four hands one-hot, seat-relative (me, LHO, partner, RHO), plus
    trump one-hot and team/opp trick counts (normalised /7)."""
    x = torch.zeros(CentralCritic.IN_DIM)
    for rel in range(4):
        seat = (my_seat + rel) % 4
        base = rel * 52
        for c in game.players[seat].hand:
            x[base + card_to_index(c)] = 1.0
    off = 4 * 52
    if game.trump_suit:
        x[off + suits.index(game.trump_suit)] = 1.0
    my_team = 1 if my_seat % 2 == 0 else 2
    x[off + 4] = game.scores[my_team] / 7.0
    x[off + 5] = game.scores[2 if my_team == 1 else 1] / 7.0
    return x


def _aux_targets(game: Hokm, my_seat: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Labels: for [LHO, partner, RHO] × 52, is that card in that hand?
    Mask covers only cards actually hidden from `my_seat` (not mine, not
    already played)."""
    target = torch.zeros(3 * 52)
    mask = torch.ones(52, dtype=torch.bool)
    me = game.players[my_seat]
    for c in me.hand:
        mask[card_to_index(c)] = False
    for c in game.cards_played_this_hand:
        mask[card_to_index(c)] = False
    for j, rel in enumerate((1, 2, 3)):  # LHO, partner, RHO
        seat = (my_seat + rel) % 4
        base = j * 52
        for c in game.players[seat].hand:
            target[base + card_to_index(c)] = 1.0
    mask3 = mask.float().repeat(3)
    return target, mask3


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------

class League:
    """Opponent sampling: current policy / frozen PFSP pool / heuristic /
    random. PFSP weight for a frozen member ∝ (1 − winrate_vs_it) + floor,
    so opponents that beat us are revisited more often."""

    def __init__(self, pool_dir: str, rng: random.Random,
                 p_current=0.35, p_frozen=0.35, p_heuristic=0.25, p_random=0.05):
        self.pool_dir = pool_dir
        self.rng = rng
        self.p = (p_current, p_frozen, p_heuristic, p_random)
        self.results: Dict[str, List[int]] = {}   # path -> [wins, games]
        os.makedirs(pool_dir, exist_ok=True)

    def _pool(self) -> List[str]:
        return sorted(
            os.path.join(self.pool_dir, f)
            for f in os.listdir(self.pool_dir)
            if f.endswith(".pt")
        )

    def snapshot(self, net: DMCNet, tag: str, max_pool: int = 10) -> None:
        path = os.path.join(self.pool_dir, f"league_{tag}.pt")
        torch.save({"dmc_net": net.state_dict()}, path)
        pool = self._pool()
        while len(pool) > max_pool:
            victim = pool.pop(0)
            self.results.pop(victim, None)
            os.remove(victim)

    def record_result(self, opponent_key: Optional[str], won: bool) -> None:
        if opponent_key is None:
            return
        w, g = self.results.get(opponent_key, [0, 0])
        self.results[opponent_key] = [w + (1 if won else 0), g + 1]

    def sample(self, current_net: DMCNet) -> Tuple[str, Optional[str], list]:
        """Returns (kind, pool_key, [seat1_player, seat3_player])."""
        pc, pf, ph, pr = self.p
        pool = self._pool()
        if not pool:
            pc, pf, ph, pr = pc + pf, 0.0, ph, pr
        r = self.rng.random() * (pc + pf + ph + pr)
        if r < pc:
            from dmc import DMCPlayer
            mk = lambda nm: DMCPlayer(nm, net=current_net, epsilon=0.02,
                                      rng=random.Random(self.rng.getrandbits(32)))
            return "current", None, [mk("Opp E (cur)"), mk("Opp W (cur)")]
        r -= pc
        if r < pf:
            weights = []
            for path in pool:
                w, g = self.results.get(path, [0, 0])
                wr = (w / g) if g >= 5 else 0.5
                weights.append((1.0 - wr) + 0.15)
            pick = self.rng.choices(pool, weights=weights, k=1)[0]
            from dmc import DMCPlayer
            net = DMCNet()
            blob = torch.load(pick, map_location="cpu", weights_only=True)
            net.load_state_dict(blob["dmc_net"])
            net.eval()
            mk = lambda nm: DMCPlayer(nm, net=net,
                                      rng=random.Random(self.rng.getrandbits(32)))
            return "frozen", pick, [mk("Opp E (frz)"), mk("Opp W (frz)")]
        r -= pf
        if r < ph:
            return "heuristic", None, [
                HeuristicAgent("Opp E (heur)", rng=random.Random(self.rng.getrandbits(32))),
                HeuristicAgent("Opp W (heur)", rng=random.Random(self.rng.getrandbits(32))),
            ]
        return "random", None, [
            RandomAgent("Opp E (rnd)", rng=random.Random(self.rng.getrandbits(32))),
            RandomAgent("Opp W (rnd)", rng=random.Random(self.rng.getrandbits(32))),
        ]


# ---------------------------------------------------------------------------
# Episode generation (step API — no pandas, no engine learning hooks)
# ---------------------------------------------------------------------------

def play_episode(
    net: DMCNet,
    opponents: list,
    rng: random.Random,
    epsilon: float,
) -> Tuple[List[Decision], bool, int]:
    """Play one hand. Returns (learner decisions, team1_won, trick_diff)."""
    south, north = _Seat("L South"), _Seat("L North")
    players = [south, opponents[0], north, opponents[1]]
    g = Hokm(players, minimal_logging=True,
             rng=random.Random(rng.getrandbits(32)))
    g.start_game()
    g.choose_trump_suit()
    g.round_count = 0

    decisions: List[Decision] = []
    net.eval()
    for _ in range(300):
        if g.scores[1] >= 7 or g.scores[2] >= 7:
            break
        if all(len(p.hand) == 0 for p in g.players):
            break
        if len(g.current_trick) == 4:
            g.resolve_trick_if_complete()
            continue
        nxt = g.get_next_to_play()
        seat = g.players.index(nxt)
        legal = g.legal_cards_for_player(nxt)
        if seat in LEARNER_SEATS:
            cand = [card_to_index(c) for c in legal]
            static = nxt.get_state().cpu()
            hist_c, hist_s = history_features(g, seat)
            if len(legal) == 1:
                pick = 0
            elif rng.random() < epsilon:
                pick = rng.randrange(len(legal))
            else:
                with torch.no_grad():
                    q = net.q_values(static, hist_c, hist_s, cand)
                pick = int(q.argmax().item())
            card = legal[pick]
            aux_t, aux_m = _aux_targets(g, seat)
            decisions.append(Decision(
                static=static,
                hist_cards=list(hist_c),
                hist_seats=list(hist_s),
                action=card_to_index(card),
                full_info=_full_info_vec(g, seat),
                aux_target=aux_t,
                aux_mask=aux_m,
            ))
        else:
            card, _ = nxt.play_card(g.lead_suit)
        err = g.apply_play(nxt, card)
        if err:
            raise RuntimeError(f"illegal play by {nxt.name}: {err}")

    t1, t2 = g.scores[1], g.scores[2]
    won = t1 > t2
    ret = (WIN_BONUS if won else -WIN_BONUS) + TRICK_DIFF_W * (t1 - t2)
    for d in decisions:
        d.ret = ret
    return decisions, won, t1 - t2


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

def _collate(batch: List[Decision]):
    static = torch.stack([d.static for d in batch])
    lens = torch.tensor([len(d.hist_cards) for d in batch], dtype=torch.long)
    tmax = max(1, int(lens.max().item()))
    hc = torch.zeros(len(batch), tmax, dtype=torch.long)
    hs = torch.zeros(len(batch), tmax, dtype=torch.long)
    for i, d in enumerate(batch):
        if d.hist_cards:
            hc[i, : len(d.hist_cards)] = torch.tensor(d.hist_cards)
            hs[i, : len(d.hist_seats)] = torch.tensor(d.hist_seats)
    actions = torch.tensor([d.action for d in batch], dtype=torch.long)
    rets = torch.tensor([d.ret for d in batch], dtype=torch.float32)
    full = torch.stack([d.full_info for d in batch])
    aux_t = torch.stack([d.aux_target for d in batch])
    aux_m = torch.stack([d.aux_mask for d in batch])
    return static, hc, hs, lens, actions, rets, full, aux_t, aux_m


class DMCTrainer:
    def __init__(
        self,
        *,
        seed: int = 0,
        lr: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 200_000,
        epsilon: float = 0.10,
        distill_w: float = 0.5,
        aux_w: float = 0.3,
        snapshot_every: int = 3000,
        pool_dir: Optional[str] = None,
    ):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.net = DMCNet()
        self.critic = CentralCritic()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.distill_w = distill_w
        self.aux_w = aux_w
        self.snapshot_every = snapshot_every
        self.league = League(
            pool_dir or os.path.join(_ROOT, "models", "dmc_league"), self.rng
        )
        self.games = 0
        self.q_losses: deque = deque(maxlen=200)
        self.aux_losses: deque = deque(maxlen=200)
        self.recent_vs = {"heuristic": deque(maxlen=300),
                          "current": deque(maxlen=300),
                          "frozen": deque(maxlen=300),
                          "random": deque(maxlen=300)}
        # Best held-out win rate seen by quick_eval(); drives dmc_best.pt.
        self.best_eval: float = -1.0

    def generate_game(self) -> int:
        kind, key, opps = self.league.sample(self.net)
        decisions, won, _diff = play_episode(
            self.net, opps, self.rng, self.epsilon
        )
        self.buffer.extend(decisions)
        self.league.record_result(key, won)
        self.recent_vs[kind].append(1 if won else 0)
        self.games += 1
        if self.games % self.snapshot_every == 0:
            self.league.snapshot(self.net, f"g{self.games}")
        return len(decisions)

    def train_steps(self, n: int) -> None:
        if len(self.buffer) < self.batch_size:
            return
        self.net.train()
        for _ in range(n):
            batch = self.rng.sample(range(len(self.buffer)), self.batch_size)
            recs = [self.buffer[i] for i in batch]
            static, hc, hs, lens, actions, rets, full, aux_t, aux_m = _collate(recs)

            v_c = self.critic(full)
            critic_loss = F.mse_loss(v_c, rets)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            q, v, aux = self.net(static, hc, hs, lens, actions)
            q_loss = F.smooth_l1_loss(q, rets)
            distill = F.mse_loss(v, v_c.detach())
            bce = F.binary_cross_entropy_with_logits(
                aux, aux_t, reduction="none"
            )
            denom = aux_m.sum().clamp(min=1.0)
            aux_loss = (bce * aux_m).sum() / denom
            loss = q_loss + self.distill_w * distill + self.aux_w * aux_loss
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.q_losses.append(float(q_loss.item()))
            self.aux_losses.append(float(aux_loss.item()))
        self.net.eval()

    def quick_eval(
        self,
        games: int = 60,
        opponent: str = "heuristic",
        *,
        seed: int = 20240917,
    ) -> float:
        """Greedy held-out win rate: the current net at seats 0+2 vs `opponent`.

        Distinct from `recent_vs` in `stats()`, which is a rolling average over
        *training* games — those carry exploration noise and a shifting league
        mixture, so they can't be compared across runs. This is ε=0 against a
        fixed family with a fixed seed, so successive calls are comparable.

        Deliberately uses a local `random.Random`: drawing from `self.rng`
        would make the training trajectory depend on how often we evaluated.
        """
        from dmc import DMCPlayer

        from agents import make_team

        eval_rng = random.Random(seed)
        me = [
            DMCPlayer(
                f"DMC {tag}",
                net=self.net,
                epsilon=0.0,
                rng=random.Random(eval_rng.getrandbits(32)),
            )
            for tag in ("S", "N")
        ]
        opps = make_team(
            opponent,
            seed=eval_rng.getrandbits(32),
            names=("Eval E", "Eval W"),
        )
        g = Hokm(
            [me[0], opps[0], me[1], opps[1]],
            minimal_logging=True,
            rng=random.Random(eval_rng.getrandbits(32)),
        )

        was_training = self.net.training
        self.net.eval()
        wins = 0
        try:
            for _ in range(games):
                g.play_game(save_excel_log=False)
                if g.scores[1] > g.scores[2]:
                    wins += 1
        finally:
            if was_training:
                self.net.train()
        return wins / games if games else 0.0

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"dmc_net": self.net.state_dict(),
             "critic": self.critic.state_dict(),
             "games": self.games},
            path,
        )

    def stats(self) -> str:
        import statistics
        parts = [f"games={self.games}", f"buf={len(self.buffer)}"]
        if self.q_losses:
            parts.append(f"q_loss={statistics.mean(self.q_losses):.4f}")
        if self.aux_losses:
            parts.append(f"aux={statistics.mean(self.aux_losses):.4f}")
        for k, dq in self.recent_vs.items():
            if dq:
                parts.append(f"wr_{k}={sum(dq)/len(dq):.2f}({len(dq)})")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Single-process driver + multiprocess actors
# ---------------------------------------------------------------------------

def best_checkpoint_path(out: Optional[str]) -> Optional[str]:
    """`dmc_best.pt` next to the regular `--out` checkpoint."""
    if not out:
        return None
    return os.path.join(os.path.dirname(out) or ".", "dmc_best.pt")


def periodic_eval(tr: DMCTrainer, out: Optional[str], games: int,
                  opponent: str) -> float:
    """Run quick_eval, log it, and checkpoint the net if it's a new best.

    `dmc_best.pt` is kept separately from `--out` because `--out` is the
    *latest* net: DMC win rate is noisy game-to-game, so the last checkpoint of
    a run is not reliably its strongest one.
    """
    wr = tr.quick_eval(games=games, opponent=opponent)
    print(f"[dmc-eval] games={tr.games} wr_{opponent}={wr:.3f}", flush=True)
    if wr > tr.best_eval:
        tr.best_eval = wr
        best = best_checkpoint_path(out)
        if best:
            tr.save(best)
            print(f"[dmc-eval] new best {wr:.3f} -> {best}", flush=True)
    return wr


def train_single(games: int, *, seed=0, out=None, log_every=500,
                 steps_per_game=2, trainer: Optional[DMCTrainer] = None,
                 eval_every=2000, eval_games=60, eval_opponent="heuristic"):
    tr = trainer or DMCTrainer(seed=seed)
    t0 = time.time()
    for i in range(games):
        tr.generate_game()
        tr.train_steps(steps_per_game)
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[dmc] {tr.stats()} | {rate:.1f} games/s", flush=True)
        if out and (i + 1) % 2000 == 0:
            tr.save(out)
        if eval_every and (i + 1) % eval_every == 0:
            periodic_eval(tr, out, eval_games, eval_opponent)
    if out:
        tr.save(out)
    return tr


def _actor_proc(actor_id, weights_path, version, episodes_q, seed, epsilon, pool_dir):
    """Actor process: regenerate episodes with the freshest weights."""
    torch.set_num_threads(1)
    rng = random.Random(seed)
    net = DMCNet()
    league = League(pool_dir, rng)
    my_version = -1
    while True:
        if version.value != my_version and os.path.isfile(weights_path):
            try:
                blob = torch.load(weights_path, map_location="cpu", weights_only=True)
                net.load_state_dict(blob["dmc_net"])
                net.eval()
                my_version = version.value
            except Exception:
                pass  # partially-written file; retry next round
        kind, key, opps = league.sample(net)
        try:
            decisions, won, diff = play_episode(net, opps, rng, epsilon)
        except Exception as e:  # never kill the actor on one bad game
            episodes_q.put(("error", actor_id, repr(e)))
            continue
        # Plain lists only: tensor payloads would use torch's shared-memory
        # fd-passing, which sandboxed environments commonly forbid.
        payload = [
            (d.static.tolist(), d.hist_cards, d.hist_seats, d.action,
             d.full_info.tolist(), d.aux_target.tolist(), d.aux_mask.tolist(),
             d.ret)
            for d in decisions
        ]
        episodes_q.put(("episode", kind, key, won, payload))


def train_parallel(games: int, *, actors=3, seed=0, out=None, log_every=1000,
                   steps_per_episode=2, eval_every=2000, eval_games=60,
                   eval_opponent="heuristic"):
    import torch.multiprocessing as mp

    torch.set_num_threads(2)
    tr = DMCTrainer(seed=seed)
    ctx = mp.get_context("spawn")
    episodes_q = ctx.Queue(maxsize=64)
    version = ctx.Value("i", 0)
    weights_dir = os.path.join(_ROOT, "models")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "dmc_live_weights.pt")
    tr.save(weights_path)
    version.value = 1

    procs = []
    for a in range(actors):
        eps = 0.10 if a % 3 else 0.16   # one wider-exploration actor
        p = ctx.Process(
            target=_actor_proc,
            args=(a, weights_path, version, episodes_q, seed * 1000 + a, eps,
                  tr.league.pool_dir),
            daemon=True,
        )
        p.start()
        procs.append(p)

    t0 = time.time()
    errors = 0
    try:
        while tr.games < games:
            msg = episodes_q.get()
            if msg[0] == "error":
                errors += 1
                if errors <= 5:
                    print(f"[dmc] actor error: {msg[2]}", flush=True)
                continue
            _, kind, key, won, payload = msg
            for (st, hcds, hsts, act, full, auxt, auxm, ret) in payload:
                self_d = Decision(
                    static=torch.tensor(st, dtype=torch.float32),
                    hist_cards=hcds, hist_seats=hsts, action=act,
                    full_info=torch.tensor(full, dtype=torch.float32),
                    aux_target=torch.tensor(auxt, dtype=torch.float32),
                    aux_mask=torch.tensor(auxm, dtype=torch.float32),
                    ret=ret,
                )
                tr.buffer.append(self_d)
            tr.league.record_result(key, won)
            tr.recent_vs[kind].append(1 if won else 0)
            tr.games += 1
            tr.train_steps(steps_per_episode)
            if tr.games % 500 == 0:
                tr.save(weights_path)
                version.value += 1
            if tr.games % tr.snapshot_every == 0:
                tr.league.snapshot(tr.net, f"g{tr.games}")
            if log_every and tr.games % log_every == 0:
                rate = tr.games / (time.time() - t0)
                print(f"[dmc] {tr.stats()} | {rate:.1f} games/s", flush=True)
            if out and tr.games % 2000 == 0:
                tr.save(out)
            if eval_every and tr.games % eval_every == 0:
                periodic_eval(tr, out, eval_games, eval_opponent)
    finally:
        for p in procs:
            p.terminate()
    if out:
        tr.save(out)
    return tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30000)
    ap.add_argument("--actors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(_ROOT, "models", "dmc_latest.pt"))
    ap.add_argument("--eval-every", type=int, default=2000,
                    help="Greedy held-out eval every N games (0 disables).")
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--eval-opponent", default="heuristic",
                    help="Any agents.py spec, e.g. heuristic, random, pimc:8.")
    args = ap.parse_args()
    common = dict(seed=args.seed, out=args.out, log_every=args.log_every,
                  eval_every=args.eval_every, eval_games=args.eval_games,
                  eval_opponent=args.eval_opponent)
    if args.actors <= 0:
        train_single(args.games, **common)
    else:
        train_parallel(args.games, actors=args.actors, **common)

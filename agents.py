"""
Agent factory: build a two-seat Hokm team from a short spec string.

One place that knows how to turn `"dmc:models/dmc_latest.pt"` into a pair of
ready-to-seat player objects, so every caller (evaluate.py, dmc_train.py's
eval-in-the-loop, the dev console, ad-hoc scripts) agrees on what a family
name means and how it is configured for *evaluation* — greedy, no learning,
no exploration.

Spec grammar
------------
    random                 uniform-random legal card         (baselines.RandomAgent)
    heuristic              rule-based baseline               (baselines.HeuristicAgent)
    untrained              fresh EnhancedPlayer, random init (control group)
    nfsp:<path.pth>        NFSP checkpoint, greedy           (EnhancedPlayer)
    dmc:<path.pt>          Deep Monte-Carlo checkpoint       (dmc.DMCPlayer)
    pimc                   determinized search, 24 samples   (pimc.PIMCPlayer)
    pimc:<n>               ...with n determinizations per decision

Determinism
-----------
`make_team` derives two *independent* `random.Random` streams from `seed`
(one per seat) so a team's two seats never move in lockstep, while the whole
team stays reproducible from a single integer. Net-based seats (nfsp/dmc at
ε=0) are deterministic and ignore their RNG; it is still constructed so that
swapping families does not shift any other stream.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence, Tuple, Union

from baselines import HeuristicAgent, RandomAgent
from enhanced_player import EnhancedPlayer
from game_constants import ACTION_DIM, STATE_DIM

# Families that take no argument, and those that require/accept one.
_PLAIN_FAMILIES = ("random", "heuristic", "untrained")
_PATH_FAMILIES = ("nfsp", "dmc")
_FAMILIES = _PLAIN_FAMILIES + _PATH_FAMILIES + ("pimc",)

# Baseline families usable as an opponent without any checkpoint on disk.
BASELINE_SPECS = ("random", "heuristic", "untrained")

_EXT = {"nfsp": ".pth", "dmc": ".pt"}

SPEC_HELP = (
    "Valid specs: 'random', 'heuristic', 'untrained', 'nfsp:<path.pth>', "
    "'dmc:<path.pt>', 'pimc', 'pimc:<determinizations>'."
)


class SpecError(ValueError):
    """Raised for malformed agent specs (a ValueError, so callers can just
    catch ValueError)."""


def parse_spec(spec: str) -> Tuple[str, Optional[Union[str, int]]]:
    """
    Split a spec into `(family, argument)` and validate it.

    Returns the argument as a path string for nfsp/dmc, an int for
    `pimc:<n>`, and None where the family takes no argument (including a
    bare `pimc`, which means "use the PIMCPlayer default").

    Raises `SpecError` (a ValueError) with an actionable message otherwise.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise SpecError(f"Empty agent spec. {SPEC_HELP}")

    raw = spec.strip()
    family, sep, arg = raw.partition(":")
    family = family.strip().lower()
    arg = arg.strip()

    if family not in _FAMILIES:
        raise SpecError(
            f"Unknown agent family {family!r} in spec {spec!r}. {SPEC_HELP}"
        )

    if family in _PLAIN_FAMILIES:
        if sep or arg:
            raise SpecError(
                f"Agent family {family!r} takes no argument, got {spec!r}. "
                f"{SPEC_HELP}"
            )
        return family, None

    if family in _PATH_FAMILIES:
        if not arg:
            raise SpecError(
                f"Spec {spec!r} needs a checkpoint path, "
                f"e.g. '{family}:models/checkpoint{_EXT[family]}'. {SPEC_HELP}"
            )
        return family, arg

    # pimc / pimc:<n>
    if not sep:
        return family, None
    if not arg:
        raise SpecError(
            f"Spec {spec!r} has an empty determinization count; use 'pimc' for "
            f"the default or e.g. 'pimc:32'. {SPEC_HELP}"
        )
    try:
        n = int(arg)
    except ValueError:
        raise SpecError(
            f"PIMC determinizations must be an integer, got {arg!r} in "
            f"{spec!r}. {SPEC_HELP}"
        ) from None
    if n <= 0:
        raise SpecError(
            f"PIMC determinizations must be >= 1, got {n} in {spec!r}."
        )
    return family, n


def spec_checkpoint(spec: str) -> Optional[str]:
    """The on-disk checkpoint a spec depends on, or None if it needs no file.
    Lets callers fail fast with a clear message before playing any games."""
    family, arg = parse_spec(spec)
    return arg if family in _PATH_FAMILIES else None


def _nfsp_seat(name: str, checkpoint: Optional[str]) -> EnhancedPlayer:
    """Greedy NFSP seat: exploration and learning fully disabled.

    `checkpoint=None` yields the 'untrained' control — identical wiring, just
    random-init weights — so an untrained-vs-trained gap is attributable to
    the weights alone.
    """
    p = EnhancedPlayer(name, STATE_DIM, ACTION_DIM, epsilon=0.0, eta=0.0)
    p.learning_enabled = False
    if checkpoint:
        p.load_policy_state(checkpoint)
    return p


def make_team(spec: str, *, seed: int, names: Tuple[str, str]) -> List[Any]:
    """
    Build the two seat objects for one team from `spec`.

    Parameters
    ----------
    spec  : see the module docstring for the grammar.
    seed  : base seed; each seat gets its own derived `random.Random`.
    names : (first_seat_name, second_seat_name) — used in engine logs.

    Returns a list of exactly two players, ready to drop into
    `Hokm([...])`. Raises `SpecError` (a ValueError) on a bad spec and
    `FileNotFoundError` when a checkpoint path does not exist.
    """
    if len(names) != 2:
        raise SpecError(f"make_team needs exactly two seat names, got {names!r}")

    family, arg = parse_spec(spec)
    rngs = (random.Random(seed + 1), random.Random(seed + 2))

    if family == "random":
        return [RandomAgent(names[i], rng=rngs[i]) for i in range(2)]

    if family == "heuristic":
        return [HeuristicAgent(names[i], rng=rngs[i]) for i in range(2)]

    if family == "untrained":
        return [_nfsp_seat(names[i], None) for i in range(2)]

    if family == "nfsp":
        return [_nfsp_seat(names[i], str(arg)) for i in range(2)]

    if family == "dmc":
        from dmc import DMCPlayer  # local: keeps `import agents` torch-light

        return [
            DMCPlayer(names[i], checkpoint=str(arg), epsilon=0.0, rng=rngs[i])
            for i in range(2)
        ]

    # family == "pimc"
    from pimc import PIMCPlayer

    kwargs = {} if arg is None else {"determinizations": int(arg)}
    return [PIMCPlayer(names[i], rng=rngs[i], **kwargs) for i in range(2)]


def describe(spec: str) -> str:
    """Short human label for tables — `nfsp:a/b/c.pth` -> `nfsp:c.pth`."""
    family, arg = parse_spec(spec)
    if family in _PATH_FAMILIES:
        import os

        return f"{family}:{os.path.basename(str(arg))}"
    return spec.strip()

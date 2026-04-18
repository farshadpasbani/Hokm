"""Reproducibility helpers: seed Python/NumPy/Torch from one call."""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_all(seed: Optional[int]) -> None:
    """
    Seed stdlib random, NumPy (if importable), and PyTorch (if importable).

    Pass `None` to explicitly opt out (no-op); passing 0 does seed with 0.
    Environment variable PYTHONHASHSEED is also set for child processes,
    but note it only affects interpreters started *after* this call.
    """
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

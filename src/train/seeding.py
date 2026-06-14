"""Determinism helpers (AGENT.md §2: same seed + data + config ⇒ same metrics)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    # Determinism is non-negotiable (AGENT.md §2): same seed + data + config must
    # reproduce the same metrics. Seed every RNG that touches the pipeline.
    os.environ["PYTHONHASHSEED"] = str(seed)  # stabilise hash-based ordering
    random.seed(seed)  # Python's stdlib RNG (e.g. shuffles, samplers)
    np.random.seed(seed)  # NumPy RNG used across feature/data code
    try:
        # torch is optional at import time so non-DL code paths still work.
        import torch

        torch.manual_seed(seed)  # CPU RNG
        torch.cuda.manual_seed_all(seed)  # all GPU RNGs, if any
        # Force deterministic kernels; warn_only so ops lacking one don't hard-crash.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def resolve_device(device: str) -> str:
    # "auto" picks CUDA when available, else CPU; any explicit value is honoured as-is.
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"  # no torch -> CPU is the only meaningful target

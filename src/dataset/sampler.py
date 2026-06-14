"""Imbalance handling (AGENT.md §5.3).

Either class-balanced sampling *or* loss weighting — configurable, not both blindly.
This module provides the sampling side: per-sample weights for a weighted sampler and
effective-number class weights for the loss side.
"""

from __future__ import annotations

import numpy as np


def class_counts(labels: np.ndarray, n_classes: int) -> np.ndarray:
    # minlength pins the output to n_classes even if a rare class is absent here, so the
    # returned array always indexes by class id (a missing class gets count 0).
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float64)
    return counts


def sample_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weight per sample for a class-balanced sampler.

    Strategy 1 of 2 (the *sampling* side): weight each sample by 1/count of its class, so a
    weighted sampler draws rare-class windows proportionally more often. NORMAL dominates
    the 3W data, so without this the model would barely ever see a fault during training.
    """
    counts = class_counts(labels, n_classes)
    inv = np.where(counts > 0, 1.0 / counts, 0.0)  # absent class -> 0 weight (never drawn)
    return inv[labels.astype(np.int64)]  # broadcast the per-class weight back onto each sample


def effective_number_weights(labels: np.ndarray, n_classes: int, beta: float = 0.999) -> np.ndarray:
    """Class-balanced loss weights via the effective-number scheme (Cui et al., 2019).

    Strategy 2 of 2 (the *loss* side): instead of raw inverse frequency, weight by the
    inverse "effective number" of samples. Intuition: near-duplicate samples add little
    new information, so a class's *useful* size saturates rather than growing linearly with
    count. beta->1 trusts large counts (closer to inverse-frequency); beta=0 gives equal
    weights. This down-weights the majority less aggressively than plain 1/count.
    """
    counts = class_counts(labels, n_classes)
    # Effective number E_n = (1 - beta^n) / (1 - beta): a geometric saturation of n.
    eff = np.where(counts > 0, (1.0 - np.power(beta, counts)) / (1.0 - beta), 1.0)
    weights = np.where(eff > 0, 1.0 / eff, 0.0)
    # Normalise so weights average to 1 across present classes — keeps the loss scale
    # (and effective learning rate) comparable to the unweighted case.
    present = counts > 0
    if present.any():
        weights = weights * present.sum() / weights[present].sum()
    return weights

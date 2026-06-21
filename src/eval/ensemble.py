"""Probability-averaging ensemble of two classifiers (AGENT.md §5.6, idea #10).

The GBT and the ST-MoE make complementary errors (engineered-feature vs learned
temporal representation), so averaging their per-window class probabilities usually
beats either alone. Both inputs must be aligned to the same test windows and use the
full 10-class column space.
"""

from __future__ import annotations

import numpy as np

from eval.metrics import ClassificationReport, classification_report
from schema import N_CLASSES


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Expand a classifier's ``(n, len(classes))`` proba to the full ``(n, 10)`` space.

    LightGBM only emits columns for classes seen in training; this scatters them into the
    canonical 0..9 layout (missing classes -> 0) so two models can be averaged.
    """
    full = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    return full


def ensemble_report(
    proba_a: np.ndarray, proba_b: np.ndarray, y_true: np.ndarray, weight_a: float = 0.5
) -> ClassificationReport:
    """Weighted-average two aligned ``(n, 10)`` probability matrices and score the result."""
    p = weight_a * proba_a + (1.0 - weight_a) * proba_b
    return classification_report(y_true, p.argmax(axis=1))

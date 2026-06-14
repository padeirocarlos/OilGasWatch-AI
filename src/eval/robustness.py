"""Robustness probes under sensor freeze / channel dropout (AGENT.md §5.6, §7).

Re-scores a model when sensors are degraded: ``dropout`` zeros a channel, ``freeze``
holds it at its window-start value. The model must not crash and the metric drop must
stay bounded (AGENT.md §7). Operates on raw sequence windows ``(n, T, 27)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from eval.metrics import classification_report
from models.network import subsystem_indices
from schema import SUBSYSTEM_CHANNELS

PredictFn = Callable[[np.ndarray], np.ndarray]


def _apply(X: np.ndarray, channels: list[int], mode: str) -> np.ndarray:
    # Build a perturbed copy of the windows with the given channels degraded.
    out = X.copy()
    if mode == "dropout":
        out[:, :, channels] = 0.0  # channel dropout: sensor reads dead/zero
    elif mode == "freeze":
        # Sensor freeze: hold each affected channel at its first time-step value across
        # the window (a stuck sensor emitting a constant), per AGENT.md §7 edge case.
        out[:, :, channels] = X[:, :1, channels]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out


@dataclass
class RobustnessResult:
    baseline_macro_f1: float
    perturbations: dict[str, float]  # name -> macro-F1 under that perturbation

    def summary(self) -> str:
        lines = [f"clean macro-F1: {self.baseline_macro_f1:.4f}"]
        for name, f1 in self.perturbations.items():
            # Δ is the macro-F1 drop vs. clean; AGENT.md §7 requires it stay bounded.
            drop = self.baseline_macro_f1 - f1
            lines.append(f"  {name:<28} {f1:.4f}  (Δ {drop:+.4f})")
        return "\n".join(lines)


def robustness_probe(
    predict: PredictFn,
    X: np.ndarray,
    y_event: np.ndarray,
    mode: str = "dropout",
) -> RobustnessResult:
    """Score the clean input, then each subsystem perturbed in turn."""
    # Clean baseline macro-F1 to measure every perturbation's drop against.
    base = classification_report(y_event, predict(X)).macro_f1
    perturbed: dict[str, float] = {}
    # Degrade one subsystem's channels at a time; the model must still run (no crash)
    # and its macro-F1 drop must stay bounded (graceful degradation, AGENT.md §7).
    for (name, _chans), idx in zip(SUBSYSTEM_CHANNELS.items(), subsystem_indices(), strict=True):
        Xp = _apply(X, idx, mode)
        perturbed[f"{mode}:{name}"] = classification_report(y_event, predict(Xp)).macro_f1
    return RobustnessResult(baseline_macro_f1=base, perturbations=perturbed)

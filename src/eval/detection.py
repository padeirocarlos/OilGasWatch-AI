"""Time-to-detection / anticipation metrics (AGENT.md §5.6).

Latency from the true onset (first transient-labelled window of an instance) to the
first window where the model correctly fires the transient flag. Negative latency =
the model anticipated the onset (fired before the labelled transient began), which is
the early-detection behaviour the project optimises for (AGENT.md §1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    detected: int
    n_instances: int
    detection_rate: float
    median_latency_windows: float
    mean_latency_windows: float
    latencies: list[float]

    def summary(self) -> str:
        return (
            f"detection rate: {self.detection_rate:.3f} "
            f"({self.detected}/{self.n_instances})\n"
            f"median latency: {self.median_latency_windows:.1f} windows | "
            f"mean: {self.mean_latency_windows:.1f} windows"
        )


def time_to_detection(
    instance_id: np.ndarray,
    order: np.ndarray,
    transient_true: np.ndarray,
    transient_pred: np.ndarray,
) -> DetectionResult:
    """Per-instance latency between true and predicted first transient firing.

    ``order`` is a monotonically increasing within-instance window index (e.g. window
    start time) used to sort each instance's windows chronologically.
    """
    latencies: list[float] = []
    detected = 0
    instances = np.unique(instance_id)
    for inst in instances:
        m = instance_id == inst  # mask selecting this instance's windows
        if not transient_true[m].any():
            continue  # no labelled onset in this instance
        idx = np.argsort(order[m])  # sort this instance's windows chronologically
        tt = transient_true[m][idx]
        tp = transient_pred[m][idx]
        onset = int(np.argmax(tt))  # first true transient window (the ground-truth onset)
        fired = np.flatnonzero(tp)  # window indices where the model raised the flag
        if len(fired) == 0:
            continue  # never detected -> excluded from latency, counts against rate
        detected += 1
        # Latency in windows: pred_first - true_onset. Negative => model fired BEFORE the
        # labelled onset, i.e. it anticipated the event (the early-detection win, §1).
        latencies.append(float(fired[0] - onset))

    # Denominator = instances that actually have an onset (detectable events only).
    n = int(sum(1 for inst in instances if transient_true[instance_id == inst].any()))
    return DetectionResult(
        detected=detected,
        n_instances=n,
        detection_rate=detected / max(1, n),
        median_latency_windows=float(np.median(latencies)) if latencies else float("nan"),
        mean_latency_windows=float(np.mean(latencies)) if latencies else float("nan"),
        latencies=latencies,
    )

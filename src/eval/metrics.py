"""Classification metrics (AGENT.md §5.6).

Macro-F1, per-class F1, and the confusion matrix — reported over the official 10
event classes regardless of which classes appear in a given split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from schema import N_CLASSES, EventClass


@dataclass
class ClassificationReport:
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    support: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"macro-F1: {self.macro_f1:.4f}", "per-class F1:"]
        for name, f1 in self.per_class_f1.items():
            lines.append(f"  {name:<28} {f1:.4f}  (n={self.support.get(name, 0)})")
        return "\n".join(lines)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    # Pin the label set to ALL 10 event classes, not just those present in this split.
    labels = list(range(N_CLASSES))
    names = [EventClass(i).name for i in labels]
    # Per-class F1; zero_division=0 makes an absent/never-predicted class score 0.0.
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    # Macro-F1 = unweighted mean of per-class F1. Averaging over the fixed 10-label set
    # means a class unseen in this split honestly counts as 0 (no free credit for rares).
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Support = true-sample count per class, for context alongside each F1.
    support = {names[i]: int((y_true == i).sum()) for i in labels}
    return ClassificationReport(
        macro_f1=macro,
        per_class_f1={names[i]: float(per_class[i]) for i in labels},
        confusion=cm,
        support=support,
    )

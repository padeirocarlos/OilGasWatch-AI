"""Layer 6 — metrics, time-to-detection, robustness (AGENT.md §5.6)."""

# Public surface of the EVAL layer: classification quality (macro-F1), how early the
# model fires (time-to-detection / anticipation), and how it degrades under sensor faults.
from eval.detection import DetectionResult, time_to_detection
from eval.metrics import ClassificationReport, classification_report
from eval.robustness import RobustnessResult, robustness_probe

__all__ = [
    "ClassificationReport",
    "classification_report",
    "DetectionResult",
    "time_to_detection",
    "RobustnessResult",
    "robustness_probe",
]

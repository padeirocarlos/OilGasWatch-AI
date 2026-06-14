"""Layer 5 — losses, SSL pretraining, multi-task training loop (AGENT.md §5.5)."""

# Public surface of the TRAIN layer: the loss zoo for the imbalanced multi-task
# objective, the seeded training loop, the SSL warm-start, and determinism helpers.
from train.loop import TrainResult, train_stmoe
from train.losses import ClassBalancedCE, FocalLoss, MultiTaskLoss
from train.seeding import resolve_device, seed_everything
from train.ssl_pretrain import pretrain

__all__ = [
    "FocalLoss",
    "ClassBalancedCE",
    "MultiTaskLoss",
    "train_stmoe",
    "TrainResult",
    "pretrain",
    "seed_everything",
    "resolve_device",
]

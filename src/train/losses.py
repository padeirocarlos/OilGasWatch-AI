"""Losses for the imbalanced multi-task objective (AGENT.md §5.5)."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weighting (Lin et al., 2017).

    Focal loss multiplies the standard cross-entropy by ``(1 - p_t)**gamma`` so that
    confidently-correct (easy) examples are down-weighted and the gradient focuses on
    hard / rare cases. Critical here: NORMAL windows hugely outnumber every fault
    class, so without this the loss would be dominated by trivially-easy NORMALs.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        # gamma controls how aggressively easy examples are suppressed (0 == plain CE).
        self.gamma = gamma
        # Optional per-class weight vector (e.g. class-balanced weights); a buffer so it
        # rides along with .to(device) / state_dict without being a learnable parameter.
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=-1)
        weight = cast("torch.Tensor | None", self.weight)
        # Per-sample (un-reduced) cross-entropy = -log p_t of the true class.
        ce = nn.functional.nll_loss(logp, target, weight=weight, reduction="none")
        # pt = exp(-ce) recovers the model's probability on the true class.
        pt = torch.exp(-ce)
        # Scale each sample by the focal modulator before averaging over the batch.
        return ((1 - pt) ** self.gamma * ce).mean()


class ClassBalancedCE(nn.Module):
    """Cross-entropy with effective-number class weights (Cui et al., 2019).

    "Effective number" / class-balanced weighting re-weights each class by
    ``(1 - beta) / (1 - beta**n_c)`` rather than raw inverse frequency, accounting for
    the fact that extra samples of an already-common class add diminishing new
    information. The weight tensor is computed upstream and injected here.
    """

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        # Pre-computed per-class effective-number weights (buffer, not a parameter).
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = cast("torch.Tensor | None", self.weight)
        return nn.functional.cross_entropy(logits, target, weight=weight)


class MultiTaskLoss(nn.Module):
    """``w_e·event + w_t·transient + w_s·state`` (weights from config, AGENT.md §5.5).

    The model is trained on three heads at once (a multi-task objective): the 10-way
    event class, the binary transient-onset flag, and an auxiliary operating-state
    label. Their losses are summed with config-tuned scalar weights so one head does
    not drown out the others.
    """

    def __init__(
        self,
        w_event: float,
        w_transient: float,
        w_state: float,
        event_loss: nn.Module,
        state_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        # Relative importance of each head; tuned in config (AGENT.md §5.5).
        self.w_event = w_event
        self.w_transient = w_transient
        self.w_state = w_state
        # Event head loss is injected (typically class-balanced FocalLoss above).
        self.event_loss = event_loss
        # Transient onset is a single binary flag -> sigmoid + BCE on the logit.
        self.transient_loss = nn.BCEWithLogitsLoss()
        # State is auxiliary; ignore_index=-1 skips windows whose operating state is
        # unknown so they contribute no gradient to the state head.
        self.state_loss = nn.CrossEntropyLoss(weight=state_weight, ignore_index=-1)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        le = self.event_loss(out["event"], batch["event"])
        lt = self.transient_loss(out["transient"], batch["transient"].float())
        ls = self.state_loss(out["state"], batch["state"])
        # Weighted sum is the single scalar that backprop optimises.
        total = self.w_event * le + self.w_transient * lt + self.w_state * ls
        # Return per-head scalars too, for logging which task dominates the loss.
        return total, {"event": le.item(), "transient": lt.item(), "state": ls.item()}

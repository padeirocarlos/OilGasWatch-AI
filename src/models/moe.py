"""Mixture-of-experts gating (AGENT.md §5.4).

Soft routing over the fused subsystem embeddings. The gate must produce a valid
pooled representation with *any* expert masked out, so robustness probes (channel /
subsystem dropout, AGENT.md §5.6) never crash and degrade gracefully.
"""

from __future__ import annotations

import torch
from torch import nn

from config import MoEConfig


class Gating(nn.Module):
    """Compute soft gate weights over experts and return the gated mixture."""

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        # Per-expert scorer: maps each expert embedding to one scalar relevance logit.
        # Soft (not top-k) routing — every present expert contributes, weighted; this
        # keeps the mixture differentiable and stable under any masking pattern.
        self.gate = nn.Sequential(
            nn.Linear(cfg.expert_dim, cfg.gating_hidden),
            nn.GELU(),
            nn.Linear(cfg.gating_hidden, 1),
        )

    def forward(
        self, experts: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """experts: (B, n_experts, dim). mask: (B, n_experts) with 1 = present.

        Returns ``(mixture (B, dim), gate_weights (B, n_experts))``.
        """
        logits = self.gate(experts).squeeze(-1)  # (B, n_experts)
        # Masked softmax: set absent experts to -inf so softmax gives them weight 0 and
        # the surviving experts' weights renormalise to sum to 1 (graceful degradation).
        if mask is not None:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        # If a row is *fully* masked, softmax-of-all--inf yields NaN; zero it so the
        # robustness probes (AGENT.md §5.6) get a defined output instead of crashing.
        weights = torch.nan_to_num(weights, nan=0.0)
        # Gated combine: weighted sum of expert embeddings → single pooled vector (B, d).
        mixture = torch.einsum("be,bed->bd", weights, experts)
        return mixture, weights

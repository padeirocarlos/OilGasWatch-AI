"""Output heads (AGENT.md §5.4).

Three heads share the fused MoE representation: a 10-way event head, a binary
transient (onset) head trained on the ``+100`` labels, and an auxiliary state head.
"""

from __future__ import annotations

from torch import nn

from config import HeadsConfig
from schema import N_CLASSES


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class Heads(nn.Module):
    # All three heads read the *same* pooled MoE vector z: multi-task learning shares
    # one representation, and the auxiliary heads regularise the main event classifier.
    def __init__(self, in_dim: int, cfg: HeadsConfig, n_states: int) -> None:
        super().__init__()
        # 10-way undesirable-event classifier — the primary deliverable (AGENT.md §3).
        self.event = _mlp(in_dim, cfg.event_hidden, N_CLASSES)
        # Binary onset detector trained on the `+100` transient labels; this is the
        # head that powers early detection / time-to-detection (AGENT.md §2).
        self.transient = _mlp(in_dim, cfg.transient_hidden, 1)
        # Auxiliary operating-state head: extra supervision, not a shipped output.
        self.state = _mlp(in_dim, cfg.state_hidden, n_states)

    def forward(self, z):
        # Logits only (no softmax/sigmoid here) — losses apply the activation.
        return {
            "event": self.event(z),
            "transient": self.transient(z).squeeze(-1),  # (B,1) -> (B,) for BCE
            "state": self.state(z),
        }

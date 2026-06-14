"""Physics-informed spatio-temporal mixture-of-experts (AGENT.md §1, §5.4).

Per-subsystem TCN encoders → graph-attention fusion → MoE gating → multi-task heads.
The model accepts the raw windowed signal ``(B, T, 27)`` in :data:`schema.SIGNAL_COLUMNS`
order, and an optional per-expert mask so subsystem dropout degrades gracefully.
"""

from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig
from models.encoders import SubsystemEncoder
from models.fusion import WellGraphAttention
from models.heads import Heads
from models.moe import Gating
from schema import SIGNAL_COLUMNS, SUBSYSTEM_CHANNELS


def subsystem_indices() -> list[list[int]]:
    """Column indices (into SIGNAL_COLUMNS) for each subsystem, in expert order.

    Bridges the §3 subsystem→channel map to integer slices, so each expert can be fed
    `x[:, :, idx]` — its own physical channels — straight from the raw signal tensor.
    """
    pos = {c: i for i, c in enumerate(SIGNAL_COLUMNS)}
    return [[pos[c] for c in chans] for chans in SUBSYSTEM_CHANNELS.values()]


class STMoE(nn.Module):
    def __init__(self, cfg: ModelConfig, n_states: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        self.subsystems = subsystem_indices()
        dim = cfg.moe.expert_dim
        # One encoder per subsystem — experts are physically grounded, not learned
        # partitions — each sized to its own channel count but emitting a shared `dim`.
        self.experts = nn.ModuleList(
            SubsystemEncoder(len(idx), cfg.encoder, out_dim=dim) for idx in self.subsystems
        )
        self.fusion = WellGraphAttention(len(self.subsystems), dim)
        self.gate = Gating(cfg.moe)
        self.heads = Heads(dim, cfg.heads, n_states=n_states)

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor | None = None) -> dict:
        """x: (B, T, 27). expert_mask: (B, n_experts) with 1 = present (optional)."""
        # 1) Encode: each expert sees only its own channel slice of the window.
        embeds = []
        for idx, enc in zip(self.subsystems, self.experts, strict=True):
            embeds.append(enc(x[:, :, idx]))
        nodes = torch.stack(embeds, dim=1)  # (B, n_experts, dim)
        # 2) Fuse: experts attend to one another to surface cross-component faults.
        # 3) Gate: soft-route into one pooled mixture vector.
        # expert_mask threads through both stages so a dropped subsystem is ignored
        # end-to-end — the single source of graceful degradation (AGENT.md §7).
        fused = self.fusion(nodes, mask=expert_mask)
        mixture, gate_w = self.gate(fused, mask=expert_mask)
        # 4) Heads: multi-task outputs share the mixture; expose gate weights for
        # interpretability / robustness analysis of which subsystem drove the call.
        out = self.heads(mixture)
        out["gate_weights"] = gate_w
        return out

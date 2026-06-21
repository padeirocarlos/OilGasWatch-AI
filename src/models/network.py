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
    def __init__(self, cfg: ModelConfig, n_states: int = 8, n_eng_features: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.subsystems = subsystem_indices()
        dim = cfg.moe.expert_dim
        # One encoder per subsystem — experts are physically grounded, not learned
        # partitions — each sized to its own channel count but emitting a shared `dim`.
        self.experts = nn.ModuleList(
            SubsystemEncoder(len(idx), cfg.encoder, out_dim=dim) for idx in self.subsystems
        )
        # Fusion and gating are ablatable (cfg.moe.use_fusion / use_gating). When off, the
        # module is omitted and forward() falls back to identity / mean over experts.
        self.fusion = WellGraphAttention(len(self.subsystems), dim) if cfg.moe.use_fusion else None
        self.gate = Gating(cfg.moe) if cfg.moe.use_gating else None

        # Hybrid branch: project the per-window engineered feature vector (the GBT's exact
        # inputs) to `dim` and concatenate to the learned mixture before the heads. This hands
        # the network the physics signal (hydrate margin, choke coeffs, differentials) directly
        # — bypassing the lossy temporal pooling — so it can match the GBT where the GBT wins
        # while keeping its own temporal / early-detection edge. Features are per-feature
        # standardised upstream (train stats), so the MLP sees well-conditioned inputs.
        self.n_eng = n_eng_features
        self.eng_proj: nn.Module | None = None
        if n_eng_features > 0:
            self.eng_proj = nn.Sequential(
                nn.Linear(n_eng_features, dim),
                nn.GELU(),
                nn.Dropout(cfg.encoder.dropout),
                nn.Linear(dim, dim),
            )
            head_in = 2 * dim
        else:
            head_in = dim
        self.heads = Heads(head_in, cfg.heads, n_states=n_states)

    def forward(
        self,
        x: torch.Tensor,
        eng: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
    ) -> dict:
        """x: (B, T, 27). eng: (B, n_eng) engineered features (hybrid). expert_mask: (B, E)."""
        # 1) Encode: each expert sees only its own channel slice of the window.
        embeds = []
        for idx, enc in zip(self.subsystems, self.experts, strict=True):
            embeds.append(enc(x[:, :, idx]))
        nodes = torch.stack(embeds, dim=1)  # (B, n_experts, dim)
        # 2) Fuse: experts attend to one another to surface cross-component faults.
        #    (ablation: identity passthrough when fusion is disabled)
        fused = self.fusion(nodes, mask=expert_mask) if self.fusion is not None else nodes
        # 3) Gate: soft-route into one pooled mixture vector. expert_mask threads through
        #    both stages so a dropped subsystem is ignored end-to-end (graceful degradation).
        if self.gate is not None:
            mixture, gate_w = self.gate(fused, mask=expert_mask)
        else:
            # ablation: uniform (mean) combination of experts, mask-aware
            if expert_mask is not None:
                w = expert_mask.float()
                w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
                mixture = torch.einsum("be,bed->bd", w, fused)
                gate_w = w
            else:
                mixture = fused.mean(dim=1)
                gate_w = torch.full(fused.shape[:2], 1.0 / fused.shape[1], device=fused.device)
        # 4) Hybrid: graft the engineered-feature embedding onto the learned mixture.
        if self.eng_proj is not None and eng is not None:
            mixture = torch.cat([mixture, self.eng_proj(eng)], dim=-1)
        # 5) Heads: multi-task outputs share the representation; expose gate weights for
        # interpretability / robustness analysis of which subsystem drove the call.
        out = self.heads(mixture)
        out["gate_weights"] = gate_w
        return out

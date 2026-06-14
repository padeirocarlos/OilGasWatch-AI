"""Cross-subsystem fusion via graph attention (AGENT.md §5.4).

The four subsystem embeddings form the nodes of a small fully-connected graph; a
single multi-head attention layer lets each expert attend to the others (a lightweight
stand-in for GAT over the well topology). This couples, e.g., downhole pressure with
production-choke behaviour, which is where many cross-component faults reveal themselves.
"""

from __future__ import annotations

import torch
from torch import nn


class WellGraphAttention(nn.Module):
    """Self-attention over the set of expert embeddings (nodes)."""

    def __init__(self, n_experts: int, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        # n_heads must divide dim; fall back to 1 head rather than error on odd configs.
        heads = n_heads if dim % n_heads == 0 else 1
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # Learned per-node positional bias: distinguishes the (orderless) experts so
        # the attention can treat "downhole" and "production" as distinct graph nodes.
        self.node_bias = nn.Parameter(torch.zeros(n_experts, dim))

    def forward(self, nodes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """nodes: (B, n_experts, dim). mask: (B, n_experts) with 1 = present.

        Returns fused per-node embeddings (B, n_experts, dim); masked experts are
        excluded from attention so the model degrades gracefully under dropout.
        """
        x = nodes + self.node_bias
        # key_padding_mask=True means "ignore this key": a dropped subsystem is removed
        # from every other node's attention, so faults are read only from live sensors.
        key_padding = None if mask is None else (mask == 0)
        # Self-attention (q=k=v=x): every expert mixes in evidence from the others, e.g.
        # downhole pressure attending to choke behaviour where cross-faults surface.
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding)
        # Residual + LayerNorm: keep each node's own embedding, add the fused context.
        return self.norm(nodes + attn_out)

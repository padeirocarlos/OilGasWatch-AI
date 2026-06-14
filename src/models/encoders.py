"""Subsystem encoders (AGENT.md §5.4).

One dilated-TCN encoder per physical subsystem (expert). Each consumes its own
channel subset and emits a fixed-width embedding. A TCN gives an exponentially
growing receptive field with stable, parallel training — well suited to the
onset-transient detection the project targets.
"""

from __future__ import annotations

import torch
from torch import nn

from config import EncoderConfig


class _TCNBlock(nn.Module):
    """Residual dilated 1-D conv block (causal padding so it can run online)."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        # `dilation` spaces out the kernel taps; stacking blocks with dilation 1,2,4,…
        # grows the receptive field exponentially so a few layers see long history.
        # `pad` is the exact left-pad that keeps the output length unchanged.
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _causal(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        # Pad only on the left: output at time t depends on t and earlier, never the
        # future. This causality is what lets the encoder run online for early detection.
        return conv(nn.functional.pad(x, (self.pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm1(self._causal(x, self.conv1))))
        h = self.drop(self.act(self.norm2(self._causal(h, self.conv2))))
        # Residual add: the block learns a correction to x, easing gradient flow so
        # many dilated layers can be stacked without vanishing-gradient pain.
        return x + h


class SubsystemEncoder(nn.Module):
    """Project a subsystem's channels, run stacked dilated TCN blocks, pool to a vector."""

    def __init__(self, in_channels: int, cfg: EncoderConfig, out_dim: int) -> None:
        super().__init__()
        # 1x1 conv lifts the few raw subsystem channels to the shared `hidden` width.
        self.proj = nn.Conv1d(in_channels, cfg.hidden, kernel_size=1)
        # dilation = 2**i: each block doubles reach, so receptive field is ~2**n_blocks.
        self.blocks = nn.ModuleList(
            _TCNBlock(cfg.hidden, cfg.kernel_size, dilation=2**i, dropout=cfg.dropout)
            for i in range(cfg.n_blocks)
        )
        # Statistics pooling: mean + std + max over time, so the head sees 3*hidden.
        self.head = nn.Linear(3 * cfg.hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_sub) -> embedding (B, out_dim)."""
        # Conv1d wants (B, C, T); the window arrives as (B, T, C), hence the transpose.
        h = self.proj(x.transpose(1, 2))  # (B, hidden, T)
        for blk in self.blocks:
            h = blk(h)
        # Statistics pooling over time. Mean alone averages an oscillation away to its
        # offset; std encodes per-feature oscillation amplitude and max captures extremes
        # (e.g. dP_tree spikes) — both discriminative cues the GBT reads from its std/max
        # window aggregates but plain mean-pooling throws away. Concatenate all three.
        pooled = torch.cat(
            [h.mean(dim=-1), h.std(dim=-1, unbiased=False), h.amax(dim=-1)], dim=-1
        )
        return self.head(pooled)

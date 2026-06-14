"""Self-supervised masked-reconstruction pretraining (AGENT.md §5.5).

Pretrains the subsystem encoders by masking spans of the input and reconstructing
them — usable on NORMAL + simulated + hand-drawn data where labels are abundant. The
pretrained encoder weights warm-start the supervised ST-MoE.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig
from models.network import STMoE


class _Reconstructor(nn.Module):
    """Wraps the encoders with a linear decoder back to channel space.

    The SSL pretext task reuses the real subsystem encoders, then bolts on a throwaway
    linear decoder mapping fused embeddings back to channel space so we can compute a
    reconstruction loss. Only the encoder weights are kept afterwards.
    """

    def __init__(self, net: STMoE, n_channels: int) -> None:
        super().__init__()
        self.net = net
        # Fused embedding width = per-expert dim x number of subsystem experts.
        dim = net.cfg.moe.expert_dim * len(net.subsystems)
        self.decoder = nn.Linear(dim, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run each subsystem's channel slice through its dedicated expert encoder.
        embeds = [
            enc(x[:, :, idx])
            for idx, enc in zip(net_subsystems(self.net), self.net.experts, strict=True)
        ]
        # Concatenate the per-subsystem embeddings, then decode to channel space.
        z = torch.cat(embeds, dim=-1)
        return self.decoder(z)


def net_subsystems(net: STMoE) -> list[list[int]]:
    return net.subsystems


def pretrain(
    net: STMoE,
    X: np.ndarray,
    mcfg: ModelConfig,
    device: str = "cpu",
) -> STMoE:
    """Masked-reconstruction pretraining over windowed signals ``X`` (n, T, C).

    Self-supervised: corrupt (mask) part of each window's input and train the encoders
    to reconstruct the clean target, learning useful signal structure from abundant
    unlabelled-friendly data (NORMAL + simulated + hand-drawn) before any supervision.
    """
    tcfg = mcfg.train
    if tcfg.ssl_epochs <= 0:
        return net  # SSL disabled -> return the cold-started encoders unchanged
    model = _Reconstructor(net, n_channels=X.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=tcfg.batch_size,
        shuffle=True,
    )
    # Seeded generator so the random masking pattern is reproducible (AGENT.md §2).
    gen = torch.Generator().manual_seed(tcfg.seed)
    for _ in range(tcfg.ssl_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            # Mask ~ssl_mask_ratio of (sample, timestep) positions across all channels.
            mask = (torch.rand(batch.shape[:2], generator=gen) < tcfg.ssl_mask_ratio).to(device)
            corrupted = batch.clone()
            corrupted[mask] = 0.0  # zero out the masked timesteps (the corruption)
            target = batch.mean(dim=1)  # reconstruct the window's channel means
            pred = model(corrupted)  # encoder must infer the means despite the mask
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # Encoder weights inside `net` are now warm-started in place; decoder is discarded.
    return net

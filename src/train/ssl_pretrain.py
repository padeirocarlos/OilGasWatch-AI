"""Self-supervised masked-reconstruction pretraining (AGENT.md §5.5).

Pretrains the subsystem encoders by masking spans of the input and reconstructing
them — usable on NORMAL + simulated + hand-drawn data where labels are abundant. The
pretrained encoder weights warm-start the supervised ST-MoE.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from config import ModelConfig
from models.network import STMoE

log = logging.getLogger("oilgaswatch.train")


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

    Memmap-safe: ``X`` may be a read-only np.memmap; batches are loaded one at a time
    (never the whole array), so this works on full-data runs without exhausting RAM.
    """
    tcfg = mcfg.train
    if tcfg.ssl_epochs <= 0:
        return net  # SSL disabled -> return the cold-started encoders unchanged
    n, _, n_channels = X.shape
    model = _Reconstructor(net, n_channels=n_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    # Seeded generators so both the shuffle and the masking pattern are reproducible (§2).
    gen = torch.Generator().manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    bs = tcfg.batch_size
    for ep in range(tcfg.ssl_epochs):
        perm = rng.permutation(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, bs):
            # Sort the batch indices so memmap reads stay roughly sequential (disk locality);
            # order within a batch is irrelevant to the reconstruction objective.
            idx = np.sort(perm[i : i + bs])
            batch = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
            # Mask ~ssl_mask_ratio of (sample, timestep) positions across all channels.
            mask = (torch.rand(batch.shape[:2], generator=gen) < tcfg.ssl_mask_ratio).to(device)
            corrupted = batch.clone()
            corrupted[mask] = 0.0  # zero out the masked timesteps (the corruption)
            target = batch.mean(dim=1)  # reconstruct the window's channel means
            pred = model(corrupted)  # encoder must infer the means despite the mask
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        log.info(
            "ssl pretrain epoch %d/%d  recon_loss=%.4f", ep + 1, tcfg.ssl_epochs, tot / max(1, nb)
        )
    # Encoder weights inside `net` are now warm-started in place; decoder is discarded.
    return net

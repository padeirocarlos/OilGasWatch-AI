"""Determinism + model wiring (AGENT.md §2: same seed + data + config ⇒ same result)."""

from __future__ import annotations

import numpy as np
import torch

from config import ModelConfig
from dataset.windows import label_windows, window_bounds
from models.network import STMoE
from train.seeding import seed_everything


def test_seed_makes_torch_init_reproducible():
    # Determinism (§2): same seed ⇒ same metrics requires same starting weights. Re-seeding
    # before each build must yield bit-identical parameters; if random init drifted, two
    # "identical" runs could diverge and reproducibility claims would be meaningless.
    cfg = ModelConfig()
    seed_everything(123)
    a = STMoE(cfg, n_states=4)
    seed_everything(123)
    b = STMoE(cfg, n_states=4)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_stmoe_degrades_gracefully_under_expert_mask():
    # Single-subsystem dropout (§7): a whole sensor subsystem can go offline in the field.
    # Masking out one MoE expert must still produce a valid 10-class output with no NaN —
    # the model degrades gracefully (bounded drop) rather than crashing on missing inputs.
    cfg = ModelConfig()
    net = STMoE(cfg, n_states=4)
    x = torch.randn(3, cfg.encoder.kernel_size * 8, 27)
    mask = torch.ones(3, 4)
    mask[:, 2] = 0  # drop a whole subsystem
    out = net(x, expert_mask=mask)
    assert out["event"].shape == (3, 10)  # still all 10 event classes
    assert not torch.isnan(out["event"]).any()


def test_window_labels_decode_transient(synthetic_frame, model_cfg):
    # The +100 transient-onset label (§2) must survive windowing: the windower has to
    # DECODE 101 back into event class 1 plus a transient flag, so the transient head
    # actually receives onset windows to learn from instead of losing them.
    bounds = window_bounds(len(synthetic_frame.df), model_cfg.window, 1.0)
    labels = label_windows(synthetic_frame.df["class"], synthetic_frame.df["state"], bounds)
    # The synthetic instance contains both NORMAL and transient windows.
    assert labels.event.max() == 1  # event class 1 recovered from the 101 code
    assert labels.transient.max() == 1  # onset windows flagged transient
    assert set(np.unique(labels.valid)) <= {0, 1}  # validity is a clean 0/1 mask

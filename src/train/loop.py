"""Multi-task training loop (AGENT.md §5.5, §6).

Trains the ST-MoE with the weighted ``event + transient + state`` objective. Every
run is seeded and writes structured metadata to ``runs/<id>/`` (AGENT.md §6).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig
from dataset.build import SequenceDataset
from dataset.sampler import effective_number_weights
from models.network import STMoE
from schema import N_CLASSES
from train.losses import FocalLoss, MultiTaskLoss
from train.seeding import resolve_device, seed_everything

log = logging.getLogger("oilgaswatch.train")


class _SeqTorchDataset(Dataset):
    # Thin torch.Dataset adapter exposing the windowed arrays as per-sample tensors.
    def __init__(self, ds: SequenceDataset) -> None:
        self.X = ds.X  # (n, T, C) windowed sensor signals
        self.event = ds.event.astype(np.int64)  # 10-way event class label
        self.transient = ds.transient.astype(np.int64)  # binary onset flag
        # Remap state: unknown (-1) stays -1 (ignored); others are dense 0..K-1 ids.
        self.state = ds.state.astype(np.int64)
        # Optional hybrid engineered features (already standardised upstream); None = raw-only.
        self.feat = ds.feat

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> dict:
        item = {
            # X may be float16 RAM or a read-only memmap; np.asarray(..., float32) gives a
            # writable contiguous copy of just this window (memmap-safe, no whole-array load).
            "x": torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)),
            "event": int(self.event[i]),
            "transient": int(self.transient[i]),
            "state": int(self.state[i]),
        }
        if self.feat is not None:
            item["eng"] = torch.from_numpy(np.asarray(self.feat[i], dtype=np.float32))
        return item


def _collate(items: list[dict]) -> dict:
    batch = {
        "x": torch.stack([it["x"] for it in items]),
        "event": torch.tensor([it["event"] for it in items], dtype=torch.long),
        "transient": torch.tensor([it["transient"] for it in items], dtype=torch.long),
        "state": torch.tensor([it["state"] for it in items], dtype=torch.long),
    }
    if "eng" in items[0]:
        batch["eng"] = torch.stack([it["eng"] for it in items])
    return batch


@dataclass
class TrainResult:
    run_id: str
    run_dir: str
    history: list[dict]
    n_states: int


def _new_run_dir(runs_dir: str) -> tuple[str, Path]:
    # Timestamp run id; each run gets its own runs/<id>/ for reproducible auditing.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    d = Path(runs_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def train_stmoe(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset | None,
    mcfg: ModelConfig,
    runs_dir: str = "runs",
    config_snapshot: dict | None = None,
) -> tuple[STMoE, TrainResult]:
    tcfg = mcfg.train
    # Seed first so every downstream RNG (init, shuffle, dropout) is deterministic.
    seed_everything(tcfg.seed)
    device = resolve_device(tcfg.device)

    # Number of state classes is inferred from the data (max dense id + 1); the -1
    # "unknown" sentinel is excluded via max(.., 0) so it never inflates the count.
    n_states = int(max(train_ds.state.max(), 0)) + 1
    # Hybrid: if the dataset carries engineered features, size the injection branch to them.
    n_eng = int(train_ds.feat.shape[1]) if train_ds.feat is not None else 0
    net = STMoE(mcfg, n_states=n_states, n_eng_features=n_eng).to(device)

    # Class-balanced focal loss for the imbalanced event head.
    # effective_number_weights down-weights common classes (NORMAL) per Cui et al. 2019.
    weights = effective_number_weights(train_ds.event, N_CLASSES, mcfg.loss.class_balanced_beta)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    event_loss = FocalLoss(gamma=mcfg.loss.focal_gamma, weight=w)
    # Combine the three heads into the weighted multi-task objective.
    criterion = MultiTaskLoss(mcfg.loss.event, mcfg.loss.transient, mcfg.loss.state, event_loss).to(
        device
    )

    loader = DataLoader(
        _SeqTorchDataset(train_ds),
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        # Seeded generator -> identical batch order across runs (reproducibility).
        generator=torch.Generator().manual_seed(tcfg.seed),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    run_id, run_dir = _new_run_dir(runs_dir)
    history: list[dict] = []
    log.info(
        "training %d epochs on %s | %d train windows, %d batches/epoch",
        tcfg.epochs,
        device,
        len(train_ds),
        len(loader),
    )

    for epoch in range(tcfg.epochs):
        net.train()
        tot = 0.0
        t0 = time.time()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = net(batch["x"], eng=batch.get("eng"))  # event/transient/state logits
            loss, parts = criterion(out, batch)  # weighted multi-task scalar + parts
            opt.zero_grad()
            loss.backward()
            # Clip the global grad norm: a single outlier batch can otherwise spike the update
            # and tip the whole run into NaN — cheap insurance for stable convergence.
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item()
        # Record mean train loss plus the last batch's per-head breakdown for the epoch.
        row = {"epoch": epoch, "train_loss": tot / max(1, len(loader)), **parts}
        if val_ds is not None:
            row["val_event_acc"] = _quick_event_acc(net, val_ds, device, tcfg.batch_size)
        history.append(row)
        # Per-epoch progress so a long CPU run isn't a black box (and the trend is visible).
        log.info(
            "epoch %2d/%d  loss=%.4f  val_acc=%.3f  (%.1fs)",
            epoch + 1,
            tcfg.epochs,
            row["train_loss"],
            row.get("val_event_acc", float("nan")),
            time.time() - t0,
        )

    result = TrainResult(run_id=run_id, run_dir=str(run_dir), history=history, n_states=n_states)
    # Persist config + history (audit trail) and the trained weights under runs/<id>/.
    _write_metadata(run_dir, mcfg, result, config_snapshot)
    torch.save(net.state_dict(), run_dir / "model.pt")
    return net, result


@torch.no_grad()
def _quick_event_acc(net: STMoE, ds: SequenceDataset, device: str, bs: int) -> float:
    # Cheap per-epoch validation signal (plain accuracy); the real evaluation is the
    # macro-F1 / detection harness in the eval layer.
    net.eval()
    correct = total = 0
    X = ds.X  # numpy array or memmap; slice per batch and convert (never load whole X)
    y = ds.event
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.asarray(X[i : i + bs], dtype=np.float32)).to(device)
        eng = None
        if ds.feat is not None:
            eng = torch.from_numpy(np.asarray(ds.feat[i : i + bs], dtype=np.float32)).to(device)
        pred = net(xb, eng=eng)["event"].argmax(-1).cpu().numpy()
        correct += int((pred == y[i : i + bs]).sum())
        total += len(pred)
    return correct / max(1, total)


def _write_metadata(
    run_dir: Path, mcfg: ModelConfig, result: TrainResult, snapshot: dict | None
) -> None:
    # Everything needed to reproduce/audit the run: config, inferred state count, and
    # the full per-epoch loss/metric history (AGENT.md §6 structured run metadata).
    meta = {
        "run_id": result.run_id,
        "model_config": mcfg.model_dump(),
        "n_states": result.n_states,
        "history": result.history,
    }
    if snapshot:
        meta["config_snapshot"] = snapshot
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

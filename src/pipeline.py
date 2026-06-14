"""End-to-end orchestration: baseline and ST-MoE runs (AGENT.md §1, §7).

Glues the layers into reproducible, well-disjoint runs whose metrics are logged to
``runs/<id>/``. Importable so both the CLI and the tests exercise the same path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DataConfig, FeaturesConfig, ModelConfig
from dataset.build import build_sequence_dataset, build_window_dataset
from dataset.splitter import WellDisjointSplitter
from eval.detection import time_to_detection
from eval.metrics import classification_report
from ingest import discover_instances
from models.baseline import GBTClassifier
from train.seeding import seed_everything


@dataclass
class BaselineRun:
    run_id: str
    run_dir: str
    macro_f1: float
    metrics: dict


def _runs_subdir(runs_dir: str, tag: str) -> tuple[str, Path]:
    # Each run gets a unique timestamped dir under runs/ so its metrics.json never clobbers a
    # prior run — the durable, reproducible record the eval subcommand later replays (§6, §7).
    run_id = f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    d = Path(runs_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def _cap_per_class(metas: list, limit_per_class: int, seed: int) -> list:
    """Deterministically keep at most ``limit_per_class`` instances per event-class dir.

    Used for fast/debug runs: caps each class so a few huge classes don't dominate, while
    keeping per-class balance roughly intact.
    """
    # Seeded RNG + sorted iteration => same subset every run with the same seed (§2 determinism).
    rng = np.random.default_rng(seed)
    by_dir: dict[int, list] = {}
    for m in metas:
        by_dir.setdefault(m.event_class_dir, []).append(m)
    out: list = []
    for _, group in sorted(by_dir.items()):
        # Permute then truncate => an unbiased random sample of this class, not just the head.
        idx = rng.permutation(len(group))[:limit_per_class]
        out.extend(group[i] for i in idx)
    return out


def run_baseline(
    dcfg: DataConfig,
    fcfg: FeaturesConfig,
    mcfg: ModelConfig,
    classes: list[int] | None = None,
    test_frac: float = 0.25,
    limit_per_class: int | None = None,
    stratified: bool = False,
    progress: bool = True,
) -> BaselineRun:
    """Fit the GBT baseline on a well-disjoint split and log macro-F1 + detection.

    End-to-end flow: discover instances -> (optional cap) -> well-disjoint split ->
    window each side into per-window feature matrices -> fit GBT -> evaluate -> log to runs/.
    With ``stratified=True`` rare classes are guaranteed test wells (see splitter).
    """
    # Seed first so discovery, splitting, sampling and the GBT are all reproducible (§2).
    seed_everything(dcfg.seed)
    metas = discover_instances(dcfg, classes=classes)
    if not metas:
        raise ValueError("no instances discovered; check dataset_dir / classes")
    if limit_per_class:
        metas = _cap_per_class(metas, limit_per_class, dcfg.seed)

    # Split BY WELL, never by window: train/test wells must not overlap or the model can
    # memorise a well and inflate metrics (§2). assert_disjoint turns that invariant into a
    # hard check, not a hope.
    splitter = WellDisjointSplitter(seed=dcfg.seed)
    split = (
        splitter.stratified_train_test_split(metas, test_frac=test_frac)
        if stratified
        else splitter.train_test_split(metas, test_frac=test_frac)
    )
    split.assert_disjoint()

    # Window both sides. Train fits and exposes its feature_columns; test reuses them
    # (feature_columns=None => inherit train's schema) so columns line up exactly at predict.
    train = build_window_dataset(split.train, dcfg, fcfg, mcfg, progress=progress)
    test = build_window_dataset(
        split.test, dcfg, fcfg, mcfg, feature_columns=None, progress=progress
    )

    # Primary task: 10-way event classifier. macro-F1 weights every class equally, so rare
    # fault classes count as much as NORMAL despite heavy imbalance.
    event_clf = GBTClassifier(mcfg.baseline, seed=dcfg.seed).fit(train.X, train.event)
    event_pred = event_clf.predict(test.X)
    report = classification_report(test.event, event_pred)

    # A SEPARATE binary transient classifier is fit purely for time-to-detection. The event
    # head answers "which fault?"; this head answers "is a fault onset happening *now*?" — a
    # cleaner target for measuring detection latency, and it can fire before the event head
    # commits to a class. Skipped when no transient (onset) windows exist in train (§2, §3).
    detection = None
    if train.transient.max() > 0:
        trans_clf = GBTClassifier(mcfg.baseline, seed=dcfg.seed).fit(train.X, train.transient)
        trans_pred = trans_clf.predict(test.X)
        # Latency from true onset to first correct firing; uses per-instance window order so
        # windows are read in time sequence. Negative latency = fired before the label (early).
        detection = time_to_detection(test.instance_id, test.order, test.transient, trans_pred)

    # Persist the full run record: config provenance (seed, classes, test_frac), dataset sizes
    # and well counts (so disjointness is auditable), plus all metrics. This is what makes a
    # result reproducible and replayable via the eval subcommand (§7).
    run_id, run_dir = _runs_subdir(dcfg.runs_dir, "baseline")
    metrics = {
        "run_id": run_id,
        "kind": "baseline",
        "backend": mcfg.baseline.backend,
        "classes": classes,
        "test_frac": test_frac,
        "seed": dcfg.seed,
        "n_train_windows": int(len(train)),
        "n_test_windows": int(len(test)),
        "n_train_wells": len({w for w in train.well_id}),
        "n_test_wells": len({w for w in test.well_id}),
        "macro_f1": report.macro_f1,
        "per_class_f1": report.per_class_f1,
        "support": report.support,
        "confusion": report.confusion.tolist(),
    }
    if detection is not None:
        metrics["detection"] = {
            "detection_rate": detection.detection_rate,
            "median_latency_windows": detection.median_latency_windows,
            "mean_latency_windows": detection.mean_latency_windows,
            "n_instances": detection.n_instances,
        }
    # default=str lets numpy scalars / arrays serialise without a custom encoder.
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return BaselineRun(
        run_id=run_id, run_dir=str(run_dir), macro_f1=report.macro_f1, metrics=metrics
    )


def run_stmoe(
    dcfg: DataConfig,
    fcfg: FeaturesConfig,
    mcfg: ModelConfig,
    classes: list[int] | None = None,
    test_frac: float = 0.25,
    limit_per_class: int | None = None,
    stratified: bool = False,
    streaming: bool = False,
    hybrid: bool = False,
    progress: bool = True,
) -> dict:
    """Train the ST-MoE on a well-disjoint split and evaluate on held-out wells.

    Mirrors run_baseline's flow but builds sequence (windowed time-series) datasets for the
    deep model instead of flattened per-window feature matrices, and trains the multi-task net.
    Passing the same ``classes``/``test_frac``/``limit_per_class``/``stratified`` as
    run_baseline (with the same ``dcfg.seed``) yields the identical well split — the basis
    for a fair comparison (§7). ``streaming=True`` builds the sequence windows on disk
    (memmap) so the full dataset can be used without exhausting RAM.
    """
    # torch/train.loop imported lazily so the lightweight baseline path needn't pull in torch.
    import torch

    from dataset.build import build_sequence_memmap
    from eval.detection import time_to_detection
    from train.loop import train_stmoe

    # Train loop uses its own seed (mcfg.train.seed); the split still uses dcfg.seed so the
    # well partition is identical to the baseline's — an apples-to-apples comparison (§7).
    seed_everything(mcfg.train.seed)
    metas = discover_instances(dcfg, classes=classes)
    # Apply the SAME deterministic per-class cap as run_baseline before splitting, so both
    # models train and test on exactly the same wells (not just the same policy).
    if limit_per_class:
        metas = _cap_per_class(metas, limit_per_class, dcfg.seed)
    splitter = WellDisjointSplitter(seed=dcfg.seed)
    split = (
        splitter.stratified_train_test_split(metas, test_frac=test_frac)
        if stratified
        else splitter.train_test_split(metas, test_frac=test_frac)
    )
    split.assert_disjoint()

    # Sequence datasets keep the temporal axis (vs. flattened features) for the TCN encoders.
    # streaming=True writes windows to disk-backed memmaps so full-data runs don't OOM; the
    # cache files live under runs/_seqcache and are removed in the finally block below.
    cache_dir = Path(dcfg.runs_dir) / "_seqcache"
    if streaming:
        train_ds = build_sequence_memmap(
            split.train, dcfg, fcfg, mcfg, cache_dir / "train.f16",
            with_features=hybrid, progress=progress,
        )
        test_ds = build_sequence_memmap(
            split.test, dcfg, fcfg, mcfg, cache_dir / "test.f16",
            with_features=hybrid, progress=progress,
        )
    else:
        train_ds = build_sequence_dataset(
            split.train, dcfg, fcfg, mcfg, with_features=hybrid, progress=progress
        )
        test_ds = build_sequence_dataset(
            split.test, dcfg, fcfg, mcfg, with_features=hybrid, progress=progress
        )

    # Hybrid: robustly standardise the engineered features using TRAIN stats only
    # (leakage-clean), applied to both splits. Engineered features are heavy-tailed, so
    # mean/std + a single outlier yields extreme z-scores that NaN the injection MLP — use
    # median/IQR and clip to ±10 so the MLP always sees well-conditioned, bounded inputs.
    if hybrid and train_ds.feat is not None:
        med = np.median(train_ds.feat, axis=0)
        iqr = np.percentile(train_ds.feat, 75, axis=0) - np.percentile(train_ds.feat, 25, axis=0)
        iqr[iqr < 1e-6] = 1.0  # guard constant/near-constant features

        def _robust(a: np.ndarray) -> np.ndarray:
            return np.clip((a - med) / iqr, -10.0, 10.0).astype(np.float32)

        train_ds.feat = _robust(train_ds.feat)
        test_ds.feat = _robust(test_ds.feat)

    try:
        # Training owns its own runs/ logging; config_snapshot records the same provenance
        # fields as the baseline so any logged run is reproducible from the snapshot alone (§7).
        net, result = train_stmoe(
            train_ds,
            test_ds,
            mcfg,
            runs_dir=dcfg.runs_dir,
            config_snapshot={"classes": classes, "test_frac": test_frac, "seed": dcfg.seed},
        )

        # eval() + no_grad() => inference mode (dropout/BN frozen, no autograd) for scoring.
        net.eval()
        # Score on the net's device; inputs must be moved there too (a cuda net fed CPU
        # tensors raises a device mismatch in the conv layers).
        device = next(net.parameters()).device
        with torch.no_grad():
            event_preds, trans_preds = [], []
            X = test_ds.X  # numpy array or memmap; sliced per batch (never loaded whole)
            bs = mcfg.train.batch_size
            # Batch through the held-out set; the net is multi-head, so read both the event
            # head (argmax = class) and the transient head (sigmoid > 0.5 = onset firing) in
            # one pass for a fair macro-F1 + detection comparison vs the baseline.
            feat = test_ds.feat
            for i in range(0, len(X), bs):
                # np.asarray(..., float32) gives a writable contiguous batch (memmap-safe).
                xb = torch.from_numpy(np.asarray(X[i : i + bs], dtype=np.float32)).to(device)
                eb = None
                if feat is not None:
                    eb = torch.from_numpy(np.asarray(feat[i : i + bs], dtype=np.float32)).to(device)
                out = net(xb, eng=eb)
                event_preds.append(out["event"].argmax(-1).cpu().numpy())
                trans_preds.append((torch.sigmoid(out["transient"]) > 0.5).long().cpu().numpy())
        event_pred = np.concatenate(event_preds) if event_preds else np.array([])
        trans_pred = np.concatenate(trans_preds) if trans_preds else np.array([])
        report = classification_report(test_ds.event, event_pred)

        metrics = {
            "run_id": result.run_id,
            "kind": "stmoe",
            "classes": classes,
            "test_frac": test_frac,
            "seed": dcfg.seed,
            "streaming": streaming,
            "hybrid": hybrid,
            "n_train_windows": int(len(train_ds)),
            "n_test_windows": int(len(test_ds)),
            "n_train_wells": len(set(train_ds.well_id)),
            "n_test_wells": len(set(test_ds.well_id)),
            "macro_f1": report.macro_f1,
            "per_class_f1": report.per_class_f1,
            "support": report.support,
            "confusion": report.confusion.tolist(),
        }
        # Detection latency from the transient head, scored exactly like the baseline's
        # separate transient classifier so the two are directly comparable.
        if test_ds.transient.max() > 0:
            detection = time_to_detection(
                test_ds.instance_id, test_ds.order, test_ds.transient, trans_pred
            )
            metrics["detection"] = {
                "detection_rate": detection.detection_rate,
                "median_latency_windows": detection.median_latency_windows,
                "mean_latency_windows": detection.mean_latency_windows,
                "n_instances": detection.n_instances,
            }
        # Write the held-out scores into the run dir the train loop created so eval can
        # replay this ST-MoE run exactly like a baseline run (§7).
        out_path = Path(result.run_dir) / "metrics.json"
        out_path.write_text(json.dumps(metrics, indent=2, default=str))
        return metrics
    finally:
        if streaming:
            # Release the memmaps, then delete the (multi-GB) on-disk cache files.
            del train_ds, test_ds
            for fname in ("train.f16", "test.f16"):
                (cache_dir / fname).unlink(missing_ok=True)

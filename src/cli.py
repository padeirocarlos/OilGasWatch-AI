"""Command-line entrypoints (AGENT.md §4: ``ingest | features | train | eval``).

Run as ``uv run python src/cli.py <command> [options]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# sys.path shim: when invoked as a script (``python src/cli.py``) Python only puts the
# script's own dir on the path, so sibling layer packages (config, ingest, features, ...)
# would be unresolvable. Prepending this file's dir makes them importable as top-level
# modules without forcing a ``python -m`` invocation or an installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    load_data_config,
    load_features_config,
    load_model_config,
)

log = logging.getLogger("oilgaswatch")


def _configs(args: argparse.Namespace):
    # Load all three YAML configs up front (AGENT.md §6: configs are loaded once and passed
    # explicitly). Every subcommand uses this so paths and knobs stay consistent across a run.
    return (
        load_data_config(args.data_config),
        load_features_config(args.features_config),
        load_model_config(args.model_config),
    )


def _classes(args: argparse.Namespace) -> list[int] | None:
    # ``--classes`` restricts the run to specific event-class dirs (e.g. only HYDRATE).
    # ``None`` means "all classes"; the empty-list case collapses to None for that reason.
    return list(args.classes) if args.classes else None


def cmd_ingest(args: argparse.Namespace) -> int:
    # Stage 1 sanity check: discover instances and resample one to confirm the schema and
    # quality channels (per-sensor is_missing/is_frozen) are produced as expected.
    # Heavy layer imports are deferred into each cmd so unrelated subcommands stay cheap.
    from ingest import discover_instances, load_and_resample

    dcfg, _, _ = _configs(args)
    metas = discover_instances(dcfg, classes=_classes(args))
    if args.limit:
        metas = metas[: args.limit]
    log.info("discovered %d instances", len(metas))
    # Tally by provenance (real / simulated / hand-drawn) since eval must be able to
    # stratify or exclude synthetic sources (AGENT.md §3).
    by_source: dict[str, int] = {}
    for m in metas:
        by_source[m.source.name] = by_source.get(m.source.name, 0) + 1
    log.info("by source: %s", by_source)
    if metas:
        inst = load_and_resample(metas[0].path, dcfg)
        log.info(
            "sample %s -> %d rows, %d quality channels",
            metas[0].instance_id,
            inst.n_samples,
            len(inst.quality_columns),
        )
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    # Stage 2 smoke test: build the physics-informed feature set for a single instance and
    # report its shape, optionally persisting it. Operates on one instance for fast iteration.
    from features.pipeline import build_features
    from ingest import discover_instances, load_and_resample

    dcfg, fcfg, _ = _configs(args)
    metas = discover_instances(dcfg, classes=_classes(args))
    if not metas:
        log.error("no instances discovered")
        return 1
    metas = metas[: args.limit] if args.limit else metas
    inst = load_and_resample(metas[0].path, dcfg)
    feats = build_features(inst, fcfg)
    # Drop the two label columns to count only true feature channels in the log line.
    body = feats.drop(columns=["class", "state"])
    log.info(
        "built %d features over %d timesteps for %s",
        body.shape[1],
        body.shape[0],
        metas[0].instance_id,
    )
    if args.out:
        # Match the dataset's on-disk format (pyarrow + brotli) per AGENT.md §3.
        feats.to_parquet(args.out, engine="pyarrow", compression="brotli")
        log.info("wrote %s", args.out)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    # Stage 3: fit + evaluate the gradient-boosted-tree baseline end-to-end and print its
    # metrics. This is the reference number the ST-MoE must beat (AGENT.md §7).
    from pipeline import run_baseline

    dcfg, fcfg, mcfg = _configs(args)
    if args.stride_s:
        mcfg.window.stride_s = args.stride_s
    run = run_baseline(
        dcfg,
        fcfg,
        mcfg,
        classes=_classes(args),
        test_frac=args.test_frac,
        limit_per_class=args.limit_per_class or None,
        stratified=args.stratified,
        progress=True,
    )
    log.info("baseline macro-F1=%.4f  ->  %s", run.macro_f1, run.run_dir)
    print(json.dumps(run.metrics, indent=2, default=str))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # Stage 4: train + evaluate the spatio-temporal mixture-of-experts (ST-MoE) deep model.
    from pipeline import run_stmoe

    dcfg, fcfg, mcfg = _configs(args)
    # CLI overrides for the schedule, so a full run can be driven without editing model.yaml.
    if args.epochs:
        mcfg.train.epochs = args.epochs
    if args.batch_size:
        mcfg.train.batch_size = args.batch_size
    if args.stride_s:
        mcfg.window.stride_s = args.stride_s
    if args.ssl_epochs >= 0:
        mcfg.train.ssl_epochs = args.ssl_epochs
    metrics = run_stmoe(
        dcfg,
        fcfg,
        mcfg,
        classes=_classes(args),
        test_frac=args.test_frac,
        limit_per_class=args.limit_per_class or None,
        stratified=args.stratified,
        streaming=args.streaming,
        hybrid=args.hybrid,
        progress=True,
    )
    log.info("stmoe macro-F1=%.4f", metrics["macro_f1"])
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    # Stage 5: replay a past run by printing the metrics.json a training run logged. No
    # recomputation — runs/ is the durable record of reproducible results (AGENT.md §6, §7).
    from pathlib import Path

    dcfg, _, _ = _configs(args)
    metrics_path = Path(dcfg.runs_dir) / args.run / "metrics.json"
    if not metrics_path.exists():
        log.error("no metrics.json under run %s", args.run)
        return 1
    print(metrics_path.read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Parent-parser pattern: shared flags (config paths, class filter, limit) are declared
    # once on a non-help ``common`` parser, then passed via ``parents=[common]`` to BOTH the
    # top-level parser and every subparser. This lets the same flag be accepted before or
    # after the subcommand (e.g. ``--classes 1 baseline`` and ``baseline --classes 1``).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-config", default="configs/data.yaml")
    common.add_argument("--features-config", default="configs/features.yaml")
    common.add_argument("--model-config", default="configs/model.yaml")
    common.add_argument("--classes", type=int, nargs="*", help="restrict to these class dirs")
    common.add_argument("--limit", type=int, default=0, help="cap instances (debugging)")

    p = argparse.ArgumentParser(
        prog="oilgaswatch", description="OilGasWatch-AI early-detection pipeline", parents=[common]
    )
    # One subcommand per pipeline stage; each binds its handler via ``func`` (dispatched in
    # main). ``required=True`` forces the user to pick a stage rather than defaulting silently.
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ingest", parents=[common], help="load + resample, report schema/quality"
    ).set_defaults(func=cmd_ingest)
    f = sub.add_parser("features", parents=[common], help="build features for one instance")
    f.add_argument("--out", help="write features parquet")
    f.set_defaults(func=cmd_features)

    b = sub.add_parser("baseline", parents=[common], help="fit + evaluate the GBT baseline")
    b.add_argument("--test-frac", type=float, default=0.25)
    b.add_argument("--limit-per-class", type=int, default=0, help="cap instances per class dir")
    b.add_argument("--stratified", action="store_true", help="stratify the well split by class")
    b.add_argument("--stride-s", type=float, default=0.0, help="window stride (0=config)")
    b.set_defaults(func=cmd_baseline)

    t = sub.add_parser("train", parents=[common], help="train + evaluate the ST-MoE")
    t.add_argument("--test-frac", type=float, default=0.25)
    t.add_argument("--limit-per-class", type=int, default=0, help="cap instances per class dir")
    t.add_argument("--epochs", type=int, default=0, help="override training epochs (0 = config)")
    t.add_argument("--batch-size", type=int, default=0, help="override batch size (0 = config)")
    t.add_argument("--stratified", action="store_true", help="stratify the well split by class")
    t.add_argument("--stride-s", type=float, default=0.0, help="window stride (0=config)")
    t.add_argument(
        "--streaming", action="store_true", help="build windows on disk (memmap) for full data"
    )
    t.add_argument(
        "--hybrid", action="store_true", help="inject engineered features into the ST-MoE"
    )
    t.add_argument(
        "--ssl-epochs", type=int, default=-1, help="SSL pretrain epochs (-1 = config, 0 = off)"
    )
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", parents=[common], help="print metrics for a run id")
    e.add_argument("run", help="run id under runs/")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    # Dispatch to the handler the chosen subparser attached via set_defaults(func=...).
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

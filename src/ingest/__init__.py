"""Layer 1 — parquet IO, schema validation, resampling (AGENT.md §5.1)."""

from __future__ import annotations

from pathlib import Path

from config import DataConfig
from ingest.instance import InstanceFrame, InstanceMeta, parse_meta
from ingest.io import SchemaError, load_instance
from ingest.resample import resample
from schema import Source

# Provenance is encoded in the filename prefix (AGENT.md §3): real field recordings
# (WELL-), physics simulations (SIMULATED_), and expert hand-drawn instances (DRAWN_).
# Globbing per-source lets `include_sources` config gate which provenances are loaded.
_SOURCE_GLOBS = {
    "real": "WELL-*.parquet",
    "simulated": "SIMULATED_*.parquet",
    "hand_drawn": "DRAWN_*.parquet",
}


def discover_instances(cfg: DataConfig, classes: list[int] | None = None) -> list[InstanceMeta]:
    """List instance metadata under ``cfg.dataset_dir``, filtered by source and class dir."""
    root = Path(cfg.dataset_dir)
    metas: list[InstanceMeta] = []
    # The 3W dataset is laid out as dataset/<event_class>/<instance>.parquet, so each
    # immediate numeric subdir is one of the 10 event classes (§3). Filter to the
    # requested classes, else scan every digit-named class directory.
    class_dirs = (
        [root / str(c) for c in classes]
        if classes is not None
        else sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()))
    
    for cdir in class_dirs:
        if not cdir.exists():
            continue
        for src in cfg.include_sources:
            for f in sorted(cdir.glob(_SOURCE_GLOBS[src])):
                metas.append(parse_meta(f))
    return metas


def load_and_resample(path: str | Path, cfg: DataConfig) -> InstanceFrame:
    """Convenience: ``load_instance`` then ``resample`` in one call.

    This is the canonical entry point: raw parquet in, gap-free regular-grid
    ``InstanceFrame`` out, ready for the feature layer.
    """
    return resample(load_instance(path), cfg)


__all__ = [
    "InstanceFrame",
    "InstanceMeta",
    "Source",
    "SchemaError",
    "load_instance",
    "resample",
    "parse_meta",
    "discover_instances",
    "load_and_resample",
]

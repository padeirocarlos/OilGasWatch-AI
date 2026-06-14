"""The typed in-memory representation of a single 3W instance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from schema import Source

# Filename grammar drives provenance + well grouping. Only real instances embed a
# physical well id (WELL-<n>) plus an acquisition timestamp; simulated/hand-drawn have none.
# WELL-00001_20140124083303.parquet | SIMULATED_00014.parquet | DRAWN_00001.parquet
_WELL_RE = re.compile(r"^(WELL-\d+)_\d+$")
_SIM_RE = re.compile(r"^SIMULATED_\d+$", re.IGNORECASE)
_DRAWN_RE = re.compile(r"^DRAWN_\d+$", re.IGNORECASE)


# An "instance" = one continuous time-series recording of a single well's sensors,
# labelled with the undesirable-event class it captures. Frozen so it is hashable and
# safe to use as a stable grouping key throughout the pipeline.
@dataclass(frozen=True)
class InstanceMeta:
    """Provenance for one instance, parsed from path + filename."""

    path: str
    instance_id: str
    well_id: str  # grouping key for the well-disjoint splitter (synthetic for sim/drawn)
    source: Source
    event_class_dir: int  # the dataset/<n> directory it came from

    @property
    def is_real(self) -> bool:
        # Eval may exclude or stratify by source (§3); real = field-acquired ground truth.
        return self.source is Source.REAL


def parse_meta(path: str | Path) -> InstanceMeta:
    """Derive provenance from a parquet file path.

    The well id is what the :class:`WellDisjointSplitter` groups on, so simulated
    and hand-drawn instances (which have no physical well) get synthetic group ids
    that can never collide with a real ``WELL-xxxxx``.
    """
    p = Path(path)
    stem = p.stem
    # Parent dir name is the event class (dataset/<n>/...); -1 flags an off-layout path.
    try:
        event_dir = int(p.parent.name)
    except ValueError:
        event_dir = -1

    # Real instances share the physical well id across recordings, so the splitter can
    # keep all data from one well on the same side of a train/eval split (no leakage).
    if (m := _WELL_RE.match(stem)) is not None:
        return InstanceMeta(str(p), stem, m.group(1), Source.REAL, event_dir)
    # Synthetic sources have no physical well; give each its own unique id (prefixed so it
    # can never collide with a real WELL-xxxxx) -> they group as singletons and never leak.
    if _SIM_RE.match(stem):
        return InstanceMeta(str(p), stem, f"SIM::{stem}", Source.SIMULATED, event_dir)
    if _DRAWN_RE.match(stem):
        return InstanceMeta(str(p), stem, f"DRAWN::{stem}", Source.HAND_DRAWN, event_dir)
    # Unknown naming: treat the whole stem as its own group to stay leakage-safe.
    return InstanceMeta(str(p), stem, f"UNK::{stem}", Source.REAL, event_dir)


@dataclass
class InstanceFrame:
    """A validated, time-indexed instance plus its metadata and quality channels.

    ``df`` is indexed by a ``DatetimeIndex`` named ``timestamp`` and always holds
    the 27 signal columns + ``class`` + ``state``. Quality indicator columns
    (``is_missing``, ``<chan>__is_frozen``) are added by :mod:`ingest.resample`.
    """

    df: pd.DataFrame
    meta: InstanceMeta
    quality_columns: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        # Number of grid ticks (rows); post-resample this is a regular cadence.
        return len(self.df)

    @property
    def well_id(self) -> str:
        # Convenience pass-through so consumers can group without reaching into meta.
        return self.meta.well_id

    def signal(self, columns: list[str]) -> pd.DataFrame:
        # Slice out a subset of channels, e.g. one MoE expert's subsystem map (§3).
        return self.df[columns]

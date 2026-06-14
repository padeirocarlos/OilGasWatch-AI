"""Authoritative 3W Dataset schema (AGENT.md §3).

This module is the single source of truth in code for column names, dtypes, the
class taxonomy, the subsystem channel map, and the transient-offset convention.
Nothing downstream may invent columns or relabel classes (AGENT.md §8).
"""

from __future__ import annotations

from enum import IntEnum

# --- Label convention -------------------------------------------------------

# 3W encodes the early-onset window of a fault by adding 100 to its class code:
# e.g. 1 (settled BSW fault) vs 101 (BSW onset transient). This lets one ``class``
# column carry both regime and onset timing. We must preserve it end-to-end so the
# transient detection head can train on the onset (AGENT.md §2 "Respect the transient labels").
#: Offset added to an event code to mark its transient (onset) window.
TRANSIENT_OFFSET = 100


class EventClass(IntEnum):
    """The 10 undesirable-event classes. Values match the ``class`` column (pre-offset).

    Codes are the official 3W taxonomy; do not renumber or relabel (AGENT.md §8).
    """

    NORMAL = 0
    # BSW = Basic Sediment & Water: the water+solids fraction of produced fluid;
    # an abrupt rise signals water breakthrough into the well.
    ABRUPT_INCREASE_OF_BSW = 1
    # DHSV = DownHole Safety Valve; spurious closure chokes flow from deep in the well.
    SPURIOUS_CLOSURE_OF_DHSV = 2
    # Slugging = intermittent gas/liquid surges in the riser; an oscillatory regime.
    SEVERE_SLUGGING = 3
    FLOW_INSTABILITY = 4
    RAPID_PRODUCTIVITY_LOSS = 5
    # PCK = Production ChoKe (surface valve regulating flow); restriction throttles output.
    QUICK_RESTRICTION_IN_PCK = 6
    # Scaling = mineral deposits gradually plugging the choke.
    SCALING_IN_PCK = 7
    # Hydrate = ice-like gas+water solids that can block a flowline at low T / high P.
    HYDRATE_IN_PRODUCTION_LINE = 8
    HYDRATE_IN_SERVICE_LINE = 9


# These faults have a discrete onset moment, so 3W marks a ``code+100`` onset window
# for them — early detection is meaningful here.
#: Classes that carry a transient onset label (``code + 100``). AGENT.md §3.
TRANSIENT_CLASSES: frozenset[int] = frozenset({1, 2, 5, 6, 7, 8, 9})

# Slugging (3) and flow instability (4) have no clean start instant — they are
# persistent oscillatory regimes — so no onset label exists for them.
#: Steady-state regimes with no clean onset (characterised by oscillation).
STEADY_STATE_CLASSES: frozenset[int] = frozenset({3, 4})

N_CLASSES = len(EventClass)

# --- Signal columns ---------------------------------------------------------

# Channels are grouped by the physical subsystem they instrument. Each group feeds
# its own MoE expert so an encoder can specialise in one subsystem's dynamics; the
# gating network then fuses experts. Prefixes: P-=pressure, T-=temperature,
# ABER-=choke opening %, ESTADO-=valve/equipment state, Q*=flow. MON=upstream
# ("montante"), JUS=downstream ("jusante") of a valve — so dP across a choke = MON−JUS.
#: 27 sensor channels, grouped by physical subsystem (used by the MoE experts).
SUBSYSTEM_CHANNELS: dict[str, list[str]] = {
    # Expert A: surface production tree — flow path from well to platform via the
    # production choke (CKP), TPT (Temperature/Pressure Transducer) and PDG-side gauges.
    "A_production": [
        "ABER-CKP",
        "P-MON-CKP",
        "P-JUS-CKP",
        "T-MON-CKP",
        "T-JUS-CKP",
        "P-MON-SDV-P",
        "PT-P",
        "P-TPT",
        "T-TPT",
        "ESTADO-M1",
        "ESTADO-W1",
        "ESTADO-SDV-P",
    ],
    # Expert B: downhole sensors — PDG (Permanent Downhole Gauge) reads P/T deep in
    # the well, plus the DHSV state. Most sensitive to reservoir-side faults.
    "B_downhole": [
        "P-PDG",
        "T-PDG",
        "ESTADO-DHSV",
    ],
    # Expert C: gas-lift & annulus — gas injected down the annulus (CKGL choke, QGL
    # flow) lightens the fluid column to boost production; faults here perturb lift.
    "C_gaslift_annulus": [
        "ABER-CKGL",
        "QGL",
        "P-ANULAR",
        "P-MON-CKGL",
        "P-JUS-CKGL",
        "ESTADO-M2",
        "ESTADO-W2",
        "ESTADO-SDV-GL",
        "ESTADO-XO",
        "ESTADO-PXO",
    ],
    # Expert D: service line (BS) — auxiliary injection line; only flow + downstream P.
    "D_service_line": [
        "QBS",
        "P-JUS-BS",
    ],
}

# The below lists are derived from SIGNAL_COLUMNS by prefix so column membership has
# one source of truth; downstream code groups channels by physical type via these.
#: Flat list of all 27 signal channels, in subsystem order.
SIGNAL_COLUMNS: list[str] = [c for chans in SUBSYSTEM_CHANNELS.values() for c in chans]

# Valve states are discrete {0=closed, 0.5=transitioning, 1=open}; interpolating
# them would invent physically impossible intermediate states (AGENT.md §3, §5.1).
#: Valve/state channels are ordinal in {0, 0.5, 1}; never interpolate them.
VALVE_COLUMNS: list[str] = [c for c in SIGNAL_COLUMNS if c.startswith("ESTADO-")]

#: Choke-opening channels, in percent.
CHOKE_COLUMNS: list[str] = [c for c in SIGNAL_COLUMNS if c.startswith("ABER-")]

# "P" matches both pressures and ABER- chokes (which also start with a letter run),
# so chokes are excluded explicitly. Stored in Pa (~1e6 scale), not bar.
#: Pressure channels (Pa).
PRESSURE_COLUMNS: list[str] = [
    c for c in SIGNAL_COLUMNS if c.startswith("P") and c not in CHOKE_COLUMNS
]

#: Temperature channels (°C).
TEMPERATURE_COLUMNS: list[str] = [c for c in SIGNAL_COLUMNS if c.startswith("T")]

# Only two flow meters exist (gas-lift QGL, service-line QBS); listed explicitly
# rather than by prefix since "Q" also is not a unique physical-type marker elsewhere.
#: Flow channels (m³/s).
FLOW_COLUMNS: list[str] = ["QBS", "QGL"]

# Everything except valve states — these are the channels that may be resampled,
# interpolated and robustly scaled (AGENT.md §2 within-instance normalisation).
#: Continuous (non-valve) channels — safe to interpolate / scale.
CONTINUOUS_COLUMNS: list[str] = [c for c in SIGNAL_COLUMNS if c not in VALVE_COLUMNS]

# ``class`` = event taxonomy (possibly +100 onset), ``state`` = auxiliary state label.
LABEL_COLUMNS: list[str] = ["class", "state"]

# Validation target: an ingested instance must contain exactly these columns —
# unknown columns raise (AGENT.md §5.1), so the schema cannot silently drift.
#: Full expected column set in a raw instance (order-independent).
ALL_COLUMNS: list[str] = SIGNAL_COLUMNS + LABEL_COLUMNS


class Source(IntEnum):
    """Provenance of an instance, parsed from its filename prefix.

    Provenance is kept so eval can exclude or stratify by source: simulated and
    hand-drawn instances are usable for augmentation/pretraining but must not be
    treated as real wells when reporting metrics (AGENT.md §3).
    """

    REAL = 0
    SIMULATED = 1
    HAND_DRAWN = 2


def decode_label(raw: int | None) -> tuple[int | None, bool]:
    """Split a raw ``class`` value into ``(event_code, is_transient)``.

    Inverts the ``+100`` onset encoding so downstream heads see a clean event code
    plus a boolean onset flag, rather than reasoning about the magic offset.

    ``101`` -> ``(1, True)``; ``1`` -> ``(1, False)``; ``None`` -> ``(None, False)``.

    Args:
        raw: The raw ``class`` cell value, or ``None`` for unlabeled samples.

    Returns:
        ``(event_code, is_transient)`` where ``event_code`` is the offset-stripped
        class and ``is_transient`` marks whether the +100 onset offset was present.
    """
    if raw is None:
        return None, False
    # Values >= 100 are onset windows; strip the offset to recover the base class.
    if raw >= TRANSIENT_OFFSET:
        return raw - TRANSIENT_OFFSET, True
    return raw, False

"""Layer 2 — physics-informed feature construction (AGENT.md §5.2)."""

from features.choke import choke_coeff
from features.differentials import differentials
from features.hydrate import hydrate_margin
from features.normalise import normalise
from features.oscillation import oscillation
from features.pipeline import FEATURE_FUNCTIONS, build_features
from features.spectral import spectral
from features.valve import valve_transitions

__all__ = [
    "FEATURE_FUNCTIONS",
    "build_features",
    "differentials",
    "choke_coeff",
    "hydrate_margin",
    "spectral",
    "oscillation",
    "valve_transitions",
    "normalise",
]

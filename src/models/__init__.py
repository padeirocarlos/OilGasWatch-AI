"""Layer 4 — baseline GBT and the ST-MoE network (AGENT.md §5.4).

Public surface of the models layer: the must-beat GBT baseline plus the ST-MoE
building blocks (per-subsystem encoders, graph fusion, MoE gating, multi-task heads)
and the assembled network. Build/prove the baseline before the network (AGENT.md §1).
"""

from models.baseline import GBTClassifier
from models.encoders import SubsystemEncoder
from models.fusion import WellGraphAttention
from models.heads import Heads
from models.moe import Gating
from models.network import STMoE, subsystem_indices

__all__ = [
    "GBTClassifier",
    "SubsystemEncoder",
    "WellGraphAttention",
    "Gating",
    "Heads",
    "STMoE",
    "subsystem_indices",
]

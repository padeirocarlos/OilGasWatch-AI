"""Typed configuration (AGENT.md §6: YAML, loaded once, passed explicitly).

No global state, no env-var magic. Each config file maps to a pydantic model so
magic numbers live in ``configs/*.yaml`` rather than function bodies (AGENT.md §2, §8).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# --- data.yaml --------------------------------------------------------------


class DataConfig(BaseModel):
    """Paths, resampling, and split policy — controls how raw parquet becomes a grid.

    These knobs live here (not in ingest code) so resampling/quality behaviour is
    reproducible from config alone, with no magic numbers in function bodies (§2/§8).
    """

    dataset_dir: str = "dataset"
    runs_dir: str = "runs"
    # 3W instances are irregularly sampled; resample onto a fixed grid so windows and
    # spectral features have a well-defined sample rate. 1 s is the native 3W cadence.
    resample_rate_s: float = Field(1.0, gt=0, description="Regular grid spacing, seconds.")
    # Short sensor dropouts are forward-filled; gaps longer than this are too long to
    # trust, so the sample is flagged is_missing rather than fabricated (§5.1).
    max_gap_s: float = Field(
        30.0, gt=0, description="Forward-fill gaps up to this; beyond -> is_missing."
    )
    # A "frozen" (stuck) sensor reads near-constant; we detect it as ~zero variance
    # over this rolling window and emit an explicit indicator channel (§2 quality).
    frozen_window_s: float = Field(
        60.0, gt=0, description="Rolling window for is_frozen detection."
    )
    # Variance threshold below which a channel counts as frozen; in (signal-unit)^2,
    # so this floor is generous given Pa-scale pressures.
    frozen_eps: float = Field(1e-9, ge=0, description="Variance below this counts as frozen.")
    # Which provenance tiers to load; eval can later exclude simulated/hand-drawn (§3).
    include_sources: list[str] = Field(default_factory=lambda: ["real", "simulated", "hand_drawn"])
    # Global RNG seed — same seed + data + config ⇒ identical metrics (§2 determinism).
    seed: int = 42


# --- features.yaml ----------------------------------------------------------


class HydrateCurve(BaseModel):
    """Antoine-style hydrate-equilibrium curve: ``P_eq = exp(a + b / (T + c))`` (Pa, °C).

    Hydrates form when (T, P) crosses into the stable region of this curve; the
    ``hydrate_margin`` feature measures distance from it. Coefficients are physical
    constants, so per §8 they are exposed here (never hardcoded) with a validation TODO.

    Defaults are placeholders. # TODO: validate against PVT data (AGENT.md §8).
    """

    a: float = 30.0  # intercept of the log-pressure curve (dimensionless)
    b: float = -8000.0  # slope term; negative so P_eq falls as 1/(T+c) grows
    c: float = 273.15  # °C→K-style offset keeping the denominator positive


class FeaturesConfig(BaseModel):
    """Window sizes, spectral bands, and physics constants for feature construction."""

    # Rolling FFT window; a power of two for an efficient transform. Longer windows
    # resolve lower frequencies (slow slugging) at the cost of time localisation.
    spectral_window_s: float = Field(256.0, gt=0)
    # Channels worth spectral analysis: the pressure/flow signals where oscillatory
    # faults (slugging, instability) show up as dominant frequencies (§5.2).
    spectral_channels: list[str] = Field(default_factory=lambda: ["P-PDG", "P-TPT", "PT-P", "QGL"])
    # Frequency bands (Hz) for band-power features; tuned to slow production dynamics
    # (sub-0.2 Hz). Band power separates severe slugging from generic flow noise.
    spectral_bands: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2)]
    )
    # Window for amplitude/period of detrended oscillation features.
    oscillation_window_s: float = Field(256.0, gt=0)
    # ε floor inside the choke coefficient Cv = Q / (opening·sqrt(max(ΔP, ε))); guards
    # against divide-by-zero when a choke sees ~zero pressure drop (§5.2, §7 edge case).
    differential_eps: float = Field(1e-6, gt=0, description="Floor for ΔP in choke coefficient.")
    # Lag (seconds) over which rolling differentials (dP/dt etc.) are computed.
    rolling_diff_window_s: float = Field(10.0, gt=0)
    hydrate_curve: HydrateCurve = HydrateCurve()


# --- model.yaml -------------------------------------------------------------


class BaselineConfig(BaseModel):
    """Gradient-boosted-tree baseline over flattened per-window features (§5.4).

    The baseline must be proven before the deep MoE is built (§1, §8).
    """

    backend: str = Field("lightgbm", description="'lightgbm' or 'sklearn'.")
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63  # LightGBM grows leaf-wise; 63 ≈ a depth-6 tree's leaf budget.
    max_depth: int = -1  # -1 = unbounded depth, leaf count alone caps complexity.
    # Faults are rare vs NORMAL; "balanced" reweights classes inversely to frequency
    # so the tree does not collapse to predicting the majority class (§5.3 imbalance).
    class_weight: str | None = "balanced"


class WindowConfig(BaseModel):
    """Sliding-window geometry for turning a time series into model samples (§5.3)."""

    # Each training sample spans 5 min of signal — long enough to see fault dynamics.
    window_s: float = Field(300.0, gt=0)
    # 50% overlap (stride = window/2): more samples and better onset coverage without
    # full redundancy.
    stride_s: float = Field(150.0, gt=0)
    # Drop windows shorter than this (e.g. at an instance's tail) so every sample is full.
    min_window_s: float = Field(300.0, gt=0)


class EncoderConfig(BaseModel):
    """Per-subsystem temporal encoder (dilated TCN / SSM); one instance per expert (§5.4)."""

    hidden: int = 64
    # Stacking blocks with growing dilation widens the receptive field exponentially,
    # so a few blocks still cover the full window.
    n_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.1


class MoEConfig(BaseModel):
    """Mixture-of-experts fusion: one expert per physical subsystem (§5.4)."""

    expert_dim: int = 64
    # Must equal the 4 subsystem groups (A/B/C/D) in schema.SUBSYSTEM_CHANNELS.
    n_experts: int = 4  # one per subsystem
    gating_hidden: int = 64  # gating net soft-routes over experts; valid with any masked.
    # Ablation toggles. use_fusion=False skips the cross-subsystem graph attention;
    # use_gating=False replaces soft routing with a plain mean over experts. Both True
    # is the full ST-MoE; flipping them measures each component's contribution.
    use_fusion: bool = True
    use_gating: bool = True


class HeadsConfig(BaseModel):
    """Three task heads share the fused trunk (multi-task; §5.4)."""

    transient_hidden: int = 64  # onset (+100) detector — the early-detection objective.
    event_hidden: int = 64  # 10-way event classifier.
    state_hidden: int = 32  # auxiliary state head; smaller as it is a side task.


class LossWeights(BaseModel):
    """Multi-task weights + imbalance-loss params: L = w_e·event + w_t·transient + w_s·state."""

    event: float = 1.0  # primary classification objective.
    transient: float = 0.5  # onset detection; weighted below event but still material.
    state: float = 0.25  # auxiliary regulariser, lowest weight.
    # Focal loss focusing parameter: γ>0 down-weights easy (well-classified) samples
    # so training concentrates on hard, rare fault windows (§5.5).
    focal_gamma: float = 2.0
    # Class-balanced loss "effective number" β; →1 makes reweighting more aggressive
    # toward rare classes (Cui et al. 2019). 0.999 suits heavy imbalance.
    class_balanced_beta: float = 0.999


class TrainConfig(BaseModel):
    """Optimisation loop hyperparameters, incl. optional SSL pretraining (§5.5)."""

    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42  # mirrors DataConfig.seed; keeps the train loop deterministic (§2).
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    # >0 enables masked-reconstruction self-supervised pretraining on NORMAL +
    # simulated + hand-drawn data before fine-tuning; 0 skips it (§5.5).
    ssl_epochs: int = 0
    # Fraction of timesteps masked during SSL reconstruction.
    ssl_mask_ratio: float = 0.5


class ModelConfig(BaseModel):
    """Aggregate of all model.yaml sub-configs (window → encoder → MoE → heads → loss)."""

    window: WindowConfig = WindowConfig()
    baseline: BaselineConfig = BaselineConfig()
    encoder: EncoderConfig = EncoderConfig()
    moe: MoEConfig = MoEConfig()
    heads: HeadsConfig = HeadsConfig()
    loss: LossWeights = LossWeights()
    train: TrainConfig = TrainConfig()

# --- loader -----------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    # Missing/empty YAML yields {} so pydantic falls back to the typed defaults above —
    # the code runs config-free while still keeping every knob overridable from YAML.
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    return DataConfig(**_load_yaml(path))


def load_features_config(path: str | Path = "configs/features.yaml") -> FeaturesConfig:
    return FeaturesConfig(**_load_yaml(path))


def load_model_config(path: str | Path = "configs/model.yaml") -> ModelConfig:
    return ModelConfig(**_load_yaml(path))

# OilGasWatch-AI

Real-time detection and classification of **undesirable events in oil & gas wells**,
built on the **3W Dataset 2.0.0** (Petrobras) and optimised for *early detection* — catching
the transient onset of a fault, not just the settled regime.

The full driving specification is in [`AGENT.md`](AGENT.md). This README is the quick-start.

## Pipeline

Layered, in dependency order (`src/`):

| Layer | Package | What it does |
|-------|---------|--------------|
| Schema/config | `schema.py`, `config.py` | Authoritative column/class/channel map; typed YAML config (`configs/*.yaml`). |
| 1. Ingest | `ingest/` | Parquet IO + schema validation, regular-grid resampling, per-sensor `__is_missing` / `__is_frozen` quality channels (gap-free output, valves carry-forward only). |
| 2. Features | `features/` | Physics-informed pure functions: cross-component differentials, guarded choke coefficients, hydrate margin, rolling spectral, oscillation, valve transitions, within-instance robust normalisation. |
| 3. Dataset | `dataset/` | `WellDisjointSplitter` (zero well leakage, asserted), sliding-window construction + labelling, class-balanced sampling / effective-number weights. |
| 4. Models | `models/` | `GBTClassifier` baseline; ST-MoE = per-subsystem dilated-TCN encoders → graph-attention fusion → MoE gating → transient/event/state heads (degrades gracefully under expert masking). |
| 5. Train | `train/` | Focal + class-balanced losses, masked-reconstruction SSL pretrain, seeded multi-task loop logging to `runs/<id>/`. |
| 6. Eval | `eval/` | Macro-F1 / per-class F1 / confusion, time-to-detection (anticipation), robustness under sensor freeze / channel dropout. |

## Setup

```bash
uv sync                      # create .venv from pyproject + uv.lock
```

## Usage

```bash
# Inspect ingestion + quality channels for a class.
uv run python src/cli.py ingest --classes 1

# Build features for one instance (optionally write parquet).
uv run python src/cli.py features --classes 1 --out /tmp/feats.parquet

# Fit + evaluate the GBT baseline on a well-disjoint split (logs to runs/).
uv run python src/cli.py baseline --limit-per-class 12

# Train + evaluate the ST-MoE.
uv run python src/cli.py train --classes 1 5 8

# Print the metrics for a logged run.
uv run python src/cli.py eval baseline-YYYYMMDD-HHMMSS
```

All knobs (resample rate, window sizes, spectral bands, hydrate-curve coefficients,
model hyperparameters) live in `configs/data.yaml`, `configs/features.yaml`,
`configs/model.yaml` — never hardcoded in function bodies.

## Reference result

GBT baseline, well-disjoint split, 12 instances/class, all 10 classes:
**macro-F1 ≈ 0.69**, transient detection rate 1.0 with **mean latency ≈ −1.3 windows**
(fires *before* the labelled onset — the early-detection behaviour we optimise for).
This is the benchmark the ST-MoE must beat (AGENT.md §7).

## Development

```bash
uv run pytest -q             # unit + leakage + reproducibility + golden-file tests
uv run ruff check src tests  # lint
uv run black src tests       # format
```

Tests that need the dataset are skipped automatically when `dataset/` is absent.

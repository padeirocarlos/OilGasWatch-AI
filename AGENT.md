# AGENT.md — 3W Early-Detection System

> Driving specification for autonomous coding agents and the human dev team building the
> Oil and Gas undesirable-event detector. Read this file top-to-bottom before writing code.
> It is the single source of truth for scope, conventions, and acceptance criteria.

---

## 1. Mission

Build a reproducible pipeline that detects and classifies undesirable events in Oil and Gas production, using the
**3W Dataset 2.0.0** (Petrobras) from multivariate sensor time series, optimised for
**early detection** (catching the transient onset, not just the settled fault).

Target deliverables, in dependency order:

1. `ingest` — load/validate parquet instances into a typed, resampled representation.
2. `features` — physics-informed feature construction (differentials, spectral, quality).
3. `baseline` — gradient-boosted-tree benchmark over engineered features.
4. `model` — physics-informed spatio-temporal mixture-of-experts (ST-MoE).
5. `eval` — well-disjoint, anticipation-weighted evaluation harness.

Ship 1→3 first and prove the feature set works before investing in 4.

---

## 2. Non-negotiable principles

- **Differentials over absolutes.** Cross-component ΔP/ΔT and normalised flow coefficients
  are the signal. Never feed a model raw absolute pressures as primary features without
  within-instance normalisation — it destroys cross-well generalisation.
- **Well-disjoint evaluation.** Train and test wells must never overlap. A split that leaks
  a well across folds is a bug, not a config choice. Enforce in code, assert in tests.
- **Respect the transient labels.** Event code `+100` marks the onset window
  (`TRANSIENT_OFFSET = 100`). Preserve it end-to-end; the transient head trains on it.
- **Quality is a feature, not a nuisance.** Missing/frozen sensors get explicit indicator
  channels. Never silently impute and forget — record what was imputed.
- **Determinism.** Every run is seeded. Same seed + same data + same config ⇒ same metrics.
- **No silent magic numbers.** Window sizes, thresholds, hydrate-curve params live in config,
  not in function bodies.

---

## 3. Dataset facts the code must encode

- 29 signal columns + 2 labels (`class`, `state`). Units: pressures Pa, temps °C, flows m³/s,
  choke openings %, valve states ∈ {0, 0.5, 1} (ordinal; 0.5 = transitioning/intermediate).
- 10 classes: `0 NORMAL`, `1 ABRUPT_INCREASE_OF_BSW`, `2 SPURIOUS_CLOSURE_OF_DHSV`,
  `3 SEVERE_SLUGGING`, `4 FLOW_INSTABILITY`, `5 RAPID_PRODUCTIVITY_LOSS`,
  `6 QUICK_RESTRICTION_IN_PCK`, `7 SCALING_IN_PCK`, `8 HYDRATE_IN_PRODUCTION_LINE`,
  `9 HYDRATE_IN_SERVICE_LINE`.
- Transient events (`TRANSIENT = True`): 1, 2, 5, 6, 7, 8, 9 → carry a `code+100` onset label.
  Steady-state regimes (`TRANSIENT = False`): 3, 4 → no clean onset; characterised by oscillation.
- Sources: real, simulated, hand-drawn. Use simulated + hand-drawn for augmentation/pretraining
  but keep them flagged so eval can exclude or stratify by source.
- Parquet engine `pyarrow`, compression `brotli`.

### Subsystem channel map (used for the MoE experts)

| Expert          |           Channels |
|-----------------|--------------------|
| A — Production  | `ABER-CKP, P-MON-CKP, P-JUS-CKP, T-MON-CKP, T-JUS-CKP, P-MON-SDV-P, PT-P, P-TPT, T-TPT, ESTADO-M1, ESTADO-W1, ESTADO-SDV-P` |
| B — Downhole    | `P-PDG, T-PDG, ESTADO-DHSV` |
| C — Gas-lift/annulus | `ABER-CKGL, QGL, P-ANULAR, P-MON-CKGL, P-JUS-CKGL, ESTADO-M2, ESTADO-W2, ESTADO-SDV-GL, ESTADO-XO, ESTADO-PXO` |
| D — Service line | `QBS, P-JUS-BS` |

---

## 4. Repository layout (target)

```
OilGasWatch-AI/
├── AGENT.md                 # this file
├── pyproject.toml           # deps + tooling config
├── docs                     # project documents files 
├── configs/
│   ├── data.yaml            # paths, resample rate, split policy
│   ├── features.yaml        # window sizes, hydrate-curve coeffs, spectral bands
│   └── model.yaml           # encoder/MoE/head hyperparameters
├── src/
│   ├── ingest/              # parquet IO, schema validation, resampling
│   ├── features/            # differentials, spectral, quality, valve transitions
│   ├── dataset/            # windowing, well-disjoint splitter, samplers
│   ├── models/              # encoders, graph fusion, MoE, heads, baseline
│   ├── train/               # loops, losses (focal/class-balanced), SSL pretrain
│   ├── eval/                # metrics, time-to-detection, robustness probes
│   └── cli.py               # entrypoints: ingest|features|train|eval
├── 01_feature_starter.ipynb   # exploratory feature builder (delivered)
└── tests/                   # unit + leakage + reproducibility tests
```

---

## 5. Module contracts

### 5.1 `ingest`
- `load_instance(path) -> InstanceFrame`: read parquet, coerce dtypes, parse timestamp to
  `Instant`, validate against the 29+2 schema. Raise on unknown columns.
- `resample(frame, rate) -> InstanceFrame`: regular grid; forward-fill ≤ `max_gap`, beyond
  that emit `is_missing` = 1. Never interpolate valve states — carry-forward only.
- Emit per-channel `is_frozen` when variance over a rolling window ≈ 0.

### 5.2 `features`
Implement each as a pure function `f(InstanceFrame, cfg) -> DataFrame` so they compose and test in isolation:
- `differentials`: `dP_CKP`, `dP_CKGL`, `dT_CKP`, `dP_grad = P-PDG − P-TPT`, `dT_grad`.
- `choke_coeff`: `Cv_eff = Q / (opening · sqrt(max(ΔP, ε)))` per choke; guard division.
- `hydrate_margin`: distance from (T, P) to the parameterised hydrate-equilibrium curve.
- `spectral`: rolling FFT → dominant frequency, spectral entropy, band-power; for `P-PDG`,
  `P-TPT`, `PT-P`, `QGL`. Window length from `features.yaml`.
- `oscillation`: amplitude + period of detrended signal.
- `valve_transitions`: Δstate flags + `time_since_change`.
- `normalise`: within-instance robust scaling using NORMAL-segment stats only.

### 5.3 `datasets`
- `WellDisjointSplitter`: groups by well id; guarantees zero well overlap; assert in tests.
- `make_windows`: sliding windows with stride; attach `class`, `state`, and transient flag.
- Imbalance handling via class-balanced sampler **or** loss weighting (configurable, not both blindly).

### 5.4 `models`
- `baseline.GBTClassifier`: LightGBM/XGBoost over flattened per-window engineered features.
- `encoders.SubsystemEncoder`: dilated TCN or structured SSM; one instance per expert.
- `fusion.WellGraphAttention`: GAT over the well topology; edge features = differentials.
- `moe.Gating`: soft routing over experts; must produce valid output with any expert masked.
- `heads`: `TransientHead`, `EventHead` (10-way), `StateHead` (auxiliary).

### 5.5 `train`
- `losses`: focal and class-balanced cross-entropy.
- `ssl_pretrain`: masked-reconstruction / forecasting on NORMAL + simulated + hand-drawn.
- Multi-task objective = `w_e·event + w_t·transient + w_s·state`; weights in config.

### 5.6 `eval`
- `metrics`: macro-F1, per-class F1, confusion matrix.
- `time_to_detection`: latency from true onset to first correct transient firing.
- `robustness`: re-score under simulated sensor freeze / channel dropout.
- Report against the official 3W benchmark definitions where applicable.

---

## 6. Coding conventions

- Python ≥ 3.11, type hints everywhere, `ruff` + `black` clean, `mypy` on `src/`.
- Configs are YAML, loaded once and passed explicitly — no global state, no env-var magic.
- Pandas/Polars for tabular, PyTorch for the deep model. Pick one array lib per module.
- Logging via `logging`, not `print`. Structured run metadata to `runs/<id>/`.
- Tests live beside the contract they verify; CI runs `pytest -q` on every PR.

---

## 7. Definition of done (per layer)

- **ingest/features:** unit tests for each function, including the ΔP-with-frozen-sensor edge
  case and the division-guard in `Cv_eff`. A golden-file test on one known instance.
- **baseline:** reproducible macro-F1 on a fixed well-disjoint split, logged to `runs/`.
- **model:** beats the baseline on macro-F1 **and** time-to-detection on the same split;
  degrades gracefully (no crash, bounded metric drop) under single-subsystem dropout.
- **always:** seeded, leakage test green, `ruff`/`mypy` clean, README updated.

---

## 8. Guardrails for the agent

- Do **not** invent dataset columns or relabel classes. The schema in §3 is authoritative.
- Do **not** merge a change that makes evaluation non-well-disjoint.
- Do **not** hardcode hydrate-curve or spectral-band constants in function bodies.
- If a physical assumption is uncertain (e.g. hydrate-curve coefficients), expose it in config
  with a cited default and a `# TODO: validate against PVT data` note — never guess silently.
- Prefer the simplest layer that satisfies the acceptance test. Build the baseline before the network.

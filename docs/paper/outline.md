# Paper outline — Honest evaluation of undesirable-event detection on 3W

> Working scaffold. Numbers in *italics with a single split, seed 42* are from our current
> runs and **must be re-run as multi-seed / k-fold mean±std** before submission (marked
> `[NEED]`). The paper's spine is **trustworthy evaluation**, not a new SOTA model.

## Candidate titles
- *"How trustworthy are 3W fault-detection benchmarks? Source-stratified, leakage-aware
  evaluation and a physics-informed hybrid."*
- *"Physics meets sequence learning for subsea well fault detection: what actually helps,
  and what the benchmark hides."*

## Target venue
Primary: *Geoenergy Science and Engineering* (home of the 3W dataset paper) or *Engineering
Applications of AI*. Secondary: *Expert Systems with Applications*, *Computers & Chemical
Engineering*, *IEEE Access*.

## Core contributions (the selling points — lead with these)
1. **A rigorous evaluation protocol for 3W**: well-disjoint splitting, **source
   stratification** (real vs simulated vs hand-drawn), and a concrete **label-leakage**
   caution. We show standard scores are inflated and propose an honest alternative.
2. **Empirical demonstration of simulation inflation**: 5/10 fault classes are evaluated
   mostly/entirely on *simulated* data; real-only macro-F1 is far lower, and most fault
   classes are **unvalidated on real wells**.
3. **A physics-informed hybrid** (engineered features × learned temporal model) and the
   finding that the two are **complementary** — neither alone detects the hardest real
   class, their fusion does.
4. **Honest ablations / negative results**: ~10 intuitive enhancements measured and
   rejected; *physics features beat architectural sophistication* in this regime.

---

## 1. Introduction
- Motivation: early detection of undesirable events in offshore O&G (safety, cost, §AGENT).
- The 3W dataset as the de-facto benchmark; the problem: reported scores may not reflect
  real-well performance.
- Gap: leakage, source composition, and well-disjointness are inconsistently handled.
- **Contributions list** (the 4 above).

## 2. Related work
- 3W dataset + toolkit; prior fault-detection methods on 3W `[NEED: literature survey]`.
- Physics-informed ML for industrial time series.
- Mixture-of-experts / TCN / hybrid feature+deep models.
- Evaluation pitfalls in time-series ML (leakage, non-i.i.d. splits).

## 3. Dataset & problem
- 3W 2.0.0: 27 channels + class/state, 10 classes, ~2228 instances, 1 Hz.
- **Source composition table** (a headline figure — already computed):

  | class | total | real | sim | drawn |
  |---|---|---|---|---|
  | NORMAL | 594 | 594 | 0 | 0 |
  | ABRUPT_BSW | 128 | 4 | 114 | 10 |
  | SPURIOUS_DHSV | 38 | 22 | 16 | 0 |
  | SEVERE_SLUGGING | 106 | 32 | 74 | 0 |
  | FLOW_INSTABILITY | 343 | **343** | 0 | 0 |
  | RAPID_PROD_LOSS | 450 | 11 | 439 | 0 |
  | QUICK_RESTRICTION | 221 | 6 | 215 | 0 |
  | SCALING_IN_PCK | 46 | 36 | 0 | 10 |
  | HYDRATE_PRODUCTION | 95 | 14 | 81 | 0 |
  | HYDRATE_SERVICE | 207 | 57 | 150 | 0 |

  → *most fault classes have ≤14 real instances; this drives the whole honest-evaluation story.*

## 4. Methodology
### 4.1 Evaluation protocol (the spine)
- **Well-disjoint** splitting (group by well id; assert zero overlap); simulated/hand-drawn
  as synthetic singleton wells.
- **Stratified** split so rare classes appear in test.
- **Source-aware** metrics: report all-source AND real-only.
- **Leakage control**: exclude instance-level diagnostics (the `__norm_fallback` case).
- `[NEED]` Promote to **k-fold (e.g. 5-fold) well-disjoint CV × ≥3 seeds**; report mean±std.

### 4.2 Physics-informed features
- 7 families (differentials, choke coefficients, hydrate margin, spectral, oscillation,
  valve transitions, within-instance normalisation) + quality channels → 139 features ×
  5 aggregates = 695-d per window (see `docs/feature_layout.md`).
- Within-instance robust scaling (cross-well generalisation).

### 4.3 Models
- **Baseline**: gradient-boosted trees (LightGBM) over the 695-d window vector.
- **ST-MoE**: per-subsystem dilated-TCN experts → graph fusion → MoE gating → multi-task
  heads (event / transient / state).
- **Hybrid**: inject the engineered vector into the ST-MoE post-pooling (the key method).

### 4.4 Metrics
- macro-F1, per-class F1 (all-source + real-only).
- **Time-to-detection** (anticipation) via the transient head `[NEED: formal stats]`.
- **Robustness** under sensor freeze / channel dropout (we have `eval/robustness`) `[NEED: run]`.

## 5. Experiments & results
### 5.1 Main comparison *(single split, seed 42 — `[NEED]` multi-seed)*
| model | macro-F1 (all) | FLOW_INSTABILITY F1 | latency |
|---|---|---|---|
| GBT (de-leaked) | *0.880* | *0.01* | *+66 s* |
| ST-MoE (raw) | *0.847* | *0.07* | *−570 s* |
| **Hybrid** | ***0.965*** | ***0.81*** | ***−396…−776 s*** |

### 5.2 The leakage finding
- `__norm_fallback` perfectly flags steady-state classes → baseline *0.944 → 0.880* when
  removed. Cautionary result for 3W feature engineering.

### 5.3 Simulation inflation / real-only evaluation (headline)
- All-source *0.880* vs real-only per-class: NORMAL *0.82*, HYDRATE_PROD *0.96*,
  FLOW_INSTABILITY *0.01 (GBT)/0.81 (hybrid)*; 5 classes unevaluable on real data.
- **Honest real-evaluable macro-F1 ≈ 0.85–0.90 (hybrid)**, not 0.965.
- `[NEED]` real-only k-fold for the classes with sufficient real data.

### 5.4 Ablations / negative results (rigor section)
- Pull the table from `docs/ablations.md`: richer pooling (−0.045), SSL (−0.067), cap-60
  fusion result that **reverses** at scale, coherence/trend (no gain), specialist (0.00).
- **Synergy table** (the conceptual centerpiece):

  | input | FLOW_INSTABILITY F1 |
  |---|---|
  | features only (GBT) | 0.01 |
  | features only (one-vs-rest specialist) | 0.00 |
  | sequence only (ST-MoE) | 0.07 |
  | **sequence + features (hybrid)** | **0.81** |

### 5.5 Early detection — `[NEED]` formalize + multi-seed.
### 5.6 Robustness — `[NEED]` run the freeze/dropout probes.

## 6. Discussion
- Why physics features dominate; why the MoE machinery's value is data-dependent.
- The real bottleneck = real-data scarcity, not modelling.
- **Limitations** (be candid): single-seed in current draft, no external baselines yet,
  binary-separability confound, hydrate-curve coefficients unvalidated.

## 7. Conclusion
- Trustworthy evaluation changes the 3W performance picture; a physics-informed hybrid is
  a strong, honest method; most fault classes need more real data, not better models.

## Reproducibility statement
- Public code, seeded, well-disjoint, config-driven; `docs/ablations.md` + `docs/feature_layout.md`.

## Appendices
- A. Full feature layout (695-d). B. Hyperparameters/configs. C. Per-class confusion matrices.
- D. Full ablation log.

---

## Experiment checklist before submission (`[NEED]`)
- [ ] 5-fold well-disjoint CV × ≥3 seeds → mean±std for §5.1, §5.3, §5.4
- [ ] Real-only k-fold for real-evaluable classes
- [ ] ≥1–2 literature/published-method baselines on the same protocol
- [ ] Run robustness probes (`eval/robustness`) and tabulate
- [ ] Formal time-to-detection definition + statistics
- [ ] External validity: train-on-simulated → test-on-real transfer numbers
- [ ] Figures: source-composition bar, real-vs-all per-class, ablation summary, latency CDF

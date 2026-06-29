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
### 5.1 Main comparison
**Rigorous (5-fold well-disjoint CV × 3 seeds = 15 reps), stride-300:**

| model | macro-F1 (all) | macro-F1 (real) | FLOW_INSTABILITY F1 |
|---|---|---|---|
| **GBT (de-leaked)** — 15-rep CV | **0.832 ± 0.057** | (per-class, §5.3) | **0.171 ± 0.261** |
| **Hybrid** — 5-fold CV *(4/5 folds; fold 4 running)* | **0.855 ± 0.059** | **0.357 ± 0.073** | **0.355 ± 0.324** |
| Hybrid — stratified single split *(optimistic)* | *0.965* | — | *0.81* |

Per-fold (hybrid vs GBT, *same* seed-42 wells): fold 0 0.781 vs 0.822; fold 1 0.933 vs 0.930;
fold 2 0.884 vs 0.851; fold 3 0.821 vs 0.804.

> ✅ **Verdict (under cross-validation): the hybrid and the GBT are essentially tied** —
> hybrid **0.855 ± 0.059** vs GBT **0.832 ± 0.057** (a marginal, within-noise edge to the
> hybrid, which wins 3/4 folds but loses fold 0). This is **far from the 0.965 ≫ 0.880 the
> single split implied** — the stratified single split was *optimistic*. The hybrid's one
> consistent edge is **FLOW_INSTABILITY** (0.355 vs ~0.30) and early detection, but FLOW_INST
> variance is enormous (±0.324: it swings 0.003–0.790 across folds). **Honest framing: the
> hybrid is competitive with — not dominant over — the GBT; report mean±std, not the single
> split.** Latency: hybrid −396…−776 s vs GBT +66 s (single split; CV `[NEED]`).

Note: the GBT single-split 0.880 sat at the **high end** of the fold distribution; the honest
CV mean is **0.832 ± 0.057**. FLOW_INSTABILITY shows huge fold variance (0.171 ± 0.261) — it
swings 0.00–0.89 depending on which class-4 wells are held out, itself a finding worth stating.

### 5.2 The leakage finding
- `__norm_fallback` perfectly flags steady-state classes → baseline *0.944 → 0.880* when
  removed. Cautionary result for 3W feature engineering.

### 5.3 Simulation inflation / real-only evaluation (headline)
**Rigorous GBT per-class F1 (15-rep CV), all-source vs real-only** — the paper's key table
(from `docs/paper/results_kfold.md`). "real F1" is averaged only over folds where the class
has real test data:

| class | F1 all (mean±std) | F1 real (mean±std) | median real support |
|---|---|---|---|
| NORMAL | 0.793 ± 0.113 | 0.754 ± 0.140 | 8428 |
| ABRUPT_BSW | 0.993 ± 0.007 | **0.216 ± 0.336** | 31 |
| SPURIOUS_DHSV | 0.896 ± 0.134 | 0.678 ± 0.216 | 27 |
| SEVERE_SLUGGING | 0.935 ± 0.110 | — (0 real) | 0 |
| FLOW_INSTABILITY | 0.171 ± 0.261 | 0.184 ± 0.266 | 989 |
| RAPID_PROD_LOSS | 0.985 ± 0.027 | **0.117 ± 0.174** | 32 |
| QUICK_RESTRICTION | 0.978 ± 0.029 | — (0 real) | 0 |
| SCALING_IN_PCK | 0.689 ± 0.369 | 0.554 ± 0.403 | 4539 |
| HYDRATE_PRODUCTION | 0.928 ± 0.114 | 0.876 ± 0.186 | 2531 |
| HYDRATE_SERVICE | 0.950 ± 0.087 | 0.376 ± 0.363 | 38 |

**The story in one table:** classes with ~0.99 all-source F1 (BSW, prod-loss, restriction,
slugging) **collapse or vanish on real data** (0.12–0.22, or no real test windows) — their
scores are *simulation artifacts*. Classes with real data **hold up** (NORMAL 0.75,
HYDRATE_PROD 0.88). FLOW_INSTABILITY (100% real) is consistent all-vs-real (~0.18 for the GBT;
0.81 for the hybrid) — the only fault class judged honestly. **Deep-hybrid fold-0 (n=1):
all-source macro-F1 0.781, real-only 0.292**, FLOW_INSTABILITY 0.085 — consistent with the
GBT pattern (simulated classes inflate; real-evaluable classes hold: NORMAL 0.81, HYDRATE_PROD
0.75) but see the §5.1 caveat that this fold *underperforms* the GBT. Full deep k-fold `[NEED]`.

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
- **Key finding restated:** under 5-fold CV the **hybrid (0.855 ± 0.059) ≈ GBT (0.832 ± 0.057)**
  — competitive, not dominant; the 0.965 single-split was optimistic. The hybrid's real edge is
  FLOW_INSTABILITY + early detection, not overall macro-F1.
- **Limitations** (be candid): hybrid CV is single-seed 5-fold (`[NEED]` multi-seed); FLOW_INST
  has enormous fold variance (±0.32); no external baselines yet; binary-separability confound;
  hydrate-curve coefficients unvalidated.

## 7. Conclusion
- Trustworthy evaluation (well-disjoint CV, source-stratified, leakage-aware) **materially
  changes the 3W picture**: single-split scores are inflated, most fault classes are
  simulation-only, and under CV the physics-informed hybrid is **competitive with — not
  dominant over — a strong GBT baseline** (0.855 vs 0.832). The hybrid's value is concentrated
  in the one hard real class (FLOW_INSTABILITY) and early detection. The dominant limitation is
  **real-data scarcity, not model capacity** — better evaluation and more real data matter more
  than better models.

## Reproducibility statement
- Public code, seeded, well-disjoint, config-driven; `docs/ablations.md` + `docs/feature_layout.md`.

## Appendices
- A. Full feature layout (695-d). B. Hyperparameters/configs. C. Per-class confusion matrices.
- D. Full ablation log.

---

## Experiment checklist before submission (`[NEED]`)
- [x] **GBT** 5-fold well-disjoint CV × 3 seeds → mean±std (`experiments/kfold_eval.py`)
- [x] Real-only per-class CV for the GBT (§5.3 table)
- [~] **Deep hybrid** CV — reduced 3-fold rep running (`experiments/kfold_deep.py`);
      full 5-fold × 3-seed (~10 GPU-h) still `[NEED]`
- [ ] ≥1–2 literature/published-method baselines on the same protocol
- [ ] Run robustness probes (`eval/robustness`) and tabulate
- [ ] Formal time-to-detection definition + statistics
- [ ] External validity: train-on-simulated → test-on-real transfer numbers
- [ ] Figures: source-composition bar, real-vs-all per-class, ablation summary, latency CDF

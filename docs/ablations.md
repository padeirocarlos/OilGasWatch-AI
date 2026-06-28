# Ablation findings & experiment log

Empirical results from tuning the ST-MoE on the 3W dataset. Read this before
"simplifying" or "improving" the model — several intuitive changes were measured and
**rejected**, and one ablation result does **not** transfer across data regimes.

All runs: well-disjoint stratified split, seed 42, GPU. "macro-F1" is over the 10
event classes on held-out wells. Reproduce model variants via `src/cli.py train` flags.

---

## ⚠️ Fusion / gating: do NOT simplify on the cap-60 result

A 2×2 ablation of the graph **fusion** (`WellGraphAttention`) and the MoE **gating**
(`Gating`) gives **opposite conclusions in two data regimes** — so a small-data ablation
must not be used to justify removing them from the production model.

**Cap-60 raw model** (no hybrid features, stride-300, 30 ep) — *mean-of-experts wins:*

| config | macro-F1 | class-4 F1 |
|---|---|---|
| full (fusion + gating) | 0.678 | 0.396 |
| fusion, no gating | 0.776 | 0.496 |
| no fusion, gating | 0.656 | 0.179 |
| **neither (mean over experts)** | **0.798** | **0.501** |

→ tempting conclusion: gating is net-harmful, fusion ~neutral, *just average the experts.*

**Full-data hybrid model** (stride-150, no-SSL, 30 ep) — *fusion + gating wins:*

| config | macro-F1 | class-4 F1 | latency (win) |
|---|---|---|---|
| **fusion + gating** | **0.9646** | **0.810** | −2.64 |
| mean-of-experts (`--no-fusion --no-gating`) | 0.9487 | 0.676 | −3.68 |

→ simplifying here **costs −0.016 macro-F1 and −0.134 class-4** (it does fire ~1 window
earlier — a minor latency edge — but classification dominates).

**Why the reversal:** the value of fusion/gating is *data-dependent*. At cap-60 the raw
model is data-starved, so the gate's extra parameters overfit/degenerate and a plain mean
is more robust. At full data — with the engineered features also present — there is enough
signal and samples for fusion+gating to help. They are **harmful when starved, helpful at
scale.**

**Decision: keep `use_fusion=True, use_gating=True` (the defaults).** Do not remove them
based on a small-data ablation. The `--no-fusion` / `--no-gating` toggles exist for
*measurement*, not as a recommended production setting.

---

## Other measured findings

| change | result | verdict |
|---|---|---|
| **`__norm_fallback` leak** removed from model features | baseline 0.944 → **0.880** (honest); it perfectly flagged the steady-state classes 3/4 | **fixed** (correctness) — earlier numbers were inflated |
| **Hybrid: inject engineered features** into the ST-MoE | raw 0.847 → **0.965**; FLOW_INSTABILITY 0.07 → **0.81** | **kept** — the single biggest win |
| **Richer pooling** (mean+std+max in `SubsystemEncoder`) | −0.045 macro-F1 vs mean-only | **rejected**, reverted to mean |
| **SSL pretraining** (masked reconstruction, `ssl_epochs>0`) | stride-150: −0.067 macro-F1, −0.63 class-4 (biases encoders toward the dominant signal) | **rejected** for classification; only helps absolute latency |
| **stride-150 vs stride-300** (hybrid) | finer detection grid + more data → best classifier *and* earlier warning | **stride-150 preferred** |
| **Coherence features** (cross-channel covariance top-eigenvalue / |corr|, for FLOW_INSTABILITY) | GBT: ~0.5% gain, class-4 F1 unchanged (0.002) | **rejected** — flow instability is not a coherent multi-channel oscillation in this formulation |
| **Trend / rolling-slope features** | GBT: used a little but noisy on the well-disjoint split → −0.015 macro-F1 | **rejected** |

### Best model to date
**Full data · stratified · stride-150 · hybrid · no SSL · fusion+gating** →
**macro-F1 0.965**, FLOW_INSTABILITY 0.81, detects onset ~6.6 min early. Beats the
de-leaked GBT baseline (0.880) on every axis.

### Per-aggregate / per-feature notes
- The GBT leans most on **mean** and **min** aggregates (~33% gain each), then **max**
  (17%), with **last** and **std** smaller (~9% each). `std` is small *because* dedicated
  oscillation/spectral features already carry variability — see `docs/feature_layout.md`.

---

## Data/label audit & the honest performance picture

After feature/architecture levers stopped paying off, we audited *why* FLOW_INSTABILITY
(class 4) is the weak class. The audit changed the conclusion twice — record it so the
reasoning isn't repeated.

### Real-only re-evaluation exposes heavy simulation inflation
Re-scoring the de-leaked GBT's test set on **real instances only** (same trained model):

| | macro-F1 |
|---|---|
| all-source test | 0.880 |
| real-only test | 0.306\* |

\*The 0.306 is itself misleading — 5 fault classes have **0–86 real test windows**, so they
score F1=0 for lack of data, not for model failure. Per-class on classes with real data:

| class | F1 all | F1 real | real test windows | source |
|---|---|---|---|---|
| NORMAL | 0.87 | 0.82 | 4600 | real |
| ABRUPT_BSW | 0.999 | — | 0 | sim |
| SEVERE_SLUGGING | 1.00 | — | 0 | sim/mix |
| **FLOW_INSTABILITY** | 0.01 (GBT) / 0.81 (hybrid) | same | 1863 | **100% real** |
| HYDRATE_PRODUCTION | 0.98 | 0.96 | 2379 | sim-trained, holds |
| (BSW, slugging, prod-loss, restriction, scaling) | ~0.99 | n/a | ≤86 | **mostly simulated** |

**Takeaway:** the headline 0.88/0.96 is **inflated by simulated data** — most fault classes
are evaluated mostly/entirely on *simulated* instances and are **unvalidated on real wells**.
FLOW_INSTABILITY is simply the one fault class with abundant real data and therefore the only
one judged honestly. Honest **real-evaluable** macro-F1 for the hybrid is **≈ 0.85–0.90**,
not 0.965.

### The class-4 specialist fails — "headroom" was an artifact
A binary GBT (4-vs-NORMAL, real, well-disjoint) scores 0.90, suggesting recoverable headroom.
But a **one-vs-rest** specialist (4 vs *all 9 others*, full data, well-disjoint) scores
**F1 = 0.000** — it cannot recognise class-4 against the full population on held-out wells.
So the 0.90 was the **two-population (operating-point/well-type) confound**, not real
separability. Confirmed by the full picture:

| approach | class-4 F1 |
|---|---|
| GBT multiclass (features only) | 0.01 |
| GBT one-vs-rest specialist (features only) | 0.00 |
| raw ST-MoE (sequence only) | 0.07 |
| **hybrid (sequence + features)** | **0.81** |

Neither features alone nor sequence alone can detect FLOW_INSTABILITY — **only their joint
representation** (the hybrid) can. The hybrid's **0.81 is the genuine, well-generalising
ceiling** for this all-real class; specialists and features add nothing.

---

## Modeling effort: CLOSED

Conclusions after exhausting features, architecture, pooling, SSL, fusion-simplification,
coherence/trend features, ensembling, and the class-4 specialist:

1. **Two changes produced essentially all the value:** the `__norm_fallback` **leak fix**
   (honesty) and the **hybrid feature injection** (the sequence×feature synergy, 0.85→0.96).
2. **FLOW_INSTABILITY at 0.81 (all-real) is near its achievable ceiling** — not a labelling
   floor, but no modelling lever recovers more; the apparent "headroom" was a confound.
3. **The dominant real-world limitation is data composition, not modelling:** 5 of 10
   classes are simulation-only and unvalidated on real wells. Further real accuracy needs
   **real fault data**, which no model change can substitute for.
4. **Report honestly:** lead with real-evaluable macro-F1 (≈0.85–0.90) + the source caveat,
   not the simulation-inflated 0.965.

**Recommended config (frozen):** full data · stratified · stride-150 · hybrid · no SSL ·
fusion+gating. Recommended `model.yaml`: `ssl_epochs: 0` (SSL hurts classification).

---

## Reproducing the variants

```bash
# full ST-MoE (default)
uv run python src/cli.py train --stratified --stride-s 150 --streaming --hybrid --ssl-epochs 0
# mean-of-experts ablation
uv run python src/cli.py train ... --no-fusion --no-gating
# raw (no engineered features): drop --hybrid ;  SSL on: --ssl-epochs 10
```

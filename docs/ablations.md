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

### Best model to date
**Full data · stratified · stride-150 · hybrid · no SSL · fusion+gating** →
**macro-F1 0.965**, FLOW_INSTABILITY 0.81, detects onset ~6.6 min early. Beats the
de-leaked GBT baseline (0.880) on every axis.

### Per-aggregate / per-feature notes
- The GBT leans most on **mean** and **min** aggregates (~33% gain each), then **max**
  (17%), with **last** and **std** smaller (~9% each). `std` is small *because* dedicated
  oscillation/spectral features already carry variability — see `docs/feature_layout.md`.

---

## Reproducing the variants

```bash
# full ST-MoE (default)
uv run python src/cli.py train --stratified --stride-s 150 --streaming --hybrid --ssl-epochs 0
# mean-of-experts ablation
uv run python src/cli.py train ... --no-fusion --no-gating
# raw (no engineered features): drop --hybrid ;  SSL on: --ssl-epochs 10
```

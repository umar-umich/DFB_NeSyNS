# Framework results — the three-view detector, and what limits it

Consolidated report for the branch-combination phase. Every number is video-level, computed
under the single video-identity rule (`analysis/tbiom/video_id.py`), with `τ` frozen once per
model at its own EER on FF++ val and applied unchanged to every test set. Multi-seed rows are
means over three seeds (42 / 1337 / 7).

Companion files: `FULL_RESULTS.md` (the raw 11 × 8 grid), `FUSION_OPERATORS.md` (nine operators
× every dataset), `SEED_SUMMARY.md` (seed spreads), `BRANCH3_REPORT.md` (how Branch 3 was
arrived at), `STAGE7*.md` (the conflict-aware line).

---

## 1. The framework

Three evidence branches over one image, fused by a subjective-logic operator:

| view | what it reads | dim |
|---|---|---:|
| **semantic** | CLIP ViT-L/14 + LoRA | 64 |
| **artifact** | β-VAE manifold projector; orthogonal residual of `f_s − f_c` | 64 |
| **Bona-fide face prior** | frozen real-face prior | 1024 |

Each emits Dirichlet evidence through an `EvidentialHead`; the three opinions are combined and
projected to `p(fake)`. Trained on FF++ c23 only, 20 epochs, batch 128, seed 42/1337/7.

Two Branch-3 variants are reported throughout because they behave differently:

| arm | Branch 3 | standalone AUROC |
|---|---|---:|
| **B** | our LoRA preservation student | 0.8314 |
| **C** | discriminative Br3 (externally fine-tuned ViT-L) | 0.8539 |

---

## 2. Headline

| configuration | mean OOD AUROC | vs P0-DS | mean FPR_real | seed spread |
|---|---:|---:|---:|---:|
| P0-DS (2-view baseline) | 0.8633 | — | 0.247 | — |
| best single branch (artifact) | 0.8566 | −0.0067 | 0.206 | ±0.0089 |
| 3 views, simple CE fusion | 0.8639 | +0.0006 | **0.316** | — |
| **B, 3 views, DS** | 0.8779 | +0.0145 | **0.148** | **±0.0025** |
| **B, 3 views, Murphy** | **0.8805** | **+0.0172** | **0.138** | ±0.0025 |
| C, 3 views, DS | 0.8780 | +0.0147 | 0.164 | ±0.0083 |
| **C, 3 views, Murphy + fitted q** | **0.8825** | **+0.0192** | 0.175 | ±0.0083 |
| C, 2 views, Murphy | **0.8841** | **+0.0208** | 0.177 | ±0.0056 |

**The claim the data supports:** the three-view evidential framework beats the two-view baseline
by **+0.015 to +0.021** mean OOD AUROC and roughly **halves the false-alarm rate** (0.138–0.177
against 0.247). That margin is far outside the seed spread.

**The claim it does not support:** any ordering *among* the top configurations. Those differ by
0.002–0.006 against seed spreads of 0.0025–0.0083.

---

## 3. What each component contributes

| step | mean OOD AUROC | gain |
|---|---:|---:|
| best single branch | 0.8566 | — |
| + a second view (P0-DS) | 0.8633 | +0.0067 |
| + a third view, **simple CE** fusion | 0.8639 | +0.0006 |
| + a third view, **evidential** fusion | 0.8779 | **+0.0146** |
| + per-view reliability (`Murphy + q`) | 0.8825 | +0.0046 |

Adding a third view under **cross-entropy buys +0.0006 — nothing.** Adding the same third view
under evidential fusion buys **+0.0146**. The gain is in how the views are combined, not in
having a third one.

The per-branch numbers say why fusion is needed at all: semantic 0.8552, artifact 0.8566,
FS-VFM 0.8539 — three views of near-identical strength, and every fusion beats all of them.

---

## 4. Limitations

### 4.1 The semantic view is redundant on arm C, and two methods agree

Given freedom to weight the views, the data removes semantic:

| method | parameterisation | fitted weight/reliability for semantic |
|---|---|---:|
| Learned (F2) | linear weights, softmax | **0.000** |
| Shafer discounting | `b′ = q·b`, `u′ = (1−q)+q·u` | **0.00 – 0.05** |

Two independent parameterisations, same conclusion. Semantic and artifact both derive from the
same CLIP trunk, and Dempster–Shafer assumes **independent** sources — feeding it two correlated
views double-counts them.

**Consequence:** on arm C the framework nominally has three branches and effectively has two.
Arm B does not have this problem — its weaker Branch 3 leaves genuine work for all three, and
fitting `q` there makes things *worse* (0.8805 → 0.8793).

**What fixes it, and what does not.** Reliability from **accuracy** (`q = 2·AUROC − 1`) does not
work: all three views score 0.71–0.74 on dev, so accuracy cannot tell them apart. Redundancy is
the problem, and only a **fitted** `q` sees it. A principled redundancy-aware `q` — from
inter-view correlation rather than from a fit — is unexplored and is the obvious next step.

### 4.2 Nothing separates the top configurations

| source of variation | effect on mean OOD AUROC |
|---|---:|
| operator choice (nine operators) | 0.0060 – 0.0115 |
| view choice (3-view vs 2-view) | 0.0032 |
| seed (same config, new init) | 0.0025 – 0.0166 |

All three are the same size. "Murphy beats DS by +0.0026" is inside one seed's noise. Settling
any of these needs 5+ seeds per cell — free in GPU terms for the operators (replay only), but a
training run per seed for the arms.

### 4.3 Deepfake-Eval-2024 is a ceiling nothing lifts

Every configuration sits at **0.68–0.69**; P0-DS is 0.6922. The best three-view number
(0.6931, Murphy) is +0.0009 — nothing. On modern in-the-wild data the framework does not help,
and this is the dataset a reviewer is most likely to ask about.

### 4.4 Conflict-aware fusion was tested and failed

Nine operators — DS, Yager, Dubois–Prade, PCR6, Deng-entropy, Murphy, CCF, averaging, learned
weights. **None fixes the suppression** that motivated Stage 7: on every suppressed dataset the
best conflict-aware operator is beaten by plain simple fusion. Two results do hold:

- **CCF is consistently worst** (0.8724 on arm B 3-view; 0.8782 on arm C 2-view) because its
  consensus step takes the **minimum** belief across sources — one low-belief view caps the
  fused belief. Protective on the real side, costly for ranking.
- **Dubois–Prade reproduces Yager exactly** (max |diff| = 0.00e+00), which is the two-class
  algebra rather than a coincidence: `{real}` and `{fake}` are the only conflicting pair and
  their union *is* Θ, where Yager already sends conflict. It would differ only on ≥ 3 classes.

### 4.5 Standalone strength does not predict framework contribution

FSFM's fine-tuned checkpoint beats P0-DS standalone on DFDCP (+0.048) and CDFv3 (+0.034). In
the framework, arm C's 3-view DS (0.8780) is **level with** arm B's (0.8779), whose Branch 3 is
the weaker student. A stronger view is not automatically a better branch.

### 4.6 Smaller, but real

- **DFD does not reproduce FSFM's reported standalone number**: ours 0.8651 against their
  0.9717, where CDFv2 agrees to 0.032. Protocol difference, not diagnosed.
- **Arm B checkpoints were selected on best FF++ val AUROC**, not the VALmix macro-AUROC
  protocol used earlier. Candidates were within 0.0001 so it cannot move the headline, but it is
  a deviation.
- **P2a's third-view vacuity remains inference**, not measurement — it exported no per-view
  operator evidence. Its checkpoints survive, so the direct measurement is runnable.
- **`concat_ce`** (concatenate the three features, one CE head) is implemented but **not
  trained**; arm D (mean-of-logits + CE) stands in as the simple-CE baseline.

---

## 5. What to fix next, in order

1. **A redundancy-aware reliability.** Derive `q` from inter-view correlation on dev data rather
   than fitting it, so the framework down-weights a redundant view for a stated reason instead of
   because an optimiser found it. This is the direct fix for §4.1 and needs no GPU.
2. **More seeds on the operators.** Replay-only, no training. Would settle §4.2's operator
   question at zero compute cost.
3. **Deepfake-Eval-2024.** Nothing in this phase moves it. It likely needs different training
   data rather than a different fusion rule.
4. **Decouple semantic from artifact.** They share a CLIP trunk, which is the root cause of
   §4.1. A genuinely independent second view would let all three contribute on arm C too.

---

## 6. Reproduction

| what | where |
|---|---|
| full grid | `analysis/tbiom/full_results.py` → `FULL_RESULTS.md` |
| nine operators | `analysis/tbiom/fusion_operators.py` → `FUSION_OPERATORS.md` |
| discounted fusion | `analysis/tbiom/discounted_fusion.py` |
| seed aggregation | `analysis/tbiom/seed_summary.py` → `SEED_SUMMARY.md` |
| figures | `analysis/tbiom/make_figures.py`, `make_embeddings.py` → `figures/` |
| training configs | `DISCERN_Ext/eval_adaptation/configs/run1*.yaml`, `armB_*.yaml`, `armC_*.yaml` |
| exports | `logs/tbiom/stage23/*.csv` |

Environments: `discern_ext` (unmodified, peft 0.14.0 / diffusers 0.32.2 pinned) and
`discern_ext_timm` (a clone plus `timm`, for the FS-VFM ViT).

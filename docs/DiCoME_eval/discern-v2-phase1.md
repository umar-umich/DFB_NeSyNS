# Claude Code — DISCERN v2 Integration, Phase 1 instructions

**Scope of THIS task**: Track A (analysis that selects the branches) + Track B scaffolding
and ports. **Do not run D0–D5 training** — those are UMAR-RUNS after Track A returns and
the projector is selected. Produce code, analysis, plots, and a written selection
recommendation. No architecture is committed until the analysis lands.

Read `README-discern-v2-integration.md` first for the phase framing, the branch contract,
and the parked-items list. The north star and success criterion there govern this task.

## Contract (do not violate)

- 🟢 You write: code, configs, scripts, analysis, plots.
- 🔴 Umar runs: all training/eval, dataset conversions.
- 🟡 Ask Umar before changing: hyperparameters, seed, eval sources, data paths, scope,
  and the D1 projector choice.
- **No invented numbers.** Leave result cells as `TODO(run)` until a real run fills them.
- **Port from the validated pilot code, not from memory.** If you cannot locate the
  original pilot implementation of a mechanism, stop and ask — do not reconstruct it.
- Work in the DISCERN repo (confirm path with Umar; assumed `../DISCERN/`). Do not modify
  DiCoME core config/model — copy mechanisms out.

---

## Track A — analysis (write the scripts; Umar runs them)

Order matters: A3 is cheapest and gates the most, so make it runnable first.
Sequence: **A3 → A1 → A2**.

### Threshold discipline (applies to A1 and A2 — read before writing either)

Do **not** choose an operating threshold per OOD source. Optimizing a threshold on the
OOD test source leaks the target distribution and weakens the zero-shot claim (this bites
hardest on CDFv3 / DF40 / Deepfake-Eval-2024).

- Use **one** threshold determined only from the permitted training/validation protocol
  (FF++ train / approved val slice) and **freeze it for every OOD dataset and generator**.
- If a suitable single threshold is unavailable, report **both** the threshold-dependent
  rescue/harm numbers **and** threshold-free score-based analyses — but never tune a
  threshold on the OOD test source itself.

### A3 — rate-response + embedding structure (run first)

Compute the P1d rate-response `R(x) = [r_β1, …, r_βM]` on DF40 and evaluate **three**
questions — family clustering is not the only thing that keeps P1d alive:

1. **Forgery separability** — real/fake linear probe from `R(x)`.
2. **Family structure** — silhouette / Fisher ratio across FS/FR/EFS/FE (descriptive).
3. **Complementarity / error-predictiveness** — P0-error prediction from `R(x)`, and
   rescue/harm separation.

Plus UMAP + per-family distribution plots (descriptive).

- **Held-out-generator discipline for the supervised probes.** The real/fake probe (Q1)
  and the P0-error probe (Q3) must **not** be trained and evaluated on the same DF40
  generators — use generator-held-out evaluation, same discipline as A2. UMAP and
  silhouette/Fisher are descriptive and can run on the full set.
- Output: `analysis/A3_rate_response/` with plots + a one-paragraph verdict written from
  the actual statistics, not asserted.
- **Revised gate**: P1d survives if its rate response provides **incremental
  forensic/applicability information** on any of the three questions — even if families do
  **not** form clean clusters (e.g. it may separate real/fake or P0-correct/P0-error
  without naming the family). Only if `R(x)` adds no incremental signal on any question
  does P1d lose privileged status.

### A1 — sample-level complementarity matrices

Across `{P0, P1a, P1b, P1c, P1d, P2a, P3a, P4}`, at video level, per source, using the
**single frozen threshold** above:

- **rescue(op)** = fraction where P0 wrong, op right
- **harm(op)** = fraction where P0 right, op wrong
- **error-overlap(op_i, op_j)** = agreement on the wrong samples
- evidence correlation between branches

Output: `analysis/A1_complementarity/` matrices as CSV + heatmaps. Break out the inversion
rows (danet/mcnet/tpsm/facevid2vid-cdf) and heygen explicitly — they are the
decision-relevant cells.

### A2 — oracle headroom + realizable gate (feasibility vs deployment)

**Oracle metrics** (primary — a label-aware oracle makes AUROC conceptually awkward, so
AUROC is not a headline number here):

- oracle **accuracy** and **balanced accuracy**
- **union-of-correct coverage**: `OracleCoverage = P(∃b : ŷ_b = y)`
- **remaining shared-error rate** (no expert correct)
- Oracle-routed AUROC, if computed, goes in **supplementary only**, explicitly labeled
  *label-peeking / non-deployable*.

**Realizable gate target — learn specialist utility, not an expert-ID label.** For each
specialist `b`, define per-sample utility relative to the visual baseline:

```
Δℓ_b = ℓ_vis − ℓ_b        # positive → specialist b improves on the visual baseline
                          # negative → specialist b harms
```

The gate predicts `Δℓ_b` (or its sign / a calibrated version) from the **label-free rich
evidence state**:
`q = g(e_vis, u_vis, e_b, u_b, C_vis,b, r_b, response-stats)`.
This maps naturally to `q_b(x)` later and handles rescue/harm better than a vague
expert-ID classifier.

- **`Δℓ_b` is a training-time TARGET only** (it is computed from the loss, which uses the
  label). The gate's **inputs stay label-free** — never feed `Δℓ_b`, the label, or
  generator/dataset/family ID as an inference feature.
- **Not** P0 probability as the primary feature (P0 is confidently wrong on the inverted
  rows).
- Keep the gate deliberately **simple** (logistic / 1-hidden-layer MLP). The realizable
  gate reports **AUROC normally** on the detection task.

Split the gate work into two explicit levels:

- **A2a — feasibility diagnostic (this phase).** Leave-one-generator-out OOD
  cross-validation over the DF40/CDFv3 generators. Question: *can observable evidence
  predict specialist utility at all?* This is the **Phase-1 go/no-go for whether to build
  D4** — if even the cross-generator gate can't recover oracle headroom, applicability is
  not inferable and D4 is a negative result. Report per held-out generator.
- **A2b — deployment-valid protocol (spec now, run later).** The final DISCERN gate must
  be trained using **only data permitted by the generalization protocol** (FF++ train /
  internal manipulation holdouts / approved augmentation), with **no generator ID, dataset
  ID, or forgery-family label** as inputs at inference. A2a is an optimistic feasibility
  ceiling (the gate has seen sibling generators); A2b is the real deployment test. Any D4
  built on A2a alone must carry that caveat in the paper.

**Per-fold hygiene (critical for the LOGO diagnostic).** In every A2 (and A3-probe) fold,
fit normalization / scalers / calibration **only on that fold's training generators** —
never globally before splitting, or the leave-one-generator-out diagnostic leaks.

**Gate-Recovery** (report in **balanced-accuracy** terms):
`Recovery = (BA_gate − BA_base) / (BA_oracle − BA_base)`, per candidate set and per
held-out generator. Candidate sets: start `{P0, P1d, P2a}`, then the strongest-rescue P1
variant, e.g. `{P0, P1b, P2a}`.

Output: `analysis/A2_gate/` with oracle-metric tables, the A2a recovery table, the A2b
protocol spec, and a written go/no-go for D4/D5.

### Track A deliverable back to Umar — `analysis/SELECTION.md`

Answer these seven questions, each from the actual numbers:

1. Does manifold-residual evidence add information beyond direct CLIP evidence?
2. Which P1 projector provides the most **usable complementary** evidence (not highest
   standalone AUC)?
3. Does P2a contribute errors/rescues sufficiently different from the manifold and visual
   branches?
4. What is the oracle headroom (coverage, balanced-accuracy gain)?
5. Can a simple observable gate recover a meaningful fraction of it (A2a Gate-Recovery)?
6. Does `R(x)` predict forgery, failure, or applicability even without family clusters?
7. Therefore, which two branches enter D1–D3, and is D4 justified?

**Flag the projector choice as 🟡 ASK-UMAR** — recommend, do not commit.

---

## Track B — scaffolding and ports (write in parallel with A)

### B1 — branch interfaces (independently toggleable + centralized Dirichlet)

Each branch fundamentally outputs:

```
evidence:    [B, 2]   # real/fake
features:    [...]    # representation for analysis/gate
diagnostics: {...}    # branch-specific
```

**Do not let each branch compute its own α/S/p/u.** One common DISCERN utility derives
them centrally:

```
α_b = e_b + 1 ;  S_b = Σ_k α_{b,k} ;  p_b = α_b / S_b ;  u_b = K / S_b
```

The public interface may still return `evidence, alpha, p, vacuity, features`, but the
four evidential quantities come from the shared utility — this prevents subtle
inconsistencies between DiCoME-ported and DISCERN-native branches.

**Each branch must be independently toggleable by config** so the D1 sub-ablation
(`D1-V` visual only / `D1-M` manifold only / `D1-VM` both) is a config switch, not a code
change. Three branch classes: `VisualEvidenceBranch`, `ManifoldEvidenceBranch` (swappable
projector), `ProcessEvidenceBranch` (P2a). Wire into DISCERN's existing EDL abstraction
and video-aggregation path. Keep the CLIP branch unchanged for now.

### B2 — port the manifold-projector interface + validated projectors (incl. β-VAE control)

Do **not** hard-assume MR-VAE — Track A may select P1a or P1b. Port the common
manifold-projector interface and the validated implementations under one DISCERN interface:

```
projector:
    beta_vae             # original DiCoME β-VAE — port as the manifold CONTROL
    deterministic_ae     # P1a
    beta_tcvae           # P1b
    mr_vae               # P1d  (default placeholder only)
```

- Add the **original β-VAE** so we have β-VAE / AE / β-TCVAE / MR-VAE under exactly the
  same interface — it is the reference control the P1 variants were defined against.
- MR-VAE remains the **wiring placeholder** default; projector selection stays
  **config-driven** and swapping is a one-line change.
- WAE (P1c) stays **unported** unless Track A unexpectedly selects it.
- Do not hardwire MR-VAE-specific assumptions (rate-distortion, β-schedule) into the
  branch or fusion. Best standalone projector ≠ best specialist for a gated architecture —
  P1b's larger FR rescue could beat P1d in the final system.
- Port from validated pilot code, not from memory.

### B3 — port the P2a SDXL-VAE process residual (strict augmentation rule)

Port the AEROBLADE-style SDXL-VAE reconstruction-residual extraction exactly from the P2a
pilot: `x → SDXL-VAE → x̂ → r_LDM → e_proc`. Frozen VAE.

**Caching hazard — be strict.** The residual `r_LDM(x) = D(x, x̂)` corresponds to a
particular image. If DISCERN applies stochastic spatial/color augmentation to `x` but the
process branch loads a cached residual computed from the *unaugmented* image, the two
branches no longer describe the same observation (we've hit this clean-feature /
augmented-image mismatch in DISCERN before).

- Cache process residuals **only** when the process-branch input is deterministic and
  exactly aligned with the cached sample.
- **Never** pair cached clean residuals with independently augmented visual inputs without
  explicitly validating that design.
- For the first integration, **reproduce P2a preprocessing exactly** before optimizing
  runtime.

### Config flags (add, defaulted per README)

`semantic_v1=false`, `structural_sem_v1=false`, `handcrafted_freq=false`, `ccv=false`,
`symbolic_rules_v1=false`, `manifold_v2=true`, `process_v2=true` — old branches stay
available for ablation but off the v2 forward path.

---

## Explicit stop points

1. After B1–B3 land, the next runnable step is **D0 (DISCERN v1 reproduction)** —
   that is UMAR-RUNS. Hand it over; do not launch training. **The D0 config must
   explicitly restore v1**: `semantic_v1=true`, `structural_sem_v1=true` (and the other v1
   branches on), `manifold_v2=false`, `process_v2=false` — so a stripped v2 configuration
   can never be mislabeled "D0."
2. Do not implement **D4 applicability-aware fusion** until Track A's **A2a Gate-Recovery**
   verdict is in and Umar greenlights. Near-zero recovery → D4 is a written negative
   result, not code.
3. Do not touch P3a, P4, SBI, or VALmix retraining — all parked (see README).

## What to hand back at the end of Phase 1

- Track A: the three analysis dirs + `SELECTION.md` (seven answers, 🟡 for sign-off).
- Track B: the branch interfaces (independently toggleable, centralized Dirichlet utility)
  + ported β-VAE / P1a / P1b / P1d projectors and the P2a module; the explicit-v1 **D0
  config** ready for Umar to run; and **D1-V / D1-M / D1-VM** sub-configs stubbed so the
  visual-necessity ablation is a config switch once the projector is chosen. Note the
  confirmed DISCERN repo path.
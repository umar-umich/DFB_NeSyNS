# Claude Code — DISCERN v2 Phase 2: three-slot spatial forensics (audits + build)

**Supersedes** the projector-selection Track A/B of Phase 1. The swappable-manifold-projector
D0–D5 ladder is retired: the β-TCVAE projector received no generative objective (the bug),
so the reals-only **frozen reference** replaces the co-trained manifold branch entirely.
This phase builds and audits the three-slot architecture and its reasoning layer.

## The one principle

The contribution is the **reasoning layer** — applicability-aware fusion + explicit
reliability separation — **not** the priors. A prior enters only if it carries conditional
forensic information beyond what's already in, measured on evidence we have or a cheap audit
here. "We combine several foundation models" is explicitly *not* the story.

## Target architecture (three SLOTS, not three fixed branches)

```
CLIP ViT-L/14 (generalist anchor)   → e_sem
Real-only frozen reference           → e_ref     (Specialist 1)
P2a LDM-AE process consistency       → e_proc    (Specialist 2)
        │
        ▼
Utility-supervised applicability → EDL fusion → (V, C, A) → Real / Fake / Defer
```

Any specialist that fails its audit is **removed**, not kept for richness. If P2a fails,
the slot opens for a later earned specialist (DINO/frequency) — it is not auto-filled.

## Contract (do not violate)

- 🟢 You write: code, configs, scripts, analysis, plots.
- 🔴 Umar runs: all training/eval, dataset conversions.
- 🟡 Ask Umar before changing: hyperparameters, seed, eval sources, data paths, scope; and
  the encoder decision (Task 0) and any specialist promotion.
- **No invented numbers** — result cells `TODO(run)` until a real run fills them.
- **Port from validated code, not memory.** Can't find the original? Stop and ask.
- **FF++-only fairness protocol holds.** No external real corpus (VGGFace2 etc.) this cycle;
  if ever, it's a clearly-separated supplementary track.
- Work in the DISCERN repo (confirm path; assumed `../DISCERN/`).

---

## Task 0 — encoder decision (cheap; gates the reference's input space)

The reals-only reference must be fit on a **stationary** CLIP feature space — that is the
root lesson of the bug (a frozen projector reading a drifting encoder is miscalibrated).
Measure how much LN/LoRA tuning contributes to the visual branch on the conventional axis:

- Compare LN-tuned-CLIP visual vs frozen-CLIP visual (e_sem only).
- Within noise → **single frozen CLIP** for both `e_sem` and the reference input `f_0`
  (simplest; reference input is trivially stationary).
- Material drop → **dual encoder**: `E_vis` (LN-tuned) for `e_sem`, `E_ref` (frozen
  snapshot) for `f_0` and the reference.
- Either way, **the reference is always fit on a frozen CLIP feature space.** 🟡 report the
  number and let Umar lock the encoder before Stage II.

## Common infrastructure (build once)

- **Centralized Dirichlet utility.** Branches output `evidence / features / diagnostics`;
  one shared util computes `α=e+1, S=Σα, p=α/S, u=K/S`. `u` is evidential **vacuity**, not
  auto-labeled epistemic uncertainty.
- **Residual calibration.** Standardize each branch's residual with **authentic-training**
  stats, `r̃=(r−μ_R)/(σ_R+ε)`, `μ_R,σ_R` fit on FF++ reals only, before the evidence head.
  For process trajectories keep `[μ,σ,max,upper-pct,slope,…]` but normalize with frozen
  training stats.
- **Guards that actually run.** Wire `assert_projector_config` into the construction/training
  path. Add tests: reference params receive **zero** gradient under the frozen protocol, and
  nonzero gradient only in the offline reals-only fitting stage. This is the guard that would
  have caught the original bug.
- **Threshold discipline.** One threshold from FF++ train/val, frozen across all OOD sources;
  never tuned per OOD source. No threshold available → report threshold-free score analyses.
- **Per-fold hygiene.** Fit scalers/normalization/calibration only on each fold's training
  generators, never globally before splitting.

---

## The four audits (write the scripts; Umar runs them)

### E1 — Reference audit: C0 → C1 → C2 → C3 (CLIP-feature space, real-only, frozen)

- **C0** current discriminative projector — relabel honestly as "discriminatively-learned
  residual transform" (the bugged arm, kept as a control, not discarded).
- **C1** frozen **random-matched** projector — same dims, normalization, head. Capacity floor.
- **C2** real-only **PCA / linear** reference — the cheapest honest reference.
- **C3** real-only **deterministic AE** — nonlinear, frozen after the reals-only fit.
- All fit on **FF++ authentic training features only**, frozen; `r_ref = f_0 − P(f_0)`;
  calibrated; → `H_ref` → `e_ref`. Extract magnitude and angular disagreement
  (`‖f−f̂‖`, `1−cos(f,f̂)`).
- **Reference-fitting objective is itself a choice.** DiCoME's aligned-VAE loss (the gap
  analysis corrected our earlier reading) is `1 − cos(z.detach(), ẑ)` + KL — **direction-only,
  magnitude-free**, and its target is detached. For the C2/C3 fit, evaluate both a cosine
  (direction-only) and an MSE (magnitude-aware) reconstruction target — with cosine, the
  residual magnitude carries the signal; with MSE it doesn't, which changes what `r_ref`
  measures. Borrow the **loss form**; do **not** borrow the co-training — ours fits on reals
  only and freezes, which is the whole point (see the out-of-scope note below).
- **Mandatory domain-detector audit (gates entry).** For each arm, quantify/plot
  `p(r | R_FF++)`, `p(r | R_OOD)`, `p(r | F_OOD)`. If OOD-real shifts as much as fake, the
  branch is a **dataset detector**, not a forensic residual — it fails, regardless of AUC.
- **Promotion criterion is conditional information beyond CLIP** (rescues CLIP errors, makes
  different errors, stable on OOD real) — **not** standalone AUC.
- Escalate to real-only **β-VAE / MR-VAE + rate-response** only if C2/C3 earn it. The C1 floor
  matters: if C1 random ≈ C0, the current branch is capacity-only; that does **not** imply
  C2/C3 fail — they test a different hypothesis, which is why they run in the same sweep.

### E2 — Process audit (P2a)

- Quantify P2a **rescue / harm** vs CLIP, OOD-real behavior, conditional info
  `I(Y ; P2a | CLIP, Ref)`.
- **Run the domain-detector audit hardest here** — reconstruction methods are the classic
  domain detectors.
- **Signed response.** AEROBLADE: LDM-generated content can reconstruct *more* faithfully, so
  **low** residual can be evidence of generation. The head learns polarity; do **not**
  hardcode "large residual ⇒ fake."
- **Gate**: P2a keeps its slot only if it shows conditional info beyond CLIP+Ref and passes
  the audit. If `rescue ≈ harm` and `I ≈ 0`, **remove it** — do not keep it to look richer.

### E3 — Three-branch complementarity (frozen outputs)

- Freeze qualified branches; save per-sample evidence / prediction / vacuity.
- Compute **rescue / harm / error-overlap / evidence-correlation** across
  `{CLIP, Ref, Process}`.
- Go/no-go for the multi-branch premise: do Ref and Process actually complement CLIP, or
  collapse into it (evidence-correlation ≈ 1)? A branch that's near-collinear with CLIP in
  evidence space adds nothing for the gate to use.

### E4 — Applicability-aware fusion + reliability (the novelty)

- Compare fusion variants: **(a)** plain sum/EDL, **(b)** confidence-weighted, **(c)**
  utility-supervised applicability, **(d) DiCoME-style DS combination as the baseline to
  match**, and **(e) applicability-weighting applied *within* a DS-style rule**.
- **Why DS is in the comparison, and why (e) exists.** The gap analysis + our own P0-UW
  control show DS is genuinely load-bearing: replacing it with an (inverse-vacuity) weighted
  mean cost up to −0.077 on CDFv3 and −0.123 on FaceReenact — and nothing else changed. The
  property that carries it is DS's **conflict reweighting** (divide by `1−C`): a weighted
  *sum/mean* cannot let a confident view override a diffuse-but-wrong one, so it fails exactly
  where OOD conflict lives. Our proposed `e_fused = e_sem + λ·Σ w_b e_b` is **additive** and
  therefore inherits that weakness — applicability weighting answers "is this branch
  applicable?" but not "do the applicable branches conflict?" So variant (e) applies the
  `w_b` gate *inside* a DS-style conflict-aware combination rather than as a plain weighted
  sum, and the test is whether the applicability variants **match DS on the conflict families**
  (CDFv3 FaceReenact/TalkingFace, the rows where P0-UW cratered), not just on the easy ones.
  If additive applicability fusion regresses there versus DS, keep the DS-style combination and
  layer applicability on top.
- If any DS-style variant is run, **guard the `1−C` divisor with an epsilon** — DiCoME's is
  unclamped and divides by zero at total conflict.
- **Utility-supervised `q_b`.** Target `t_b(x) = 1[ ℓ_b(x) < ℓ_sem(x) − δ ]` from
  **held-out / leave-one-manipulation-out** branch predictions; `q_b = σ(H_app,b(h_b))`
  learns to predict it. `t_b` and the label are **train-time only** — gate inputs stay
  label-free; never feed `t_b`, the label, or generator/dataset/family ID at inference.
- Fusion: `w_b = q_b(1−u_b)`, `e_fused = e_sem + λ·(Σ_b w_b e_b)/(1+Σ_b w_b)`,
  with `e_b ≈ 0 ⇒ Real` (an inapplicable specialist must not push toward Fake). CLIP is the
  always-on anchor; specialists are gated.
- **Reliability = V / C / A**, not one fused vacuity:
  - `V` = fused vacuity `K/S_fused` (insufficient evidence).
  - `C` = applicability-weighted informative conflict
    `(Σ_{b<c} w_b w_c D_JS(p_b,p_c)) / (Σ_{b<c} w_b w_c + ε)` — so uniform+uniform doesn't
    read as strong agreement.
  - `A` = unsupportedness `1 − mean_b q_b(1−u_b)`.
  - `R = g(V, C, A, margin) → Real / Fake / Defer`, with selective-risk / coverage analysis.
  - **No OOD test labels enter calibration.**

---

## Training strategy (for the runs Umar executes)

**Recipe alignment from the DiCoME gap analysis (all [R], zero novelty cost).** Our infra
reproduces DiCoME, so any DISCERN-v2 underperformance is a recipe/architecture delta on our
side. The analysis found the EDL formulation already matches exactly — the KL-annealing is
**not** the gap (both anneal; at DiCoME's selected checkpoint its KL is weaker than ours). The
real recipe deltas, in order of cost-to-test:

- **Score existing checkpoints at epochs 3–8 first — no retraining.** DiCoME's best is epoch
  1–4 of 20; our arms select 27–35. If part of their edge is "stop before FF++ overfitting,"
  it's free for us. This is the cheapest test in the whole phase; run it before any retrain.
- **Weight decay 0.01 with bias/norm excluded** (we run `0.0`). Late-epoch FF++ overfitting is
  exactly the flat-cross-dataset failure mode; regularizing the tail is the most likely single
  recipe fix. Closes: our long unregularized runs vs their regularized epoch-1–4 selection.
- **Stronger, always-on photometric augmentation** — DiCoME blurs and colour-jitters *every*
  training image (blur σ 0.1–2.0, jitter 0.2); we blur 10% and at half amplitude. Our weakest
  rows are compression/resolution-shifted (DFDC, DFDCP). Closes: the exact shift our worst
  sources exhibit. Test whether DFDC + DFDCP improve *together*.
- **LR 1e-4 vs our 3e-4** — smaller and worth a sweep, but lower priority than the above.

Fold these into Stage II as DISCERN-v2 defaults to test, not as blind adoption — each is a
one-line falsifiable change against a named metric.

- **Degradation / vacuity consistency** is the one focused training ablation worth keeping —
  it's DISCERN-native, not generic augmentation: `L_pred-cons = D_JS(p(x), p(Tx))` and
  `L_vac-cons = max(0, u(x) − u(Tx))` under benign transforms `T` (JPEG, resize, blur, mild
  noise, illumination). Idea: a degraded observation may become *less* certain but must not
  become artificially *more* certain via a shortcut.
- **Five-stage schedule**, not joint-from-scratch: (I) reals-only reference pretrain, then
  freeze permanently; (II) per-branch heads on FF++ c23 with identical supervision; (III)
  branch qualification = the E1–E3 audits; (IV) applicability fusion; (V) reliability/defer
  calibration on held-out/LOMO, no OOD labels.

## Parked this cycle (do NOT build — reasons matter)

DINOv3 as a committed branch (later **earned swap** by `I(Y;DINO|CLIP)>0` — rescue / different
errors / OOD-real stability / family coverage, **not** "beats CLIP"); FSFM region-masking /
face-parser; VGGFace2-scale real pretraining (breaks FF++-only fairness); SigLIP2 bake-off;
frequency 4th branch; SBI; FreqDebias / Fourier augmentation; diffusion-noise-consistency
beyond P2a; VGGT; all temporal. Five simultaneous prior/distribution changes would destroy
the causal story — if the model improves you wouldn't know whether the reasoning method or
the swapped priors did it.

**Understood but not adopted from DiCoME (per the gap analysis).** The co-trained β-VAE inside
the CLIP manifold (ours is reals-only + frozen, or the residual becomes definitionally
circular); the orthogonal-projection artifact view (buys *statistical* independence but leaves
the two views *epistemically* dependent — both fail when CLIP fails, the shared-ignorance case
our V/C/A exists to fix); and DS *as the novelty* (it's the baseline we must match — its
`u_f = u_0·u_1` blind spot, where two views ignorant *for the same reason* fuse to
over-confidence, is our contribution surface, and it's aggravated by both-views-in-one-manifold
which we don't have). Borrow their **recipe**, not their benchmark-shaped architecture.

## Stop points

1. No training launches — those are UMAR-RUNS.
2. A specialist enters fusion only after passing its E1/E2 audit (conditional info + the
   domain-detector test).
3. V/C/A calibration only after E3 confirms complementarity.
4. Escalation to a richer reference (β-VAE/MR-VAE) or DINO/frequency only on earned audit
   results — 🟡 ASK-UMAR.

## Hand back at end of Phase 2

Audit report (E1–E3) + fusion/reliability results (E4) + a `SELECTION.md` (which specialists
earned slots, with audit evidence) + the Task-0 encoder decision + confirmed repo path.

## T-BIOM framing (writing track — start in parallel)

Frame the paper as **reliable face biometric-integrity assessment under heterogeneous
manipulation shift**. The branches are *measurement mechanisms* — appearance evidence,
authentic-reference consistency, generative-process consistency. The biometric question sits
one level above them: *is the observed facial biometric evidence trustworthy enough to support
an authenticity decision — do independent forensic cues agree, and should the system defer
under unseen manipulation?* Do **not** revive the identity/behavior/structural/generative
taxonomy — a spatial-only design has no "behavior" mechanism, and the reliability contribution
is the real biometric hook. The bug-as-finding (why naive projector-swapping fails; the need
to separate reference construction from forensic interpretation) and the audit design are
writable now. 
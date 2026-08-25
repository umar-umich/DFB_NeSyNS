# Claude Code — DISCERN v2 Phase 2: authoritative V1 build spec

**Status.** Authoritative. This document is the single source of truth for the V1 build.
It supersedes the earlier Phase-2 Claude Code specs and all pre-pivot plans (audit-gated
staging, LN-first anchor, C0–C3 ladder as a gate, DINO/CLIP-reference-first). Where those
documents disagree with this one, this one wins. The companion `why` document (the DISCERN v2
handoff) remains valid as background; this is the `what to run`.

**What changed from the prior spec (read once).**
1. Reference branch is **frozen FS-VFM-L/16**, not frozen CLIP. Committed for V1, not deferred.
2. Anchor is CLIP-L/14 with the **ported DiCoME LoRA implementation**, not LN-tuning and not a
   hyperparameter graft.
3. A mandatory **FS-VFM preprocessing parity pre-flight** runs before any training, because
   FS-VFM's face-crop convention differs from ours and getting it wrong invalidates Branch B.
4. Meta split is **60/40 with 5-fold cross-fitting** of the applicability gates, not a thin
   40/35/25 three-way partition.
5. A mandatory **frozen FS-VFM direct-probe control** (same-capacity head) sits alongside the
   residual reference.
6. **Baseline bookkeeping** and a **result-significance rule** are now explicit.
7. The applicability gate is **fusion-utility-supervised** (target defined from pairwise DS), not
   standalone-classifier-supervised.
8. FS-VFM's **native DLIB+30% crop is the default** input; the aligned crop is the logged control.
9. **Specialist unavailability maps to ignorance** (q=0, vacuous), not dropped or zero-filled
   samples.
10. The **domain-detector distribution audit** is a required diagnostic output, not just a
    success criterion.

---

## 0. Mode: build first, instrument everything, iterate second

Stand the full framework up end-to-end **once**, with a strong novel+SOTA component set, get a
trustworthy cross-dataset number, then iterate. This is **one experiment pipeline, not one
optimizer call**. Some components are fitted, frozen, and calibrated in sequence to preserve
their meaning and avoid leakage. Do not reopen architecture search before V1 runs.

The V1 paper story is not `we invented the branches`. It is: strong heterogeneous forensic
priors, plus a new applicability-aware conflict-sensitive evidential reasoning layer, plus
explicit reliability separation. The branches may use strong existing components.

---

## 1. Contract

- 🟢 Claude Code writes, integrates, tests, analyses. 🔴 Umar launches all training, evaluation,
  and dataset conversion. 🟡 Ask Umar before changing seed, data split, dataset list, major
  hyperparameters, eval sources, data paths, or replacing a V1 component.
- Never invent numbers. Cite runs and `file:line`. Port from validated code, not memory.
- Main supervised training is **FF++ c23 only**. External pretrained priors are allowed and
  must be disclosed as the same category: CLIP (external), FS-VFM (external real-face SSL),
  SDXL/LDM VAE (external). No OOD test labels enter checkpoint selection, gate training, risk
  calibration, thresholds, or hyperparameter selection.
- Work in the DISCERN repo. Confirm the path before writing anything (assumed `../DISCERN/`).

---

## 2. V1 architecture (fixed)

```text
                    ┌─ CLIP-L/14 + ported DiCoME LoRA ─────────> e_sem
aligned face x -----┼─ frozen FS-VFM-L/16 ─┬─ P_R (real-only) ─> e_ref
                    │                       └─ direct probe ────> e_direct   (control, logged)
                    └─ frozen SDXL/LDM VAE ────────────────────> e_proc
                                      │
                                      ▼
                     utility-supervised applicability  q_ref, q_proc   (q_sem = 1)
                                      │
                           Shafer applicability discounting
                                      │
                          conflict-aware Dempster-Shafer fusion
                                      │
                              V / C / A reliability
                                      │
                            Real / Fake / Defer
```

Exactly three spatial evidence branches feed fusion. `e_direct` is a control, logged and
compared, never fused. No temporal net, V-JEPA, TCN, optical flow, frequency branch, DINO,
SigLIP, SBI, FS-Adapter, or new foundation-model pretraining in V1.

---

## 3. Branch A — semantic evidence `e_sem`

Backbone CLIP ViT-L/14. **Port the validated DiCoME semantic implementation from the local
reproduced clone**, do not graft LoRA settings onto the LN code. Reuse the actual encoder class,
PEFT injection, parameter grouping, feature projection, optimizer treatment, and relevant
training-path behavior, then adapt the output to DISCERN v2's common evidential interface. The
goal is that `e_sem` behaves identically to the DiCoME semantic module already reproduced. The
rest of DISCERN v2 stays in our own staged trainer.

Trace the local DiCoME config before trusting any value. Expected form to confirm, not assume:

```text
target modules: q_proj, v_proj
rank: 8 · alpha: 16 · dropout: 0.1 · bias: none
```

Freeze all CLIP parameters except the intended LoRA adapters. If the reproduced encoder uses a
learned hidden→64-D bottleneck before the evidence head and code inspection shows it is recipe
rather than tied to DiCoME's Geometric View Purification, reuse it. Do **not** import DiCoME's
beta-VAE manifold or orthogonal-projection artifact view.

Head output is non-negative evidence `e_sem >= 0`, shape `[B, 2]`. The branch does not implement
its own Dirichlet math; it uses the centralized utility (Section 7).

---

## 4. Branch B — bona-fide face reference `e_ref` (+ direct-probe control)

Use the released **FS-VFM ViT-L/16** real-face checkpoint as the frozen coordinate system.
Confirmed facts (verify locally by checksum and state-dict diff before writing run commands):

- Repo `wolo-wolo/FSFM-CVPR25`, branch **`FSVFM-extension-R1`**. If only `FSVFM-extension` is
  present, inspect both and document R1 differences before porting.
- Pretraining VGGFace2 (~3M real faces), from scratch, **nominal 600-epoch** training, all
  scales S/B/L.
- **Checkpoint artifact.** HF repo `Wolowolo/fsfm-3c`, directory
  `pretrained_models/FS-VFM_ViT-L_VF2_600e/`. The directory is labeled `600e` (training length)
  but the actual Online Network file is **`checkpoint-599.pth`** (0-indexed final epoch), with
  Target `checkpoint-te-599.pth` (not needed for the frozen downstream encoder). Do not hardcode
  `-600`; the table-vs-filename mismatch is only the epoch-count label vs the final index.
  Resolve and record the exact artifact, checksum, and state-dict keys in `V1_BUILD_REPORT.md`.
- Downstream loads the **Online Network** and applies **FS-VFM's own normalization** from the
  shipped `pretrain_ds_mean_std.txt`, **not** ImageNet mean/std.

### 4.1 Reference model

FS-VFM is the large diverse bona-fide prior. We do **not** retrain a foundation model on FF++.
On frozen FS-VFM features `z_ref = E_FSVFM_frozen(x)`, fit a small deterministic real-only
reference `P_R` using **FF++ authentic training samples only**:

```text
z_hat = P_R(z_ref)
L_ref = 1 - cosine(z_ref.detach(), z_hat)      # no fake image or label used
```

Reuse the validated deterministic-AE implementation from DISCERN, adapting dimensions only. Do
not redesign it. After the offline fit, `E_FSVFM` and `P_R` are frozen permanently.

```text
r_ref = z_ref - z_hat
diagnostics: L2 residual magnitude, 1 - cosine(z_ref, z_hat), per-dim standardized residual
standardize using FF++ REAL TRAINING statistics only  ->  H_ref -> e_ref
```

Interpretation to state in the paper. FF++ reals are **not** the authentic-face prior. The broad
real-face prior is the VGGFace2-pretrained FS-VFM. The FF++ real-only fit is a lightweight
reference calibration in that frozen face-specific space.

### 4.2 Mandatory direct-probe control `e_direct`

Isolate the value of the `P_R` reference transformation by holding head capacity constant. Two
paths on the same frozen features, **same-capacity** evidence head on each:

```text
main:    frozen FS-VFM -> same-capacity evidence head ---------------> e_direct   (logged, NOT fused)
against: frozen FS-VFM -> P_R -> r_ref -> same-capacity evidence head -> e_ref
```

This answers one clean question: does the reference transformation add anything over the raw
pretrained representation, with capacity controlled. If the residual reference is worse than the
capacity-matched direct probe, that is a finding, and we do not spend iterate cycles improving a
reference operator that destroys an already strong representation.

Separately and only as an external implementation sanity/baseline, reproduce FS-VFM's official
R1 linear-probe protocol. That is a correctness check against the authors' numbers, not the
internal `e_ref`-vs-`e_direct` comparison above, and the two must not be conflated.

Do **not** use FS-Adapter in the main branch; it confounds the reference mechanism and stays a
later baseline.

---

## 5. Branch C — generative-process specialist `e_proc`

Use the validated **P2a LDM/SDXL first-stage VAE reconstruction operator**, fully frozen.

```text
x -> frozen VAE (VAE-native normalization) -> reconstruction/process residual -> H_proc -> e_proc
```

Start from the original aligned RGB crop with the exact normalization the VAE expects. Do not
feed CLIP or FS-VFM features here. Keep the response **signed/learned** (low residual can mean
fake, per AEROBLADE); do not hardcode `large error = fake`. Standardize process diagnostics with
FF++ real training statistics. Keep the existing validated P2a feature summary. No diffusion-noise
consistency in V1.

---

## 6. Shared input handling and branch-specific preprocessing

One aligned face crop feeds three branch-specific preprocessing functions with independent
normalization. Never normalize once globally and reuse that tensor across three pretrained models.

Branches key off the same source frame but do not share a tensor. The reference branch defaults
to FS-VFM's native crop (Section 8), so all three preprocessing functions start from the source
frame rather than from one shared aligned crop.

```text
source frame
    |-- DISCERN aligned crop  -> CLIP preprocessing --------> semantic branch
    |-- DLIB + 30% crop       -> FS-VFM normalization ------> reference branch   (native, §8)
    `-- DISCERN aligned crop  -> VAE-native normalization --> process branch
```

Add tests that print input range, mean, and std per branch on the same batch. Add a unit test
comparing our extracted FS-VFM feature against the official FS-VFM downstream path within
numerical tolerance, matching the authors' **full** input pipeline (crop convention, normalization,
pooling rule, CLS vs mean-patch, final normalization), not normalization alone.

---

## 7. Centralized evidential representation

All branches output non-negative evidence `e_b = [e_real, e_fake]`, `b in {sem, ref, proc}`
(and `direct` for the control). One shared implementation computes:

```text
alpha_b = e_b + 1 · S_b = sum_k alpha_b,k · p_b = alpha_b / S_b · u_b = K / S_b   (K = 2)
belief_b,k = e_b,k / S_b
assert sum_k belief_b,k + u_b == 1
```

Use the term **evidential vacuity / lack of evidence** for `u`. Ported and native branches share
this exact utility so they cannot drift.

---

## 8. FS-VFM native preprocessing (MANDATORY, before any training)

FS-VFM was pretrained and released for downstream deepfake detection on **DLIB face detection
with 30% additional cropping, resized to 224**, and it uses this convention for both pretraining
and downstream DfD. FACER parsing generates the CRFR-P masking maps during pretraining-data
construction only; it is **not** part of the frozen downstream encoder path, so we do not run it.

Because the goal is a strong V1 number, the reference branch **defaults to FS-VFM's native
DLIB+30% crop**. The aligned crop is the control arm, not the default. Do not feed a
non-native aligned crop to a frozen encoder by default and then decide whether it is bad enough
to replace.

```text
CLIP    <- DISCERN aligned crop        (unchanged)
FS-VFM  <- official DLIB + 30% crop    (native default)
VAE     <- DISCERN aligned crop        (unchanged)
all three derived from the same source frame
```

**Consequence to plan for.** The DLIB+30% crop cache is now on the critical path for **every**
split the reference branch sees, including FF++ train and val (Stage A fits `P_R` on FF++ real
DLIB crops; Stage B trains `H_ref` on FF++ DLIB crops). Generating this cache across all splits
is 🔴 preflight work Umar runs before training, not a fallback. The CLIP and VAE aligned-crop
caches are untouched.

**Second-detector correctness (composes with Section 14.1).** FS-VFM's DLIB detection runs
independently of the primary pipeline detector. On multi-face frames the two can select different
faces, which would make branches describe different identities and render fusion meaningless.
Rule: on frames with multiple DLIB detections, select the DLIB face with the highest bbox overlap
(IoU) to the primary pipeline's chosen crop. If no DLIB face overlaps sufficiently, or DLIB fails
to detect, set `branch_valid_ref = 0` and treat the reference opinion as vacuous (Section 14.1).

**Parity control (logged, does not gate training).** On a small fixed FF++ subset, run the frozen
FS-VFM same-capacity probe on (a) the native DLIB+30% crop and (b) the aligned crop, both with
FS-VFM normalization. Record the feature-stat and probe-performance gap in the parity report.
This number informs a later, operational question: if the aligned crop costs essentially nothing,
we may retire the second detector to avoid cross-detector mismatch. It does not change the V1
default, which is native.

---

## 9. Build order (staged, leakage-safe)

The applicability target cannot be defined before the branch experts exist, and the risk model
must not train on the gates' training data. Sequence:

```text
Pre. FS-VFM preprocessing parity pre-flight (Section 8).
A.   Fit real-only P_R offline on FF++ real FS-VFM features (cosine). Verify nonzero grad in A.
     Freeze E_FSVFM and P_R. Save reference statistics.
B.   Train CLIP-LoRA + H_sem, H_ref, H_proc (and e_direct control head) on FF++ c23 train.
     E_FSVFM frozen, P_R frozen, VAE frozen. Per-branch auxiliary EDL losses so each branch is
     individually usable. Do not train applicability or risk yet.
C.   Select the expert checkpoint on VAL_select (Section 11).
D.   Train applicability gates q_ref, q_proc on VAL_meta via 5-fold cross-fitting (Section 12).
E.   Fit the V/C/A risk/defer model on out-of-fold q_b over VAL_meta. Refit gates on all
     VAL_meta for deployment.
F.   Evaluate FF++ test + full OOD suite (Section 21). Log everything (Section 20).
```

---

## 10. Training recipe (Stage B)

Use the reproduced DiCoME recipe where it is architecture-independent, verified from local code,
not prose memory.

```text
optimizer: AdamW · base LR: 1e-4 · weight decay: 0.01
LoRA: exact reproduced DiCoME settings (ported, §3)
photometric aug: exact reproduced DiCoME ranges/order (verify from code)
```

Verify actual optimizer parameter groups. Exclude bias/norm from weight decay only if the
reproduced code really does. Preserve the reproduced scheduler/warmup if present. Do **not** add
MixUp/CutMix/random-erasing merely because FS-VFM fine-tuning uses them; they destroy the process
residual and complicate comparability. Fit `P_R` on clean standard training crops, no fake-label
aug, no MixUp/CutMix.

---

## 11. Checkpoint selection

Train the reproduced practical horizon and save every epoch. Do not hardcode an epoch window.
Select on **FF++ VAL_select only**: primary video-level AUROC, tie-break lower ECE / val loss.
Never select using CDF/DFDC/DF40 or any OOD source. Save epoch-wise OOD scores for later analysis
only.

---

## 12. Meta split for applicability and reliability (60/40 + cross-fit)

Use only original FF++ validation videos. Deterministic **video-level** split, grouped by source
video so no frames from one source cross partitions:

```text
VAL_select = 60%    (expert checkpoint selection, Section 11)
VAL_meta   = 40%    (applicability + risk, cross-fit below)
```

Stratify by real/fake and manipulation source where possible. Expose fractions in config and
report actual video counts. On `VAL_meta`, run **5-fold cross-fitting** of the small applicability
heads: each fold yields out-of-fold `q_b`; the out-of-fold predictions fit the logistic V/C/A risk
model; finally refit the applicability heads on all of `VAL_meta` for deployment. This cross-fits
only the tiny gates, not CLIP or FS-VFM, so it is nearly free and gives cleaner risk training than
a small third partition. LOMO is deferred to the publication version.

---

## 13. Fusion-utility-supervised applicability

The gate must learn the quantity it actually controls: whether admitting specialist `b` into the
conflict-aware fusion improves the decision for this sample. Do **not** supervise on standalone
specialist quality (`CE(p_sem,y) − CE(p_b,y)`), which asks a different and weaker question and can
diverge from fusion utility. Define the target from the same DS mechanism, using **plain
undiscounted pairwise DS** of the anchor with the specialist:

```text
q_sem = 1 (always-on generalist)
q_ref(x), q_proc(x) in [0,1], learned on VAL_meta folds

p_sem⊕b^DS = DS(omega_sem, omega_b)            # undiscounted, same centralized DS + eps as deploy
Delta_b^fuse = CE(p_sem, y) - CE(p_sem⊕b^DS, y)
t_b = 1[Delta_b^fuse > delta] · delta = 0 (config, never tuned on OOD)
```

`q_b` then predicts the probability that letting specialist `b` into fusion improves the outcome.
Compute this independently for CLIP+Reference and CLIP+Process. `p_sem`, `p_b`, and the pairwise
DS opinion are all computed from the **frozen selected-checkpoint** experts on `VAL_meta`.

**Named approximation (write this into the paper).** The target is pairwise and full-admission,
while deployment is three-way and soft-discounted (`omega_sem ⊕ omega'_ref ⊕ omega'_proc`). This
is a much better proxy than standalone-classifier quality, but it cannot see interactions where
reference and process each help alone yet jointly overshoot or conflict. Three-way Shapley-style
marginal utility is deferred; pull it in only if the Section 20 plain-DS-vs-DS+applicability
diagnostic shows the applicability layer failing to beat plain DS. Call the V1 method
**fusion-utility-supervised applicability**, not generic utility supervision.

Gate inputs are label-free at inference. Recommended compact inputs: `p_sem, u_sem, p_b, u_b`,
branch residual diagnostics, and a branch hidden summary if already available. Never input the
label, `t_b`, dataset ID, generator/manipulation ID, or family ID. Train each gate with BCE. Log
gate AUROC/accuracy against its fusion-utility target.

---

## 14. Shafer applicability discounting before Dempster-Shafer fusion

### 14.1 Specialist unavailability is ignorance, not a missing sample

When a specialist cannot produce a valid input on a frame (FS-VFM's DLIB detector fails or finds
no face matching the primary crop per Section 8; catastrophic P2a failure), do **not** discard
the frame, inject zero features, duplicate another crop, or let the evidence head interpret
failure as Real or Fake. Instead force the specialist to complete ignorance:

```text
branch_valid_b = 0  ->  q_b = 0  ->  belief'_b,k = 0, u'_b = 1   (set at the opinion level)
```

The override sets the opinion vacuous directly; the invalid crop is never fed to `H_b`. This is a
clean property of the framework: availability failure becomes vacuous evidence rather than a
forced forensic opinion, and the anchor carries the sample. During Stage B, mask the per-branch
auxiliary EDL loss for any branch on frames where `branch_valid_b = 0`. Log `branch_valid_ref` and
`branch_valid_proc` per sample.

### 14.2 Discounting

Not a plain weighted evidence sum. Discount each valid specialist opinion by its applicability:

```text
belief'_b,k = q_b * belief_b,k
u'_b        = (1 - q_b) + q_b * u_b
assert sum_k belief'_b,k + u'_b == 1
```

`q_b -> 1` trusts the specialist; `q_b -> 0` turns its opinion into ignorance rather than evidence
for Real; conflicting applicable specialists stay visible to DS. The anchor is undiscounted
(`q_sem = 1`). Combine `omega_sem ⊕ omega'_ref ⊕ omega'_proc` using the **validated DiCoME DS
combination logic only** (port the combination, not the view construction), with epsilon guards
around near-total conflict.

---

## 15. Mandatory DS unit tests (before real training)

Synthetic opinions covering: both vacuous; anchor confident / specialist vacuous; specialist
confident / anchor vacuous; confident agreement; confident disagreement; near-total conflict;
`q_b = 0`; `q_b = 1`. Assert finite outputs, normalized opinion, sensible limiting behavior, and
that `q_b = 0` has no specialist influence. Fix and document the combination order and test whether
order changes the result beyond numerical tolerance.

---

## 16. Fused opinion back to EDL

```text
S_f = K / clamp(u_f, eps, 1) · e_f,k = belief_f,k * S_f · alpha_f = e_f + 1 · p_f = alpha_f / sum(alpha_f)
```

Centralize this. Do not mix inconsistent EDL and subjective-logic conversions across modules.

---

## 17. Reliability decomposition V / C / A

```text
V = u_f                                   # insufficient fused evidence
w_b = q_b * (1 - u_b), with q_sem = 1
C = sum_{b<c} w_b w_c JS(p_b, p_c) / (sum_{b<c} w_b w_c + eps)   # informative experts disagree
A = 1 - mean_{b in {ref,proc}} q_b (1 - u_b)                    # specialist support absent
```

Also log raw DS conflict separately as a diagnostic.

---

## 18. Real / Fake / Defer

A deliberately small risk model, default logistic regression on `[V, C, A, fused_margin]` where
`fused_margin = abs(p_f,fake - 0.5)`, trained on the out-of-fold `q_b` predictions over VAL_meta
(Section 12). Target is `1` if the fused prediction is wrong. No large MLP in V1. Report
error-detection AUROC, risk-coverage curve, selective risk, coverage at a fixed 10% abstention
budget, and a few fixed coverage points. Any threshold comes from FF++ validation only and is
frozen on all OOD tests.

---

## 19. Correctness guards (non-negotiable, pass before Umar launches)

**Gradient.** LoRA nonzero in B; base CLIP zero except LoRA; FS-VFM zero everywhere; `P_R`
nonzero only in Stage A, zero in B+; VAE zero; evidence heads nonzero in B; applicability heads
only in the gate stage; risk model only on VAL_meta cross-fit.

**Mechanism.** The reference-config assertion (`assert_projector_config` or its replacement) runs
in the real build path, not a dead guard. `L_ref` is nonzero and in the Stage-A optimizer graph.
The saved checkpoint contains fitted reference weights and statistics. The restored model
reproduces reference residuals.

**Data leakage.** FF++ train source videos absent from val partitions; VAL_select and VAL_meta
source-disjoint, and folds within VAL_meta source-disjoint; no OOD sample enters training or
calibration; no test label enters thresholding.

---

## 20. Instrumentation (from the same V1 pipeline)

Save per-frame and per-video: dataset, sample/video ID, label, manipulation/family/method if
available, `e_sem/p_sem/u_sem`, `e_ref/p_ref/u_ref`, `e_proc/p_proc/u_proc`, `e_direct` and its
prediction, `branch_valid_ref/branch_valid_proc`, `q_ref/q_proc`, reference residual diagnostics,
process diagnostics, discounted opinions, DS conflict diagnostics, fused `p/u`, `V/C/A`, risk,
prediction, correctness.

**Post-hoc contribution diagnostics** (evaluate the already-trained model with toggles): semantic
only, +reference, +process, full, plain-DS without applicability, DS+applicability, and residual
reference vs `e_direct`. Label these **contribution diagnostics, not retrained ablations**.
Publication-quality ablations require retraining when appropriate.

**Domain-detector distribution audit (required output in `V1_BRANCH_DIAGNOSTICS.md`).** A branch
that scores well may be exploiting dataset acquisition or compression differences rather than
forensic signal. For the reference and the process branch, compare the residual distributions:

```text
reference: p(r_ref | R_FF++), p(r_ref | R_OOD), p(r_ref | F_OOD)
process:   p(r_proc | R_FF++), p(r_proc | R_OOD), p(r_proc | F_OOD)
key test:  D(R_FF++, R_OOD)  <<  D(R_OOD, F_OOD)  ?
```

If real-vs-real cross-dataset drift is as large as real-vs-fake separation within OOD, the branch
is behaving as a dataset detector and its headline AUC is suspect. This is a required diagnostic,
not a build gate; flag failures for the iterate stage (scale reference data), do not block V1.

---

## 21. Evaluation and baseline bookkeeping

Video aggregation is mean frame fake probability unless the validated DISCERN/DiCoME code uses a
different verified rule, in which case reproduce and document both. Report frame and video AUC
where available; video AUC is primary. Evaluate on already-prepared datasets without blocking the
first run on new conversion.

```text
FF++ c23 test · Celeb-DF-v1/v2 · CDFv3 (FS/FR/TF breakdown) · DFD · DFDCP · DFDC
UADFV if supported · Deepfake-Eval-2024 (zero-shot) · DF40 family/method (R1 DF40 support)
```

No OOD result may change the checkpoint or hyperparameters after the fact.

**Baseline bookkeeping (mandatory in every table).** Distinguish four rows and never conflate
them: DiCoME paper / released-checkpoint result; **our reproduced DiCoME result**; DISCERN-v1;
DISCERN-v2 V1. Compare against the reproduced numbers. For DFD this means the in-house baseline
(0.9419 per the handoff, confirm against the run log), **not** the paper's 0.982, so we do not
diagnose a gap that does not exist.

---

## 22. Result-significance rule

Do not promote or remove any component on a one-run swing at the noise floor.

```text
|Delta AUC| < 0.01, or inconsistent family-level effects  ->  second seed required before any
promotion, removal, or architecture change. Report bootstrap CIs where convenient.
```

Our prior controlled experiments already showed several differences near the noise floor, so this
rule governs every read-off decision.

---

## 23. V1 success criteria

V1 succeeds if: the three-prior framework trains and evaluates without mechanism bugs; OOD
performance is in a reasonable competitive range; the applicability+DS+V/C/A layer does not
destroy the strong component baselines; at least one specialist shows nontrivial rescue relative
to the anchor; reference and process branches are not merely dataset detectors; reliability
signals are meaningfully associated with errors and selective risk. If these hold, **stop
architecture expansion** and let diagnostics choose the next change.

---

## 24. Iterate order (only after V1 numbers exist)

- Semantic weak: reproduced DiCoME LoRA vs LN; LoRA with/without the 64-D bottleneck.
- Reference weak or source-biased: frozen FS-VFM alone vs FS-VFM+PCA vs FS-VFM+deterministic AE;
  then additional diverse real-face data, FS-Adapter baseline, native FS-VFM masked-recon
  response. Never retrain FS-VFM from scratch.
- Process adds little: drop or replace the slot only after rescue/harm analysis; later candidates
  diffusion-noise consistency or an earned frequency specialist.
- Fusion weak: plain DS vs DS+applicability vs additive-applicability (negative control only).
- Reliability weak: degradation/vacuity consistency (`L_pred-cons = D_JS(p(x),p(Tx))`,
  `L_vac-cons = max(0, u(x)-u(Tx))`); better meta cross-fitting / LOMO for gate targets.

---

## 25. Parked for V1 (do not build)

DINOv3; SigLIP2; V-JEPA/temporal; VGGT; frequency fourth branch; SBI; FreqDebias; new
VGGFace2/CelebA/FFHQ pretraining; FSFM/FS-VFM pretraining from scratch; FSFM face-parser region
masking; **FS-Adapter inside the main branch** (allowed only as the §4.2 frozen direct-probe
control, which is not FS-Adapter); diffusion-noise branch beyond P2a; DiCoME beta-VAE manifold;
DiCoME orthogonal artifact projection.

---

## 26. Deliverables before Umar runs V1

Create `V1_BUILD_REPORT.md`, `V1_CONFIG.yaml`, `V1_GUARDS_REPORT.md`, `V1_PARITY_REPORT.md`,
`V1_RUN_COMMANDS.md`.

`V1_BUILD_REPORT.md` covers: confirmed DISCERN repo path; confirmed local DiCoME path and exact
recipe items borrowed; confirmed FS-VFM repo/branch/checkpoint/normalization (checksum + state-dict
keys); parameter counts (frozen, LoRA, reference, heads, applicability); the staged sequence; the
FF++ train / VAL_select / VAL_meta partitions with counts; the evidence/DS formulas implemented;
all passed unit/gradient/leakage guards; the parity-check decision; and any deviations from this
document.

`V1_RUN_COMMANDS.md` gives Umar commands in order: (0a) generate the FS-VFM DLIB+30% crop cache
across all splits including FF++ train/val, (0b) FS-VFM parity control on the fixed subset,
(1) fit reference, (2) train experts, (3) select checkpoint, (4) train applicability gates
(cross-fit), (5) calibrate risk/defer, (6) FF++ test, (7) OOD suite, (8) instrumentation and
read-off including the domain-detector audit. Do not launch these jobs.

---

## 27. Deliverables after Umar runs V1

Generate `V1_RESULTS.md`, `V1_BRANCH_DIAGNOSTICS.md`, `V1_RELIABILITY.md`, `READOFF.md`.

`READOFF.md` answers only: is V1 competitive enough to continue; which branch contributes the most
unique rescue; which branch is the largest source of harm or domain shift; does applicability
improve over plain DS; do V/C/A improve error detection and selective prediction; what are the
**top two** changes to run next. Apply the Section 22 significance rule to every claim. Do not
propose ten new modules.

---

## 28. Novelty framing (writing track, in parallel)

Frame as reliable face biometric-integrity assessment under heterogeneous manipulation shift.
Branches are measurement mechanisms; the biometric question of whether the evidence is trustworthy
enough to decide, whether independent cues agree, and whether to defer sits above them. Do not
revive the identity/behavior/structural/generative taxonomy. Claim no novelty for CLIP, LoRA,
FS-VFM, SDXL-VAE, or DS individually. Cite FSFM (CVPR'25) and CFB/UTOM (TPAMI'26) as real-face
prior art we use, not claim, and beat CFB on four axes: utility-supervised applicability `q_b` vs
inverse-vacuity weighting (our P0-UW control shows inverse-vacuity is the fusion that lost);
V/C/A vs a single fused `u`; per-sample defer vs a source-data-calibrated threshold; heterogeneous
forensic references vs two same-type backbones. Verify FADNet and LaP-Forensics actually exist and
say what is claimed before positioning against them.

Working contribution statement, provisional until V1 validates it: DISCERN-v2 treats heterogeneous
pretrained forensic priors as conditionally useful evidence sources rather than universally
reliable detectors, learns sample-level specialist applicability from held-out utility, discounts
inapplicable opinions before conflict-aware evidential combination, and separates fused vacuity,
informative conflict, and unsupportedness to support calibrated Real/Fake/Defer decisions under
heterogeneous manipulation shift.
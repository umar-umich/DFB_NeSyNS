# Generation-Process Consistency Probing — Pilot Plan (Revised)

> **AMENDMENT 2026-08-07 — fusion rule for P1a–P4 changed from uncertainty-weighted mean to
> N-view Dempster-Shafer.** P0-UW ran as specified and its verdict is **Harmful**, not
> neutral: the fusion swap costs −0.018 on CDFv2, −0.077 on full CDFv3 (−0.123 on the
> FaceReenact family), with damage monotone in distribution shift and exactly +0.0000
> in-domain on FF++. Full evidence and CIs in `../../../DiCoME/experiments/pilot_P0-UW/DELTA.md`.
>
> The original mandate ("Fusion for all pilots except P0-DS: uncertainty-weighted mean … Do
> NOT use DS-fusion") assumed the swap was roughly neutral so it could serve as a clean
> control. Since it is not, every downstream pilot built on it would start −0.005 to −0.077
> below P0-DS, worst on precisely the unseen-family transfer that the Generalist / Specialist
> verdicts are defined by — an operator carrying real signal could be misclassified Harmful
> purely from the fusion it was required to sit on.
>
> The constraint that motivated UW was genuine: `DS_Combin` hard-codes exactly two views and
> P2a/P3a/P4 each add a third. It is removed by exploiting the associativity of Dempster's
> rule — `experiments/common/fusion_ds_nview.py` does iterated pairwise combination through
> DiCoME's own `_combine_two_opinions`, and is **bit-identical** to the shipped fusion for two
> views (verified, max abs diff 0.00e+00), so P1a–P1d are unaffected.
>
> Consequently: **P1a–P4 use N-view DS fusion and report Δ against P0-DS.** P0-UW is retained
> as the control row that produced this finding, not as the baseline. Sections "Baselines",
> "Contract", and the prompt template's fusion line are superseded on this point only.

Operational README for the DISCERN v2 T-BIOM pilot. Supersedes `README-dicome-component-swap-pilot.md` and the earlier version of this file. Companion to `README-dicome-reproduction.md` (eval) and `README-dicome-training-audit.md` (training verification).

---

## Framing

Different image-generation processes leave different forensic traces. A single foundation-model detector (CLIP semantic features, DiCoME's basis) is one measurement. Latent-diffusion autoencoder reconstruction, diffusion denoising discrepancy, and CNN-generation gradient transforms are others — each grounded in a published forensic-detection paradigm.

The pilot tests two hypotheses:

- **H1**: Forensic representations derived from operators intrinsic to distinct generative processes contain complementary evidence beyond a semantic foundation-model representation.
- **H2**: Such process-derived evidence transfers not only to its corresponding generation family but also to unseen face-generation/manipulation mechanisms.

Before the process-operator pilots, a manifold-projector mini-suite (P1a–P1d) benchmarks DiCoME's β-VAE against alternatives — the specific β-VAE choice has never been ablated against other latent-regularization schemes in this setting.

---

## Pilot catalog

| ID | Category | Change from baseline | Theoretical basis | Purpose |
|----|----------|----------------------|-------------------|---------|
| **P0-DS** | Reference | Original DiCoME + DS fusion | — | Established baseline; self-trained checkpoint exists |
| **P0-UW** | Reference | Original DiCoME + **uncertainty-weighted mean fusion** | Fusion isolation | Controls for fusion change in every downstream pilot |
| **P1a** | Manifold-projector control | Deterministic feature AE | Generic AE control | Is DiCoME's KL/probabilistic regularization load-bearing? |
| **P1b** | Manifold-projector variant | β-TCVAE | Chen et al., NeurIPS'18 | Best direct β-VAE refinement — TC penalty for disentanglement |
| **P1c** | Manifold-projector variant | WAE (Wasserstein AE) | Tolstikhin et al., ICLR'18 | Aggregate-distribution matching instead of per-sample KL |
| **P1d** | Manifold-projector variant | MR-VAE (Multi-Rate VAE) | Multi-rate rate–distortion | Rate–distortion response across β values as discriminative signal — highest-interest variant |
| **P2a** | Process operator | Frozen LDM first-stage AE, reconstruction residual only | AEROBLADE, CVPR'24 | LDM-generated images reconstruct faithfully through their AE |
| **P2b** *(conditional)* | Extension | P2a + pixel + latent + CLIP residual combination | Extended AEROBLADE | Only if P2a produces Generalist or Specialist verdict |
| **P3a** | Process operator | LaRE²-style latent reconstruction / denoising error | LaRE², CVPR'24 | Score-based dynamics expose modern synthesis |
| **P3b** *(conditional)* | Extension | P3a + FIRE-style frequency decomposition | FIRE, CVPR'25 | Only if P3a produces Generalist or Specialist verdict |
| **P4** | Process operator | **Official LGrad** (LGrad's provided pretrained network + target class scheme) | LGrad, CVPR'23 | CNN synthesis leaves systematic traces exposed by gradients |

**Explicitly not in this sweep**: FactorVAE, InfoVAE, Denoising Multi-β VAE, VQ-VAE, FSFM, flow-matching reconstruction operators. Consider only if a parent pilot signals demand.

**Naming discipline**: P4 uses the official LGrad implementation. Any deviation (different pretrained network, different target class scheme) must be labeled "LGrad-inspired" in the pilot notes and paper.

---

## Pilot ordering

```
P0-DS → P0-UW → P1a → P1b → P1c → P1d → P2a → P3a → P4
                                          ↓         ↓
                                        (P2b)     (P3b)
```

P2b and P3b only run if their parent produces a Generalist or Specialist verdict.

---

## Baselines

Every downstream pilot's DELTA reports against **both**:

- **P0-DS** (0.9644 CDFv2, 0.9419 DFD, 0.8825 DFDC, 0.8576 DFDCP, 0.9516 CDFv3 face-swap, 0.9929 FF++ from `TRAINING_REPRODUCTION.md`) — total gain vs the established DiCoME reference.
- **P0-UW** (numbers TBD from this pilot) — isolates operator contribution from the fusion change.

Comparing only to P0-DS confounds operator effect with fusion change; comparing only to P0-UW hides total pilot-stack gain. Report both.

---

## Success classification

Each operator's DELTA classifies the operator as one of:

- **Generalist**: improves multiple families with positive rescue-minus-harm across the evaluation set. Port as a core evidence stream in DISCERN v2.
- **Specialist**: large gain on its intended process family plus unseen generators of the same or related family, with little damage elsewhere. Reserve for the reasoning layer's mechanism-support role, not as always-on branch.
- **Redundant**: little incremental information beyond CLIP. Drop.
- **Harmful**: meaningful negative transfer. Drop and investigate.

The prior "must be stable across every family" bar is too strict — it would defer useful process-specific detectors that belong in the mechanism-support layer.

---

## Evaluation set (expanded)

Every pilot's DELTA evaluates on:

- **Continuity set**: CDFv2, DFD, DFDC, DFDCP, FF++
- **CDFv3**: all 8 generators broken out separately (SimSwap, GHOST, MobileFaceSwap, HifiFace, Celeb-DF-v2 inherited, InSwapper, BlendFace, UniFace)
- **DF40**: all methods separately + 4 family aggregates (swap / reenactment / synth / editing)
- **Celeb-DF++**: per manipulation category. If not yet integrated on our infrastructure, mark pending and proceed.
- **Deepfake-Eval-2024**: zero-shot deployment-style benchmark. This is the T-BIOM headline target if DISCERN v2 beats DiCoME here.

---

## Prerequisites

- Env `dicome`, repo at `../DiCoME`, self-trained P0-DS checkpoint in place.
- FF++ train/val H5s prepared; continuity-set test H5s ready.
- New H5 conversions: CDFv3 per-generator (currently face-swap only), DF40, Celeb-DF++ (if available), Deepfake-Eval-2024.
- **New downloads**:
  - LDM first-stage AE (SD 2.1 VAE) — for P2a
  - SD 2.1 UNet + scheduler — for P3a
  - Official LGrad pretrained network + target class scheme — for P4 (do not substitute)
- Compute: 2× H200. ~0.5 day setup + 1.5h train + ~0.5 day analysis + ~2h expanded eval per pilot.

---

## Contract

- 🟢 YOU-RUN: operator implementation, evidential head plumbing, config, verification, analysis, reporting.
- 🔴 UMAR-RUNS: training runs, eval runs, dataset conversions.
- 🟡 ASK-UMAR: any change to hyperparameters, seed, evaluation sources, or stacking pilots.
- No invented numbers. Every AUC from an actual run.
- Do NOT modify DiCoME's central config or core model code. Work in `../DiCoME/experiments/pilot_{PILOT_ID}/`.
- Each pilot adds exactly one change on top of P0-UW (except P0-UW itself, which changes only fusion vs P0-DS).

---

## Prompt template (specify pilot ID at top)

Paste inside the code block. Fill `PILOT_ID` and pilot-specific fields.

````markdown
# Task: Generation-process pilot — Pilot PILOT_ID

## Pilot specification
- **Pilot ID**: P0-DS / P0-UW / P1a / P1b / P1c / P1d / P2a / P2b / P3a / P3b / P4
- **Operator or change**: [copy from catalog]
- **Theoretical basis (citation)**: [copy from catalog]
- **Comparison baselines**: P0-DS (always), P0-UW (for P1a–P4)

Refer to `README-generation-process-pilot.md` for the full plan.

## Context
- Repo at `../DiCoME`, env `dicome`, self-trained P0-DS checkpoint is the reference.
- Training: seed 42, 20 epochs, batch 128, LR 1e-4. Same as the self-trained baseline.
- Pilot adds exactly one change on top of P0-UW. Do NOT stack.
- Fusion for all pilots except P0-DS: uncertainty-weighted mean using DiCoME's vacuity u^v. Do NOT use DS-fusion.
- Do NOT force process-operator evidence (P2a/P2b/P3a/P3b/P4) through orthogonal projection — P3 and P4 have no reconstruction, so their evidence goes directly to a fresh evidential head.

## Phase 1 — Implementation (YOU-RUN)

Create `../DiCoME/experiments/pilot_PILOT_ID/` with standard files (`operator.py`, `model.py`, `fusion.py`, `notes.md`). Pilot-specific implementation:

- **P0-DS**: no code change. Use self-trained checkpoint from `TRAINING_REPRODUCTION.md`. Exists as the fixed reference row.
- **P0-UW**: keep DiCoME's β-VAE + orthogonal projection + evidential heads unchanged. Swap DS-fusion for uncertainty-weighted mean. Retrain from scratch on FF++ c23, same hyperparameters, seed 42.
- **P1a**: replace β-VAE with a deterministic AE of matched architecture (no KL, no reparameterization; retain cosine reconstruction loss). Everything else identical to P0-UW.
- **P1b**: replace β-VAE with β-TCVAE. Implement total-correlation penalty per Chen et al. Preserve latent dim and cosine reconstruction loss.
- **P1c**: replace β-VAE with WAE. Use MMD-based aggregate distribution matching. Preserve latent dim and cosine reconstruction loss.
- **P1d**: replace β-VAE with MR-VAE. Train across several β values simultaneously (specify schedule in `notes.md`). Feed the rate–distortion response across β to the evidential head — this is the discriminative claim being tested.
- **P2a**: frozen LDM first-stage AE cycle. `image → LDM VAE encoder → LDM VAE decoder → reconstructed image`. Reconstruction residual only (pixel-level or simple fixed statistic). No pixel + latent + CLIP combination in this pilot. Feed residual statistics to a fresh evidential head as the second view.
- **P2b (conditional)**: extends P2a with pixel + latent + CLIP residual combination. Only if P2a passes.
- **P3a**: frozen SD 2.1 UNet + scheduler. LaRE²-style single-timestep latent reconstruction error. Pick one timestep (start with t=100 or t=250) and stick with it. Feed error map statistics to a fresh evidential head. No frequency decomposition.
- **P3b (conditional)**: extends P3a with FIRE-style frequency-band decomposition (low / mid / high). Only if P3a passes.
- **P4**: official LGrad. LGrad's provided pretrained network + their exact target class scheme. Do not substitute. Any deviation must be labeled LGrad-inspired in `notes.md`.

Sanity check: instantiate the modified model, print total trainable params. P1a–P1d and P2b should stay close to P0-UW (~868K). P2a/P3a/P4 add ~10K–100K on the operator's evidential head. Frozen components (LDM VAE, LDM UNet, LGrad backbone) must stay frozen — verify `requires_grad`.

🟡 ASK-UMAR if the operator requires input-preprocessing changes (resolution, normalization, color space) — document explicitly.

## Phase 2 — Operator preparation (YOU-RUN)

- **P2a**: verify LDM VAE loads and reconstructs FF++ frames sensibly. Save 5 examples to `sanity/` for visual check.
- **P3a**: verify UNet runs at target timestep without errors; single-step denoising produces sensible latent updates.
- **P4**: verify official LGrad gradient extraction works. Gradient magnitudes non-trivial and sample-varying. Backbone in eval mode.

Skip for P0-DS, P0-UW, P1a–P1d.

## Phase 3 — Training run (YOU-RUN prepare, UMAR-RUNS launch)

Point at the pilot's config, log to `logs/train.log`. Same hyperparameters, seed, epochs, batch size, LR as P0-DS training. Skip for P0-DS (checkpoint exists).

## Phase 4 — Evaluation (YOU-RUN prepare, UMAR-RUNS launch)

Expanded evaluation set (continuity + CDFv3 per-generator + DF40 per-method-and-family + Celeb-DF++ per-category + Deepfake-Eval-2024). Results per source → `results/{source}.json`.

## Phase 5 — DELTA reporting (YOU-RUN)

Write `experiments/pilot_PILOT_ID/DELTA.md`:

**Table**: Source | P0-DS | P0-UW (if applicable) | This pilot | Δ vs P0-DS | Δ vs P0-UW | 95% CI (paired bootstrap 2000 resamples) | Notes

**Complementarity analysis** (for P1a–P4):
- HSIC / MI between operator evidence and CLIP evidence at feature and evidence levels
- Error-rescue rate: fraction of CLIP-wrong samples the operator gets right
- Error overlap: fraction both get wrong
- Rescue-minus-harm: rescue rate minus new-error rate introduced by fusion

**Verdict classification**: {Generalist, Specialist, Redundant, Harmful}. Justify from the response pattern across generators / families, not from aggregate AUC alone.

**Interpretation** (10–20 lines): what the pilot's response pattern says about the operator's forensic content and what it implies for DISCERN v2.

## Non-goals
- No modification of DiCoME's central config or core model code outside the pilot dir.
- No stacking pilots.
- No hyperparameter or seed changes.
- No deltas from partial runs.
- No running P2b or P3b before their parent's verdict.
- No calling P4 "LGrad" if any component was substituted — must be "LGrad-inspired."

## Success criteria
- Operator implementation exists, self-contained.
- Trained checkpoint from full 20 epochs (except P0-DS reusing existing).
- Expanded evaluation completed.
- DELTA.md exists with response-matrix analysis and verdict classification.
````

---

## Response matrix

After each pilot completes, append its per-source AUC row to `experiments/RESPONSE_MATRIX.md`. Rows = operators (P0-DS, P0-UW, P1a, …). Columns = evaluation sources (CDFv2, DFD, DFDC, DFDCP, FF++, CDFv3-SimSwap, …, DF40-methods, DF40-families, Celeb-DF++, Deepfake-Eval-2024).

This matrix is the input to reasoning-layer rule derivation.

---

## Reasoning layer — derived, not predefined

Do **not** hardcode mechanism-support predicates in advance. After all pilots complete, examine the operator × generator/family response matrix. Predicate candidates emerge from actual observed behavior:

- Where an operator's evidence correlates with a specific family's failures, that operator becomes a candidate predicate for that family's mechanism-support hypothesis.
- Where two operators consistently disagree on a subset, that disagreement is a candidate rule.
- Where an operator succeeds on unseen families it wasn't designed for, that transfer pattern informs the reasoning layer's coverage claim.

Reasoning layer construction happens in Phase D of the merged plan, after the response matrix is complete. Not before.

---

## Expected directory layout

```text
../DiCoME/experiments/
├── pilot_P0-DS/       # reference (existing checkpoint)
├── pilot_P0-UW/       # fusion-only control
├── pilot_P1a/         # deterministic AE
├── pilot_P1b/         # β-TCVAE
├── pilot_P1c/         # WAE
├── pilot_P1d/         # MR-VAE
├── pilot_P2a/         # LDM AE reconstruction only
├── pilot_P3a/         # LaRE² latent reconstruction error
├── pilot_P4/          # official LGrad
├── pilot_P2b/         # (conditional) P2a + pixel/latent/CLIP residual
├── pilot_P3b/         # (conditional) P3a + FIRE frequency
└── RESPONSE_MATRIX.md
```

---

## Timeline

- P0-DS: 0 GPU-days (checkpoint exists)
- P0-UW: 1 day (retrain + expanded eval)
- P1a–P1d: 1.5 days each × 4 = 6 days
- P2a: 2 days
- P3a: 2 days
- P4: 2 days
- P2b / P3b: 2 days each, conditional

Unconditional total: ~13 GPU-days. With conditionals: ~17 GPU-days. Full T-BIOM extension: 25–30 days including reasoning layer and writeup.

---

## Known gotchas

- **P0-UW is mandatory before any downstream verdict**. Otherwise operator gains vs P0-DS are confounded with the fusion change.
- **CLIP suppression risk (P2a)**: CLIP may suppress tiny LDM reconstruction artifacts. If P2a fails, do not conclude the LDM signal is useless — try P3a (score-based, doesn't route through CLIP) before dropping. If both fail, LDM/diffusion operators genuinely aren't complementary at our evaluation resolution.
- **Frozen model versioning**: pick LDM version (SD 1.5 or SD 2.1) and stick with it across P2a and P3a.
  **RESOLVED 2026-08-08 — SD 1.5.** `stabilityai/stable-diffusion-2-1` and `-2-1-base` both
  return HTTP 404 on HuggingFace *even with a valid token* — removed/renamed, not gated — so
  SD 2.1 is not obtainable. P2a and P3a both use the ungated community re-upload
  `stable-diffusion-v1-5/stable-diffusion-v1-5` (full VAE + UNet + scheduler from one repo,
  so the "same version across both pilots" requirement holds). SD 1.5 is also the model
  AEROBLADE and LaRE² predominantly used, so this is closer to the source papers than SD 2.1
  would have been.
- **LGrad discipline**: substituting components in P4 while calling it LGrad conflates our contribution with theirs. Label deviations LGrad-inspired.
- **DFD gap propagates**: baseline DFD is 0.9419 vs paper 0.982 (frame-extraction pipeline gap). Interpret DFD deltas against 0.9419.
- **Single-seed pilots**: |Δ| in the 0.005–0.015 range needs a second seed before conclusion.
- **Celeb-DF++ availability**: if not integrated, mark pending and proceed on the rest. Do not block.
- **Deepfake-Eval-2024**: zero-shot on every pilot's checkpoint. It's the deployment-realism test, not a training extension.
- **MR-VAE β schedule**: needs to be documented explicitly; a bad schedule can make the rate–distortion response uninformative.

---

## Next steps after pilots

Three outcome branches:

**If P2a/P3a/P4 produce at least two Generalist verdicts** (ideally P3a + P4 covering diffusion and CNN generation families): proceed to full DISCERN v2 construction with surviving operators as evidence streams, P0-UW's uncertainty-weighted fusion, and response-matrix-derived reasoning layer.

**If mostly Specialist verdicts**: DISCERN v2 becomes a mixture-of-specialists architecture where the reasoning layer routes samples to the appropriate operator based on preliminary signal, rather than always-on ensemble. Different but still-defensible T-BIOM thesis.

**If mostly Redundant / Harmful**: generation-process consistency framing does not hold at our evaluation resolution. Reconsider (CLIP may suppress everything — try direct operator-evidence evaluation without CLIP fusion), or fall back to alternative framings.

The manifold-projector sub-suite (P1a–P1d) drives a smaller decision: whether to preserve or replace DiCoME's β-VAE in DISCERN v2. If any of P1b/P1c/P1d clearly beats P1a and P0-UW, port that variant. If not, deterministic AE is sufficient and β-VAE machinery is decorative.
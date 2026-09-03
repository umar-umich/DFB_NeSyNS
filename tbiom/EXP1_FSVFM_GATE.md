# Experiment 1 — does an FS-VFM view clear the Branch-3 gate?

Candidate (b), the **preservation student** (LoRA-adapted FS-VFM, `logs/fpad/studentA_preserve_seed42` epoch 9), is the default candidate because it carries the existing Step-5 and Step-7b evidence. `ordinary` is its sibling. `clip` is the V1 anchor's fused output, scored on the same videos in the same sweep.

`tau` frozen per arm at its own EER on **FF++ val**, applied unchanged everywhere. Every number is read from parquet already on disk — no GPU, no training.

| arm | tau |
|---|---:|
| preserve | 0.4239 |
| ordinary | 0.4405 |
| clip | 0.7091 |

## Zero-shot video AUROC and real-side FPR

| dataset | n | preserve AUROC | clip AUROC | Δ | preserve FPR_real | clip FPR_real |
|---|---:|---:|---:|---:|---:|---:|
| FF++ | 700 | **0.9867** | 0.9879 | -0.0012 | 0.036 | 0.029 |
| Celeb-DF-v2 | 518 | **0.8713** | 0.9246 | -0.0533 | 0.045 | 0.062 |
| DFD | 3431 | **0.8877** | 0.9210 | -0.0334 | 0.074 | 0.041 |
| DFDC | 4704 | **0.8552** | 0.8486 | +0.0067 | 0.199 | 0.157 |
| DFDCP | 654 | **0.8397** | 0.9022 | -0.0625 | 0.361 | 0.178 |
| Deepfake-Eval-2024 | 814 | **0.6821** | 0.6281 | +0.0540 | 0.320 | 0.248 |
| Celeb-DF-v1 | 100 | **0.8285** | 0.9045 | -0.0760 | 0.026 | 0.105 |
| UADFV | 98 | **0.9825** | 0.9975 | -0.0150 | 0.041 | 0.000 |

Mean over 8 sets: preserve **0.8667** vs clip **0.8893** (**-0.0226**); FPR_real 0.138 vs 0.102 (**+0.035**).

## Complementarity against CLIP — the promotion criterion

| dataset | n | CLIP wrong | rescue | harm | **margin** | error_overlap |
|---|---:|---:|---:|---:|---:|---:|
| FF++ | 700 | 34 | 0.382 | 0.015 | **+0.367** | 0.618 |
| Celeb-DF-v2 | 518 | 124 | 0.202 | 0.241 | **-0.040** | 0.798 |
| DFD | 3431 | 927 | 0.280 | 0.131 | **+0.150** | 0.720 |
| DFDC | 4704 | 1067 | 0.366 | 0.094 | **+0.272** | 0.634 |
| DFDCP | 654 | 114 | 0.465 | 0.163 | **+0.302** | 0.535 |
| Deepfake-Eval-2024 | 814 | 325 | 0.308 | 0.129 | **+0.179** | 0.692 |
| Celeb-DF-v1 | 100 | 22 | 0.182 | 0.244 | **-0.062** | 0.818 |
| UADFV | 98 | 2 | 0.000 | 0.042 | **-0.042** | 1.000 |

Mean margin **+0.141**, positive on **5/8** datasets.

## Sibling arm — ordinary

| dataset | ordinary AUROC | margin vs CLIP |
|---|---:|---:|
| FF++ | 0.9864 | +0.360 |
| Celeb-DF-v2 | 0.8758 | -0.080 |
| DFD | 0.8884 | +0.031 |
| DFDC | 0.8511 | +0.268 |
| DFDCP | 0.8259 | +0.231 |
| Deepfake-Eval-2024 | 0.6782 | +0.200 |
| Celeb-DF-v1 | 0.8646 | -0.126 |
| UADFV | 0.9858 | -0.021 |

## Verdict

- Near-peer AUROC: mean gap **-0.0226** — **yes**
- Positive rescue-harm margin: mean **+0.141**, positive on 5/8 — **yes**

**PROMOTE.** The preservation expert clears both gate conditions. It is the version to wire as Branch 3 in Experiment 2, under fused-only supervision, one variable against the Stage-5 baseline.

## Gaps

- **Celeb-DF-v3 is absent** from this sweep, so the seven-set suite is covered by six of its members plus Celeb-DF-v1 and UADFV as extras. CDFv3 would need a scoring run.
- Candidate (a), the frozen FS-VFM probe, needs an OOD feature-extraction pass and is reported separately.
- `clip` here is the **V1 anchor's** fused output, not DISCERN-Ext's P0-DS. It is the correct comparator for these parquet files because both were produced in the same sweep on the same videos, but it is not the Stage-5 framework.

---

# The controlled contrast that changes the diagnosis

The V1 anchor's crossdataset parquet carries `p_proc` / `u_proc` — **V1's own process branch**,
an independent implementation of the same AEROBLADE idea in a different framework. It is **not
vacuous**:

| framework | supervision | fusion | mean u_proc | process AUROC |
|---|---|---|---:|---:|
| **DISCERN-Ext Stage 5** | fused-only | DS | **0.996** (vacuous) | 0.51 – 0.60 |
| **V1** | fused **+ per-branch EDL** | CCF / DS | **0.351** | **0.621** (0.710 on DFDCP) |

Per dataset, V1's process branch: FF++ 0.6164, Celeb-DF-v2 0.6695, DFD 0.6411, DFDC 0.5959,
DFDCP **0.7098**, DFEval24 0.4914. Score range `[0.53, 0.90]` against our `[0.5008, 0.5229]`.

`discern_v1_model.py:276` — `auxiliary_losses()`, *"Per-branch EDL losses so each branch is
individually usable (§9 Stage B)."* V1 supervises each branch directly. We do not.

**This is the strongest available evidence for the vacuity mechanism, and it is evidence the fix
works** — the same operator, the same six statistics, supervised per-view, yields an informative
branch reaching 0.71 on DFDCP where ours reaches 0.60 and contributes nothing.

## What this is NOT

It is **not** a controlled experiment, and it must not be reported as one. V1 differs from
DISCERN-Ext in at least four ways beyond supervision, any of which could contribute:

1. **Fusion operator** — V1's primary is CCF (consensus & compromise), not the DS orthogonal sum.
   CCF routes conflicting belief rather than renormalising it, so a low-mass view is not the
   identity element there.
2. **Branch composition** — V1 fuses semantic / reference / process; DISCERN-Ext fuses semantic /
   artifact / process. The incumbent pair a third view competes against is different.
3. **Head construction** — different projection width and normalisation.
4. **Applicability discounting** — V1 discounts opinions by `q` before fusing; DISCERN-Ext does
   not.

So the honest claim is a **multi-view optimisation pathology in this DS-plus-fused-only setup**,
with per-view supervision as the leading — not the established — explanation.

## Alternative explanations that remain open

Listed so the eventual claim stays paper-safe. Each is a live candidate for why a competent view
goes vacuous here:

- **Correlated errors.** If the third view's mistakes coincide with the incumbent pair's, its
  evidence adds variance without adding information, and the loss prefers silence. Our
  `error_overlap` of 0.535–1.000 for the preservation expert is consistent with this.
- **Evidence-scale mismatch.** The two CLIP views arrive at `u ≈ 0.12–0.17`; a third view
  starting at `u ≈ 0.9` is an order of magnitude out. DS combination is not scale-free, so the
  weaker view may be numerically swamped before optimisation begins.
- **The EDL loss itself.** The evidential objective penalises confident errors sharply. For a
  view with 0.6 accuracy, the safest policy under that penalty is low evidence everywhere —
  independent of fusion.
- **Sequential DS chaining.** Three views fold pairwise, so the third is combined against an
  already-sharpened opinion rather than against the raw pair. Order effects were verified
  invariant for the *result*, but not for the *gradient*.
- **Gradient domination.** The semantic and artifact heads share the CLIP trunk and receive
  gradient through two paths; the process head has one. Its effective learning rate is lower
  regardless of anything about evidence.

Discriminating among these needs the isolated per-view-supervision experiment, which stays
deferred.

## P2a's third view — still inference

The direct measurement the plan asked for is **not possible from disk**. P2a exported
`evidence_semantic`, `evidence_artifact`, `evidence_fused` and their uncertainties for all eight
datasets, but **no operator evidence and no `u_operator`** — the analysis computed them in memory
and did not persist them. `analysis/distributions/uncertainty_*.csv` holds *fused* vacuity only.

P2a's checkpoints do survive at `DiCoME/runs/pilots/P2a/checkpoints/`, so the measurement is
runnable — its `P2aModule` stashes `last_operator_evidence` on the module for exactly this
purpose. Until then the claim stays labelled **inference**, resting on `error_overlap = 1.0` with
`rescue = 0` alongside `auc_operator = 0.943`, plus the identical fused-only loss construction.

The V1 contrast above does not settle it either way: V1 shows a process branch *can* be
non-vacuous under per-view supervision, which makes P2a's fused-only setup more suspect, not less.

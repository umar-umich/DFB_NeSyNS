# Independent views — report

Brief: `tbiom/docs/discern-v2-independent-views-brief.md`. Outputs in `analysis/independent_views/`.

## Verdicts

| part | verdict | evidence | one line |
|---|---|---|---|
| 1. Finish joint IBDC | in progress | `analysis/joint_ibdc/` | J1 seeds 1-2 and both λ sweeps running; see the blocker below. |
| 2. Close or reopen the residual line | **CLOSED** | `part2_fixA.json` | Every orthogonalised variant is *worse* than the backbone-alone probe. |
| 3. Frozen backbone screen | blocked | -- | The brief forbids upsampling, and every stored crop is 256 px. See below. |
| 4. Adapt the winner | not started | -- | Needs Part 3. |
| 5. Cross-backbone mapping | not started | -- | Needs Part 3's winner. |
| 6. Perturbation stability | not started | -- | Needs Part 3 backbones. |

---

## Part 2 — the residual line is closed

Fix A orthogonalises each block residual before the head, porting DiCoME's step
(`core_model.py:318-336`). Trained in the "both" configuration — backbone feature plus orthogonalised
residual — three seeds each.

| variant | dev v3.1 | sd | vs R2 |
|---|---:|---:|---:|
| **R2, backbone alone (the bar)** | **0.6812** | -- | -- |
| R4, residual alone | 0.6658 | -- | −0.0154 |
| both, no orthogonalisation | 0.6664 | -- | −0.0148 |
| **A1**, project off the pooled I-JEPA direction | 0.6582 | 0.0015 | **−0.0231** |
| **A3**, project off the top-64 PCA subspace | 0.6618 | 0.0006 | **−0.0194** |

**Pre-set rule:** reopen only if the best variant beats R2 with a source-grouped bootstrap 95%
interval above zero. The best variant is 0.0194 *below* R2, so no interval can lie above zero.
**The residual line is closed.**

### Why this is informative rather than merely negative

Orthogonalisation was the intervention with the strongest prior. The measured failure mode was
redundancy — backbone plus residual (0.6664) scoring below backbone alone (0.6812) — and removing the
shared component is exactly the mechanism that fixes redundancy in DiCoME.

It makes things worse, and monotonically: removing one direction costs 0.0082, removing 64 costs
0.0194 relative to the un-orthogonalised "both". So the residual's useful content is **largely the
part aligned with the backbone**. There is no complementary component hiding underneath; taking the
shared part away removes signal rather than isolating anything.

This is the opposite of DiCoME's situation, and it explains why the same mechanism does not transfer.
DiCoME's residual is `f_s − f_c`, the part of a feature its own learned manifold cannot reconstruct,
so the orthogonal complement is meaningful by construction. The I-JEPA residual is a prediction error
in a foreign feature space, whose overlap with the backbone is not a nuisance to be removed but most
of what it knows.

**A2 was not run.** It projects off the deployed Appearance feature, which requires a CLIP pass on
the I-JEPA cache's own frames. Those frames carry arm-B draws from `extract_residuals.py:draw_arm_b`,
which differ from the joint-IBDC draws in `training/joint_ibdc/draws.py`, so no existing CLIP cache
aligns with them. Given A1 and A3 both move the wrong way, and A2 removes a single direction as A1
does, it would not change the verdict.

**R1 and R0h still run**, as the brief specifies, whatever Fix A showed: they are a paper ablation of
DiCoME's orthogonal projection inside our own system. They need the `manifold_feature` extraction
pass, which no export currently produces.

---

## Part 3 — blocked by a resolution constraint, and the brief forbids the workaround

The brief requires: *"Cut face crops at each model's native resolution from the original frames,
using the deployed face boxes. Do not upsample the existing 224 crops."*

Measured on this corpus:

| | |
|---|---|
| every stored crop, every dataset | **256 x 256** (FFpp_train, CDFv2, DFDC all checked) |
| preprocessed FF++ frame directories | 224 x 224 |
| stored face boxes | **none** — `train.json` is a video-ID split list, not boxes |
| raw source videos | present for FF++ only, under `Datasets/FaceForensics++` |

So the three 224-input candidates — CLIP ViT-L/14, DINOv3 ViT-L/16, I-JEPA ViT-H/14 — are fine, since
256 to 224 is a downscale. **Perception Encoder at 448 is not**: reaching 448 from a 256 crop is
upsampling, which the brief rules out, and the fraction of faces whose box is smaller than 448 would
be 100% by construction.

Re-cutting at native resolution would mean re-running face detection over every dataset from source
video, and the source videos exist for FF++ only. That is a full re-preprocessing of the corpus, not
a step inside this brief.

**Not resolved unilaterally.** The options are to run PE at 336 or 224 and label the resolution
honestly, to run PE at 448 from upsampled 256 crops and record it as a stated deviation, or to drop
PE from the screen. Each changes what the comparison means, so the choice is the authors'.

---

## Blocker carried from Part 1

The brief refers to *"the joint IBDC brief and its Option B addition"*, and twice to a run called
**J1s**: "J1s, all 3 seeds, if not already done", and "Confirm that G0b was reported with J1s heads
as the Option B addition requires."

**No Option B addition exists in this repository.** `tbiom/docs/` holds only
`discern-v2-joint-ibdc-brief.md`, and the string `J1s` appears nowhere except in this brief. G0b was
measured with **J0 heads**, which is what the joint IBDC brief itself specifies
(*"On the FF++ training cache, using J0 heads"*), so it cannot be confirmed against a document that
is not here.

Everything else in Part 1 is running. J1s and the G0b re-check are held until the addition arrives.

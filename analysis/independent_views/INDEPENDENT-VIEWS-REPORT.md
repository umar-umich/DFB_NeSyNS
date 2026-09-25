# Independent views — report

Brief: `tbiom/docs/discern-v2-independent-views-brief.md`. Outputs in `analysis/independent_views/`.

## Verdicts

| part | verdict | evidence | one line |
|---|---|---|---|
| 1. Finish joint IBDC | in progress | `analysis/joint_ibdc/` | J1 seeds 1-2 and both λ sweeps running; see the blocker below. |
| 2. Close or reopen the residual line | **CLOSED** | `part2_fixA.json` | Every orthogonalised variant is *worse* than the backbone-alone probe. |
| 3. Frozen backbone screen | **no candidate beats frozen CLIP** | `screen_*.json` | CLIP wins on fused AUROC; the most *independent* candidate fuses worst. |
| 4. Adapt the winner | **stopped** | -- | The pre-set rule stops Parts 4 and 5 when no candidate beats frozen CLIP. |
| 5. Cross-backbone mapping | **stopped** | -- | Same rule. |
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


## Part 3 — the screen, and why Parts 4 and 5 stop

Three seeds per candidate. The head, recipe and selection are identical across candidates; only the
frozen backbone differs.

### Standalone

| backbone | dim | FF++ val AUROC | VALmix-v2 | dev v3.1 | best epoch |
|---|---:|---:|---:|---:|---:|
| **CLIP ViT-L/14, frozen** | 1024 | **0.9123** ±0.0014 | **0.7723** ±0.0003 | **0.7541** ±0.0007 | 9.7 |
| DINOv3 ViT-L/16 | 1024 | 0.8140 ±0.0004 | 0.7314 ±0.0011 | 0.7026 ±0.0011 | 1.0 |
| PE-Core B16/224 | 768 | 0.8418 ±0.0005 | 0.6983 ±0.0013 | 0.6761 ±0.0008 | 18.0 |

### Fused, with the candidate in the Appearance slot

| backbone | fused dev AUROC | real FPR | LOO(APP) | folds APP zeroed | phi(APP, OM) |
|---|---:|---:|---:|---:|---:|
| **CLIP, frozen** | **0.9126** ±0.0003 | **0.0750** | **+0.0033** | 2.00 / 5 | +0.2501 |
| DINOv3 | 0.9090 ±0.0002 | 0.1071 | −0.0003 | 3.67 / 5 | **+0.1768** |
| PE-Core B16 | 0.9093 ±0.0000 | 0.1131 | +0.0000 | **5.00 / 5** | +0.2906 |

**Pre-set rule:** the winner needs the highest fused out-of-fold development AUROC and must beat
frozen CLIP. CLIP is highest. **No candidate beats it, so Parts 4 and 5 stop.**

### The finding that matters: independence did not buy complementarity

This brief exists because Appearance and Off-manifold share a CLIP trunk, and the hypothesis was that
an independent backbone would supply the complementarity four fusion studies could not find.

DINOv3 **is** measurably more independent. Its error correlation with Off-manifold is the lowest of
the three, phi +0.1768 against CLIP's +0.2501 and PE's +0.2906. That is the quantity the brief's own
tie-break uses as a proxy for independence, and DINOv3 wins it clearly.

It fuses **worse**. Its Appearance-slot contribution is −0.0003, its weight is discarded in 3.67 of 5
folds, and the fused system lands 0.0036 below CLIP's with a real-video FPR 3.2 points higher.

So lower error correlation did not produce a better fusion. Being wrong on *different* videos is not
the same as being right on videos the others miss, and the reliability fit — which sees the
development objective, not the correlation — discards the more independent view anyway. This is a
direct negative for the premise, measured rather than argued, and it is the fourth distinct route to
improving this fusion that has failed.

### An unexpected result about the deployed Appearance view

The deployed system, with its LoRA-adapted Appearance view in the same slot, scores fused AUROC
**0.9070** with real FPR **0.1035**, LOO(APP) **−0.0023**, and Appearance retained in 2 of 5 folds
(`analysis/ijepa/fusion_eval_selfcheck.json`).

Frozen CLIP with a 20-epoch EDL head beats it on every one of those: **0.9126** fused, **0.0750** FPR,
**+0.0033** LOO. A 0.0056 gain in fused AUROC and a 2.9-point drop in real-video FPR, from *removing*
the adaptation.

This is not a clean frozen-versus-adapted contrast — the deployed view was trained end to end inside
DiCoME's full multi-task objective, with the β-VAE and alignment terms, while this head sees only the
EDL loss on a frozen embedding. But the direction is consistent with everything else measured here:
whatever the adaptation and the shared trunk do for the Appearance view alone, they cost the fusion.
That is worth a controlled follow-up, and it is a more promising lead than any of the six parts of
this brief.

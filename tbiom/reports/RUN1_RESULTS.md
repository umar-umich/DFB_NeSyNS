# Run 1 — results

Four trained configurations plus P0-DS, one protocol throughout: FF++ c23 seed 42, video-level
AUROC under the single video-identity rule, `tau` frozen per arm at its own EER on FF++ val.
Means are over the **six true OOD sets**; VALmix is reported but excluded from means because it
is the development set and F2's weights are fitted on it.

| arm | Branch 3 | fusion | loss |
|---|---|---|---|
| P0-DS | — (two views) | DS | EDL |
| armA | LoRA preservation student | DS | EDL, **fused-only** |
| **armB** | LoRA preservation student | DS | EDL **+ per-branch** |
| 1c | FSFM released fine-tuned ViT-L | DS | EDL + per-branch |
| 1d | FSFM released fine-tuned ViT-L | **mean of logits** | **cross-entropy** |

## Complete-framework video AUROC

| dataset | P0-DS | armA | **armB** | 1c | 1d |
|---|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | **0.9646** | 0.8638 | 0.9523 | 0.9555 | 0.9461 |
| Celeb-DF-v3 | 0.8409 | 0.8626 | **0.8888** | 0.8807 | 0.8782 |
| DFD | 0.9421 | 0.8812 | 0.9376 | 0.9308 | **0.9431** |
| DFDC | 0.8828 | 0.8559 | **0.8839** | 0.8705 | 0.8797 |
| DFDCP | 0.8573 | 0.8381 | **0.8991** | 0.8675 | 0.8801 |
| Deepfake-Eval-2024 | **0.6922** | 0.6789 | 0.6755 | 0.6669 | 0.6863 |
| VALmix (dev) | **0.8852** | 0.8276 | 0.8823 | 0.8766 | 0.8742 |
| **mean (6 OOD)** | 0.8633 | 0.8301 | **0.8729** | 0.8620 | 0.8689 |
| **vs P0-DS** | — | −0.0332 | **+0.0096** | −0.0013 | +0.0056 |

## Real-side FPR at frozen tau

| dataset | P0-DS | armA | **armB** | 1c | 1d |
|---|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | 0.129 | **0.045** | 0.067 | 0.157 | 0.157 |
| Celeb-DF-v3 | 0.129 | **0.045** | 0.067 | 0.157 | 0.157 |
| DFD | 0.174 | 0.083 | **0.044** | 0.143 | 0.298 |
| DFDC | 0.317 | **0.184** | 0.228 | 0.294 | 0.273 |
| DFDCP | 0.391 | 0.357 | 0.352 | 0.387 | **0.317** |
| Deepfake-Eval-2024 | 0.341 | **0.299** | 0.306 | 0.400 | 0.442 |
| **mean (6)** | 0.247 | **0.169** | 0.177 | 0.256 | 0.274 |

**armB meets the stated success condition**: real-side FPR does not regress on Celeb-DF-v2
(0.067 vs 0.129), Celeb-DF-v3 (0.067 vs 0.129) or DFDC (0.228 vs 0.317), and fused AUROC improves.

## Question 1 — does per-branch supervision keep the experts alive?

Yes, and it is the single largest effect in Run 1.

| arm | semantic u | artifact u | fsvfm u | semantic AUROC | artifact AUROC | fsvfm AUROC |
|---|---:|---:|---:|---:|---:|---:|
| armA fused-only | **0.9997** | **0.9966** | 0.029 | **0.5539** | **0.3906** | 0.8301 |
| armB aux-EDL | 0.049 | 0.051 | 0.043 | 0.8657 | 0.8686 | 0.8314 |
| 1c | — | — | — | 0.8421 | 0.8509 | 0.8539 |
| 1d simple-CE | n/a (logits) | n/a | n/a | 0.7975 | 0.7983 | 0.8496 |

Under fused-only supervision the **two CLIP incumbents collapsed** and the newcomer took over —
the opposite of Stage 5, where the third view died. So the pathology is **winner-take-all**, not
"weak views die", and which view survives tracks initial evidence scale: FS-VFM entered at
u = 0.09 against 0.76 and 0.62. Artifact at **0.391** is below chance, because a vacuous head's
probability ordering carries no information.

armA → armB is **+0.0428** mean AUROC. That is the intervention justified.

Plain cross-entropy on the mean of logits partially reproduces the effect but lets both CLIP
views decay to 0.798 — a 0.068 drop against armB — and gives the worst real-side FPR of any arm
(0.274, worse than P0-DS's 0.247). So the evidential parameterisation contributes beyond the
supervision itself, though its own margin (+0.0040 armB over 1d) is inside the noise band.

## Question 2 — which simple fusion?

Means over the six OOD sets. F2's weights are fitted on VALmix only and applied unchanged.

| arm | F1 averaging | F2 learned | F3 DS | verdict |
|---|---:|---:|---:|---|
| armA | 0.8295 | 0.8297 | 0.8297 | no difference |
| armB | 0.8778 | 0.8774 | 0.8768 | **no second defect** — spread 0.0010 |
| 1c | 0.8708 | **0.8778** | 0.8701 | F2 clears the band, **+0.0077** |
| 1d | — | — | — | logit arm, DS not applicable |

For armB the operator choice is noise: a 0.0005 spread says nothing. **DS is not the bottleneck
there.** For 1c it is different, and that matters — see below.

## The finding worth the paper: complementarity does not convert

FS-VFM against the incumbent **pair** (rescue = P(FS-VFM right | pair wrong)):

| arm | mean margin | positive on | framework AUROC |
|---|---:|---:|---:|
| armB (student) | +0.120 | 6/7 | **0.8729** |
| **1c (FSFM FT)** | **+0.335** | **7/7** | 0.8620 |

**1c's third view is far more complementary and produces the worse framework under DS.** On
Celeb-DF-v2 it rescues **71%** of the pair's errors at 12% harm, and fused CDFv2 still lands
below P0-DS. F2 recovers most of that (0.9660 against DS's 0.9582). So the complementarity is
real and **DS is failing to convert it** — which is the measured motivation for a conflict-aware
operator, and it appears on the arm with the most to convert rather than being assumed.

## Standalone strength did not predict framework contribution

The FSFM fine-tuned checkpoint, probed standalone through this pipeline, beat P0-DS on DFDCP
(+0.048) and Celeb-DF-v3 (+0.034). In-framework, 1c is **below** armB, whose Branch 3 is the
weaker LoRA student. Worth stating rather than hiding: a stronger view is not automatically a
better branch.

## What is not established

- **Every arm-vs-arm difference except armA is inside or near ±0.01.** Under the section-22 rule
  the ordering armB > 1d > 1c needs a confirming seed before it is a claim. The headline
  +0.0096 for armB against P0-DS clears the band; the arm ordering does not.
- **Deepfake-Eval-2024 is the one set where every arm loses to P0-DS** (best 0.6863 against
  0.6922). Nothing here helps on the modern in-the-wild set.
- **DFD does not reproduce FSFM's reported number** standalone: ours 0.8651 against their 0.9717,
  a 0.107 gap where Celeb-DF-v2 agrees to 0.032. Protocol difference, not diagnosed.
- **Checkpoints for armB, 1c and 1d were taken at best FF++ val AUROC**, not by the VALmix
  macro-AUROC protocol used earlier. armB's candidates were within 0.0001 so it cannot move the
  headline, but it is a deviation.
- **P2a's third-view vacuity remains inference**, not measurement: it exported no per-view
  operator evidence. Its checkpoints survive, so it is runnable.

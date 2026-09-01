# MR-VAE as the anchor — the P1d pilot, re-judged on the health dashboard

## The question

Can the β-VAE inside P0-DS be replaced with MR-VAE, on CLIP features, to get a better anchor?

**It was already run.** Pilot **P1d**, "MR-VAE + rate-distortion response", FF++ c23, seed 42 —
the same recipe as P0-DS with the projector swapped. Its checkpoints are on disk. It was
**rejected on a mean AUROC delta of −0.0127**, and that judgement was made before the health
dashboard existed.

P1d is not only a projector swap: its artifact branch is **69-d = 64 orthogonal-residual + the
5-point BETA_GRID rate response**. Feeding the rate curve into the artifact head is the pilot's
actual hypothesis.

## Port and correctness

`MRVAEProjector` ported into the fork behind `manifold_projector: beta_vae|mrvae`, default
`beta_vae`, so P0-DS and every existing config are untouched. Every layer shape matches P1d's
checkpoint exactly, and the port reproduces its published number:

| | published P1d | ported anchor | delta |
|---|---:|---:|---:|
| Celeb-DF-v2, fused | 0.9495 | **0.9505** | +0.0010 |

## The result the AUROC-only judgement missed

τ frozen per anchor and readout on its own FF++ val.

| anchor | readout | dataset | AUROC | EER | **FPR_real@τ** | d_RF |
|---|---|---|---:|---:|---:|---:|
| **MR-VAE** | fused | Celeb-DF-v2 | 0.9505 | 0.1365 | **0.067** | +0.519 |
| β-VAE | fused | Celeb-DF-v2 | **0.9646** | **0.0991** | 0.129 | +0.531 |
| **MR-VAE** | fused | **DFDCP** | **0.8957** | **0.1821** | **0.252** | **+0.443** |
| β-VAE | fused | DFDCP | 0.8573 | 0.2387 | 0.391 | +0.368 |
| MR-VAE | fused | VALmix | 0.8843 | 0.1970 | **0.175** | +0.457 |
| β-VAE | fused | VALmix | **0.8852** | **0.1859** | 0.213 | +0.439 |
| **MR-VAE** | artifact | Celeb-DF-v2 | 0.9513 | 0.1236 | **0.051** | +0.364 |
| β-VAE | artifact | Celeb-DF-v2 | **0.9685** | **0.0948** | 0.073 | +0.408 |
| **MR-VAE** | artifact | VALmix | 0.8742 | 0.2000 | **0.107** | +0.347 |
| β-VAE | artifact | VALmix | **0.8789** | 0.2044 | 0.127 | +0.368 |

**MR-VAE has lower real-side FPR than β-VAE in every single pairing**, and on DFDCP it wins on
both axes at once (+0.0384 AUROC and −0.139 FPR).

The pattern is consistent: MR-VAE trades a little ranking for a materially better operating
point. On Celeb-DF-v2 fused it gives up 0.0141 AUROC and halves FPR_real, 0.129 → 0.067. The
lowest real-side FPR measured on any anchor in this project is MR-VAE's artifact readout at
**0.051** on Celeb-DF-v2 and **0.107** on VALmix.

## Complete — all six OOD datasets, fused readout

τ frozen per anchor on its own FF++ val. Port verified against every published P1d number first
(max deviation 0.0010).

| dataset | MR-VAE AUROC | β-VAE AUROC | Δ AUROC | **MR-VAE FPR_real** | **β-VAE FPR_real** | **Δ FPR** |
|---|---:|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | 0.9505 | **0.9646** | −0.0141 | **0.067** | 0.129 | **−0.062** |
| Celeb-DF-v3 | **0.8849** | 0.8409 | +0.0441 | **0.067** | 0.129 | **−0.062** |
| DFD | 0.9349 | **0.9421** | −0.0073 | **0.058** | 0.174 | **−0.116** |
| DFDC | 0.8700 | **0.8828** | −0.0128 | **0.254** | 0.317 | **−0.064** |
| DFDCP | **0.8957** | 0.8573 | +0.0384 | **0.252** | 0.391 | **−0.139** |
| Deepfake-Eval-2024 | 0.6635 | **0.6922** | −0.0287 | **0.280** | 0.341 | **−0.061** |
| **mean** | | | **+0.0033** | | | **−0.084** |

**MR-VAE has lower real-side FPR on 6 of 6 datasets**, by 0.061–0.139, mean −0.084. That result
does not depend on any disputed number: both anchors were scored by the same exporter, on the
same videos, on the same day.

## A discrepancy that must be resolved before the AUROC mean is quoted

The mean AUROC delta reads **+0.0033** here against the pilot table's **−0.0127**, and the whole
difference is Celeb-DF-v3:

| | pilot table | re-measured here |
|---|---:|---:|
| P0-DS CDFv3 | 0.9516 | **0.8409** |
| P1d CDFv3 | 0.8852 | 0.8849 ✓ |

**P1d reproduces to 0.0003; P0-DS is off by 0.11 on the same 5,418 videos (178 real / 5,240
fake).** Both were scored through the same exporter in the same run, so this is not an evaluation
difference between the two arms.

The likely cause: `DiCoME/eval_adaptation/RESULTS.md` records that CDFv3 was once "face-swap-family
only (8 generators)", and the current config spans more. If the pilot table's CDFv3 column came
from that older, easier subset, its P0-DS entry is not comparable to today's — but then P1d's
should have shifted too, and it did not. **Unexplained, and it decides the sign of the mean:**

| basis | mean Δ AUROC |
|---|---:|
| all six, re-measured | +0.0033 |
| excluding CDFv3 | −0.0049 |
| pilot table (7 sources) | −0.0127 |

So the honest statement is: **MR-VAE is unambiguously better on real-side FPR, and its AUROC
standing ranges from −0.005 to +0.003 depending on a CDFv3 number that does not currently
reconcile.** It is not the −0.0127 penalty the pilot table implies, on any reading.

## What this does and does not establish

**Does:** the P1d rejection was made on a metric that could not see this. On the operational axis
the projector swap is an improvement, not a regression, and the pilot table's −0.0127 is not the
whole story.

**Does not:** these are three datasets (Celeb-DF-v2, DFDCP, VALmix), against the published
table's seven. The AUROC losses on Celeb-DF-v3 (−0.066) and Deepfake-Eval-2024 (−0.027) are large
and are **not** re-measured here; if they carry a real-side penalty too, the picture changes.
Scoring MR-VAE on the remaining four is the obvious next step and is cheap — the checkpoint
exists and the port works.

Also: P1d selected epoch 1 on saturated FF++ val AUROC, the same weak selection Step 2 found for
P0-DS, and its other checkpoints (epochs 3 and 13) have never been evaluated on anything else.

## Why MR-VAE ended up on FS-VFM rather than CLIP

It was on CLIP first — this pilot. The `mrvae_rate` expert, an MR-VAE operator fit on **FS-VFM**
features with a logistic rate probe, came later, after P1d underperformed on AUROC. That branch
was found inert (`MRVAE_OPERATOR_FINDING.md`) and failed the membership gate on every basis. The
two are different objects and only the CLIP one is an anchor candidate.

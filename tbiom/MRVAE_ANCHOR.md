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

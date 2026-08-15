# A3 verdict — does P1d's rate response carry incremental information?

Source: DF40, 13,237 frames / 2,927 videos. Leave-one-generator-out throughout; per-fold
scalers. Run 2026-08-15 from the P1d export produced with the MR-VAE projector.

Every probe is reported against a baseline built from evidence already available without
`R(x)` — P1d's own `p_fused`. A standalone AUROC proves nothing here, because a probe that
merely re-encodes the score would look excellent while adding nothing.

| Question | R(x) | baseline (`p_fused`) | **incremental** | folds |
|---|---|---|---|---|
| Q1 forgery separability | 0.8628 | 0.8504 | **+0.0122** | 70 |
| Q2 family structure | silhouette −0.109, Fisher 0.097 | — | — | descriptive |
| Q3 P0-miss prediction (fakes, video level) | 0.8708 | 0.9363 | **−0.0655** | 52 |
| Q3 rescue vs harm (fakes, video level) | 0.9639 | 0.9853 | **−0.0215** | 43 |

## Verdict

**The rate response does not justify P1d's privileged status.**

The revised gate says P1d survives if `R(x)` adds incremental forensic or applicability
information on *any* of the three questions. Read literally, it clears that bar — but only
on Q1, and only by +0.012 AUROC. On both applicability questions the response is *worse*
than the single scalar P1d already emits: −0.066 for predicting which fakes P0 will miss,
−0.022 for separating rescues from harms. Family structure is absent, confirming the prior
handoff rather than overturning it (silhouette is negative, i.e. points sit closer to other
families than their own).

So the letter of the gate is met by a hair and its spirit is not. The curve's *shape* was
the discriminative claim — that a genuine face decays gracefully under rate pressure while a
manipulated one falls off a cliff — and on the two questions that would make that shape
useful for routing, the shape underperforms the scalar. P1d should lose privileged status
going into D1.

## Two corrections worth recording

Both of these produced confidently wrong numbers before they were caught, and both are now
pinned in the code with comments explaining why.

**1. Pooling reals and fakes inverted Q3.** A P0 error means opposite things by class: a
real P0 gets wrong is a false positive (it looked fake), a fake P0 gets wrong is a miss (it
looked real). The train folds are fake-majority (~40 generators × 18) and the test folds are
real-majority (18 fakes vs ~416 reals), so a pooled probe learned one direction and was
scored on the other. It reported AUROC ≈ 0.17 — apparently strongly anti-predictive, in fact
a sign error. Measured slope deltas confirm the flip: −0.0022 on reals, +0.0077 on fakes.
Q3 is now evaluated within class.

**2. Frame-level scoring inflated the result, and the fold threshold silently ate the
folds.** Frames of one video share their outcome, so a frame-level AUROC counts the same
evidence dozens of times. Aggregating to video level was correct but exposed a second issue:
DF40 generators hold ~18 videos each, so the frame-scale `min_per_group = 20` discarded all
but 7 folds. A separate `--min-videos-per-group` (default 8) now applies to video-level
probes, restoring 52 folds.

The intermediate frame-level figure of 0.961 for Q3, computed before the baseline comparison
existed, should be disregarded — with the baseline in place the incremental is negative.

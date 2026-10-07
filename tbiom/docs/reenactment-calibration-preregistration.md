# Pre-registration: reenactment-weighted evidence calibration

Written and committed **before the run**, at UMAR's instruction (2026-10-07), because the idea it
tests was formed *after* seeing a result. Everything below is fixed now and is not revised once
numbers exist.

## Why this needs pre-registering at all

The uncalibrated grid search already ran and selected scale 1.0 — no calibration — because the
development objective is overall fused AUROC and that objective *prefers* a loud appearance view.
Reaching for a different objective after seeing that is choosing the measurement to get the answer.
So the new objective, the selection rule, the acceptance rule and the failure conditions are written
here first.

## The intervention

The adapted appearance view dominates the fusion by evidence MAGNITUDE, not by trust weight: the
temporal view holds weight 0.25 in every row, while the appearance view's trust rises from 0.15
(frozen) to 0.85 (adapted) and it emits near-saturated evidence (FF++ val AUROC 0.994).

`calibrate(t, tau)` scales the appearance view's **vacuity** by `tau`, leaving its probability
untouched. Its ranking and standalone AUROC therefore cannot move; only how loudly it speaks inside
the fusion changes. `tau > 1` makes it quieter.

Grid, fixed: **1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0**.

## The reenactment development population

Fitting uses development only. A development fake video is REENACTMENT if its subset is in this
list, which is fixed here and not adjusted afterwards:

```
Face2Face  NeuralTextures
DaGAN  FSRT  HyperReenact  hyperreenact  LIA  lia  MCNET  mcnet  TPSMM  tpsm
LivePortrait  liveportrait  AniTalker  EDTalk  EchoMimic  FLOAT  R3DPortrait  hellomeme
SadTalker  sadtalker  fomm  facevid2vid  danet  MRAA  one_shot_free  pirender
```

Face swap, whole-image and unlabelled subsets (`Deepfakes`, `FaceSwap`, `BlendFace`, `Celeb-DF-v2`,
`GHOST`, `HifiFace`, `InSwapper`, `MobileFaceSwap`, `SimSwap`, `UniFace`, `facefusion*`,
`CDFv2val-*`, `DFDCPval-*`, `DFEval24val-*`, `fake`) are excluded. **All real development videos are
kept**: a slice changes which manipulations are asked about, never which genuine videos must not be
flagged.

## Selection rule (fitted on development, fixed now)

Choose the `tau` that maximises **fused development AUROC on the reenactment population**, subject
to a guard:

> the overall fused development AUROC must not drop more than **0.005** below the `tau = 1` value.

If no `tau > 1` satisfies the guard, the selection is `tau = 1` and the row is reported as *no
calibration selected*, exactly as the unweighted attempt was.

## Acceptance rule on test (fixed now, evaluated once)

The calibrated row replaces the uncalibrated winner only if **all three** hold.

1. **CDFv3 reenactment** AUROC improves over the uncalibrated winner's 0.9591.
2. **KoDF first-order motion** AUROC improves over 0.9680.
3. The **ten-set mean** AUROC drops by at most **0.005** from 0.9212.

DF40 reenactment (0.9316) is reported beside these as corroboration but does **not** enter the rule,
because DF40 informed earlier operator design and is not a clean confirmation set.

## What failure looks like, and it is reported either way

- The guard rejects every `tau > 1` → the development objective cannot be moved without paying for
  it overall, and the trade is intrinsic rather than a tuning artefact.
- A `tau` is selected on development but conditions 1–3 fail on test → the calibration overfits the
  development reenactment population.
- Conditions 1–2 pass and 3 fails → the trade is real and quantified, and the decision becomes an
  explicit choice about what the system is for, not a free win.

A result in any of these shapes is the finding. The uncalibrated winner stays the reference unless
all three acceptance conditions hold.

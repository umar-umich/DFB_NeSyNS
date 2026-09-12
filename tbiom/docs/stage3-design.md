# Stage 3 design — fixed before any held-out result is examined

Recorded ahead of time so that feature choices, estimators and thresholds are frozen before the
OOD sets are touched. If any held-out result later drives a revision, those sets become
development data from that point on and the writeup says so.

## Correction 1 — preserved AUROC does not mean the drift was absorbed

Stage 2 reports three quantities per cue and severity, not two:

| observed | meaning |
|---|---|
| large drift, stable AUROC, stable FPR/TPR | drift tolerable under this evaluation |
| large drift, stable AUROC, **FPR rises** | ranking survives, calibration or threshold fails |
| AUROC declines | discrimination genuinely lost; no threshold fixes it |

FPR/TPR are measured at a threshold **frozen on baseline dev** and applied unchanged at every
severity, per cue, direction fixed a priori (higher `p` = more fake). A threshold refitted per
severity would re-centre on the drift and hide it.

AUROC is a group-level ranking measure and is not a statement about any individual sample's
reliability. It is reported as such.

## Correction 2 — severity estimation is a candidate, not a required component

The pipeline is **not** committed to `estimate severity → look up reliability`. Three arms, in
this order:

- **Main comparison (deployable).** M1/M2/M3 all receive the same deployment-available
  information: the cue measurements plus the model-free quality indicators. No processing
  settings, no oracle severity.
- **Diagnostic comparator (not deployable, reported separately).** The same mechanisms given
  the *known* experimental processing settings, purely to quantify whether knowing them helps
  at all. If knowing the settings buys little, explicit severity estimation is not worth
  building.
- **Optional follow-up.** Explicit estimation of processing settings, only if the diagnostic
  shows a meaningful benefit.

This avoids committing to a single severity scale across JPEG, resize, blur and their
combinations, which do not share one.

## Correction 3 — reconstruction error is not the severity signal

`recon_err` responds strongly to JPEG, but it is produced by a branch trained to detect
manipulation, so it may also depend on manipulation, identity and content. The risk is a
shortcut in which genuine manipulation evidence is reinterpreted as poor quality and discounted.

**Early evidence that this risk is real.** On FF++ val at `jpeg:70`, `recon_err` shifts
**−0.664 for real** against **−0.381 for fake** — the severity response is ~1.75× stronger on
the real class, and the real/fake gap *widens* from 1.527 to 1.811. One severity only, so
preliminary, but it already shows the response is class-dependent.

Therefore severity-related inputs come from **model-free quality indicators** computed from
pixels with no network in the path:

| indicator | what it targets | measured response |
|---|---|---|
| `blocking` | DCT block lattice, i.e. compression | 0.0067 → **2.407** under `jpeg:30`; 0.0162 under resize |
| `blur` | softening, i.e. resolution loss | 91.0 → **13.7** under `resize:0.35`; 109.8 under jpeg |
| `hf_energy` | effective resolution | 0.169 → 0.133 under resize |
| `contrast` | content control | ~54 under every condition, as intended |

`blocking` and `blur` respond to different processing families and barely to each other's, which
is what makes them usable as separate axes rather than one severity scale.

`recon_err` may still be offered to the mechanisms, but only as an **increment** on top of these
indicators, and its contribution is reported separately. Before it is used for reliability at
all it must pass: (a) severity relationship holding separately within each class, (b) transfer
across held-out source videos and processing families, (c) added value beyond the model-free
indicators.

## Correction 4 — same inputs, and the evaluation stays untouched

M1, M2 and M3 receive **identical** deployment-available information. If only M2/M3 saw
estimated severity or `recon_err`, any improvement would be attributable to the extra input
rather than to probabilistic or symbolic reasoning, and the comparison would be worthless.

Any predicted quality or reliability input used to train Stage 3 is generated **out of fold**,
with all processed versions of the same source video held in the same fold. Otherwise a model
sees the same face at `jpeg:70` in training and `jpeg:30` in validation and the split leaks.

Known processing settings stay out of the main comparison entirely.

## What each outcome would mean

- stable ranking with threshold failure → the problem is **calibration**
- lost discrimination → **cue replacement or conditional weighting**, not reasoning
- M1 ≈ M2 ≈ M3, all improved → the gain came from **better inputs**, and no reasoning advantage
  has been demonstrated

Stage 2 is useful in all three cases, including the ones that say simplify.

## Manifest correction

Stage 2 issues **20** passes: 2 dev datasets × 7 severities (fit) + 2 OOD datasets × 3
severities (held out). An earlier message said 16; that was an arithmetic error in the prose,
not a gap in the grid.

---

## Sequencing decision (2026-09-13)

**Stage 3 runs on the winner of the SBI training comparison, not on arm B.**

Stage 3's prerequisites are met — the Stage 1 measurement record exists for all eight datasets
and Stage 2's reliability curves are fitted — so it *could* run now against the frozen arm B.
It will not, and the reason is in the evidence-diagnostic brief itself:

> Select the improved training configuration using dev only. Freeze that detector, recompute its
> measurements, and run M1/M2/M3 with identical deployment-available inputs. New training
> invalidates old feature caches, reliability curves, and calibration parameters; characterize
> the new detector rather than reusing Arm B's fitted relationships.

Running Stage 3 on arm B now would produce M1/M2/M3 numbers describing a detector we may be
about to replace. If an SBI arm wins on dev, arm B's reliability curves, its calibration and its
`tau` all stop describing the deployed model, and every M2/M3 result fitted against them would
have to be discarded and recomputed. The cost of waiting is a delay; the cost of not waiting is
doing the stage twice and risking the first set of numbers being quoted.

### What this does not defer

Stage 4's third test — processing discrimination, the central empirical claim — is also
unblocked, and it is a property of *a* detector rather than of the eventual winner. It can run
on arm B now as a baseline reading, provided the writeup says which detector it describes. If
the winner differs, it is cheap to repeat.

### Order of operations

1. arms A / B / C finish training and export (running)
2. authenticity teacher trains, unblocking arm D
3. arm D runs once its FPNet dependency is satisfied
4. select on **dev only**, freeze the winner
5. re-export the winner's Stage 1 measurement record; refit Stage 2 reliability curves against it
6. Stage 3: M1 / M2 / M3 on identical deployment-available inputs
7. Stage 4's processing test against the frozen winner

Step 5 is not optional bookkeeping. It is the step that makes Stage 3's inputs describe the
model Stage 3 is about.

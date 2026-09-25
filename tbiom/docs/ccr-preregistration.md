# Conflict-conditioned reliability (CCR) — pre-registration

Written before any CCR number was produced. Fixes the arms, the features, the folds, the controls and
the acceptance rule, so that a negative result is reportable rather than discarded.

## The question

The deployed system fits ONE reliability triple per fold and applies it to every video. This asks
whether making reliability a function of **inter-view conflict** improves the fusion.

## Why this is not the study that already failed

`DECISIONS_AND_OPTIONS.md` records a reliability-controller ladder (MLP / GRU / Mamba over four
temporal segments) that predicted w from **quality descriptors** — blur, blockiness, contrast, motion,
luma — and failed: best learned GRU 0.8215 against constant DS-R 0.8235, delta −0.0020, interval
above zero in 0 of 5 folds. So "reliability should depend on the input" is already a documented
negative **for quality**.

CCR conditions on a different signal, and one with direct evidence behind it:

- **G0a** showed that conflict adds to vacuity when predicting the fused system's own errors:
  +0.0312 with a source-grouped 95% interval of [+0.0201, +0.0424]. Quality descriptors had no access
  to that.
- **The two-case diagnostic** showed the complementarity is family-structured, not quality-structured:
  the video view uniquely rescues 244 videos against 80 on reenactment-heavy DF40train, and the
  spatial pair rescues 168 against 21 on face-swap CDFv2val. One global triple cannot exploit that.

The bar is therefore specific: **does conflict carry what quality did not?**

## Fixed before running

- **Views and order** (T, APP, OM), the deployed spatial exports, the cache-derived temporal opinions.
- **Discount**: evidence scaling e' = r·e, the deployed convention, not Shafer.
- **Operator**: PCR6+-R, the differentiable implementation verified in Task 1 to reproduce the
  deployed fusion exactly (0.0 maximum absolute difference).
- **Folds**: the five source-grouped 50/50 development splits, `dsr_system.splits`. The controller is
  fitted on the fitting half and scored on the selection half. Never both.
- **Objective for fitting w**: binary cross-entropy on the fused projected probability, back-propagated
  through the fusion. The fusion is differentiable, so no surrogate is needed.
- **Selection metric**: the v3.1 development score, as everywhere else in this project.

## Arms

| arm | w comes from |
|---|---|
| `const` | one fitted triple per fold — the deployed baseline |
| `ccr` | a small network on CONFLICT features |
| `ccr-shuffled` | the same network on conflict features permuted across videos within a fold |
| `quality` | the same network on the quality descriptors — the known-negative reference |

## Conflict features, per video

The three pairwise conflicts kappa_ij computed from the undiscounted masses, the three per-view
vacuities u_b, and the three per-view margins −|p_b − 0.5|. Nine inputs. All are functions of the
opinions being fused, which is the same self-reference the controller study had, and is why the
fitting and selection halves are disjoint.

## Acceptance rule, pre-set

CCR is accepted only if **both** hold against `const`, on the selection halves:

1. mean v3.1 improves by at least **+0.005**, the same bar the controller study used; and
2. the paired source-grouped bootstrap interval of the difference is above zero in at least **3 of 5**
   folds.

`ccr-shuffled` must NOT pass. If it does, the gain is an artefact of the extra capacity rather than of
conflict, and CCR is rejected whatever its own numbers say.

## What is reported either way

Per-fold v3.1 for every arm, the paired differences with intervals, the fitted weight distributions,
and the per-forgery-type breakdown. A negative is written up as a finding, alongside the two operator
studies and the quality controller, as evidence that the fusion is not where this system's headroom
is.

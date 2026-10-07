# Strong branches, weak fusion: what we measured and where it points

Self-contained for discussion. Everything below is measured on this system; nothing is projected.
Source files are named so each number can be checked.

## 1. The system

Three evidential views are combined with **PCR6+ under reliability discounting (PCR6+-R)**.

| view | what it is |
|---|---|
| **APP** | appearance. Deployed: CLIP ViT-L/14 + LoRA. Candidate: PE-Core-L/336 adapted |
| **OM** | off-manifold, a beta-VAE orthogonal residual of the appearance feature |
| **T** | temporal, V-JEPA 2 ViT-L/16 + LoRA on 16-frame clips |

Each view emits Dirichlet **evidence**; `alpha = e + 1`, `S = sum(alpha)`, `p = alpha_fake / S`, and
**vacuity** `u = K / S` with `K = 2`. Vacuity is *how little evidence there is*, not `1 - p`.

Fusion fits a reliability triple `w` on 5 source-grouped development folds. The discount is
**evidence scaling**, `e' = r e` (`analysis/tbiom/grouped/ops.py:53`), over a 5-point grid
`{0, 0.25, 0.5, 0.75, 1}` constrained so `max(w) = 1`.

## 2. The pattern: four independent cases, same shape

Every one improved a branch and the system did not follow.

| case | branch, measured alone | the same branch, in the fusion |
|---|---|---|
| **DINOv3** vs CLIP | most *independent* of OM: error correlation phi **0.177** against CLIP's 0.250 | contribution **−0.0002**, weight driven to zero in **3.7 of 5** folds; fused 0.9091 vs CLIP's 0.9130 |
| **Adapted appearance** | dev v3.1 **0.743 → 0.868** (+0.125) | ten-set mean +0.008, but **reenactment −0.019** (CDFv3-reen 0.978 → 0.959) |
| **V-JEPA ViT-H** vs ViT-L | dev v3.1 **+0.053** (0.742 vs 0.689) | ten-set mean **−0.0165**; real-video FPR **0.067 → 0.117** |
| **Multi-layer readout** | best frozen row on PE-Core: contribution +0.0060 vs `last`'s +0.0008 | on CLIP the same readout goes **negative**, −0.0003 |

Evidence: `analysis/upgrade/screen_*.json`, `analysis/upgrade/final/*_s0.json`,
`analysis/independent_views/INDEPENDENT-VIEWS-REPORT.md`.

**Standalone branch quality does not predict fused behaviour.** Every keep rule in this study reads
the fused metric because of this, and the four cases above are why.

## 3. The mechanism, measured rather than assumed

The obvious explanation is that a louder view takes trust away from the others. **It does not.**

| row | w(T) | w(APP) | w(OM) |
|---|---:|---:|---:|
| CLIP frozen | 0.25 | **0.15** | 1.00 |
| PE-Core + LoRA | 0.25 | **0.95** | 0.80 |
| PE-Core + LayerNorm | 0.25 | **0.85** | 0.85 |

**The temporal view holds 0.25 in every row.** What changes is the appearance view's own trust and,
with it, the magnitude of evidence it emits: the adapted view reaches FF++ val AUROC **0.994** and
sits near saturation, so its vacuity is tiny. In the combination it dominates wherever it is
confident — including reenactment, where it is confidently mediocre and the temporal view is the one
that knows.

So the failure is **evidence-magnitude calibration across views**, not the operator and not the
weights. A branch trained to be better also trains to be *louder*, and PCR6+ has no mechanism that
asks whether a view's confidence is earned. It sees only the opinion.

## 4. Evidence that this is fixable

A **vacuity scale** on one view: `u' = tau u`, with `p` untouched. Because `p` does not move, the
view's ranking and standalone AUROC **cannot** change; only its loudness does. This is NOT the trust
weight, which scales evidence and therefore drags `p` toward 0.5 as well.

Pre-registered before running (`tbiom/docs/reenactment-calibration-preregistration.md`), fitted on
development against a reenactment objective under a guard that overall AUROC may not fall more than
0.005. Selected `tau = 2.0` on the appearance view. All three test conditions passed:

| condition, fixed in advance | result | |
|---|---|---|
| CDFv3 reenactment improves over 0.9591 | 0.9610 | **PASS** (+0.0019) |
| KoDF first-order motion improves over 0.9680 | 0.9720 | **PASS** (+0.0040) |
| ten-set mean drops at most 0.005 from 0.9212 | 0.9216 | **PASS** (+0.0004) |

Reenactment recovered **and** the overall mean rose slightly. The guard was never tested, because
there was no loss to bound.

**Why the first attempt failed.** The same knob fitted against the *overall* development objective
selected `tau = 1.0` — no calibration — because the grid is monotone against it (0.9329 at 1.0 down
to 0.9104 at 20). Reenactment is a minority of development, so a gain concentrated there is
invisible to an aggregate objective. The intervention was fine; the objective could not see it.

## 4b. The best configuration found (for context on what is being fused)

| recipe | dev fused | dev FPR | contribution | **test ten-set** | test FPR |
|---|---:|---:|---:|---:|---:|
| CLIP frozen (deployed-style) | 0.9130 | 0.0745 | +0.0037 | 0.9130 | 0.0698 |
| PE-Core-L + LoRA | 0.9302 | 0.0876 | +0.0224 | 0.9190 | 0.0920 |
| PE-Core-L + LayerNorm | 0.9329 | 0.0603 | +0.0248 | 0.9212 | 0.0666 |
| **PE-Core-L + LoRA & LayerNorm** | **0.9364** | 0.0810 | **+0.0308** | **0.9240** | **0.0627** |

The two adaptation methods touch disjoint parameters and compose: together they beat either alone on
development fused AUROC, on incremental contribution, and on the ten-set test mean.

## 5. Joint per-view calibration, and an unexpected result

Fitting all three scales together over `{0.5, 1, 2, 4}^3`, weights refitted inside every candidate,
same guard, reenactment objective (`upgrade/joint_calibration.py`):

| T | APP | OM | dev overall | dev reenactment |
|---:|---:|---:|---:|---:|
| 1.0 | 1.0 | 1.0 | 0.9332 | 0.9268 |
| **1.0** | **1.0** | **4.0** | 0.9286 (−0.0045) | **0.9369 (+0.0102)** |
| 1.0 | 2.0 | 4.0 | 0.9271 | 0.9377 *(fails guard)* |

**The joint fit quietens OM, not APP** — and gains five times more reenactment on development than
the appearance-only scale did. It sits right at the guard edge, trading 0.0045 overall for 0.0102.

**It then failed on test.** Ten-set mean **0.9035 against 0.9212** (−0.0177), reenactment +0.0051,
KoDF flat. It overfitted the development reenactment population, which is the risk recorded when it
was run: the development reenactment set is a fixed subset list, the test slices come from CDFv3 and
KoDF, and a scale selected on the former does not have to survive on the latter. The single-view
appearance scale, selected under a pre-registered rule, did survive.

## 5b. Vector-valued reliability: the fifth attempt, and its control

One scalar per view cannot say "trust this view on swaps but not reenactment". The premise is
measured and holds — the temporal view separates reenactment at 0.609 against swaps at 0.357, the
off-manifold view is the reverse at 0.296 and 0.481, and because the real videos are shared, the
asymmetry is entirely in the FAKE channel.

So the discount was given a vector: `e_fake' = w rho e_fake`, `e_real' = (w / rho) e_real`. Six
numbers instead of three; `rho = 1` recovers the deployed system exactly.

| | development reenactment |
|---|---:|
| fitted asymmetry | **+0.0014** |
| random asymmetry, best of 20 draws at the same magnitudes | **+0.0012** |

**0.0002 apart.** The gain is the extra parameters, not the direction — the same verdict CCR's
shuffled control delivered. Enriching the reliability REPRESENTATION has now failed five times:
grouped t-norms, direct three-source PCR6+, input-conditioned controllers, conflict conditioning,
and per-channel vectors.

## 6. Honest limits

- **Partial overlap with the trust weight.** Both act on magnitude, so some of the calibration gain
  is reachable by a finer weight grid. What calibration adds that weights cannot: it preserves `p`
  exactly, it can quieten *every* view at once (no weight triple can, since `max(w) = 1`), and it
  is fitted against a stated objective. The overlap is real and is not quantified yet.
- **Small margins, one seed.** The passing test deltas are +0.0004 to +0.0040. The pre-registration
  fixed *direction* as the test, not significance. Three seeds with source-grouped intervals are
  extracting now.
- **Development reenactment is not test reenactment.** The development population is defined by a
  fixed subset list; the test slices come from CDFv3 and KoDF. The joint result is selected on the
  former and must be judged on the latter.
- **One backbone, one operator.** Everything here is PE-Core-L under PCR6+-R.

## 6b. The distinction that actually separates the successes from the failures

Every failed intervention acts on **evidence**, and evidence scaling drags `p` toward 0.5: it
quietens a view by corrupting what it claims. The one intervention that passed a pre-registered test
scales **vacuity** with `p` untouched — the view says exactly what it said, only less loudly.

| intervention | acts on | result |
|---|---|---|
| reliability weight (deployed) | evidence | the baseline |
| grouped t-norm, direct PCR6+ | operator | negative |
| controllers (quality), CCR (conflict) | evidence, input-conditioned | negative, controls match |
| per-channel vector | evidence | negative, control matches |
| **vacuity scale, p preserved** | **vacuity** | **passed its pre-registered test** |

That is a hypothesis, not a finding: it rests on one success against six failures, and the success is
a single scalar on one view. **The experiment that would test it directly has not been run** — take
the same magnitude change, apply it once as an evidence scale and once as a vacuity scale, on the
same folds, and see whether the p-preserving form systematically wins. If it does, the question for
this fusion stops being "what should reliability be a function of" — five answers, all no — and
becomes "what is a discount allowed to change".

## 6c. The matched-magnitude experiment (specified before running)

**Question.** Holding the amount of quietening fixed, does it matter *how* a view is quietened?

**The two arms.** A view's opinion has `alpha = e + 1`, `S = sum(alpha)`, `u = 2 / S`.

| arm | operation | effect on `p` |
|---|---|---|
| **EVIDENCE** (what the deployed discount does) | `e' = r e`, so `u' = 2 / (r (S - 2) + 2)` | moves toward 0.5 |
| **VACUITY** (what passed the pre-registered test) | `u' = m u` | **unchanged** |

**Matching.** The arms are made comparable by equal MEAN VACUITY on the fold, not by equal
parameter: for each target multiplier `m` the vacuity arm uses `m` directly, and `r` is solved
numerically so the evidence arm reaches the same mean `u`. Without this the comparison would be
between two different amounts of quietening and would say nothing about the mechanism.

**Grid.** `m` in {1.25, 1.5, 2, 3, 4}, applied to each view alone and to all three together.

**Read-out.** Fused development AUROC, overall and on the reenactment population, with the trust
weights refitted inside every candidate.

**Pre-set interpretation, fixed before the numbers exist.**
- If the vacuity arm beats the evidence arm at matched vacuity across most of the grid, the relevant
  property is that `p` is preserved, and "what a discount is allowed to change" is the right question.
- If the two arms track each other, the earlier success was about the amount of quietening and not
  its form, the hypothesis is withdrawn, and the appearance-scale result stands as a tuning win
  rather than evidence of a mechanism.
- If the evidence arm wins, the hypothesis is wrong in the opposite direction and that is reported.

## 6d. Result: the hypothesis is WRONG, in the opposite direction

Twenty cells — four view-sets by five multipliers — each with the two arms matched to the same mean
vacuity to four decimal places.

| | vacuity arm wins | mean delta (vacuity − evidence) |
|---|---:|---:|
| overall | **4 / 20** | **−0.0015** |
| reenactment | **4 / 20** | **−0.0017** |

**The evidence arm is better at matched magnitude**, which is the third of the three readings fixed
in advance. Preserving `p` is not the useful property; it is slightly harmful. The hypothesis in 6b
is withdrawn.

**What this does to the earlier success.** The appearance vacuity scale that passed its
pre-registered test was therefore not evidence of a mechanism. At the same amount of quietening, the
ordinary evidence discount would have done as well or better. What made that result work was the
**objective** — fitting against reenactment instead of the aggregate — not the form of the knob. The
test-set gains it produced are real and still stand; the explanation offered for them does not.

**One thing the grid did show.** Discounting all three views together through EVIDENCE at `m` = 3–4
reaches overall **0.9351** against the baseline's 0.9332, a gain of +0.0019 from nothing but a
uniform temperature on every view at once — a region the weight grid cannot reach, since `max(w) = 1`
pins at least one view at full volume. That is a small, unexplained positive and the only live thread
left in this section.

## 7. Where this points

1. **The objective, not the knob.** This is the one conclusion that survived its own test. The same
   intervention was worthless against overall development AUROC and valuable against a reenactment
   objective, and the matched-magnitude experiment then showed the knob's form was irrelevant. Any
   future fusion tuning should state which objective it serves; aggregate AUROC cannot see a gain
   concentrated in one family.
2. **Uniform discounting of all views is unexplored and slightly positive.** +0.0019 overall at
   `m` = 3–4 on every view at once, which the constrained weight grid cannot express. Small, but it
   is the only untested direction this study turned up.
3. **Confidence should be earned, not asserted.** The deep issue is that a view's evidence magnitude
   reflects how hard it was trained, not how reliable it is on the sample at hand. A per-sample
   reliability estimate — rather than one scalar per view per fold — is the direction this points,
   and nothing here tests it yet.
4. **Stop reading standalone numbers as evidence about the system.** Four cases, no exceptions.

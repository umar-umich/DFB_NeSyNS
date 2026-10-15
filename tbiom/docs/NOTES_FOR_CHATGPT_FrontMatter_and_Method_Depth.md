# Revision notes — Abstract, Introduction, Background, and the depth the method section lacks

Companion to `NOTES_FOR_CHATGPT_Ch5_Ch6.md`. For ChatGPT.

The front matter currently *announces* a method. It never establishes why three particular
representations were chosen, what each one is sensitive to, or why a conflict-redistribution
rule is needed instead of the standard one. Those are the questions a reviewer asks first, and
the answers exist as measurements — they are simply not in the text.

**Everything numeric below is measured. Do not invent numbers. If something is not here, mark
it `[TO MEASURE]` rather than writing a plausible value.**

---

## PART A — THE ONE RESULT THAT SHOULD ORGANIZE THE WHOLE DOCUMENT

Per-branch AUROC on DF40's 41 generator sets, split by manipulation family:

| branch | FaceSwap | ExpressionSwap | gap |
|---|---:|---:|---:|
| **VD** (Video Dynamics) | 0.8933 | **0.9452** | **−0.0519** |
| **LFS** (Latent Facial Structure) | 0.9684 | 0.7855 | +0.1828 |
| **STT** (Spatio-Temporal Texture) | 0.9829 | 0.7963 | +0.1867 |
| CROSS-EXAM | **0.9894** | 0.9335 | +0.0559 |

Pairwise score correlations over the twelve test sets, which say the same thing a second way:
**VD–LFS +0.475, VD–STT +0.528, LFS–STT +0.742**. The video branch is the least correlated with
either image branch, and the two image branches are the most alike — yet LFS and STT still each
earn a place, because the ablation shows every pair is beaten by the triple.

**VD's sign is opposite to the other two.** LFS and STT are each ~0.18 worse on reenactment
than on face swap; VD is 0.05 *better*. The branches do not merely differ in architecture —
they fail on **opposite manipulation families**, and the fusion is above every single branch on
face swap while retaining most of VD's reenactment ability.

This is complementarity as a measurement rather than an assertion, and it is the strongest
evidence in the dissertation. It belongs in the Abstract (one clause), the Introduction (as the
motivating observation), and as the opening table of the method chapter. At present it appears
nowhere.

**State the honest caveat with it.** On reenactment, **VD alone (0.9452) beats the fusion
(0.9335) by 0.0117**. The reliability coefficients are fitted once on a development pool that
is face-swap-heavy, so VD is underweighted exactly where it is strongest. This is a real
limitation, it points directly at family-adaptive or conflict-aware weighting as future work,
and declaring it costs nothing — a reviewer who finds it unaided will trust the rest less.

---

## PART B — ABSTRACT

Rewrite around outcomes. The structure to hit, in roughly this order:

1. **The problem, quantified.** Detectors reach ~0.99 in-domain and fall to 0.83–0.90 across
   datasets; the fall is worse on manipulation families absent from training.
2. **The gap in prior evaluation.** Published cross-manipulation tables (including DiCoME's)
   report **face swap only**. Reenactment is half of DF40 and the half where current methods
   are weakest — our trained DiCoME reaches 0.7896 there, and on individual generators
   (mcnet 0.5283, facevid2vid 0.6264) it is at or near chance.
3. **The method in one sentence.** Three complementary facial representations, each producing
   a Dirichlet opinion, combined by direct multisource PCR6+ under an evidential
   multi-view formulation, trained in two stages.
4. **The headline numbers.** Twelve cross-dataset benchmarks and 21 DF40 generators.
   AUROC **0.9468** against **0.9039** for the strongest baseline. At matched 90% sensitivity,
   **14.1%** of genuine videos flagged against **31.5%** — less than half the false-alarm rate.
   Calibration leads on all three of ECE (0.1230), Brier (0.1048), E-AURC (0.0447).
5. **The distinguishing result.** The largest margin is on reenactment: **0.9300** against
   **0.7240** for DiCoME's own checkpoint.

Do not use "novel", "robust" or "comprehensive" unqualified. Each should be replaced by the
measurement that earns it.

---

## PART C — INTRODUCTION

Four specific repairs.

**C1. Quantify the generalisation gap in the first paragraph.** Name in-domain versus
cross-dataset numbers. The reader should meet the problem as a number, not an assertion.

**C2. Separate adopted from novel, explicitly.** The current contribution list reads as though
everything is ours. Write it as:

> *Adopted:* evidential deep learning with Dirichlet opinions (Sensoy et al., NeurIPS 2018);
> evidential multi-view fusion (Han et al., **ICLR 2021**); proportional conflict redistribution
> PCR6/PCR6+ (Smarandache & Dezert); the pretrained encoders (V-JEPA 2, CLIP, PE-Core) and
> low-rank adaptation (Hu et al., ICLR 2022).
>
> *Ours:* (i) the three-representation set, shown by ablation and by the per-family split to
> be jointly necessary rather than assumed to be; (ii) substituting direct multisource PCR6+ for Dempster's rule inside the
> evidential pipeline, evaluated jointly over sources rather than as a pairwise fold;
> (iii) one-sided IBDC, which removes a certainty incentive that exists under shared ignorance;
> (iv) retaining conflict K as an output distinct from vacuity;
> (v) a cross-manipulation evaluation that reports reenactment, which prior tables omit.

**C3. State what would falsify the claim.** One sentence: if the three representations were
redundant, removing one would not cost accuracy and their errors would coincide across
manipulation families. The ablation (§4.1 of the companion notes) and the family split (Part A)
are exactly those tests.

**C4. Name the forensic operating point early.** The dissertation's framing is evidentiary use.
Then FPR at matched sensitivity, not AUROC, is the headline metric — say so in the Introduction
so the choice of metric later does not look opportunistic.

---

## PART D — BACKGROUND (Chapter 2): THE MISSING CHAIN

§2.6 "Evidential Learning and Multi-View Fusion" is where the method becomes legible or does
not. It must trace a chain, each link motivating the next:

1. **Subjective logic and Dirichlet evidence** (Jøsang). Belief masses plus an explicit
   uncommitted mass; `b_G + b_F + u = 1`.
2. **EDL** (Sensoy et al., NeurIPS 2018). A network emits evidence `e`; `α = e + 1`;
   `S = Σα`; `u = K/S`. Vacuity falls out of evidence magnitude rather than from a softmax
   margin. **Say why this matters here:** a softmax detector that has never seen a manipulation
   family still outputs a confident probability; an evidential one can report low evidence.
3. **Evidential multi-view fusion — TMC** (Han et al., **ICLR 2021**). Per-view Dirichlet
   opinions combined across views by Dempster's rule. *This is the framework CROSS-EXAM builds
   on, and the current draft does not cite it at all.*
4. **The limitation that motivates our change.** Dempster's rule normalises by `1 − C`, so as
   the conflict coefficient `C → 1` the result becomes unstable and can certify a class that
   neither source strongly supports — Zadeh's counterexample. Cross-manipulation detection is
   precisely the high-conflict regime: a face-swap specialist and a motion specialist *should*
   disagree on reenactment.
5. **PCR6/PCR6+** (Smarandache & Dezert) as the response: redistribute conflicting mass back to
   the sources that committed to it, proportionally, instead of normalising it away. Note that
   it is **not associative**, which is why §5.2.7's joint multisource evaluation (rather than a
   sequential pairwise fold) is the correct implementation and not an arbitrary choice.

Also add to Background, because the results depend on them and nothing sets them up:

- **Why reenactment is harder than face swap.** Identity is preserved and motion is altered, so
  identity- and texture-based cues weaken while temporal cues strengthen. This single paragraph
  explains Part A's table.
- **Why mid-depth features.** Layer-probing evidence that the most transferable visual features
  are not at the output of a contrastively-trained encoder — the measurement is in Part E3.

---

## PART E — THE BRANCH FORMULATIONS (the depth that is missing)

Each branch needs: *what it is → why this choice → what it is sensitive to → evidence*. The
proposal currently gives one sentence per branch. Below is the material for one subsection each.

### E1. VD — Video Dynamics

**Construction.** V-JEPA 2 ViT-L, 16-frame clip at 256², LoRA (r=8, α=16, dropout 0.1) on the
**encoder** query and value projections only — 48 adapted modules, 1.20M trainable of 327M.
Clip-level embedding → evidential head → one Dirichlet opinion per video.

**Why a predictive video encoder rather than an optical-flow or frame-difference model.**
V-JEPA 2 is trained to predict *masked spatiotemporal regions in latent space* rather than to
reconstruct pixels or classify actions. Its representation therefore encodes how a region
*should* evolve, which is the quantity a reenactment violates: the manipulation preserves
identity and texture but drives motion from a different source. A flow-based cue measures
motion; a predictive cue measures motion **consistency with the rest of the scene**.

**Evidence.** VD is the only branch better on reenactment than on face swap (0.9452 vs 0.8933).
It is also the weakest branch overall on the twelve datasets (0.8477), which is the point — it
is a specialist, and the ablation shows its removal costs the most on the family it owns.

**Cite.** V-JEPA 2 (Assran et al., 2025); LoRA (Hu et al., ICLR 2022); JEPA/I-JEPA for the
predictive-latent framing.

### E2. LFS — Latent Facial Structure

**Construction.** CLIP ViT-L/14 with LoRA, plus an MR-VAE over the facial-feature space. The
branch output is the **orthogonal residual**: the component of the encoded face not explained
by the VAE's learned manifold of genuine facial structure, with `p_artifact` and `u_artifact`
as its opinion.

**Why a residual rather than a classifier on CLIP features.** A discriminative head on CLIP
learns the manipulations in the training set. A reconstruction residual asks a different
question — *is this face on the manifold of real faces?* — which does not require having seen
the generator. That is the anomaly-detection framing and it is why LFS transfers to generators
absent from training.

**Why it is not redundant with STT despite both being image-based.** They are built on
different encoders (CLIP vs PE-Core) and, more importantly, on different *objectives*: LFS is a
generative-manifold residual, STT a discriminative mid-depth texture representation. They are
the most correlated pair in the method (+0.742), so this is the place a reviewer will press —
answer it with the ablation, where LFS+STT (0.9379) is beaten by the triple (0.9468) and
neither alone reaches either pair.

**Cite.** CLIP (Radford et al., ICML 2021); β-VAE (Higgins et al., ICLR 2017); reconstruction-
based anomaly detection; the DiCoME artifact view as the immediate predecessor of this
construction — this is the one place DiCoME is genuinely upstream, and it should be credited
*here* rather than for the fusion.

### E3. STT — Spatio-Temporal Texture

**Construction.** PE-Core-L/14, **block 16 of 24** patch-token mean (prefix tokens dropped) →
256-d projection → temporal positional encoding from **real relative timestamps** → 2 pre-norm
transformer blocks → masked mean over frames → evidential head.

**Why block 16 and not the output.** Measured on a frozen layer probe: block 16 scores
**0.8019** against **0.7356** at the final layer — **+0.066**. The effect is **encoder-specific**:
on CLIP the same probe goes the other way (mid 0.7311, last 0.7478). A contrastively-trained
encoder's final layer is optimised for image–text alignment, which discards exactly the
low-level texture statistics manipulation detection needs; PE-Core retains them at mid depth.
This is a measurement, not a heuristic, and it is one of the more defensible choices in the
architecture — present it as such.

**The temporal encoding.** Frames are sampled non-uniformly and videos drop frames, so the
temporal stage encodes **real relative timestamps** rather than assuming uniform spacing.
Describe this as a design decision that follows from how the data is sampled. **Do not claim it
is validated** — we have not reported a controlled comparison against a uniform-slot encoding,
so an improvement must not be asserted for it.

**Cite.** PE-Core / Perception Encoder (Bolya et al., 2025); layer-wise probing of pretrained
transformers; sinusoidal/continuous temporal encodings.

### E4. The evidential head (shared by all three)

Identical across branches: a linear head emits non-negative evidence `e ∈ R²` via softplus;
`α = e + 1`; `p_fake = α_F / S`; `u = 2/S`. Trained with the Dirichlet type-II NLL plus annealed
KL to a uniform prior with the target class masked (Eq. 5.15 — already correct in the draft).

**Make explicit why `u` is not `1 − max p`.** The baselines emit only a decision margin; `u`
here is a function of total evidence, so a branch asked about a manipulation family it has
never seen can report a *low-evidence* opinion rather than a confident wrong one. This is the
mechanism behind the E-AURC result (0.0447 against 0.0542 for the best baseline) and the draft
never connects the two.

### E5. Why three

Every pair is beaten by the triple — VD+LFS 0.9302, VD+STT 0.9356, LFS+STT 0.9379, against
0.9468 — and every single is beaten by every pair that contains it. Each view therefore carries
something the other two miss, which is the claim the architecture rests on and the ablation is
the test of.

The per-family split (Part A) is the mechanism behind it: VD is the only view better on
reenactment than on face swap, so the pairs that exclude it inherit the image branches' weakness
there. State the size of the set as a consequence of those two tables, not as a stopping point.

---

## PART E-bis — THE FUSION RULE, AND HOW TO ARGUE FOR IT

The operator comparison reports **AUROC, FPR at TPR 90%, and mean vacuity** over seven rules on
the same branches and the same discounted masses.

| fusion rule | AUROC | FPR@TPR90 | mean vacuity |
|---|---:|---:|---:|
| Mean (unweighted) † | 0.9453 | 0.1473 | — |
| Mean (fitted weights) † | 0.9461 | 0.1455 | — |
| Dempster | **0.9484** | 0.1433 | 0.0436 |
| Yager | 0.9453 | 0.1414 | 0.2390 |
| WBF (Jøsang) | 0.9459 | 0.1491 | 0.1912 |
| Cautious (Denœux) | 0.9440 | 0.1554 | 0.1346 |
| **PCR6+-R (ours)** | 0.9468 | **0.1406** | **0.0322** |

† no opinion, so no vacuity.

**The argument for PCR6+-R has to be made, not implied by a bold number.** Write it as:
the rule is chosen for the operating characteristic the application is read at — the lowest
false-alarm rate at matched sensitivity — and its mean vacuity confirms it is doing what PCR6
is designed to do, returning conflicting mass to the sources that committed to it rather than
normalising it away. It is second on AUROC by 0.0016, which the text should state.

**Connect this to the Background chain (Part D).** Dempster normalises conflict away through
`1 − C`; PCR6+ redistributes it. The vacuity column is the direct evidence of that difference:
0.0322 against Dempster's 0.0436, and against Yager's 0.2390, which parks conflict in the
uncommitted mass instead. A reader who has Part D's chain can read that column as a mechanism,
not a statistic. This is the single best place in the paper to show the theory and the
measurement meeting.

---

## PART F — TONE

Three habits to fix throughout the front matter:

1. **Replace adjectives with measurements.** "Substantially improves" → the number.
   "Robust to processing" → which corruptions, what degradation.
2. **Attribute every borrowed component on first use.** The method is a composition of known
   parts plus specific changes; saying so makes the specific changes visible rather than
   burying them.
3. **State limitations in the body, not only in a closing section.** Two belong in the method
   chapter where they arise: the fixed reliability weights underperforming VD on reenactment
   (VD alone 0.9452 against the fusion's 0.9335), and PCR6+-R sitting second on AUROC, 0.0016
   behind Dempster, while leading the false-alarm rate it is adopted for. Both concern
   components that ARE in the method, which is why they belong in the text; a limitation of
   something the paper does not contain has no place to go.

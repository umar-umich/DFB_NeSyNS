# Claude Code — DiCoME deep-dive gap analysis (instructions)

**Goal.** DiCoME's cross-dataset results are strong (~0.938 avg) and we reproduced them
cleanly on our own infra. Before committing the three-slot DISCERN v2, do a disciplined gap
analysis: attribute DiCoME's performance to specific mechanisms, and decide — element by
element — whether DISCERN v2 is missing anything worth borrowing. Produce **one** analysis
`.md` for Umar to review; he'll iterate on these instructions from it.

This is a **read-only static analysis** task. No training, no eval runs. Cite `file:line`,
actual config values, and our reproduction logs — **no invented numbers**. Work read-only in
`../DiCoME/`; do not modify their code. If a claim would need a run to confirm, list it as a
🟡 proposed confirmation for Umar — do not launch it.

## The discipline (read before analyzing)

Separate two kinds of borrowing and tag every finding as one or the other:

- **Recipe / engineering — borrowable, zero novelty cost.** EDL loss form + annealing
  schedule, evidence-head activation/clamping, LoRA/LN config, augmentation + face-crop
  pipeline, LR schedule/warmup, class balancing, video-frame sampling + aggregation,
  numerical-stability tricks. These are the *most likely* missing ingredients and the ones
  that silently explain a performance gap.
- **Architectural idea — understand, do NOT adopt.** The co-trained β-VAE-in-CLIP-manifold,
  the orthogonal-projection artifact view, DS multi-view combination as the *core mechanism*.
  We deliberately diverged here (reals-only frozen reference; applicability + V/C/A as the
  novelty). Analyze these to understand *why they work for DiCoME*, not to copy them —
  copying them erodes our contribution.

Two hard rules for the write-up:
1. Every proposed borrowing must state **which specific gap/confound in DISCERN v2 it would
   close**, and be checked against our current plan and our deliberate rejections. No "DiCoME
   does X, so we should too."
2. Be **balanced**. Document where DiCoME is *weak* too — self-admitted diffusion transfer
   (MidJourney 0.689 / CollabDiff 0.678 zero-shot), no Deepfake-Eval-2024, both views living
   in the CLIP manifold — so we don't borrow things that only help on its favorable
   benchmarks. Remember our own finding that in-domain FF++ val cannot rank cross-domain
   detectors; attribute performance to mechanisms with evidence, not to the headline number.

## Sections the analysis .md must contain

**1. Loss & EDL formulation (highest priority — the classic hidden-gap area).**
Exact EDL variant (Sensoy MSE+KL Bayes-risk / cross-entropy / digamma), the evidence
activation (softplus/exp/relu) and any clamping/normalization. The evidential-regularizer
**annealing schedule** — is there a KL warmup `λ_t`, what coefficient, what shape? The VAE
loss form, its `λ_VAE` weight and any schedule, and exactly how it's added to `L_EDL`. For
each, state DISCERN v2's current equivalent and flag mismatches. *(If DiCoME anneals the KL
term and DISCERN doesn't, that alone can explain a gap.)*

**2. Multi-view construction & DS fusion.**
How the artifact view is built (β-VAE residual + orthogonal projection — the exact ops). The
DS combination rule verbatim (conflict-mass handling, normalization). Then the load-bearing
question: **why does DS beat mean fusion** — our P0-UW pilot proved uncertainty-weighted mean
is harmful, so what property of DS carries the result? How they handle (or don't) shared
ignorance / conflict — this directly informs our V/C/A design.

**3. Backbone & adaptation.**
LoRA rank + target modules, LN-tuning specifics, which CLIP, what's frozen vs trained, and
where the ~868K trainable params live. This feeds our Task 0 (frozen vs tuned CLIP) directly:
what exactly does DiCoME tune, and could freezing cost us what it buys them?

**4. Data & training recipe.**
Face detection/crop/align, input resolution, the full augmentation list + probabilities,
sampling/class-balance, batch size, LR + schedule + warmup, epochs, early-stopping, seed,
TTA, and video-frame sampling + aggregation (mean / max / learned). **Deliverable within this
section: a direct side-by-side diff of DiCoME's working training config against DISCERN v2's
current defaults**, with every difference flagged as "plausibly matters / probably neutral."
This diff is the single highest-value artifact — since our infra reproduces DiCoME, any
v2 underperformance is a recipe/architecture delta on our side.

**5. Evaluation protocol.**
Exact metric computation (frame vs video AUC and how frames are aggregated), threshold
handling, any per-dataset normalization — checked against our leakage-avoidance discipline
(single frozen threshold, source-only stats).

**6. Synthesis — the gap table.**
One row per DiCoME element: `{we have an equivalent / we deliberately rejected it and why /
MISSING — borrow because <specific gap it closes>}`, each tagged recipe-vs-idea. Close with a
**ranked shortlist (2–4)** of the concrete recipe borrowings most likely to close an
unexplained DISCERN-vs-DiCoME gap, each with a one-line falsifiable test Umar could run.

**7. Explicitly out of scope for adoption.**
The co-trained manifold, orthogonal-projection-as-core-mechanism, and DS-as-the-novelty. List
what we understood about each and why we still differentiate — so a future reader doesn't
re-open these as "borrow" candidates.

## Hand back

- `../DiCoME/analysis/DiCoME-gap-analysis.md` (tell Umar the exact path).
- A short list of any 🟡 confirmations that would need a run, so Umar can decide whether to
  run them. Do not run them yourself.
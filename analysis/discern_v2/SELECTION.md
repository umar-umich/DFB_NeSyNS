# Track A — SELECTION (Phase 1 deliverable)

Answers the seven selection questions from `docs/DiCoME_eval/claude-code-discern-v2-phase1.md`.
Every number below comes from a real run on real exports; nothing is asserted from memory.

Run 2026-08-15. DF40 (2,927 videos) and CDFv3 (1,080 videos), video level unless stated.
Exports present for all eight operators: P0-DS, P1a, P1b, P1c, P1d, P2a, P3a, P4.

Reproduce with:
```
python analysis/discern_v2/a3_rate_response.py
python analysis/discern_v2/a1_complementarity.py
python analysis/discern_v2/a2_gate.py
```

### The threshold — resolved

**threshold = 0.5110, EER on FFpp (the protocol source), frozen for every OOD dataset and
generator.** The FF++ export landed 2026-08-15 (4,736 frames / 592 videos), so the earlier
fixed-0.5 fallback is gone and every decision-based number below has real protocol
provenance. No threshold is tuned per OOD source at any point.

This mattered more than expected: it flipped the sign of P1b's gate recovery (see Q5).
Threshold-free results — AUROC, error overlap, evidence correlation, and the whole
inversion-row table — are unchanged by construction.

---

## 1. Does manifold-residual evidence add information beyond direct CLIP evidence?

**Yes.** The manifold branches are markedly less redundant with P0 than the alternatives.

| operator | error overlap with P0 (Jaccard) | evidence correlation with P0 (Spearman) |
|---|---|---|
| P1c | **0.575** | **0.834** |
| P1b | 0.637 | 0.882 |
| P1a | 0.646 | 0.893 |
| P4 | 0.653 | 0.908 |
| P2a | 0.680 | 0.916 |
| P3a | 0.754 | 0.972 |

Every P1 variant fails on a more distinct set of samples than P2a or P3a. P3a at 0.97
correlation is very nearly a restatement of P0 — consistent with it being parked as a
diagnostic rather than a branch.

## 2. Which P1 projector provides the most usable complementary evidence?

**P1b (β-TCVAE), on the evidence that exists — but see the tension in Q5.**
**🟡 ASK-UMAR: this is your sign-off; I recommend, I do not commit.**

P1b beats every other operator on **all eight** inverted rows — the regime where P0 is
confidently wrong and therefore the only place a specialist can prove itself:

| generator | P0-DS | **P1b** | P1a | P1c | P1d | P2a | P3a |
|---|---|---|---|---|---|---|---|
| sadtalker-ff | 0.253 | **0.573** | 0.433 | 0.492 | 0.324 | 0.447 | 0.249 |
| danet-cdf | 0.286 | **0.844** | 0.666 | 0.791 | 0.664 | 0.753 | 0.510 |
| tpsm-ff | 0.290 | **0.818** | 0.574 | 0.620 | 0.606 | 0.683 | 0.287 |
| danet-ff | 0.401 | **0.721** | 0.533 | 0.524 | 0.571 | 0.611 | 0.477 |
| mcnet-cdf | 0.404 | **0.831** | 0.658 | 0.745 | 0.617 | 0.750 | 0.504 |
| mcnet-ff | 0.432 | **0.842** | 0.707 | 0.762 | 0.653 | 0.720 | 0.471 |
| facevid2vid-ff | 0.480 | **0.827** | 0.691 | 0.786 | 0.686 | 0.698 | 0.509 |
| tpsm-cdf | 0.500 | **0.866** | 0.748 | 0.625 | 0.682 | 0.779 | 0.545 |

It also has the best standalone DF40 AUROC (0.8581 vs P0's 0.8367). The instructions warn
that best-standalone ≠ best-specialist; here the two happen to coincide, which is worth
stating plainly rather than dressing up as a subtle finding.

## 3. Does P2a contribute errors/rescues sufficiently different from the manifold and visual branches?

**From the visual branch, yes; from the manifold branches, it is largely dominated.**

P2a recovers 0.25–0.47 AUROC on the inverted rows where P0 fails (e.g. danet-cdf
0.286 → 0.753) — real complementarity, and invisible in the aggregate (0.8444 vs 0.8367).
But P1b beats it on all eight of those rows, and P2a is *more* redundant with P0 than any P1
variant (overlap 0.680, correlation 0.916).

The counter-evidence that must travel with this: on `heygen`, P2a actively **harms** —
accuracy 0.556 → 0.278, harming half the samples P0 gets right. That rescue/harm tension is
exactly what an applicability gate exists to resolve, and why D4 is posed as a question.

## 4. What is the oracle headroom?

| source | candidate set | base BA | oracle BA | headroom | coverage | shared-error rate |
|---|---|---|---|---|---|---|
| DF40 | {P0, P1d, P2a} | 0.7867 | 0.8421 | +0.0554 | 0.7926 | 0.2074 |
| DF40 | {P0, P1b, P2a} | 0.7867 | 0.8459 | +0.0592 | 0.7981 | 0.2019 |
| CDFv3 | {P0, P1d, P2a} | 0.7959 | 0.8790 | +0.0831 | 0.8167 | 0.1833 |
| CDFv3 | {P0, P1b, P2a} | 0.7959 | **0.8900** | **+0.0941** | **0.8389** | 0.1611 |

Real headroom on both sources, larger with P1b. Roughly a fifth of samples remain wrong for
every branch — a floor no router can beat. Oracle-routed AUROC is written to a separate file
marked *label-peeking / non-deployable* and is deliberately not a headline.

## 5. Can a simple observable gate recover a meaningful fraction of it? (A2a)

**Partly, and — awkwardly — the projector that offers the most headroom is the one the gate
exploits least.**

Under the frozen FFpp threshold (0.5110). The earlier fixed-0.5 numbers are shown for
comparison because the threshold changed the *sign* for P1b, which is worth seeing:

| candidate set | DF40 recovery | CDFv3 recovery | (at the old 0.5 fallback) |
|---|---|---|---|
| {P0, **P1d**, P2a} | **+0.027** | **+0.265** | +0.066 / +0.368 |
| {P0, P1b, P2a} | **−0.032** | **−0.031** | +0.014 / +0.091 |

Negative means the gate does *worse* than simply trusting P0. Three things follow:

- **P1d is the only projector a gate can exploit.** It is positive on both sources; P1b is
  negative on both.
- **P1b has the most headroom and the least gateable headroom.** It offers the largest
  oracle ceiling (+0.094 BA) and the biggest rescues, and the gate captures none of it.
- **DF40 recovery is weak even at its best** (+0.027). DF40 spans far more generator
  families, so even the optimistic A2a protocol — where the gate has seen sibling
  generators — barely transfers.

This tension is resolved by *splitting* the ladder rather than picking one projector for
everything — see Q7.

## 6. Does R(x) predict forgery, failure, or applicability even without family clusters?

**Barely, and not where it matters.** Full detail in `A3_VERDICT.md`.

| question | R(x) | baseline `p_fused` | incremental | folds |
|---|---|---|---|---|
| Q1 forgery separability | 0.8628 | 0.8504 | **+0.0122** | 70 |
| Q3 P0-miss prediction | 0.8708 | 0.9363 | **−0.0655** | 52 |
| Q3 rescue vs harm | 0.9639 | 0.9853 | **−0.0215** | 43 |
| Q2 family structure | silhouette −0.109, Fisher 0.097 | — | — | descriptive |

The revised gate keeps P1d alive if `R(x)` adds incremental signal on *any* question. It
clears that bar on Q1 alone, by +0.012. On both applicability questions the response is
*worse* than the scalar P1d already emits, and family structure is absent. The discriminative
claim was about the *shape* of the rate-distortion curve; on the two questions where shape
would be useful for routing, shape loses to the scalar. **P1d loses privileged status.**

## 7. Therefore: which two branches enter D1–D3, and is D4 justified?

**Branches: CLIP visual + P1b (β-TCVAE) as the manifold branch, with P2a as the third view
in D2/D3.** The visual branch is fixed — the phase criterion is that the new mechanisms
transfer without destroying it. P1b is the recommendation on complementary rescue (all eight
inverted rows), standalone AUROC (0.858, best of the suite), and oracle headroom
(+0.094 CDFv3, the largest of any set).

**DECIDED 2026-08-15 (Umar): the ladder splits by projector.** P1b is the best raw
specialist but its gate recovery is negative on both sources; P1d is the only projector a
gate can exploit. Rather than force one choice, each is used where its evidence supports it:

| arm | configs | projector | rationale |
|---|---|---|---|
| Main ladder | `D1_V` / `D1_M` / `D1_VM` → `D2_process` → `D3_full` | **beta_tcvae (P1b)** | wins all 8 inverted rows, best AUROC, largest headroom |
| D4 arm | **`D3_full_p1d`** → D4 | **mr_vae (P1d)** | the only positive gate recovery (+0.027 / +0.265) |

**The constraint that makes this valid:** D4's claim is measured as D4 − D3, so
**D4(P1d) must be compared against D3(P1d)**, never against the P1b D3. Otherwise the delta
conflates "gating helped" with "the manifold branch changed" and means nothing. That is why
`D3_full_p1d.yaml` exists as an explicit matched baseline rather than a note someone has to
remember at run time. The cost of running both arms is one extra D3, not a second ladder.

**D4 — REVISED 2026-08-15: defensible, not negative.** A2b has now been run (full results in
`A2B_FINDINGS.md`), and it overturns the earlier "large oracle, unrecoverable" reading,
which rested on A2a over two sources.

A gate trained **only on FFpp** and frozen recovers a meaningful share of the oracle headroom
on **five of seven** OOD sources, averaging **+0.106** with P1d (vs +0.033 with P1b):

| source | DF40 | CDFv3 | CDFv2 | DFEval24 | DFDC | DFDCP | DFD | mean |
|---|---|---|---|---|---|---|---|---|
| {P0,P1d,P2a} | +0.134 | +0.288 | −0.102 | **−0.290** | +0.102 | +0.363 | +0.248 | **+0.106** |
| {P0,P1b,P2a} | +0.023 | +0.176 | +0.045 | **−0.336** | −0.012 | +0.327 | +0.010 | +0.033 |

P1d beats P1b on five of seven sources and 3× on the mean — a second, independent protocol
reaching the same projector conclusion as A2a, which strengthens the D1/D4 split.

Three caveats that must travel with this, all detailed in `A2B_FINDINGS.md`:
- **A2a and A2b are not directly comparable** (per-generator folds vs whole-source), so
  A2b > A2a does not mean deployment beats the ceiling.
- **A2a is uncomputable on four of the seven sources** — CDFv2/DFEval24/DFDC/DFD lack the
  generator structure LOGO requires. A2b works everywhere.
- **DFEval24, the in-the-wild source, is where the gate does the most harm** (−0.290), and
  it has the weakest baseline (BA 0.650). Helping on curated benchmarks while hurting on
  real-world data matters for a reliability paper and should not be averaged away.

**D4 design input — the routing threshold matters more than the gate family.** A2c sweeps it
(`A2B_FINDINGS.md`). A2b's default of tau = 0.5 routes on 84% of samples and is a poor
operating point. **tau = 0.95** is strictly better on the axes that matter: worst-source harm
drops 72% (DFEval24 −0.0237 → −0.0066 BA), routing falls to 47% so the gate actually
abstains, one more source turns positive (6/7), and it costs only ~20% of the mean recovery
(0.106 → 0.085). Build D4 with a conservative routing threshold, not the default.

---

## Status of the 🟡 items

1. **D1 projector** — **RESOLVED**: P1b for the main ladder, P1d for the D4 arm (Q7).
2. **FF++ export** — **DONE** 2026-08-15; threshold is now 0.5110 from FFpp EER.
3. **D4** — still open. Reading is negative-leaning; confirm before it is written up as a
   negative result rather than built. Depends on A2b, not yet run.

## Still outstanding

- **D0 — RUNNING** (launched 2026-08-15). System of record is
  `training/config/detector/nesy_defake_ablation4_ccv.yaml`, run from the repo root:
  `python training/train.py --detector_path training/config/detector/nesy_defake_ablation4_ccv.yaml`
  (launching from inside `training/` fails at import: `fwa_blend.py` loads a dlib asset by a
  CWD-relative path). The `discern_v2/` files are flag manifests, not runnable trainer
  configs — `train.py` has no `_base_:` include mechanism.

  **One spec deviation, recorded in `D0_v1_reproduction.yaml`:** the spec asks D0 to set
  `structural_sem_v1=true` alongside `ccv`, but `causal_branch.type` is one-of
  `{simple, improved_scm, ccv}` — those two flags name *mutually exclusive* mechanisms, not
  independent branches. The system of record uses `ccv`, so `structural_sem_v1` is FALSE in
  the real v1 configuration. Setting it true would select a different causal mechanism and
  produce something that is not the reference system.

- **A2b** deployment-valid gate protocol: specified, not run.

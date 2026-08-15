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

### Standing caveat — the threshold

No FF++ export exists yet, so no threshold can be derived from the permitted protocol
source. `frozen_threshold()` returns a fixed 0.5 and reports that provenance rather than
tuning anything on an OOD source. Per the instructions, threshold-free analyses (AUROC,
score correlation, error overlap) carry every conclusion below, and each decision-based
table is reported beside its threshold-free counterpart. **`export_features.py P0-DS
--source FFpp` removes this caveat** and is worth running before the numbers are final.

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

| candidate set | DF40 recovery | CDFv3 recovery |
|---|---|---|
| {P0, P1d, P2a} | **+0.066** | **+0.368** |
| {P0, P1b, P2a} | +0.014 | +0.091 |
| {P0, P1a, P2a} | −0.073 | +0.321 |
| {P0, P1c, P2a} | −0.239 | +0.292 |

Negative means the gate does *worse* than simply trusting P0. Two things follow:

- **DF40 recovery is poor for every set** (−0.24 to +0.07). DF40 spans far more generator
  families, so even the optimistic A2a protocol — where the gate has seen sibling
  generators — barely transfers.
- **P1d is the most gateable** (+0.066 / +0.368) despite having *less* oracle headroom than
  P1b. P1b offers the largest ceiling and the biggest rescues but the gate captures little
  of it.

This is a genuine tension in the selection, not a tie to be broken quietly — see Q7.

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

**The honest caveat:** if D4 is built, P1d is the more gateable partner (+0.066/+0.368 vs
P1b's +0.014/+0.091). The choice therefore depends on a decision that has not been made yet:

- **D3 is the destination** → choose **P1b**. Raw complementarity is what fusion consumes,
  and P1b dominates it.
- **D4 is the destination** → **P1d** deserves reconsideration despite losing its
  rate-response justification, because a gate extracts more from it.

I recommend **P1b**, because D4 is not currently justified (below) and D3 is the honest
system. 🟡 Yours to confirm.

**D4: not justified on this evidence, and leaning negative.** Oracle headroom is real
(+0.055 to +0.094) but an observable gate recovers little of it and on DF40 frequently makes
things worse. Per the integration README this is the "large oracle, unrecoverable" branch —
a legitimate negative result, not a failure. Two things should be settled before it is
written up as final: the FF++ threshold export, and A2b (the deployment-valid protocol,
specified in `A2_gate/A2b_PROTOCOL.md`), since A2a is only an optimistic ceiling.

---

## Open 🟡 ASK-UMAR

1. **D1 projector** — recommendation is P1b; confirm, or choose P1d if D4 is the target.
2. **FF++ export** — approved, not yet run. Removes the 0.5 threshold fallback.
3. **D4** — my reading is negative-leaning; confirm before it is written up as a negative
   result rather than built.

# Grouped (dependency-partitioned) evidential fusion — outcome

Stage 10 of `tbiom/docs/grouped-fusion-brief.md`. Pre-registration: `tbiom/configs/grouped_fusion_preregister.yaml`,
written before any candidate was scored.

## Terminal outcome

**No demonstrated advantage.** grouped-R fails its Stage 3 confirmation gate. The requirement that it do more than a
plain two-view fusion fails at every one of the four detector configurations, for a reason that is algebraic rather
than statistical, and the falsification controls that passed on the deployed system fail once the detector changes.

All four seed configurations are now complete. The full detector-seed swap did **not** reproduce the effect, and it
exposed a stronger version of the same conclusion.

## The gate, requirement by requirement

Four configurations: the deployed warm+gate spatial views with V-JEPA 2 seed 42 and seed 1337, and SBI arm B — the
one three-branch family trained at several seeds — at seeds 42 and 1337 with the matching temporal branch. Each
system's reliability weights are fitted on **its own** development exports.

| requirement | warm+gate / T42 | warm+gate / T1337 | armB42 | armB1337 |
|---|---|---|---|---|
| DF40 reenactment gain over Dempster-R | +0.0114 | +0.0025 | **+0.0000** | +0.0020 |
| control grouped-TA-R loses | +0.0033 | +0.0012 | **+0.0000** | **−0.0003** |
| control grouped-TO-R loses | +0.0049 | +0.0051 | **+0.0010 (CI spans 0)** | **−0.0005** |
| does more than two-view T+OM | **+0.0000** | **−0.0001** | **+0.0000** | **+0.0000** |

Bold = fails. 2,000 paired source-group draws per contrast, reused real videos grouped across generator sets, and a
materiality floor of 1e-4 AUROC so that two arms differing only by floating-point noise cannot produce a "significant"
interval by repeating the same tiny difference in every draw.

**The T+OM requirement fails at all four seeds. The falsification controls, which passed under both warm+gate
configurations, fail under both arm-B seeds.** The partition effect does not survive a change of detector.

## Why the fourth requirement fails

The operator's own fitted discount sets the **appearance weight to zero** in 4 of 5 folds on the deployed system,
2 of 5 on armB42 and 3 of 5 on armB1337. An evidence-scaled opinion with r = 0 carries no evidence; its canonical
weights are both 1; the minimum of 1 and w_OM is w_OM. The cautious inner step therefore returns the off-manifold
opinion unchanged, and what remains is the outer Dempster step between two views. In those folds grouped-R **is** the
two-view fusion of temporal and off-manifold, bit for bit.

The arm-B seeds sharpen this considerably. There the **baseline's** fit discards the appearance view too —
Dempster-R zeroes it in 5 of 5 folds at armB42, as does grouped-TA-R — so Dempster-R, grouped-TA-R, grouped-R and
T+OM all collapse onto the same function and score identically (0.9607 reenactment, 0.9674 swap, 0.9638 overall).
That is why three requirements read exactly +0.0000 at that seed.

The general statement is therefore stronger than "grouped-R equals T+OM". It is: **once the reliability weight is
free to be fitted, the appearance view is discarded, and the redundancy the partition was designed to handle has
already been removed by the discount.** The partition has nothing left to do. A fused operator cannot demonstrate a
dependence correction on a pair of views when the fit has already deleted one of them.

## What the controls do and do not establish

Under the two warm+gate configurations the controls pass: pairing the temporal view with either spatial view is worse
than pairing the two CLIP heads. Read alone that looks like support for the dependence argument.

Under both arm-B seeds they do not. grouped-TA-R matches grouped-R exactly at armB42 and beats it by +0.0003 at
armB1337; grouped-TO-R beats it by +0.0005 at armB1337. So the control result is a property of one detector, not of
the architecture — which is exactly what a second detector seed is for, and exactly why the brief required one.

Even where the controls pass, they would not have rescued the operator: a grouping that beats two wrong groupings is
not evidence for a mechanism when a simpler arm matches the winner exactly.

## Development, for completeness

No arm clears the pre-registered detection bar. grouped-R is −0.0016 against Dempster-R with the CI spanning zero in
5 of 5 splits, and it fails the load-bearing vacuity guard at −0.0055 (`EXP_GROUPED_DEVELOPMENT.md` §3–4). The PCR6-
outer arms fail that guard heavily, as the audit predicted from the algebra (GCP-AO-R −0.0347, GCY-AO-R −0.0537).

The mechanism inspection describes a trade, not an advantage:

| case on the selection folds | videos | grouped-R | Dempster-R |
|---|---:|---:|---:|
| both CLIP heads wrong together, temporal right | 362 | **0.7428** | 0.5558 |
| temporal wrong, CLIP heads agree and right | 520 | 0.5796 | **0.7173** |

## The swap-family cost, stated plainly

Every configuration that gains on reenactment loses on face swap. grouped-R against Dempster-R is −0.0033 [−0.0052,
−0.0016] on the 19 swap sets at seed 42, and −0.0007 [−0.0011, −0.0004] at seed T1337. Both intervals exclude zero.
This is reported here and in every table rather than left to the favourable family.

## Decisions

1. **Dempster-R stays the deployed rule**, and stays a labelled baseline row — the standard Dempster–Shafer operator
   and the DiCoME-family reference. It is not presented as the proposed method, and it is not quietly reinstated as
   one either: nothing in this study proposes a replacement.
2. **No fusion-operator novelty claim is made.** `GROUPED_PRIOR_ART_AND_CLAIMS.md` records why, including that no
   primary source could be retrieved in this session and that Quost, Masson and Denœux (IJAR 2011) on optimised
   t-norm combination is the work most likely to subsume the construction and must be obtained first.
3. The reportable finding is diagnostic: on reenactment generators where both CLIP heads sit at chance they are wrong
   *together*, because they share a feature path, and a rule that reads their agreement as two independent
   confirmations is misled by it. The cleanest evidence for that is the fitted discount itself.
4. Prior results are preserved. `EXP_FUSION_NOVEL.md` (commit `2b3f752`), the single-seed interval-free version, is
   retained unchanged and superseded rather than deleted or relabelled.

## Deliverables

| file | stage |
|---|---|
| `GROUPED_REFERENCE_MANIFEST.md` | 4 — audit, algebra of the caches, exposure history |
| `tbiom/configs/grouped_fusion_preregister.yaml` | 8 — frozen before scoring |
| `EXP_GROUPED_AUDIT.md` / `.json` | 7 — 28/28 algebraic and mechanism checks |
| `EXP_GROUPED_DEVELOPMENT.md` / `.json` | 6 — the fixed matrix on development, both discount settings |
| `EXP_GROUPED_EXPOSED.md` / `.json` | 9 — ten test sets and DF40, both required means |
| `EXP_GROUPED_CONFIRM.md` / `.json` | 3 — the gate |
| `GROUPED_PRIOR_ART_AND_CLAIMS.md` | 9 — prior art and claim-to-evidence map |
| `GROUPED_FUSION_OUTCOME.md` | 10 — this file |

## Deficits, stated rather than papered over

- **No untouched confirmation set exists for this study.** DF40 informed the choice of this operator before the task
  began. The gate tests reproducibility across a seed and against falsification controls; neither is a
  generalisation claim to unseen generators.
- The five folds are not five detector seeds, and the bootstrap quantifies sample uncertainty conditional on the
  trained encoders, not encoder-training variability.
- The deployed warm+gate recipe exists at one seed and this task forbids training another, so the full detector swap
  uses SBI arm B, a different training recipe. Within arm B the seed is the only difference, but arm B is not the
  deployed recipe, so the four configurations differ in more than a seed.
- The reenactment effect size ranges over an order of magnitude across configurations where it is positive at all:
  +0.0114, +0.0025, +0.0020, and +0.0000.
- An earlier run of this stage fitted the arm-B weights on warm+gate development exports. That was wrong — the
  weights must come from the system being scored — and the numbers here are the corrected ones. The error is
  recorded rather than silently overwritten; it changed the arm-B conclusions.

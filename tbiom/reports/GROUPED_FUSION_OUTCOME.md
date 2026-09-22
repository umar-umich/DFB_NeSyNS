# Grouped (dependency-partitioned) evidential fusion — outcome

Stage 10 of `tbiom/docs/grouped-fusion-brief.md`. Pre-registration: `tbiom/configs/grouped_fusion_preregister.yaml`,
written before any candidate was scored.

## Terminal outcome

**No demonstrated advantage.** grouped-R fails its Stage 3 confirmation gate. Three of the four requirements pass;
the fourth fails for a reason that is algebraic rather than statistical, and it removes the basis for the claim.

Secondary outcome: **independent confirmation pending** for the full detector-seed swap (SBI arm B seeds 42 and 1337),
whose exports were still running when this was written. The temporal-seed swap is complete and reproduces every sign.

## The gate, requirement by requirement

| requirement | seed 42 | seed T1337 (temporal seed swapped) | verdict |
|---|---|---|---|
| DF40 reenactment gain over Dempster-R, CI excludes zero | +0.0114 [+0.0086, +0.0144] | +0.0025 [+0.0019, +0.0031] | **pass** |
| control grouped-TA-R loses | +0.0033 [+0.0010, +0.0054] | +0.0012 [+0.0006, +0.0018] | **pass** |
| control grouped-TO-R loses | +0.0049 [+0.0029, +0.0068] | +0.0051 [+0.0035, +0.0067] | **pass** |
| grouped-R does more than two-view T+OM | −0.0000 [−0.0002, +0.0001] | −0.0001 [−0.0002, +0.0001] | **fail** |

2,000 paired source-group draws per contrast, reused real videos grouped across generator sets.

## Why the fourth requirement fails

The operator's own fitted discount sets the **appearance weight to zero in 4 of 5 folds**. An evidence-scaled opinion
with r = 0 carries no evidence; its canonical weights are both 1; the minimum of 1 and w_OM is w_OM. The cautious
inner step therefore returns the off-manifold opinion unchanged, and what remains is the outer Dempster step between
two views. In those folds grouped-R **is** the two-view fusion of temporal and off-manifold, bit for bit — verified
numerically on all 41 DF40 sets (`EXP_GROUPED_EXPOSED.md` §0) and reproduced by the bootstrap above.

So the cross-manipulation gain that motivated this study is real, but it is not the dependency partition. It is the
decision to drop the appearance view. The partition is doing no work once the discount is free to be fitted.

## What the controls do and do not establish

The falsification controls pass under both seeds: pairing the temporal view with either spatial view is worse than
pairing the two CLIP heads. That is a genuine finding about the architecture — it is consistent with the two CLIP
heads being the redundant pair. But it does not rescue the operator, because the winning configuration is reachable
without any partition at all. A control that beats two wrong groupings is not evidence for a mechanism when a
simpler arm matches the winner exactly.

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
- The completed seed swap changes the temporal branch only; the full swap, which also changes the spatial views, is
  still exporting. The deployed warm+gate recipe exists at one seed and this task forbids training another, so the
  full swap uses SBI arm B, where seed is the only difference within the family.
- The reenactment effect size is seed-sensitive even where its sign is not: +0.0114 at seed 42 against +0.0025 at
  seed T1337, a factor of four.

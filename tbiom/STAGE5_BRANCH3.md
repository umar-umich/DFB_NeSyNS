# Stage 5 — Branch 3 is rank-informative and structurally worthless to DS fusion

## The measurement

Three-branch framework, P2a's validated head, epoch 8. Branch 3's contribution measured by
inverting each view's exported `(p, u)` back to its Dirichlet opinion and replaying the fusion
with and without it. The replay reproduces the exported `p_fused` to **2.15e-07**, so the
inversion is exact and the counterfactual is trustworthy.

| domain | 2-view AUROC | 3-view AUROC | **Δ from Branch 3** | Branch 3 alone | mean u_process |
|---|---:|---:|---:|---:|---:|
| CDFv2val | 0.9051 | 0.9051 | **−0.0000** | 0.5101 | 0.9978 |
| DFDCPval | 0.9535 | 0.9536 | **+0.0000** | 0.6038 | 0.9934 |
| DFEval24val | 0.5699 | 0.5697 | **−0.0001** | 0.5021 | 0.9958 |
| all | 0.8260 | 0.8259 | **−0.0001** | 0.5393 | 0.9958 |

Branch 3 knows something on DFDCP — AUROC 0.6038, well above chance — and delivers **none** of it.

## Why, and why no fusion tweak fixes it by itself

`u = K/S` near 1 means `S` near `K`, which means evidence near **zero**: a **vacuous** opinion. A
vacuous opinion is the **identity element** of the Dempster-Shafer orthogonal sum. So Branch 3 is
not being outvoted or suppressed — it is being *ignored exactly*, by the algebra, no matter what
it has learned.

The mechanism that drives it there is worth stating, because it is a property of the training
scheme rather than a bug:

> The loss is applied **only to the fused opinion**. Two strong views already produce a nearly
> correct fused opinion. Any non-vacuous contribution from a weaker third view perturbs that,
> which *increases* the loss. So gradient descent drives the weak view's evidence toward zero.
> **Vacuity is the loss-minimising strategy for a weak view under fused-only supervision.**

This is an attractor, not a dead zone at initialisation. That is why more capacity made it
*worse*, not better: the 290-parameter head reached `u = 0.87`, and P2a's 25.8k-parameter head
reached `u = 0.996` — more capacity means it learns to be silent more precisely.

| head | trainable | mean u_process | Branch 3 AUROC (DFDCPval) | Δ to fusion |
|---|---:|---:|---:|---:|
| v1 frozen-standardizer | 290 | 0.8709 | 0.5169 | ~0 |
| **P2a (validated)** | 25,806 | **0.9958** | **0.6038** | **~0** |

The P2a head *is* the better branch — it doubles the rank information on DFDCP — and it is *more*
completely ignored.

## This re-reads P2a's own pilot

P2a's complementarity table reports `auc_operator` 0.87–0.97, which is what made the third view
look strong. The same rows report **`error_overlap = 1.0`, `rescue = 0`, `harm = 77`** at
threshold 0.5. Those coexist only if P2a's operator scores were likewise clustered just above
0.5 — rank-informative, evidence-poor. P2a used **no auxiliary operator loss** either
(`_compute_loss` is explicitly unchanged from P0-DS, "the operator adds a view, not a loss term").

So P2a's third view was almost certainly vacuous too, and **P2a's mean Δ of −0.0092 (+0.0066
after the CDFv3 correction) cannot have come from its operator view.** It came from the two CLIP
branches. The pilot that motivated Branch 3 never actually demonstrated a working third view —
it demonstrated a well-measured probe on features the model then discarded.

Not fully verified: P2a's own `u_operator` was never exported, so this rests on the
`error_overlap = 1.0` signature plus the identical loss construction. Stated as strong inference,
not measurement.

## Consequences

1. **Do not claim a three-branch framework.** As built, it is two branches plus an exactly-ignored
   third. The honest description is a two-branch framework with a measured negative result about
   the third.
2. **The probe is the upper bound, not the branch.** The six statistics support AUROC 0.7047 on
   FF++ val by logistic probe; the branch delivers 0.51. The gap is the cost of asking a
   fused-only loss to supervise a weak view.
3. **Stage 7 is now necessary rather than conditional**, and it has a concrete target. But
   conflict-aware fusion alone is not obviously sufficient: the problem is not conflict handling,
   it is that Branch 3 arrives with no mass to weigh. An operator that renormalises per-view
   influence (rather than treating low evidence as identity) is the shape that could work.
4. **The alternative fix is a direct per-view loss on `e_proc`.** It is the obvious remedy and it
   is what P2a deliberately declined, to keep the pilot to one change. Under this brief it is a
   second variable, so it is a decision to be taken rather than slipped in — ASK-UMAR.

`analysis/tbiom/branch3_contribution.py` reproduces every number here from exports alone, no GPU.

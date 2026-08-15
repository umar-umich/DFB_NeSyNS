# A2b — deployment-valid gate results

Run 2026-08-15. Gate trained **only on FFpp** (592 videos, the permitted protocol source),
then frozen and applied to seven OOD sources it has never seen. No generator ID, dataset ID,
or family label is an input at any point; `dloss_b` is a training target only. One frozen
threshold (0.5110, FFpp EER) throughout.

This is the protocol a shipped DISCERN gate would have to satisfy. A2a was the optimistic
ceiling.

## Gate-Recovery per OOD source

| source | base BA | oracle BA | **{P0,P1d,P2a}** | **{P0,P1b,P2a}** |
|---|---|---|---|---|
| DF40 | 0.7872 | 0.8409 | **+0.134** | +0.023 |
| CDFv3 | 0.7971 | 0.8762 | **+0.288** | +0.176 |
| CDFv2 | 0.8782 | 0.9293 | −0.102 | +0.045 |
| DFEval24 | 0.6500 | 0.7318 | **−0.290** | **−0.336** |
| DFDC | 0.7750 | 0.8728 | +0.102 | −0.012 |
| DFDCP | 0.7299 | 0.8751 | **+0.363** | +0.327 |
| DFD | 0.8611 | 0.9261 | +0.248 | +0.010 |
| **mean** | | | **+0.106** | +0.033 |
| **sources positive** | | | 5 / 7 | 5 / 7 |

Oracle headroom is substantial and positive on every source (+0.051 to +0.145 BA,
union-of-correct coverage 0.79–0.93), so there is real headroom to compete for everywhere.

## What this changes

**D4 is defensible after all, with P1d.** The earlier read — "large oracle, unrecoverable,
write it up as a negative result" — came from A2a on two sources, where DF40 recovery was
≈0 and P1b went negative. Under the deployment-valid protocol a gate trained purely on FF++
recovers a meaningful share of the oracle headroom on five of seven OOD sources, averaging
+0.106 with P1d. That is not a negative result.

**P1d is confirmed as the gated partner, independently of A3.** It beats P1b on five of the
seven sources and by 3× on the mean. This is the second protocol to reach that conclusion,
which strengthens the D1/D4 projector split rather than merely restating it.

## Three caveats that must travel with these numbers

**1. A2a and A2b are not directly comparable, so do not read A2b > A2a as "deployment beats
the ceiling."** They aggregate differently: A2a averages per-generator leave-one-out folds
*within* a source; A2b trains on FF++ and evaluates each source as a whole. Different units,
different variance. The only honest comparison is within a protocol.

**2. A2a could not be computed on four of the seven sources.** CDFv2, DFEval24, DFDC and DFD
lack the generator structure LOGO needs (too few videos per `method` group), returning
`None`. A2b works everywhere because it never partitions by generator. That makes A2b the
more broadly applicable protocol as well as the more honest one — worth saying in the paper.

**3. DFEval24 — the in-the-wild source — is where the gate does the most harm** (−0.290 for
P1d, −0.336 for P1b), and it is also the source with the weakest baseline (BA 0.650). A gate
that helps on curated academic benchmarks and hurts on real-world data is a serious finding
for a paper whose identity is *reliability*, and it should not be averaged away. If one
number goes in an abstract, it should not be the mean.

**A fourth thing to watch:** the routed fraction is 0.80–0.90 on every source — the gate
hands off to a specialist for the large majority of samples rather than deferring to P0. A
gate that almost always routes is barely being selective, which sits awkwardly with the
design goal of "exploit specialists without harming what the primary detector already gets
right." Worth probing before D4 is built: a higher routing threshold, or an explicit
abstention cost, may produce a smaller but safer gain.

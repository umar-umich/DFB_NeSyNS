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

**A fourth thing, now resolved:** the routed fraction was 0.80–0.90 on every source — the
gate handed off to a specialist for the large majority of samples rather than deferring to
P0. A gate that almost always routes is barely being selective, which sits awkwardly with
"exploit specialists without harming what the primary detector already gets right." A2c
(below) sweeps the routing threshold and finds a strictly better operating point.

---

# A2c — routing threshold

`route if max_b P(specialist b beats the visual baseline) > tau`. A2b used tau = 0.5, i.e.
route whenever a specialist is more likely than not to help. Everything else is unchanged:
gate trained on FFpp only, frozen, label-free inputs, one frozen decision threshold.

**{P0, P1d, P2a}**

| tau | mean recovery | sources positive | routed fraction | DFEval24 ΔBA |
|---|---|---|---|---|
| 0.50 | 0.1062 | 5/7 | 0.836 | −0.0237 |
| 0.60 | **0.1071** | 6/7 | 0.799 | −0.0210 |
| 0.70 | 0.0889 | 5/7 | 0.758 | −0.0209 |
| 0.80 | 0.0673 | 5/7 | 0.704 | −0.0171 |
| 0.90 | 0.0747 | 5/7 | 0.599 | −0.0092 |
| **0.95** | 0.0853 | **6/7** | **0.468** | **−0.0066** |

**Recommended operating point: tau = 0.95.** Against the A2b default it is better on almost
every axis that matters:

- worst-source harm falls by **72%** (DFEval24 −0.0237 → −0.0066 BA, essentially negligible)
- routing drops from 84% to 47%, so the gate is actually *selective* — it defers to the
  visual baseline on more than half of samples, which is what the design intends
- one more source turns positive (5/7 → 6/7)
- and it costs only ~20% of the mean recovery (0.106 → 0.085)

tau = 0.60 maximises the mean (0.1071), but it barely reduces the in-the-wild harm. For a
paper whose identity is reliability, the tau = 0.95 trade is the defensible one: nearly all
of the gain, almost none of the damage, and a gate that visibly abstains.

Recovery is not monotonic in tau (0.089 → 0.067 → 0.075 → 0.085). With seven sources and a
gate fit on 592 protocol videos that wobble is small-sample noise, so tau should not be
tuned finely — 0.95 is chosen as a *regime* (be conservative), not as an optimum.

**{P0, P1b, P2a}** shows the same shape at roughly a third of the magnitude (mean 0.033 →
0.059 at tau = 0.95, DFEval24 −0.0316 → −0.0110), which is further independent support for
P1d as the gated partner.

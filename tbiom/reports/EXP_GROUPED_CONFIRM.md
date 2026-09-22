# Stage 3 — the confirmation gate for grouped-R

Three things the brief asks for, in order:
  1. a 95% interval on the DF40 reenactment gain of grouped-R over Dempster-R, from a paired source-group bootstrap
     with 2,000 draws at fixed seeds, resampling reused real videos as one group across generator sets;
  2. the same contrasts under a second detector seed;
  3. the falsification controls, grouped-TA-R and grouped-TO-R, which must lose to grouped-R under both seeds.

A fourth contrast is carried at the same status as the first, because the exposed-suite run showed the fitted
discount zeroes the appearance weight in most folds: grouped-R against the plain two-view T+OM fusion. If that
interval contains zero, the grouped operator is not doing the work, whatever the comparison against Dempster-R says.

Source grouping. A generator set's genuine videos are DF40's shared real pool, so the same real video appears in
every set of its source; grouping on the video name makes those one group across sets, which is what the brief
requires. For FF++-sourced sets the fake's first name token is its real source id, so real and fake share a group.
For Celeb-DF-sourced sets the real pool is numbered in a different namespace from the identity-pair naming of the
fakes, so the real-fake link is not recoverable from the manifest: reals group by video, fakes by identity pair, and
this is a documented limitation rather than an assumed linkage.

Wall clock 2.4 minutes, CPU. 2000 draws per contrast, fixed seed.

## Detector seed `42` — deployed warm+gate spatial views, V-JEPA 2 seed 42

41 generator sets, 22 of them reenactment.

| arm | reenactment | swap | all sets |
|---|---:|---:|---:|
| AVG-R | 0.9447 | 0.9785 | 0.9604 |
| Dempster-R | 0.9391 | 0.9789 | 0.9575 |
| T+OM | 0.9505 | 0.9756 | 0.9622 |
| grouped-R | 0.9505 | 0.9756 | 0.9621 |
| grouped-TA-R | 0.9408 | 0.9797 | 0.9588 |
| grouped-TO-R | 0.9398 | 0.9775 | 0.9573 |

| contrast | family | Δ | 95% CI | excludes zero |
|---|---|---:|---|---|
| grouped-R − Dempster-R | reenactment | +0.0114 | [+0.0086, +0.0144] | **yes** |
| grouped-R − Dempster-R | swap | -0.0033 | [-0.0052, -0.0016] | **yes** |
| grouped-R − Dempster-R | all | +0.0046 | [+0.0025, +0.0066] | **yes** |
| grouped-R − AVG-R | reenactment | +0.0058 | [+0.0037, +0.0080] | **yes** |
| grouped-R − AVG-R | swap | -0.0029 | [-0.0043, -0.0015] | **yes** |
| grouped-R − AVG-R | all | +0.0018 | [+0.0002, +0.0033] | **yes** |
| grouped-R − T+OM | reenactment | -0.0000 | [-0.0002, +0.0001] | no |
| grouped-R − T+OM | swap | +0.0000 | [-0.0001, +0.0001] | no |
| grouped-R − T+OM | all | -0.0000 | [-0.0002, +0.0001] | no |
| grouped-R − grouped-TA-R | reenactment | +0.0097 | [+0.0066, +0.0128] | **yes** |
| grouped-R − grouped-TA-R | swap | -0.0041 | [-0.0062, -0.0022] | **yes** |
| grouped-R − grouped-TA-R | all | +0.0033 | [+0.0010, +0.0054] | **yes** |
| grouped-R − grouped-TO-R | reenactment | +0.0107 | [+0.0079, +0.0135] | **yes** |
| grouped-R − grouped-TO-R | swap | -0.0019 | [-0.0034, -0.0004] | **yes** |
| grouped-R − grouped-TO-R | all | +0.0049 | [+0.0029, +0.0068] | **yes** |

## The gate

| requirement | status |
|---|---|
| reenactment interval excludes zero, seed 42 | pass — +0.0114 [+0.0086, +0.0144] |
| control grouped-TA-R loses to grouped-R, seed 42 | pass — +0.0033 [+0.0010, +0.0054] |
| control grouped-TO-R loses to grouped-R, seed 42 | pass — +0.0049 [+0.0029, +0.0068] |
| grouped-R does more than the two-view T+OM fusion, seed 42 | **fail** — -0.0000 [-0.0002, +0.0001] |

**Verdict: grouped-R NOT confirmed.**

Per the pre-registration the matrix in `EXP_GROUPED_DEVELOPMENT.md` and `EXP_GROUPED_EXPOSED.md` is therefore a diagnostic, not a comparison supporting a novelty claim, and grouped-R is reported as an exploratory trade-off. No fusion-operator novelty claim is made.

The second detector seed is still exporting; this run covers the seeds listed above only.


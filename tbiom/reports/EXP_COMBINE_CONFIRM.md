# Stage 3 — combine and confirm (both routes)

OMP_NUM_THREADS=8 python analysis/tbiom/stage3_analysis.py

Pre-registration: tbiom/docs/stage3-composition-preregistration.md.
  Route 2 -- composition on the deployed recipe: UB (warm+gate, views unfrozen, arm B) vs UE (same, arm E), and UE
             against the deployed checkpoint, which is the adoption decision.
  Route 3 -- arm E adopted as an override: E seeds 42 / 1337 / 7 each against the deployed checkpoint, per seed.
Every arm's Appearance and Off-manifold views are fused with the frozen reference temporal branch; DS-R and the
logistic comparator are refitted per arm on the same 5 source-grouped folds. Development only; no test set is used.

Selected checkpoints: deployed (warm+gate e19, views frozen) → `warmgate_e19`; UB (warm+gate, views unfrozen, arm B) → `wgUB_e00`; UE (warm+gate, views unfrozen, arm E) → `wgUE_e00`; E seed 42 (sbi_armE) → `sbiarmE_e02`; E seed 1337 → `sbiarmEs1337_e04`; E seed 7 → `sbiarmEs7_e04`.

## 1. Development, selection folds (mean of 5 splits)

| arm | rule | clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR own τ | TPR own τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed (warm+gate e19, views frozen) | DS-R | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 |
| deployed (warm+gate e19, views frozen) | LR | 0.8736 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 | 0.057 | 0.648 |
| UB (warm+gate, views unfrozen, arm B) | DS-R | 0.8703 | 0.9239 | 0.9077 | 0.7792 | 0.8129 | 0.8473 | 0.077 | 0.653 |
| UB (warm+gate, views unfrozen, arm B) | LR | 0.8711 | 0.9239 | 0.8948 | 0.7946 | 0.7953 | 0.8492 | 0.047 | 0.604 |
| UE (warm+gate, views unfrozen, arm E) | DS-R | 0.8735 | 0.9264 | 0.9043 | 0.7897 | 0.8132 | 0.8508 | 0.071 | 0.638 |
| UE (warm+gate, views unfrozen, arm E) | LR | 0.8698 | 0.9245 | 0.8889 | 0.7959 | 0.8008 | 0.8429 | 0.038 | 0.549 |
| E seed 42 (sbi_armE) | DS-R | 0.8757 | 0.9232 | 0.9000 | 0.8040 | 0.8223 | 0.8539 | 0.056 | 0.607 |
| E seed 42 (sbi_armE) | LR | 0.8690 | 0.9173 | 0.8877 | 0.8022 | 0.8089 | 0.8449 | 0.042 | 0.553 |
| E seed 1337 | DS-R | 0.8715 | 0.9183 | 0.8961 | 0.8002 | 0.8096 | 0.8506 | 0.061 | 0.581 |
| E seed 1337 | LR | 0.8676 | 0.9158 | 0.8851 | 0.8018 | 0.8060 | 0.8461 | 0.082 | 0.639 |
| E seed 7 | DS-R | 0.8753 | 0.9187 | 0.8994 | 0.8077 | 0.8092 | 0.8546 | 0.057 | 0.618 |
| E seed 7 | LR | 0.8694 | 0.9144 | 0.8907 | 0.8030 | 0.7998 | 0.8464 | 0.056 | 0.622 |

## 2. Paired comparisons (source-grouped bootstrap, 1000 draws per split, the same draws for every pair)

Criteria (pre-registered, applied to every pair for comparability; binding only for the route-2 adoption row): processed Δ ≥ +0.005 with CI > 0 in ≥ 3/5 splits; clean Δ ≥ −0.005; reenactment not significantly negative (CI < 0 in ≥ 3/5 rejects); clean FPR Δ ≤ +0.02 at each arm's own τ.

| comparison | rule | clean Δ | processed Δ (CI > 0) | reenactment Δ (CI < 0) | FPR Δ | meets criteria |
|---|---|---:|---:|---:|---:|---|
| route 2 — composition on the deployed recipe (UE − UB) | DS-R | +0.0031 | +0.0019 (1/5) | -0.0034 (2/5) | -0.006 | no |
| route 2 — composition on the deployed recipe (UE − UB) | LR | -0.0014 | -0.0004 (0/5) | -0.0060 (3/5) | -0.009 | no |
| route 2 — ADOPTION (UE − deployed) | DS-R | -0.0074 | -0.0023 (0/5) | -0.0043 (2/5) | -0.036 | no |
| route 2 — ADOPTION (UE − deployed) | LR | -0.0038 | +0.0165 (4/5) | -0.0107 (0/5) | -0.018 | **yes** |
| route 2 — unfreezing alone (UB − deployed) | DS-R | -0.0105 | -0.0042 (0/5) | -0.0009 (0/5) | -0.030 | no |
| route 2 — unfreezing alone (UB − deployed) | LR | -0.0024 | +0.0168 (5/5) | -0.0047 (0/5) | -0.010 | **yes** |
| route 3 — E seed 42 − deployed | DS-R | -0.0052 | +0.0037 (0/5) | -0.0087 (2/5) | -0.050 | no |
| route 3 — E seed 42 − deployed | LR | -0.0046 | +0.0215 (5/5) | -0.0120 (3/5) | -0.014 | no |
| route 3 — E seed 1337 − deployed | DS-R | -0.0094 | -0.0042 (0/5) | -0.0125 (3/5) | -0.045 | no |
| route 3 — E seed 1337 − deployed | LR | -0.0060 | +0.0207 (5/5) | -0.0145 (2/5) | +0.026 | no |
| route 3 — E seed 7 − deployed | DS-R | -0.0056 | -0.0023 (0/5) | -0.0092 (3/5) | -0.050 | no |
| route 3 — E seed 7 − deployed | LR | -0.0042 | +0.0178 (5/5) | -0.0088 (0/5) | -0.001 | **yes** |

## 3. Verdicts

**Route 2 (pre-registered adoption, DS-R): UE does NOT replace the deployed spatial views.** The deployed system stays unchanged under the pre-registered rule.

**Route 3 (override, reported as measured):** arm E − deployed under DS-R across seeds 42 / 1337 / 7 — clean -0.0052, -0.0094, -0.0056 (spread 0.0042); processed +0.0037, -0.0042, -0.0023; reenactment -0.0087, -0.0125, -0.0092. This route overrides the frozen criteria at the authors' request and is labelled as such wherever it is used.

One seed per route-2 arm; three seeds for route 3. Development only — no test set entered any row.

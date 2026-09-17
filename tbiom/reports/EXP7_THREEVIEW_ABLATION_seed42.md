# Experiment 7 — three-view ablation beside the temporal branch (`vjepa2_lora_seed42`)

Logistic stacking on matched inputs (p, vacuity), fitted on the fusion audit's source-grouped fitting folds, compared on the selection folds (5 splits). T = V-JEPA 2 LoRA, OM = Off-manifold residual, FP = Bona-fide face prior (warm+gate epoch 19). Paired source-group bootstrap, 1000 draws per split. Script `analysis/tbiom/temporal/safeguard_ablation.py`.

**Redundancy rule (fixed before the run):** branch X is redundant if removing it costs mean Δ < +0.005 on clean development AND the 95% CI upper bound < +0.005 in ≥ 4/5 splits AND ≤ 0.005 mean loss on each processed development condition.

## 1. Development (selection folds), mean ± sd over splits

| stack | clean v3.1 | swap | reenactment | in-the-wild | jpeg40 v3.1 | rs06 v3.1 |
|---|---:|---:|---:|---:|---:|---:|
| T+OM+FP | 0.8687 ±0.0040 | 0.9323 ±0.0077 | 0.8695 ±0.0091 | 0.8043 ±0.0041 | 0.7635 ±0.0090 | 0.8251 ±0.0111 |
| T+OM | 0.8717 ±0.0045 | 0.9341 ±0.0078 | 0.8836 ±0.0094 | 0.7973 ±0.0036 | 0.7854 ±0.0085 | 0.8275 ±0.0112 |
| T+FP | 0.8405 ±0.0057 | 0.8726 ±0.0078 | 0.8881 ±0.0085 | 0.7609 ±0.0076 | 0.7189 ±0.0128 | 0.7979 ±0.0078 |
| OM+FP | 0.8240 ±0.0070 | 0.9250 ±0.0100 | 0.7572 ±0.0188 | 0.7897 ±0.0089 | 0.7404 ±0.0076 | 0.7889 ±0.0116 |

## 2. Removing one branch — Δ = triple − pair (clean development), verdicts

| removed branch | pair | mean Δ | CI upper < +0.005 splits | CI lower > 0 splits | Δ jpeg40 | Δ rs06 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| FP | T+OM | -0.0030 | 5/5 | 0/5 | -0.0219 | -0.0024 | **REDUNDANT** |
| OM | T+FP | +0.0282 | 0/5 | 5/5 | +0.0445 | +0.0272 | earns its place |
| T | OM+FP | +0.0448 | 0/5 | 5/5 | +0.0231 | +0.0361 | earns its place |

Per split Δ [95% CI]: T+OM: -0.0020 [-0.0071, +0.0025], -0.0033 [-0.0085, +0.0016], -0.0034 [-0.0097, +0.0019], -0.0031 [-0.0071, +0.0004], -0.0031 [-0.0090, +0.0020]; T+FP: +0.0263 [+0.0162, +0.0379], +0.0265 [+0.0141, +0.0380], +0.0339 [+0.0228, +0.0453], +0.0313 [+0.0212, +0.0420], +0.0231 [+0.0128, +0.0341]; OM+FP: +0.0492 [+0.0353, +0.0637], +0.0396 [+0.0248, +0.0530], +0.0447 [+0.0289, +0.0596], +0.0477 [+0.0344, +0.0621], +0.0426 [+0.0283, +0.0578]

## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)

| test | T+OM+FP | T+OM | T+FP | OM+FP |
|---|---:|---:|---:|---:|
| CDFv2 | 0.9590 | 0.9638 | 0.8605 | 0.9722 |
| CDFv3 | 0.9569 | 0.9584 | 0.9395 | 0.8863 |
| DFD | 0.9257 | 0.9312 | 0.8668 | 0.9373 |
| DFDC | 0.8581 | 0.8608 | 0.8126 | 0.8848 |
| DFDCP | 0.8759 | 0.8668 | 0.8278 | 0.8862 |
| DFEval24 | 0.7111 | 0.7037 | 0.7176 | 0.6849 |
| FFIW | 0.9220 | 0.9218 | 0.8545 | 0.9340 |
| KoDF | 0.9575 | 0.9560 | 0.9649 | 0.8549 |
| IDForge | 0.9568 | 0.9591 | 0.9287 | 0.9674 |
| DFDM | 0.9920 | 0.9932 | 0.9821 | 0.9925 |
| CDFv2 jpeg40 | 0.8808 | 0.8953 | 0.6836 | 0.8861 |
| CDFv2 rs06 | 0.9270 | 0.9314 | 0.8411 | 0.9542 |
| DFDCP jpeg40 | 0.8030 | 0.8238 | 0.7576 | 0.8608 |
| DFDCP rs06 | 0.8168 | 0.8099 | 0.8045 | 0.8110 |
| **10-set mean** | **0.9115** | **0.9115** | **0.8755** | **0.9000** |

Conditional on one development-selected checkpoint per branch (one detector seed; one temporal-branch seed). CDFv2 and DeepfakeEval2024 have informed development. FaceShifter is sealed.

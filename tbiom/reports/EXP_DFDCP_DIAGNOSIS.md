# Experiment 3 — DFDCP / processing weakness: diagnosis for the reference triple

Two candidate causes, criteria fixed before the run, both evaluated for **DS-R** and its **logistic** comparator.

- **Cause (b) — the temporal branch's own discrimination collapses under compression:** V-JEPA's development v3.1 falls ≥ 0.05 from clean to JPEG q40. If it holds, report whether the B-augmented branch recovers DFDCP relative to the unaugmented one.
- **Cause (a) — the fusion deteriorates:** on processed development the fused score is below the best single branch, OR the fused drop exceeds the smallest single-branch drop by ≥ 0.02. Remedy, no encoder retraining: refit the fusion on clean + JPEG q40 + resize 0.60 fitting-fold videos and evaluate on the separate source-grouped selection folds; adopt if processed development improves ≥ 0.005 with clean loss ≤ 0.005.

Both can coexist and are reported separately.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development (selection folds, v3.1, mean ± sd over 5 splits)

| arm | clean | JPEG q40 | resize 0.60 |
|---|---:|---:|---:|
| branch T | 0.8094 ±0.0071 | 0.7596 ±0.0081 | 0.8027 ±0.0077 |
| branch APP | 0.8174 ±0.0062 | 0.7179 ±0.0066 | 0.7848 ±0.0115 |
| branch OM | 0.8241 ±0.0067 | 0.7429 ±0.0063 | 0.7845 ±0.0112 |
| DS-R (clean fit) | 0.8808 ±0.0048 | 0.8096 ±0.0072 | 0.8589 ±0.0086 |
| LR (clean fit) | 0.8736 ±0.0068 | 0.7564 ±0.0065 | 0.8542 ±0.0094 |
| DS-R (clean+proc fit) | 0.8800 ±0.0042 | 0.8104 ±0.0077 | 0.8589 ±0.0087 |
| LR (clean+proc fit) | 0.8764 ±0.0056 | 0.8060 ±0.0057 | 0.8539 ±0.0097 |

## 2. Verdicts

- **Cause (b):** the B-augmented V-JEPA branch drops **+0.0498** clean → JPEG q40 on development (criterion ≥ 0.05) → **does not hold**. The *unaugmented* branch drops +0.0867 on the same videos: the criterion was written against a branch that failed it, and the B-augmented reference now sits just under the line. Read it as 'augmentation removed most, not all, of the collapse' — a near-miss, not a clean pass.
- **Cause (a), DS-R, jpeg40:** fused below best single branch: False; fused drop +0.0712 vs smallest branch drop +0.0498 (excess +0.0214, criterion ≥ 0.02) → **HOLDS**. Branch drops: T +0.0498, APP +0.0994, OM +0.0813
- **Cause (a), DS-R, rs06:** fused below best single branch: False; fused drop +0.0219 vs smallest branch drop +0.0067 (excess +0.0152, criterion ≥ 0.02) → **does not hold**. Branch drops: T +0.0067, APP +0.0326, OM +0.0396
- **Cause (a), LR, jpeg40:** fused below best single branch: True; fused drop +0.1172 vs smallest branch drop +0.0498 (excess +0.0674, criterion ≥ 0.02) → **HOLDS**. Branch drops: T +0.0498, APP +0.0994, OM +0.0813
- **Cause (a), LR, rs06:** fused below best single branch: False; fused drop +0.0194 vs smallest branch drop +0.0067 (excess +0.0127, criterion ≥ 0.02) → **does not hold**. Branch drops: T +0.0067, APP +0.0326, OM +0.0396
- **Remedy (clean+processed refit), DS-R:** processed development +0.0004 (criterion ≥ +0.005), clean development loss +0.0008 (criterion ≤ 0.005) → numerical criteria not met; cause (a) holds → **not adopted**. DFDCP change under the refit: DFDCP -0.0029, DFDCP jpeg40 -0.0026, DFDCP rs06 -0.0016
- **Remedy (clean+processed refit), LR:** processed development +0.0246 (criterion ≥ +0.005), clean development loss -0.0028 (criterion ≤ 0.005) → numerical criteria met; cause (a) holds → **ADOPT**. DFDCP change under the refit: DFDCP +0.0085, DFDCP jpeg40 +0.0848, DFDCP rs06 +0.0084

## 3. Cause (b) follow-up — B-augmented vs unaugmented temporal branch

Same recipe, same fixed APP/OM checkpoints, same folds; only the temporal branch differs (`vjepa2_lora_seed42` vs `vjepa2_lora_B_seed42`).

| condition | arm | unaugmented | B-augmented | Δ |
|---|---|---:|---:|---:|
| development clean | branch T | 0.8232 | 0.8094 | -0.0139 |
| development jpeg40 | branch T | 0.7365 | 0.7596 | +0.0231 |
| development rs06 | branch T | 0.8142 | 0.8027 | -0.0115 |
| DFDCP | branch T | 0.7762 | 0.7408 | -0.0354 |
| DFDCP | DS-R (clean fit) | 0.8671 | 0.8656 | -0.0014 |
| DFDCP | LR (clean fit) | 0.8523 | 0.8557 | +0.0034 |
| DFDCP jpeg40 | branch T | 0.7191 | 0.7094 | -0.0096 |
| DFDCP jpeg40 | DS-R (clean fit) | 0.8244 | 0.8198 | -0.0046 |
| DFDCP jpeg40 | LR (clean fit) | 0.7068 | 0.7299 | +0.0231 |
| DFDCP rs06 | branch T | 0.7557 | 0.7152 | -0.0405 |
| DFDCP rs06 | DS-R (clean fit) | 0.8107 | 0.8084 | -0.0023 |
| DFDCP rs06 | LR (clean fit) | 0.7981 | 0.7935 | -0.0046 |
| CDFv2 | branch T | 0.7964 | 0.7383 | -0.0581 |
| CDFv2 | DS-R (clean fit) | 0.9556 | 0.9598 | +0.0043 |
| CDFv2 | LR (clean fit) | 0.9636 | 0.9697 | +0.0061 |
| CDFv2 jpeg40 | branch T | 0.7153 | 0.6692 | -0.0461 |
| CDFv2 jpeg40 | DS-R (clean fit) | 0.8797 | 0.8695 | -0.0102 |
| CDFv2 jpeg40 | LR (clean fit) | 0.6910 | 0.7199 | +0.0289 |
| CDFv2 rs06 | branch T | 0.7727 | 0.7116 | -0.0611 |
| CDFv2 rs06 | DS-R (clean fit) | 0.9307 | 0.9418 | +0.0111 |
| CDFv2 rs06 | LR (clean fit) | 0.9314 | 0.9470 | +0.0156 |

## 4. Test sets — video AUROC (read after fitting; reference triple, mean over the 5 fitted splits)

| test | branch T | branch APP | branch OM | DS-R (clean fit) | LR (clean fit) | DS-R (clean+proc fit) | LR (clean+proc fit) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.7383 | 0.9612 | 0.9736 | 0.9598 | 0.9697 | 0.9576 | 0.9595 |
| CDFv3 | 0.9263 | 0.8576 | 0.8801 | 0.9616 | 0.9675 | 0.9640 | 0.9584 |
| DFD | 0.7967 | 0.9304 | 0.9338 | 0.9338 | 0.9300 | 0.9311 | 0.9283 |
| DFDC | 0.7009 | 0.8758 | 0.8808 | 0.8626 | 0.8602 | 0.8570 | 0.8598 |
| DFDCP | 0.7408 | 0.8698 | 0.8727 | 0.8656 | 0.8557 | 0.8628 | 0.8641 |
| DFEval24 | 0.6732 | 0.6741 | 0.6688 | 0.7252 | 0.6865 | 0.7241 | 0.7206 |
| FFIW | 0.8011 | 0.9186 | 0.9354 | 0.9282 | 0.9262 | 0.9226 | 0.9252 |
| KoDF | 0.9656 | 0.8494 | 0.8548 | 0.9574 | 0.9630 | 0.9596 | 0.9531 |
| IDForge | 0.8899 | 0.9510 | 0.9635 | 0.9656 | 0.9622 | 0.9639 | 0.9615 |
| DFDM | 0.9798 | 0.9886 | 0.9920 | 0.9956 | 0.9963 | 0.9956 | 0.9943 |
| CDFv3 reenactment | 0.9741 | 0.8137 | 0.8416 | 0.9642 | 0.9709 | 0.9682 | 0.9602 |
| KoDF first-order motion | 0.9887 | 0.7164 | 0.7306 | 0.9814 | 0.9792 | 0.9836 | 0.9733 |
| CDFv2 jpeg40 | 0.6692 | 0.8333 | 0.8872 | 0.8695 | 0.7199 | 0.8674 | 0.8771 |
| CDFv2 rs06 | 0.7116 | 0.9451 | 0.9549 | 0.9418 | 0.9470 | 0.9386 | 0.9420 |
| DFDCP jpeg40 | 0.7094 | 0.8060 | 0.8256 | 0.8198 | 0.7299 | 0.8172 | 0.8147 |
| DFDCP rs06 | 0.7152 | 0.7981 | 0.7974 | 0.8084 | 0.7935 | 0.8068 | 0.8019 |
| **10-set mean** | **0.8213** | **0.8877** | **0.8955** | **0.9155** | **0.9117** | **0.9138** | **0.9125** |

The processed development severities (JPEG q40, resize 0.60) are the same severities as the processed test rows, so the clean+processed refit is severity-matched to those tests and its numbers there are optimistic.

DFDCP false positives at the frozen τ run high for every model on this benchmark (see Experiment 1's DFDCP rows), which is the same phenomenon the uncertainty safeguard is asked to flag.

**FreqBlender note (brief §Experiment 3).** The brief: if the fusion refit recovers DFDCP, the processing story is complete and the FreqBlender study becomes an additional study rather than a needed fix. Result: the refit is **not adopted for DS-R** (processed development +0.0004, DFDCP JPEG -0.0026) and **adopted for the logistic comparator** (processed development +0.0246, DFDCP JPEG +0.0848). DS-R has almost nothing to recover by refitting — the processed refit returns the clean weights on 3 of the 5 splits and buys +0.0004 on processed development — whereas the logistic comparator's weights move and gain 0.025, which is a statement about the comparator's fragility under processing, not a completed processing story for the reference. DFDCP stays the reference's weakest clean benchmark, so Experiment 5 is still run, as requested.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

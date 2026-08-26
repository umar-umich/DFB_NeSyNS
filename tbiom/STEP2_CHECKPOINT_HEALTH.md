# Step 2 — P0-DS checkpoint re-selection on the health dashboard

Readout: `p_fused`. Threshold frozen per checkpoint on **FF++ val**; selection metrics from **VALmix** only. No final OOD test set is touched — choosing an epoch on CDFv2/DFDC/DFD/DFDCP would be selecting on test.

`macro` averages VALmix's three domains (CDFv2val, DFDCPval, DFEval24val) so no domain dominates. **OOD-real FPR overrides AUROC as the health gate**, per the brief.

| checkpoint | tau | FF++ val AUROC | macro AUROC | EER | **macro FPR_real** | d_RF | mean p on real |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0-DS epoch 1 (the original pick) | 0.4784 | 0.9961 | 0.8647 | 0.1859 | **0.213** | +0.439 | 0.322 |
| P0-DS epoch 2 | 0.4736 | 0.9950 | 0.8589 | 0.1970 | **0.246** | +0.458 | 0.307 |
| P0-DS epoch 5 | 0.5705 | 0.9954 | 0.8603 | 0.1985 | **0.213** | +0.469 | 0.323 |
| P0-DS last epoch | 0.5298 | 0.9928 | 0.8483 | 0.2119 | **0.196** | +0.482 | 0.272 |
| DiCoME released (reference, not retrainable) | 0.4069 | 0.9935 | 0.8671 | 0.1889 | **0.212** | +0.510 | 0.241 |

## Per VALmix domain

| checkpoint | CDFv2val AUROC / FPR_real | DFDCPval AUROC / FPR_real | DFEval24val AUROC / FPR_real |
|---|---|---|---|
| P0-DS epoch 1 (the original pick) | 0.9635 / 0.142 | 0.9869 / 0.169 | 0.6436 / 0.329 |
| P0-DS epoch 2 | 0.9693 / 0.160 | 0.9866 / 0.196 | 0.6206 / 0.382 |
| P0-DS epoch 5 | 0.9679 / 0.138 | 0.9839 / 0.169 | 0.6291 / 0.333 |
| P0-DS last epoch | 0.9549 / 0.133 | 0.9786 / 0.156 | 0.6113 / 0.298 |
| DiCoME released (reference, not retrainable) | 0.9721 / 0.164 | 0.9866 / 0.156 | 0.6425 / 0.316 |

## Decision

Best real-side health: **P0-DS last epoch** at macro FPR_real 0.196 (macro AUROC 0.8483).
Best macro AUROC: **P0-DS epoch 1 (the original pick)** at 0.8647 (macro FPR_real 0.213).

Against the original epoch-1 pick, the healthiest checkpoint moves FPR_real by **-0.018** and macro AUROC by **-0.0164**.

**Mixed.** Real-side health improves but not decisively, or it improves at a real AUROC cost. Record both and treat the chassis choice as still open pending the recipe ablation.

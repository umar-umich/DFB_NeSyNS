# V1_BRANCH_DIAGNOSTICS — §20 domain-detector distribution audit

Source: `['logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet']` (epoch ?), 387913 frames across 7 sources.

**Reading this table.** `forensic` separates reals from fakes *within* an OOD source, so domain shift is common to both classes and cancels. `domain` separates FF++ **test** reals from OOD reals, where no manipulation is involved at all. §20's test is `D(R_FF++, R_OOD) << D(R_OOD, F_OOD)`; a branch whose domain number meets its forensic number is reading the dataset, not the forgery.

Both are AUROC-based separability folded to [0.5, 1], so quantities on different scales stay comparable. §20 makes this a required diagnostic, not a gate: failures are flagged for the iterate stage.

## Branch `ref`

| source | verdict | worst quantity | forensic | domain | gap |
|---|---|---|---|---:|---:|
| Celeb-DF-v2 | **DATASET DETECTOR** | `ref_residual_norm` | 0.5998 | 0.8052 | -0.2054 |
| Celeb-DF-v3 | **DATASET DETECTOR** | `ref_residual_norm` | 0.6513 | 0.8052 | -0.1539 |
| DFDC | **DATASET DETECTOR** | `ref_residual_norm` | 0.5193 | 0.7998 | -0.2805 |
| DFDCP | **DATASET DETECTOR** | `ref_residual_norm` | 0.5585 | 0.8314 | -0.2729 |
| Deepfake-Eval-2024 | **DATASET DETECTOR** | `ref_residual_norm` | 0.5219 | 0.7883 | -0.2663 |
| UADFV | **DATASET DETECTOR** | `ref_residual_norm` | 0.5177 | 0.6329 | -0.1152 |

<details><summary>every quantity</summary>

| source | quantity | forensic | domain | gap | shift R_OOD (σ) | shift F_OOD (σ) | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Celeb-DF-v2 | `ref_residual_norm` | 0.5998 | 0.8052 | -0.2054 | +1.22 | +1.56 | DATASET DETECTOR |
| Celeb-DF-v2 | `ref_angle` | 0.6050 | 0.7884 | -0.1834 | +1.09 | +1.46 | DATASET DETECTOR |
| Celeb-DF-v2 | `p_ref` | 0.6996 | 0.5477 | +0.1519 | +0.16 | +0.82 | pass |
| Celeb-DF-v2 | `u_ref` | 0.6187 | 0.5237 | +0.0951 | +0.09 | -0.36 | pass |
| Celeb-DF-v3 | `ref_residual_norm` | 0.6513 | 0.8052 | -0.1539 | +1.22 | +1.74 | DATASET DETECTOR |
| Celeb-DF-v3 | `ref_angle` | 0.6630 | 0.7884 | -0.1254 | +1.09 | +1.67 | DATASET DETECTOR |
| Celeb-DF-v3 | `p_ref` | 0.6982 | 0.5477 | +0.1505 | +0.16 | +0.81 | pass |
| Celeb-DF-v3 | `u_ref` | 0.6329 | 0.5237 | +0.1092 | +0.09 | -0.45 | pass |
| DFDC | `ref_residual_norm` | 0.5193 | 0.7998 | -0.2805 | +1.52 | +1.67 | DATASET DETECTOR |
| DFDC | `ref_angle` | 0.5273 | 0.8039 | -0.2766 | +1.61 | +1.84 | DATASET DETECTOR |
| DFDC | `p_ref` | 0.7332 | 0.6450 | +0.0882 | +0.50 | +1.21 | pass |
| DFDC | `u_ref` | 0.7280 | 0.6866 | +0.0414 | -0.72 | -1.64 | BORDERLINE |
| DFDCP | `ref_residual_norm` | 0.5585 | 0.8314 | -0.2729 | +1.96 | +2.59 | DATASET DETECTOR |
| DFDCP | `ref_angle` | 0.5648 | 0.8320 | -0.2672 | +2.19 | +3.09 | DATASET DETECTOR |
| DFDCP | `p_ref` | 0.6933 | 0.7097 | -0.0164 | +0.73 | +1.29 | DATASET DETECTOR |
| DFDCP | `u_ref` | 0.6694 | 0.7362 | -0.0668 | -0.98 | -1.63 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `ref_residual_norm` | 0.5219 | 0.7883 | -0.2663 | +2.27 | +1.82 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `ref_angle` | 0.5206 | 0.7690 | -0.2483 | +2.72 | +2.04 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `p_ref` | 0.5429 | 0.6227 | -0.0798 | +0.43 | +0.61 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `u_ref` | 0.5417 | 0.6326 | -0.0909 | -0.61 | -0.81 | DATASET DETECTOR |
| UADFV | `ref_residual_norm` | 0.5177 | 0.6329 | -0.1152 | +0.52 | +0.58 | DATASET DETECTOR |
| UADFV | `ref_angle` | 0.5282 | 0.6338 | -0.1056 | +0.51 | +0.58 | DATASET DETECTOR |
| UADFV | `p_ref` | 0.8219 | 0.5966 | +0.2253 | -0.33 | +0.89 | pass |
| UADFV | `u_ref` | 0.7188 | 0.5553 | +0.1634 | +0.20 | -0.81 | pass |

</details>

## Branch `proc`

| source | verdict | worst quantity | forensic | domain | gap |
|---|---|---|---|---:|---:|
| Celeb-DF-v2 | **DATASET DETECTOR** | `proc_mse_std` | 0.5853 | 0.7877 | -0.2024 |
| Celeb-DF-v3 | **DATASET DETECTOR** | `proc_mse_max` | 0.5658 | 0.7657 | -0.2000 |
| DFDC | **DATASET DETECTOR** | `proc_mse_p90` | 0.5326 | 0.7186 | -0.1861 |
| DFDCP | **DATASET DETECTOR** | `proc_mse_p90` | 0.6031 | 0.7124 | -0.1092 |
| Deepfake-Eval-2024 | **DATASET DETECTOR** | `p_proc` | 0.5075 | 0.6202 | -0.1127 |
| UADFV | **UNINFORMATIVE** | `proc_lpips_proxy` | 0.5320 | 0.5127 | +0.0193 |

<details><summary>every quantity</summary>

| source | quantity | forensic | domain | gap | shift R_OOD (σ) | shift F_OOD (σ) | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Celeb-DF-v2 | `proc_mse_mean` | 0.5953 | 0.7793 | -0.1840 | -0.31 | -0.37 | DATASET DETECTOR |
| Celeb-DF-v2 | `proc_mse_std` | 0.5853 | 0.7877 | -0.2024 | -0.38 | -0.44 | DATASET DETECTOR |
| Celeb-DF-v2 | `proc_mse_max` | 0.5636 | 0.7658 | -0.2022 | -0.55 | -0.63 | DATASET DETECTOR |
| Celeb-DF-v2 | `proc_mse_p90` | 0.6011 | 0.7557 | -0.1545 | -0.28 | -0.33 | DATASET DETECTOR |
| Celeb-DF-v2 | `proc_lpips_proxy` | 0.6032 | 0.7864 | -0.1833 | -0.29 | -0.33 | DATASET DETECTOR |
| Celeb-DF-v2 | `proc_center_ratio` | 0.5972 | 0.5193 | +0.0779 | -0.00 | -0.31 | pass |
| Celeb-DF-v2 | `p_proc` | 0.6539 | 0.6608 | -0.0069 | +0.52 | +0.90 | DATASET DETECTOR |
| Celeb-DF-v2 | `u_proc` | 0.6522 | 0.6513 | +0.0010 | -0.45 | -0.84 | BORDERLINE |
| Celeb-DF-v3 | `proc_mse_mean` | 0.6837 | 0.7793 | -0.0957 | -0.31 | -0.44 | DATASET DETECTOR |
| Celeb-DF-v3 | `proc_mse_std` | 0.6234 | 0.7877 | -0.1643 | -0.38 | -0.50 | DATASET DETECTOR |
| Celeb-DF-v3 | `proc_mse_max` | 0.5658 | 0.7657 | -0.2000 | -0.55 | -0.66 | DATASET DETECTOR |
| Celeb-DF-v3 | `proc_mse_p90` | 0.7026 | 0.7556 | -0.0530 | -0.28 | -0.40 | DATASET DETECTOR |
| Celeb-DF-v3 | `proc_lpips_proxy` | 0.6553 | 0.7864 | -0.1311 | -0.29 | -0.38 | DATASET DETECTOR |
| Celeb-DF-v3 | `proc_center_ratio` | 0.5102 | 0.5193 | -0.0091 | -0.00 | +0.04 | UNINFORMATIVE |
| Celeb-DF-v3 | `p_proc` | 0.5746 | 0.6608 | -0.0862 | +0.52 | +0.70 | DATASET DETECTOR |
| Celeb-DF-v3 | `u_proc` | 0.5680 | 0.6513 | -0.0833 | -0.45 | -0.62 | DATASET DETECTOR |
| DFDC | `proc_mse_mean` | 0.5380 | 0.6782 | -0.1402 | -0.04 | -0.01 | DATASET DETECTOR |
| DFDC | `proc_mse_std` | 0.5488 | 0.5986 | -0.0499 | -0.14 | -0.16 | DATASET DETECTOR |
| DFDC | `proc_mse_max` | 0.5516 | 0.5840 | -0.0324 | -0.25 | -0.34 | DATASET DETECTOR |
| DFDC | `proc_mse_p90` | 0.5326 | 0.7186 | -0.1861 | -0.00 | +0.04 | DATASET DETECTOR |
| DFDC | `proc_lpips_proxy` | 0.5381 | 0.6149 | -0.0769 | +0.13 | +0.17 | DATASET DETECTOR |
| DFDC | `proc_center_ratio` | 0.5711 | 0.5902 | -0.0191 | +0.34 | +0.05 | DATASET DETECTOR |
| DFDC | `p_proc` | 0.5858 | 0.5709 | +0.0148 | +0.16 | +0.43 | BORDERLINE |
| DFDC | `u_proc` | 0.5904 | 0.5708 | +0.0196 | -0.22 | -0.51 | BORDERLINE |
| DFDCP | `proc_mse_mean` | 0.5935 | 0.6863 | -0.0928 | -0.01 | -0.09 | DATASET DETECTOR |
| DFDCP | `proc_mse_std` | 0.5823 | 0.6239 | -0.0416 | -0.01 | -0.03 | DATASET DETECTOR |
| DFDCP | `proc_mse_max` | 0.5714 | 0.5972 | -0.0258 | -0.18 | -0.23 | DATASET DETECTOR |
| DFDCP | `proc_mse_p90` | 0.6031 | 0.7124 | -0.1092 | -0.02 | -0.17 | DATASET DETECTOR |
| DFDCP | `proc_lpips_proxy` | 0.5927 | 0.6325 | -0.0398 | +0.06 | -0.02 | DATASET DETECTOR |
| DFDCP | `proc_center_ratio` | 0.6427 | 0.5562 | +0.0865 | +0.19 | -0.49 | pass |
| DFDCP | `p_proc` | 0.6945 | 0.6075 | +0.0870 | +0.40 | +1.01 | pass |
| DFDCP | `u_proc` | 0.6808 | 0.6013 | +0.0795 | -0.40 | -1.01 | pass |
| Deepfake-Eval-2024 | `proc_mse_mean` | 0.5024 | 0.5896 | -0.0872 | +0.04 | -0.01 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `proc_mse_std` | 0.5069 | 0.5606 | -0.0538 | +0.27 | +0.14 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `proc_mse_max` | 0.5051 | 0.5507 | -0.0456 | +0.14 | +0.05 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `proc_mse_p90` | 0.5025 | 0.6076 | -0.1051 | -0.06 | -0.08 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `proc_lpips_proxy` | 0.5058 | 0.5690 | -0.0632 | +0.13 | +0.07 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `proc_center_ratio` | 0.5111 | 0.5336 | -0.0224 | -0.12 | -0.05 | UNINFORMATIVE |
| Deepfake-Eval-2024 | `p_proc` | 0.5075 | 0.6202 | -0.1127 | +0.43 | +0.41 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `u_proc` | 0.5099 | 0.6221 | -0.1122 | -0.53 | -0.47 | DATASET DETECTOR |
| UADFV | `proc_mse_mean` | 0.5334 | 0.5126 | +0.0208 | -0.13 | -0.17 | UNINFORMATIVE |
| UADFV | `proc_mse_std` | 0.5443 | 0.5109 | +0.0333 | -0.14 | -0.17 | UNINFORMATIVE |
| UADFV | `proc_mse_max` | 0.5422 | 0.5227 | +0.0195 | -0.16 | -0.22 | UNINFORMATIVE |
| UADFV | `proc_mse_p90` | 0.5347 | 0.5040 | +0.0307 | -0.13 | -0.16 | UNINFORMATIVE |
| UADFV | `proc_lpips_proxy` | 0.5320 | 0.5127 | +0.0193 | -0.10 | -0.14 | UNINFORMATIVE |
| UADFV | `proc_center_ratio` | 0.5869 | 0.5663 | +0.0206 | -0.26 | -0.52 | BORDERLINE |
| UADFV | `p_proc` | 0.6233 | 0.5581 | +0.0652 | +0.20 | +0.62 | pass |
| UADFV | `u_proc` | 0.6236 | 0.5392 | +0.0844 | -0.10 | -0.52 | pass |

</details>

## Branch `sem`

| source | verdict | worst quantity | forensic | domain | gap |
|---|---|---|---|---:|---:|
| Celeb-DF-v2 | **DATASET DETECTOR** | `u_sem` | 0.5544 | 0.6571 | -0.1027 |
| Celeb-DF-v3 | **DATASET DETECTOR** | `u_sem` | 0.5325 | 0.6571 | -0.1247 |
| DFDC | **BORDERLINE** | `u_sem` | 0.6475 | 0.6074 | +0.0400 |
| DFDCP | **DATASET DETECTOR** | `u_sem` | 0.6142 | 0.6510 | -0.0368 |
| Deepfake-Eval-2024 | **DATASET DETECTOR** | `u_sem` | 0.5282 | 0.7472 | -0.2190 |
| UADFV | **pass** | `u_sem` | 0.9579 | 0.6764 | +0.2816 |

<details><summary>every quantity</summary>

| source | quantity | forensic | domain | gap | shift R_OOD (σ) | shift F_OOD (σ) | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Celeb-DF-v2 | `p_sem` | 0.8673 | 0.6635 | +0.2038 | +0.27 | +3.04 | pass |
| Celeb-DF-v2 | `u_sem` | 0.5544 | 0.6571 | -0.1027 | +0.25 | +0.55 | DATASET DETECTOR |
| Celeb-DF-v3 | `p_sem` | 0.8318 | 0.6635 | +0.1682 | +0.27 | +2.65 | pass |
| Celeb-DF-v3 | `u_sem` | 0.5325 | 0.6571 | -0.1247 | +0.25 | +0.71 | DATASET DETECTOR |
| DFDC | `p_sem` | 0.8239 | 0.6649 | +0.1589 | +0.67 | +3.15 | pass |
| DFDC | `u_sem` | 0.6475 | 0.6074 | +0.0400 | +0.29 | +0.26 | BORDERLINE |
| DFDCP | `p_sem` | 0.8631 | 0.7078 | +0.1554 | +0.72 | +3.45 | pass |
| DFDCP | `u_sem` | 0.6142 | 0.6510 | -0.0368 | +0.37 | +0.45 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `p_sem` | 0.6244 | 0.7827 | -0.1583 | +1.07 | +1.83 | DATASET DETECTOR |
| Deepfake-Eval-2024 | `u_sem` | 0.5282 | 0.7472 | -0.2190 | +0.73 | +0.93 | DATASET DETECTOR |
| UADFV | `p_sem` | 0.9872 | 0.6653 | +0.3220 | -0.10 | +4.59 | pass |
| UADFV | `u_sem` | 0.9579 | 0.6764 | +0.2816 | +0.05 | -0.44 | pass |

</details>

## Figures

- `distributions_<quantity>.png` — one panel per source, showing `p(r | R_FF++)`, `p(r | R_OOD)` and `p(r | F_OOD)` together. This is §20's plot.
- `tsne_evidence_space.png` — the per-sample evidence space. The two pooled panels are the ones to read: clustering by DATASET means the evidence space is arranged by provenance, clustering by LABEL means it is arranged by authenticity.

The t-SNE is computed on the evidence and reliability quantities (per-branch p/u, residual diagnostics, process statistics, V/C/A) rather than on raw backbone features, because that is the space fusion and the applicability gate actually operate in.

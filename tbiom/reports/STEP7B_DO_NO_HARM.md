# Which experts can join without hurting the system?

A weaker bar than Step 5's. Step 5 asked whether complementarity is RECOVERABLE by a label-free gate and everything failed. This asks whether an expert leaves the system no worse — a deployment question, not an evidence-of-gating question. Both are kept separate so neither claim borrows the other's support.

Anchor: **P0-DS epoch 1, `p_fused`**, VALmix (1350 videos). §22 noise band = 0.01.

| arm | AUROC | Δ AUROC | FPR_real@τ | Δ FPR | d_RF | verdict |
|---|---:|---:|---:|---:|---:|---|
| anchor alone | 0.8852 | +0.0000 | 0.213 | +0.000 | +0.439 | — |
| CCF: anchor + fsvfm_ordinary | 0.8855 | +0.0003 | 0.190 | -0.024 | +0.429 | harmless |
| DS: anchor + fsvfm_ordinary | 0.8862 | +0.0010 | 0.197 | -0.016 | +0.535 | harmless |
| CCF: anchor + fsvfm_preserve | 0.8882 | +0.0030 | 0.193 | -0.021 | +0.429 | harmless |
| DS: anchor + fsvfm_preserve | 0.8865 | +0.0012 | 0.201 | -0.012 | +0.535 | harmless |
| CCF: anchor + mrvae_rate | 0.8837 | -0.0016 | 0.187 | -0.027 | +0.343 | harmless |
| DS: anchor + mrvae_rate | 0.8843 | -0.0009 | 0.188 | -0.025 | +0.366 | harmless |
| CCF: anchor + fsvfm_ordinary + fsvfm_preserve | 0.8816 | -0.0037 | 0.200 | -0.013 | +0.412 | harmless |
| DS: anchor + fsvfm_ordinary + fsvfm_preserve | 0.8726 | -0.0127 | 0.215 | +0.001 | +0.537 | **HARMS** |
| CCF: anchor + fsvfm_ordinary + mrvae_rate | 0.8755 | -0.0097 | 0.190 | -0.024 | +0.246 | harmless |
| DS: anchor + fsvfm_ordinary + mrvae_rate | 0.8850 | -0.0002 | 0.197 | -0.016 | +0.529 | harmless |
| CCF: anchor + fsvfm_preserve + mrvae_rate | 0.8802 | -0.0050 | 0.191 | -0.022 | +0.250 | harmless |
| DS: anchor + fsvfm_preserve + mrvae_rate | 0.8854 | +0.0002 | 0.200 | -0.013 | +0.526 | harmless |
| CCF: anchor + fsvfm_ordinary + fsvfm_preserve + mrvae_rate | 0.8529 | -0.0324 | 0.209 | -0.004 | +0.172 | **HARMS** |
| DS: anchor + fsvfm_ordinary + fsvfm_preserve + mrvae_rate | 0.8716 | -0.0136 | 0.215 | +0.001 | +0.538 | **HARMS** |

## Reading

11 of 14 arms leave the system no worse on both axes.

Best by AUROC: **CCF: anchor + fsvfm_preserve** at 0.8882 (+0.0030 vs anchor alone, FPR_real 0.193 vs 0.213).

**What may and may not be claimed.** A gain inside the ±0.01 band is not a demonstrated improvement, so these arms support 'this expert can be carried without cost' and NOT 'this expert improves detection'. The framework paper can include a harmless branch as an available component; it cannot cite it as evidence that applicability fusion works, which is what Steps 5-7 tested and rejected.

> Fused arms take their EER on their own VALmix scores because no FF++ val scores exist for fused combinations, while the anchor's τ is frozen on FF++ val. That flatters the fused arms; they are judged against it anyway.

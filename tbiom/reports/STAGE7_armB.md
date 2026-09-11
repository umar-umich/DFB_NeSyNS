# Stage 7 — conflict-aware fusion (run1auxedl)

Replayed over the SAME saved branch opinions; no retraining. Each view's Dirichlet opinion is recovered from its exported `(p, u)` and converted to a subjective-logic opinion, then fused N-ary.

| operator | conflict handling |
|---|---|
| DS | renormalised away — the incumbent |
| Yager | assigned to **vacuity** instead of renormalised |
| Murphy | opinions averaged, then combined N−1 times |
| CCF | multi-source consensus & compromise (FUSION 2018), genuinely N-ary |

## Complete-framework video AUROC

| dataset | DS | Yager | Murphy | CCF | best branch | best simple | P0-DS |
|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9439 | 0.9416 | **0.9500** | 0.9363 | 0.9607 | 0.9597 | 0.9646 |
| CDFv3 | 0.8866 | **0.8883** | 0.8868 | 0.8810 | 0.8750 | 0.8877 | 0.8409 |
| DFD | 0.9430 | 0.9414 | **0.9483** | 0.9374 | 0.9418 | 0.9499 | 0.9421 |
| DFDC | 0.8920 | 0.8865 | **0.8934** | 0.8815 | 0.8851 | 0.8906 | 0.8828 |
| DFDCP | **0.9038** | 0.8920 | 0.8997 | 0.8881 | 0.8898 | 0.9038 | 0.8573 |
| DFEval24 | 0.6886 | 0.6917 | **0.6938** | 0.6847 | 0.6826 | 0.6915 | 0.6922 |
| VALmix | **0.8829** | 0.8757 | 0.8828 | 0.8722 | 0.8752 | 0.8831 | 0.8852 |

| operator | mean AUROC (6 OOD) | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| DS | 0.8763 | 0.214 | +0.0130 |
| Yager | 0.8736 | 0.190 | +0.0103 |
| Murphy | 0.8787 | 0.226 | +0.0153 |
| CCF | 0.8682 | 0.175 | +0.0049 |
| *(simple F1)* | 0.8778 | — | +0.0145 |
| *(simple F2)* | 0.8774 | — | +0.0141 |
| *(simple F3)* | 0.8768 | — | +0.0135 |

## Pass condition

From the brief: *beats simple fusion on the suppressed cases while holding the strong datasets.* Suppressed = fused loses to its own best branch under DS.

| suppressed dataset | DS | best branch | best conflict-aware | best simple | fixed? |
|---|---:|---:|---:|---:|---|
| CDFv2 | 0.9439 | 0.9607 | **0.9500** (Murphy) | 0.9597 | no |

# Step 4 — single-anchor reliability, R = g(V, M)

V1's ~0.81 error-detection came from the MULTI-BRANCH setting with V, C, U_sup and M. With one anchor there is no inter-branch conflict and no support uncertainty, so the fallback spine has two features. Whether it still predicts errors is the open question this step exists to answer; C and U_sup are **not** fabricated from a single branch.

Fitted on FF++ **VAL_meta** only, cross-fit by identity group so relatives cannot straddle a fold. `tau` comes from VAL_select, so the operating point never sees the partition the risk model is fitted on.

## In-domain (FF++ VAL_meta, out-of-fold)

| readout | VAL_meta videos | anchor errors | **risk-AUROC** | selective risk @90% | base error |
|---|---:|---:|---:|---:|---:|
| artifact view (Step-3 pick) | 8960 | 346 | **0.9121** | 0.0133 | 0.0386 |
| fused (DS) | 8960 | 350 | **0.9175** | 0.0114 | 0.0391 |
| semantic (CLIP branch) | 8960 | 354 | **0.8735** | 0.0145 | 0.0395 |

## Under shift — the same fitted model applied to each OOD domain

| readout | Celeb-DF-v2 | Celeb-DF-v3 | DFD | DFDC | DFDCP | Deepfake-Eval-2024 | mean |
|---|---|---|---|---|---|---|---|
| artifact view (Step-3 pick) | 0.744 | 0.714 | 0.899 | 0.692 | 0.720 | 0.567 | **0.723** |
| fused (DS) | 0.778 | 0.799 | 0.928 | 0.683 | 0.739 | 0.563 | **0.748** |
| semantic (CLIP branch) | 0.752 | 0.952 | 0.942 | 0.607 | 0.731 | 0.516 | **0.750** |

### Selective risk at a 10% deferral budget

Error rate on the 90% of videos the risk model is most confident about, against the domain's base error rate. A useful spine cuts error here.

| readout | domain | base error | selective risk @90% | reduction |
|---|---|---:|---:|---:|
| artifact view (Step-3 pick) | Celeb-DF-v2 | 0.1796 | 0.1785 | +0.6% |
| artifact view (Step-3 pick) | Celeb-DF-v3 | 0.3674 | 0.4062 | -10.5% |
| artifact view (Step-3 pick) | DFD | 0.1789 | 0.1526 | +14.7% |
| artifact view (Step-3 pick) | DFDC | 0.2430 | 0.2123 | +12.6% |
| artifact view (Step-3 pick) | DFDCP | 0.2470 | 0.2326 | +5.8% |
| artifact view (Step-3 pick) | Deepfake-Eval-2024 | 0.3880 | 0.3749 | +3.4% |
| fused (DS) | Celeb-DF-v2 | 0.1771 | 0.1426 | +19.5% |
| fused (DS) | Celeb-DF-v3 | 0.3669 | 0.3260 | +11.1% |
| fused (DS) | DFD | 0.1833 | 0.1273 | +30.6% |
| fused (DS) | DFDC | 0.2433 | 0.2206 | +9.3% |
| fused (DS) | DFDCP | 0.2479 | 0.2221 | +10.4% |
| fused (DS) | Deepfake-Eval-2024 | 0.3864 | 0.3739 | +3.2% |
| semantic (CLIP branch) | Celeb-DF-v2 | 0.1773 | 0.1588 | +10.4% |
| semantic (CLIP branch) | Celeb-DF-v3 | 0.3681 | 0.3076 | +16.4% |
| semantic (CLIP branch) | DFD | 0.1843 | 0.1211 | +34.3% |
| semantic (CLIP branch) | DFDC | 0.2431 | 0.2370 | +2.5% |
| semantic (CLIP branch) | DFDCP | 0.2488 | 0.2282 | +8.3% |
| semantic (CLIP branch) | Deepfake-Eval-2024 | 0.3857 | 0.3872 | -0.4% |

## Verdict

### Comparability with V1's 0.8118 — they are not the same measurement

V1's number was calibrated on **FF++ val + Celeb-DF-v2 val**, which its own report flags as `NOT zero-shot for: Celeb-DF-v2:val`. That is the protocol since renamed `ffpp_cdf2_TESTCONTAMINATED`, and it put Celeb-DF-v2 inside the calibration set. It also carried a **15.24% full-coverage error rate**, against this anchor's 3.9% on FF++ VAL_meta. A risk model has roughly four times more error signal to learn from there, on data drawn from the domain it is later scored on.

So V1's 0.81 is not a target this step can be held to. It is a number from an easier and leakier setting.

This step is run at FRAME level for the same reason V1 was (25,532 frames): at VIDEO level the anchor makes **0-3 errors in 280 VAL_meta videos**, and an error predictor cannot be fitted where there are no errors. That saturation is itself a finding about the firewall — the permitted calibration partition carries almost no error signal for a strong anchor.

### What the fit actually learned

Best by risk-AUROC: **fused (DS)** — in-domain **0.9175**, mean OOD **0.7481**.

Standardised coefficients: **V +0.848**, **M -0.461**.

**Vacuity carries real weight here, with the correct sign.** V enters positively — more vacuity, more risk — so this is an evidential result and not merely margin-based confidence.

Per readout, because they differ and the difference matters:

| readout | coef V | coef M | reading |
|---|---:|---:|---|
| artifact view (Step-3 pick) | +1.896 | +0.421 | V positive — evidential |
| fused (DS) | +0.848 | -0.461 | V positive — evidential |
| semantic (CLIP branch) | -0.101 | -0.862 | V inert/negative — margin-driven |

The semantic branch is the one whose vacuity is uninformative; the artifact and fused readouts both put substantial positive weight on it. So the Dirichlet uncertainty IS doing work in the readouts that were actually selected, and the evidential framing survives for those — a claim that must be made per readout rather than in general.

### Does deferral actually reduce risk?

| readout | mean risk reduction @10% budget | domains where it made risk WORSE |
|---|---:|---:|
| artifact view (Step-3 pick) | +4.4% | 1 of 6 |
| fused (DS) | +14.0% | 0 of 6 |
| semantic (CLIP branch) | +11.9% | 1 of 6 |

**Partially viable, and weaker than the headline suggests.** Risk-AUROC of 0.748 under shift is real signal, but spending a 10% deferral budget buys only +14.0% on average and makes risk WORSE on 0 domain(s). A reliability paper can be written on this, but its claim is 'margin-based selective prediction degrades gracefully under shift', not 'evidential uncertainty predicts errors' — the vacuity term does not support the second reading. Steps 5 and 6 now matter more, not less: if no expert enters and the ladder fails the audit, this is the whole paper, and it is thin.

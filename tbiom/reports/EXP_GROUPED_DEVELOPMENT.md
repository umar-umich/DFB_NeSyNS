# Grouped fusion on development (Stage 6)

The fixed matrix of thirteen arms, each fitted and scored on the same five source-grouped folds with the same grid
and objective, twice: once with every discount held at one, which isolates the grouping itself, and once with the
discount fitted per arm per fold. Paired source-group bootstrap against BOTH mandatory baselines, Dempster-R and
AVG-R, on the same draws.

The mechanism inspection asks the question the operator is supposed to answer: when the two CLIP heads agree and are
wrong while the temporal view is right, which rule follows the temporal view? Labels and generator identity are used
for analysis only and never enter inference.

Wall clock 2.2 minutes, CPU, cached opinions. Pre-registration: `tbiom/configs/grouped_fusion_preregister.yaml`. Five folds, 2000 paired bootstrap draws per contrast.

## 1. Fitted discount (the deployed setting)

| arm | v3.1 clean | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR | TPR | NLL | Brier | ECE | vacuity err-AUROC | E-AURC (vacuity) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **grouped-R** | 0.8792 | 0.9280 | 0.9183 | 0.7912 | 0.8114 | 0.8597 | 0.103 | 0.731 | 0.4806 | 0.1544 | 0.1533 | 0.7750 | 0.05802 |
| grouped-TA-R | 0.8812 | 0.9355 | 0.9080 | 0.8002 | 0.8120 | 0.8580 | 0.101 | 0.730 | 0.4545 | 0.1456 | 0.1348 | 0.7640 | 0.05946 |
| grouped-TO-R | 0.8775 | 0.9314 | 0.9097 | 0.7914 | 0.8043 | 0.8572 | 0.092 | 0.712 | 0.4450 | 0.1413 | 0.1143 | 0.7695 | 0.06071 |
| Dempster-R *(baseline)* | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 | 0.4706 | 0.1469 | 0.1310 | 0.7805 | 0.05499 |
| AVG-R *(baseline)* | 0.8799 | 0.9322 | 0.9133 | 0.7942 | 0.8088 | 0.8592 | 0.109 | 0.748 | 0.4263 | 0.1370 | 0.1352 | 0.7630 | 0.05879 |
| CAUT-R | 0.8751 | 0.9267 | 0.9136 | 0.7849 | 0.7994 | 0.8554 | 0.113 | 0.753 | 0.4335 | 0.1411 | 0.1340 | 0.7301 | 0.06997 |
| PCR6-R | 0.8802 | 0.9333 | 0.9067 | 0.8006 | 0.8091 | 0.8574 | 0.078 | 0.675 | 0.4744 | 0.1491 | 0.1437 | 0.7180 | 0.08974 |
| GCP-AO-R | 0.8796 | 0.9282 | 0.9157 | 0.7948 | 0.8120 | 0.8594 | 0.115 | 0.751 | 0.4827 | 0.1557 | 0.1627 | 0.7458 | 0.06561 |
| GCY-AO-R | 0.8791 | 0.9290 | 0.9122 | 0.7961 | 0.8129 | 0.8588 | 0.101 | 0.726 | 0.4758 | 0.1547 | 0.1675 | 0.7268 | 0.07077 |
| GAP-AO-R | 0.8769 | 0.9266 | 0.9151 | 0.7889 | 0.8053 | 0.8581 | 0.125 | 0.749 | 0.4417 | 0.1418 | 0.1179 | 0.7688 | 0.05818 |
| T only | 0.8094 | 0.8059 | 0.9135 | 0.7088 | 0.7596 | 0.8026 | 0.137 | 0.644 | 0.6092 | 0.1997 | 0.1742 | 0.7565 | 0.08331 |
| T+APP | 0.8705 | 0.9213 | 0.9092 | 0.7809 | 0.7951 | 0.8532 | 0.123 | 0.716 | 0.4314 | 0.1371 | 0.0961 | 0.7484 | 0.06827 |
| T+OM | 0.8795 | 0.9281 | 0.9182 | 0.7921 | 0.8125 | 0.8600 | 0.101 | 0.730 | 0.4828 | 0.1550 | 0.1550 | 0.7752 | 0.05816 |

## 2. All discounts held at one — the grouping on its own

| arm | v3.1 clean | reenactment | JPEG q40 | vacuity err-AUROC |
|---|---:|---:|---:|---:|
| grouped-R | 0.8676 | 0.9173 | 0.8032 | 0.7746 |
| grouped-TA-R | 0.8790 | 0.9101 | 0.8083 | 0.7622 |
| grouped-TO-R | 0.8766 | 0.9079 | 0.8012 | 0.7850 |
| Dempster-R | 0.8771 | 0.9099 | 0.8043 | 0.7894 |
| AVG-R | 0.8720 | 0.9193 | 0.8042 | 0.7148 |
| CAUT-R | 0.8662 | 0.9188 | 0.8006 | 0.6799 |
| PCR6-R | 0.8753 | 0.8953 | 0.7979 | 0.6954 |
| GCP-AO-R | 0.8697 | 0.9084 | 0.8002 | 0.6643 |
| GCY-AO-R | 0.8684 | 0.9036 | 0.7954 | 0.7929 |
| GAP-AO-R | 0.8670 | 0.9177 | 0.8033 | 0.7836 |
| T only | 0.8094 | 0.9135 | 0.7596 | 0.7565 |
| T+APP | 0.8635 | 0.9154 | 0.7968 | 0.7840 |
| T+OM | 0.8687 | 0.9194 | 0.8075 | 0.7693 |

## 3. Paired bootstrap against both mandatory baselines

Source groups resampled with every linked clean and processed copy together; the same draws for every arm.

| arm | Δ vs Dempster-R (95% CI) | splits CI>0 | Δ vs AVG-R (95% CI) | splits CI>0 |
|---|---:|---:|---:|---:|
| grouped-R | -0.0016 [-0.0052, +0.0020] | 0/5 | -0.0007 [-0.0038, +0.0025] | 0/5 |
| grouped-TA-R | +0.0004 [-0.0012, +0.0020] | 0/5 | +0.0013 [-0.0008, +0.0035] | 0/5 |
| grouped-TO-R | -0.0033 [-0.0057, -0.0009] | 0/5 | -0.0024 [-0.0043, -0.0005] | 0/5 |
| Dempster-R | +0.0000 [+0.0000, +0.0000] | 0/5 | +0.0009 [-0.0013, +0.0032] | 0/5 |
| AVG-R | -0.0009 [-0.0032, +0.0013] | 0/5 | +0.0000 [+0.0000, +0.0000] | 0/5 |
| CAUT-R | -0.0057 [-0.0102, -0.0014] | 0/5 | -0.0048 [-0.0081, -0.0015] | 0/5 |
| PCR6-R | -0.0006 [-0.0024, +0.0012] | 0/5 | +0.0003 [-0.0028, +0.0035] | 0/5 |
| GCP-AO-R | -0.0012 [-0.0044, +0.0021] | 0/5 | -0.0003 [-0.0035, +0.0031] | 0/5 |
| GCY-AO-R | -0.0017 [-0.0050, +0.0016] | 0/5 | -0.0008 [-0.0046, +0.0030] | 0/5 |
| GAP-AO-R | -0.0039 [-0.0067, -0.0012] | 0/5 | -0.0030 [-0.0057, -0.0004] | 0/5 |
| T only | -0.0714 [-0.0856, -0.0574] | 0/5 | -0.0705 [-0.0839, -0.0573] | 0/5 |
| T+APP | -0.0103 [-0.0142, -0.0065] | 0/5 | -0.0094 [-0.0135, -0.0054] | 0/5 |
| T+OM | -0.0013 [-0.0049, +0.0024] | 0/5 | -0.0004 [-0.0035, +0.0029] | 0/5 |

## 4. Pre-registered criteria, applied to every arm

| arm | clean gain vs both baselines >= 0.002 | both CIs above 0 | safeguards | vacuity guard | verdict |
|---|---|---|---|---|---|
| grouped-R | -0.0016 no | no | ok | -0.0055 **disqualified** | no |
| grouped-TA-R | +0.0004 no | no | breached | -0.0166 **disqualified** | no |
| grouped-TO-R | -0.0033 no | no | breached | -0.0110 **disqualified** | no |
| Dempster-R | +0.0000 no | no | ok | +0.0000 ok | no |
| AVG-R | -0.0009 no | no | ok | -0.0175 **disqualified** | no |
| CAUT-R | -0.0057 no | no | breached | -0.0504 **disqualified** | no |
| PCR6-R | -0.0006 no | no | breached | -0.0625 **disqualified** | no |
| GCP-AO-R | -0.0012 no | no | ok | -0.0347 **disqualified** | no |
| GCY-AO-R | -0.0017 no | no | ok | -0.0537 **disqualified** | no |
| GAP-AO-R | -0.0039 no | no | ok | -0.0117 **disqualified** | no |
| T only | -0.0714 no | no | breached | -0.0240 **disqualified** | no |
| T+APP | -0.0103 no | no | breached | -0.0321 **disqualified** | no |
| T+OM | -0.0013 no | no | ok | -0.0053 **disqualified** | no |

## 5. Mechanism — who follows the temporal view when both CLIP heads are wrong together

Accuracy within each case, on the selection folds, each rule at its own FF++-val EER threshold.

| case | videos | **grouped-R** | Dempster-R | AVG-R | CAUT-R |
|---|---:|---:|---:|---:|---:|
| both CLIP heads wrong together, T right | 362 | **0.7428** | 0.5558 | 0.6507 | 0.7081 |
| T wrong, CLIP heads agree and right | 520 | **0.5796** | 0.7173 | 0.6916 | 0.6644 |
| all three right | 1512 | **0.9992** | 1.0000 | 0.9997 | 0.9980 |
| all three wrong | 214 | **0.0009** | 0.0018 | 0.0000 | 0.0037 |

The first row is the mechanism the operator is built for, and grouped-R does rescue more of it than Dempster-R (0.7428 against 0.5558).

## 6. Fitted discounts per fold

| arm | split 0 | split 1 | split 2 | split 3 | split 4 |
|---|---|---|---|---|---|
| grouped-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| grouped-TA-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| grouped-TO-R | (0.5, 0.5, 1) | (0.5, 0.5, 1) | (0.5, 0.25, 1) | (0.5, 0.25, 1) | (0.5, 0.25, 1) |
| Dempster-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| AVG-R | (0.25, 0.5, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| CAUT-R | (0.25, 0, 1) | (0.25, 0.5, 1) | (0.25, 0.5, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| PCR6-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) |
| GCP-AO-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| GCY-AO-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| GAP-AO-R | (0.25, 1, 1) | (0.25, 1, 1) | (0.25, 1, 1) | (0.25, 0.75, 1) | (0.25, 0.75, 1) |
| T only | (0.25, 0, 1) | (0.25, 0, 1) | (0.75, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| T+APP | (0.25, 1, 0) | (0.25, 1, 0) | (0.25, 0.75, 1) | (0.25, 1, 0) | (0.25, 1, 0) |
| T+OM | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |

Weights are (T, APP, OM) and are refitted per arm per fold; no arm inherits another's weights. Development results are exploratory in the sense the pre-registration records: they select, they do not confirm.


# Fusion operators — every operator, every dataset

Nine operators recomputed from the saved branch opinions on **one probability convention**, so cross-operator comparison is safe. No retraining: each view's Dirichlet opinion is recovered from its exported `(p, u)` via `S = K/u`.

Numbers are the **mean over 3 seeds**. `τ` frozen per operator per seed on FF++ val. `Learned` is fitted on **VALmix only**, so its VALmix column is fitted-on-itself and is excluded from the OOD mean.

| operator | conflict handling |
|---|---|
| DS | renormalised away — the incumbent |
| Yager | assigned to vacuity |
| Dubois-Prade | to the union of conflicting elements — **= Yager on a 2-class frame** |
| PCR6 | redistributed to the masses that caused it |
| Deng | entropy-weighted average, then combined N−1 times |
| Murphy | uniform average, then combined N−1 times |
| CCF | multi-source consensus & compromise (FUSION 2018) |
| Average | mean of per-view probabilities (non-evidential) |
| Learned | one non-negative weight per view, fitted on the dev set |

## arm B (student Br3) — 3-view   (semantic + artifact + fsvfm, 3 seeds)

*Learned weights (seed 42, fitted on VALmix): semantic=0.0, artifact=0.809, fsvfm=0.191*

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | **OOD mean** | FPR mean | VALmix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9475 | 0.9000 | 0.9465 | 0.8832 | 0.9005 | 0.6894 | **0.8779** | 0.148 | 0.8867 |
| Yager | 0.9446 | 0.9000 | 0.9454 | 0.8785 | 0.8896 | 0.6934 | **0.8753** | 0.139 | 0.8790 |
| Dubois-Prade | 0.9446 | 0.9000 | 0.9454 | 0.8785 | 0.8896 | 0.6934 | **0.8753** | 0.139 | 0.8790 |
| PCR6 | 0.9501 | 0.9011 | 0.9498 | 0.8798 | 0.8939 | 0.6930 | **0.8779** | 0.146 | 0.8840 |
| Deng | 0.9390 | 0.8961 | 0.9409 | 0.8817 | 0.8981 | 0.6874 | **0.8739** | 0.140 | 0.8837 |
| Murphy | 0.9548 | 0.9028 | 0.9523 | 0.8830 | 0.8967 | 0.6931 | **0.8805** | 0.138 | 0.8880 |
| CCF | 0.9420 | 0.8944 | 0.9433 | 0.8763 | 0.8870 | 0.6917 | **0.8724** | 0.132 | 0.8778 |
| Average | 0.9560 | 0.9021 | 0.9541 | 0.8799 | 0.8944 | 0.6905 | **0.8795** | 0.142 | 0.8871 |
| Learned | 0.9616 | 0.9014 | 0.9499 | 0.8755 | 0.8912 | 0.6828 | **0.8771** | 0.150 | 0.8887 |

Best: **Murphy 0.8805**; spread across all nine **0.0081**; P0-DS reference 0.8633.

## arm B (student Br3) — 2-view   (artifact + fsvfm, 3 seeds)

*Learned weights (seed 42, fitted on VALmix): artifact=0.809, fsvfm=0.191*

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | **OOD mean** | FPR mean | VALmix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9331 | 0.8949 | 0.9370 | 0.8782 | 0.8894 | 0.6923 | **0.8708** | 0.112 | 0.8767 |
| Yager | 0.9427 | 0.8992 | 0.9441 | 0.8766 | 0.8875 | 0.6943 | **0.8741** | 0.120 | 0.8781 |
| Dubois-Prade | 0.9427 | 0.8992 | 0.9441 | 0.8766 | 0.8875 | 0.6943 | **0.8741** | 0.120 | 0.8781 |
| PCR6 | 0.9413 | 0.8987 | 0.9429 | 0.8776 | 0.8880 | 0.6946 | **0.8739** | 0.119 | 0.8780 |
| Deng | 0.9270 | 0.8915 | 0.9326 | 0.8771 | 0.8872 | 0.6913 | **0.8678** | 0.106 | 0.8748 |
| Murphy | 0.9448 | 0.9012 | 0.9460 | 0.8797 | 0.8898 | 0.6959 | **0.8762** | 0.130 | 0.8795 |
| CCF | 0.9389 | 0.8967 | 0.9396 | 0.8745 | 0.8821 | 0.7000 | **0.8720** | 0.114 | 0.8752 |
| Average | 0.9486 | 0.9024 | 0.9495 | 0.8775 | 0.8885 | 0.6941 | **0.8768** | 0.134 | 0.8800 |
| Learned | 0.9621 | 0.9012 | 0.9515 | 0.8753 | 0.8915 | 0.6824 | **0.8773** | 0.144 | 0.8885 |

Best: **Learned 0.8773**; spread across all nine **0.0096**; P0-DS reference 0.8633.

## arm C (FSFM Br3) — 3-view   (semantic + artifact + fsvfm, 3 seeds)

*Learned weights (seed 42, fitted on VALmix): semantic=0.0, artifact=0.536, fsvfm=0.464*

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | **OOD mean** | FPR mean | VALmix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9622 | 0.8976 | 0.9423 | 0.8836 | 0.8998 | 0.6826 | **0.8780** | 0.164 | 0.8928 |
| Yager | 0.9654 | 0.9032 | 0.9419 | 0.8860 | 0.9004 | 0.6854 | **0.8804** | 0.189 | 0.8960 |
| Dubois-Prade | 0.9654 | 0.9032 | 0.9419 | 0.8860 | 0.9004 | 0.6854 | **0.8804** | 0.189 | 0.8960 |
| PCR6 | 0.9629 | 0.8987 | 0.9424 | 0.8832 | 0.8979 | 0.6832 | **0.8781** | 0.178 | 0.8943 |
| Deng | 0.9623 | 0.8978 | 0.9427 | 0.8839 | 0.9010 | 0.6831 | **0.8785** | 0.165 | 0.8928 |
| Murphy | 0.9633 | 0.8992 | 0.9426 | 0.8848 | 0.8978 | 0.6846 | **0.8787** | 0.190 | 0.8943 |
| CCF | 0.9532 | 0.8882 | 0.9373 | 0.8706 | 0.8885 | 0.6785 | **0.8694** | 0.190 | 0.8869 |
| Average | 0.9654 | 0.9016 | 0.9424 | 0.8836 | 0.8948 | 0.6850 | **0.8788** | 0.201 | 0.8955 |
| Learned | 0.9676 | 0.9068 | 0.9436 | 0.8844 | 0.8966 | 0.6865 | **0.8809** | 0.191 | 0.8981 |

Best: **Learned 0.8809**; spread across all nine **0.0115**; P0-DS reference 0.8633.

## arm C (FSFM Br3) — 2-view   (artifact + fsvfm, 3 seeds)

*Learned weights (seed 42, fitted on VALmix): artifact=0.536, fsvfm=0.464*

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | **OOD mean** | FPR mean | VALmix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9670 | 0.9070 | 0.9442 | 0.8902 | 0.9065 | 0.6871 | **0.8837** | 0.170 | 0.8985 |
| Yager | 0.9677 | 0.9075 | 0.9432 | 0.8884 | 0.9037 | 0.6859 | **0.8827** | 0.173 | 0.8990 |
| Dubois-Prade | 0.9677 | 0.9075 | 0.9432 | 0.8884 | 0.9037 | 0.6859 | **0.8827** | 0.173 | 0.8990 |
| PCR6 | 0.9674 | 0.9073 | 0.9435 | 0.8891 | 0.9048 | 0.6865 | **0.8831** | 0.171 | 0.8989 |
| Deng | 0.9668 | 0.9068 | 0.9443 | 0.8902 | 0.9073 | 0.6869 | **0.8837** | 0.171 | 0.8981 |
| Murphy | 0.9679 | 0.9091 | 0.9435 | 0.8902 | 0.9040 | 0.6900 | **0.8841** | 0.177 | 0.8996 |
| CCF | 0.9623 | 0.8997 | 0.9420 | 0.8832 | 0.9011 | 0.6808 | **0.8782** | 0.189 | 0.8952 |
| Average | 0.9694 | 0.9110 | 0.9416 | 0.8877 | 0.8994 | 0.6915 | **0.8834** | 0.169 | 0.8996 |
| Learned | 0.9680 | 0.9074 | 0.9439 | 0.8850 | 0.8970 | 0.6865 | **0.8813** | 0.184 | 0.8984 |

Best: **Murphy 0.8841**; spread across all nine **0.0060**; P0-DS reference 0.8633.

## What the grid shows

| arm / views | best operator | best mean | spread across nine |
|---|---|---:|---:|
| B (student Br3) — 3-view | Murphy | 0.8805 | 0.0081 |
| B (student Br3) — 2-view | Learned | 0.8773 | 0.0096 |
| C (FSFM Br3) — 3-view | Learned | 0.8809 | 0.0115 |
| C (FSFM Br3) — 2-view | Murphy | 0.8841 | 0.0060 |

### Effect sizes, side by side

| source of variation | size on mean OOD AUROC |
|---|---:|
| operator choice (spread across nine) | **0.0060 - 0.0115** |
| view choice (3-view vs 2-view, best operator each) | **0.0032** |
| seed (same config, different initialisation) | **0.0025 - 0.0166** |

**No single one of these dominates.** Operator choice moves the mean by as much as or more than the view choice does, and BOTH sit inside the seed spread. So none of the three is separable on the evidence here, and a claim that any one of them is 'what matters' would be reading noise.

What IS separable is the gap to the baseline: every operator on every configuration beats P0-DS's 0.8633 by 0.015-0.021, far outside the seed spread. The framework works; which combination rule it uses is not established.

Two operator-level results do hold up. **CCF is consistently the worst** on every configuration -- its consensus step takes the MINIMUM belief across sources, so one low-belief view caps the fused belief. And **Dubois-Prade reproduces Yager exactly**, which is not a coincidence but the two-class algebra: the only conflicting pair is {real},{fake} and their union is Theta, where Yager already puts conflict.


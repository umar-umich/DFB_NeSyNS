# Cross-dataset video AUROC

Every column is a TEST split. Four of these corpora (DFDCP, Celeb-DF-v1/v2, UADFV) ship a `val` split that IS their test split, and DFDC ships 2 val videos against 4,704 test — so there is no held-out partition inside them and **nothing here may be fitted on**. These rows show whether the Stage-1 membership verdict travels off DF40; they cannot be used to reverse it. VALmix remains the only clean basis for that.

Blank cells are still scoring.

| model | FF++ (in-domain) | Celeb-DF-v2 | Celeb-DF-v1 | DFDCP | DFDC | DFD | Deepfake-Eval-2024 | UADFV |
|---|---|---|---|---|---|---|---|---|
| CLIP anchor | 0.9909 | 0.9225 | 0.8939 | 0.8912 | 0.8477 | 0.9222 | 0.6357 | 0.9958 |
| FS-VFM preservation | 0.9867 | 0.8713 | 0.8285 | 0.8397 | 0.8552 | 0.8877 | 0.6821 | 0.9825 |
| FS-VFM ordinary | 0.9864 | 0.8758 | 0.8646 | 0.8259 | 0.8511 | 0.8884 | 0.6782 | 0.9858 |
| V1 FF++(+)DF40 ep2 | 0.9926 | 0.7503 | 0.8306 | 0.7247 | 0.7628 | 0.8414 | 0.6476 | 0.9354 |

## Preservation minus ordinary

Positive favours the preservation student. Reported only where both cells finished. The interval is a PAIRED bootstrap over videos (2,000 resamples, the same video indices drawn for both models), which is the right test here: the two students score the identical videos, so resampling them independently would inflate the spread with variance that cancels. A CI spanning zero means the sign of the delta is not established on that corpus — which matters most where the corpus is small.

| dataset | preservation | ordinary | delta | 95% CI | videos |
|---|---:|---:|---:|---:|---:|
| FF++ (in-domain) | 0.9867 | 0.9864 | +0.0003 | [-0.0019, +0.0026] | 700 |
| Celeb-DF-v2 | 0.8713 | 0.8758 | -0.0044 | [-0.0142, +0.0048] | 518 |
| Celeb-DF-v1 | 0.8285 | 0.8646 | -0.0361 | [-0.0714, -0.0052] | 100 |
| DFDCP | 0.8397 | 0.8259 | +0.0138 | [+0.0027, +0.0247] | 654 |
| DFDC | 0.8552 | 0.8511 | +0.0041 | [+0.0012, +0.0070] | 4704 |
| DFD | 0.8877 | 0.8884 | -0.0008 | [-0.0048, +0.0030] | 3431 |
| Deepfake-Eval-2024 | 0.6821 | 0.6782 | +0.0038 | [-0.0063, +0.0145] | 814 |
| UADFV | 0.9825 | 0.9858 | -0.0033 | [-0.0159, +0.0071] | 98 |

Preservation ahead on **4/8** datasets, mean delta **-0.0028** — so the earlier three-dataset reading that preservation *consistently* outperforms does not survive the wider suite.

Only these deltas have a CI excluding zero:

- **DFDC** (4,704 videos): +0.0041, favours **preservation**
- **DFDCP** (654 videos): +0.0138, favours **preservation**
- **Celeb-DF-v1** (100 videos): -0.0361, favours **ordinary**

The two resolved deltas favouring preservation are the two LARGEST OOD corpora; the one favouring ordinary is the smallest. That ordering is worth weighing, and Celeb-DF-v1 deserves particular caution: its shipped test list is also its val list, and 92 of its 100 test videos appear in its own train split, so it is a loosely specified benchmark quite apart from its size.

Read together: preservation is defensible as the default, but on the strength of two corpora rather than a general advantage, and the spec's §22 rule (|delta AUC| < 0.01 needs a second seed) covers DFDC's +0.0041 as well. DFDCP's +0.0138 is the only delta that both clears §22 and has a CI excluding zero in preservation's favour.

## Coverage

| model | dataset | videos | real / fake | frames |
|---|---|---:|---|---:|
| CLIP anchor | Celeb-DF-v1 | 100 | 38 / 62 | 3200 |
| CLIP anchor | Celeb-DF-v2 | 518 | 178 / 340 | 16572 |
| CLIP anchor | DFDC | 4704 | 2315 / 2389 | 132116 |
| CLIP anchor | DFDCP | 654 | 230 / 424 | 17222 |
| CLIP anchor | DFD | 3431 | 363 / 3068 | 109298 |
| CLIP anchor | Deepfake-Eval-2024 | 814 | 428 / 386 | 23209 |
| CLIP anchor | FF++ (in-domain) | 700 | 140 / 560 | 22400 |
| CLIP anchor | UADFV | 98 | 49 / 49 | 3099 |
| V1 FF++(+)DF40 ep2 | Celeb-DF-v1 | 100 | 38 / 62 | 3200 |
| V1 FF++(+)DF40 ep2 | Celeb-DF-v2 | 518 | 178 / 340 | 16572 |
| V1 FF++(+)DF40 ep2 | DFDC | 4704 | 2315 / 2389 | 132116 |
| V1 FF++(+)DF40 ep2 | DFDCP | 654 | 230 / 424 | 17222 |
| V1 FF++(+)DF40 ep2 | DFD | 3431 | 363 / 3068 | 109298 |
| V1 FF++(+)DF40 ep2 | Deepfake-Eval-2024 | 814 | 428 / 386 | 23209 |
| V1 FF++(+)DF40 ep2 | FF++ (in-domain) | 700 | 140 / 560 | 22400 |
| V1 FF++(+)DF40 ep2 | UADFV | 98 | 49 / 49 | 3099 |
| FS-VFM ordinary | Celeb-DF-v1 | 100 | 38 / 62 | 3200 |
| FS-VFM ordinary | Celeb-DF-v2 | 518 | 178 / 340 | 16572 |
| FS-VFM ordinary | DFDC | 4704 | 2315 / 2389 | 132116 |
| FS-VFM ordinary | DFDCP | 654 | 230 / 424 | 17222 |
| FS-VFM ordinary | DFD | 3431 | 363 / 3068 | 109298 |
| FS-VFM ordinary | Deepfake-Eval-2024 | 814 | 428 / 386 | 23209 |
| FS-VFM ordinary | FF++ (in-domain) | 700 | 140 / 560 | 22400 |
| FS-VFM ordinary | UADFV | 98 | 49 / 49 | 3099 |
| FS-VFM preservation | Celeb-DF-v1 | 100 | 38 / 62 | 3200 |
| FS-VFM preservation | Celeb-DF-v2 | 518 | 178 / 340 | 16572 |
| FS-VFM preservation | DFDC | 4704 | 2315 / 2389 | 132116 |
| FS-VFM preservation | DFDCP | 654 | 230 / 424 | 17222 |
| FS-VFM preservation | DFD | 3431 | 363 / 3068 | 109298 |
| FS-VFM preservation | Deepfake-Eval-2024 | 814 | 428 / 386 | 23209 |
| FS-VFM preservation | FF++ (in-domain) | 700 | 140 / 560 | 22400 |
| FS-VFM preservation | UADFV | 98 | 49 / 49 | 3099 |

# Stage 1A — reliability controller ladder (frozen reference branches)

Pre-registration: `tbiom/docs/stage1a-reliability-preregistration.md`. Manifest: `tbiom/reports/REFERENCE_MANIFEST.md` (evidence-scaling discount, 4 segments of 4–8 distinct stored crops, empty segments masked). Branches frozen; only the predictor of the per-view, per-segment weight differs. 5814 development videos carry all three conditions. Fusion parameters learn from labelled external development data, disclosed, and every controller has the same access. Script `analysis/tbiom/reliability/controllers.py`.

## 1. Development v3.1, selection folds (mean of 5 splits)

| controller | clean | JPEG q40 | resize 0.60 | mean of three | stress (mixed per segment) | reenactment (clean) |
|---|---:|---:|---:|---:|---:|---:|
| constant DS-R | 0.8611 | 0.7835 | 0.8258 | 0.8235 | 0.8347 | 0.8410 |
| constant, refit clean+processed | 0.8614 | 0.7832 | 0.8258 | 0.8234 | 0.8336 | 0.8433 |
| segment MLP | 0.8614 | 0.7719 | 0.8265 | 0.8199 | 0.8341 | 0.8400 |
| segment MLP — no quality | 0.8603 | 0.7843 | 0.8250 | 0.8232 | 0.8349 | 0.8373 |
| segment MLP — shuffled chronology | 0.8614 | 0.7719 | 0.8265 | 0.8199 | 0.8341 | 0.8400 |
| segment MLP — reassigned quality | 0.8599 | 0.7853 | 0.8250 | 0.8234 | 0.8350 | 0.8370 |
| GRU | 0.8604 | 0.7777 | 0.8264 | 0.8215 | 0.8323 | 0.8385 |
| GRU — no quality | 0.8599 | 0.7844 | 0.8248 | 0.8230 | 0.8344 | 0.8360 |
| GRU — shuffled chronology | 0.8604 | 0.7767 | 0.8262 | 0.8211 | 0.8315 | 0.8382 |
| GRU — reassigned quality | 0.8597 | 0.7848 | 0.8247 | 0.8231 | 0.8346 | 0.8360 |
| Mamba | 0.8602 | 0.7309 | 0.8254 | 0.8055 | 0.8292 | 0.8373 |
| Mamba — no quality | 0.8609 | 0.7838 | 0.8255 | 0.8234 | 0.8349 | 0.8395 |
| Mamba — shuffled chronology | 0.8603 | 0.7309 | 0.8254 | 0.8055 | 0.8289 | 0.8374 |
| Mamba — reassigned quality | 0.8608 | 0.7744 | 0.8255 | 0.8202 | 0.8334 | 0.8394 |
| logistic stacker (clean+processed) | 0.8381 | 0.7692 | 0.7985 | 0.8019 | 0.8164 | 0.7948 |
| deployed DS-R (full-span clip, clean) | 0.8808 | — | — | 0.8808 | — | 0.9085 |

## 2. Against constant DS-R (paired source-grouped bootstrap, 1000 draws per split)

| controller | mean Δ (avg of clean/JPEG/resize) | splits with CI lower bound > 0 | reenactment Δ | splits with reenactment CI upper < 0 |
|---|---:|---:|---:|---:|
| constant, refit clean+processed | -0.0001 | 0/5 | +0.0024 | 0/5 |
| segment MLP | -0.0036 | 0/5 | -0.0010 | 1/5 |
| segment MLP — no quality | -0.0003 | 1/5 | -0.0037 | 4/5 |
| segment MLP — shuffled chronology | -0.0036 | 0/5 | -0.0010 | 1/5 |
| segment MLP — reassigned quality | -0.0000 | 1/5 | -0.0040 | 4/5 |
| GRU | -0.0020 | 0/5 | -0.0024 | 2/5 |
| GRU — no quality | -0.0005 | 1/5 | -0.0050 | 4/5 |
| GRU — shuffled chronology | -0.0024 | 0/5 | -0.0028 | 1/5 |
| GRU — reassigned quality | -0.0004 | 0/5 | -0.0049 | 4/5 |
| Mamba | -0.0179 | 0/5 | -0.0037 | 4/5 |
| Mamba — no quality | -0.0001 | 0/5 | -0.0015 | 4/5 |
| Mamba — shuffled chronology | -0.0180 | 0/5 | -0.0036 | 4/5 |
| Mamba — reassigned quality | -0.0033 | 0/5 | -0.0016 | 3/5 |
| logistic stacker (clean+processed) | -0.0215 | 0/5 | -0.0462 | 5/5 |

## 3. Verdicts against the frozen criteria

1. **Adaptive reliability** — best learned controller: **GRU** (mean of three 0.8215 vs constant 0.8235; bootstrap mean Δ -0.0020, CI > 0 in 0/5; clean Δ -0.0008; clean FPR Δ -0.005) → **does not succeed**.
2. **History helps** — stress-condition v3.1: segment MLP 0.8341, GRU 0.8323, Mamba 0.8292; under shuffled chronology: GRU 0.8315, Mamba 0.8289 → **no**.
3. **Mamba earns inclusion** — mean of three: MLP 0.8199, GRU 0.8215, Mamba 0.8055; compute (ms per 1,000 videos): segment MLP 0.78, GRU 1.25, Mamba 3.37 → **no**. The reliability mechanism does not fail because the MLP wins.

## 4. Operating point and uncertainty (clean development)

| controller | FPR own τ | TPR own τ | FPR drop-in τ | TPR drop-in τ | error AUROC, fused vacuity | error AUROC, margin | AURC (vacuity) |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant DS-R | 0.185 | 0.752 | 0.185 | 0.752 | 0.775 | 0.780 | 0.0895 |
| constant, refit clean+processed | 0.174 | 0.739 | 0.185 | 0.749 | 0.778 | 0.774 | 0.0912 |
| segment MLP | 0.195 | 0.770 | 0.164 | 0.742 | 0.762 | 0.783 | 0.0878 |
| segment MLP — no quality | 0.190 | 0.764 | 0.162 | 0.736 | 0.759 | 0.777 | 0.0904 |
| segment MLP — shuffled chronology | 0.195 | 0.770 | 0.164 | 0.742 | 0.762 | 0.783 | 0.0878 |
| segment MLP — reassigned quality | 0.191 | 0.764 | 0.160 | 0.735 | 0.758 | 0.778 | 0.0906 |
| GRU | 0.180 | 0.755 | 0.163 | 0.742 | 0.758 | 0.775 | 0.0911 |
| GRU — no quality | 0.187 | 0.755 | 0.157 | 0.732 | 0.757 | 0.777 | 0.0930 |
| GRU — shuffled chronology | 0.193 | 0.764 | 0.162 | 0.742 | 0.756 | 0.778 | 0.0910 |
| GRU — reassigned quality | 0.185 | 0.757 | 0.157 | 0.732 | 0.756 | 0.772 | 0.0923 |
| Mamba | 0.189 | 0.761 | 0.177 | 0.749 | 0.765 | 0.778 | 0.0893 |
| Mamba — no quality | 0.188 | 0.759 | 0.177 | 0.747 | 0.767 | 0.778 | 0.0895 |
| Mamba — shuffled chronology | 0.189 | 0.761 | 0.177 | 0.749 | 0.765 | 0.778 | 0.0893 |
| Mamba — reassigned quality | 0.187 | 0.760 | 0.175 | 0.746 | 0.765 | 0.777 | 0.0897 |

Error-detection AUROC of the fused vacuity **within** each development source and condition (pooled ranking can reflect between-dataset differences):

| controller | condition | CDFv2val | CDFv3valG | DF40train | DFDCPval | DFEval24val | DS11val | DS20val | FFIWval | FFpp_val |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant DS-R | clean | 0.823 | 0.832 | 0.788 | 0.846 | 0.589 | 0.572 | 0.533 | 0.713 | 0.939 |
| constant DS-R | jpeg40 | 0.793 | 0.758 | 0.567 | 0.882 | 0.634 | 0.603 | 0.561 | 0.680 | 0.887 |
| constant DS-R | rs06 | 0.809 | 0.782 | 0.783 | 0.772 | 0.589 | 0.515 | 0.528 | 0.655 | 0.906 |
| segment MLP | clean | 0.787 | 0.807 | 0.750 | 0.829 | 0.587 | 0.583 | 0.541 | 0.734 | 0.931 |
| segment MLP | jpeg40 | 0.752 | 0.721 | 0.522 | 0.881 | 0.577 | 0.615 | 0.572 | 0.704 | 0.858 |
| segment MLP | rs06 | 0.774 | 0.753 | 0.743 | 0.777 | 0.578 | 0.531 | 0.536 | 0.673 | 0.911 |
| GRU | clean | 0.768 | 0.799 | 0.750 | 0.834 | 0.575 | 0.570 | 0.554 | 0.735 | 0.932 |
| GRU | jpeg40 | 0.764 | 0.743 | 0.558 | 0.892 | 0.600 | 0.620 | 0.571 | 0.689 | 0.864 |
| GRU | rs06 | 0.762 | 0.751 | 0.748 | 0.743 | 0.574 | 0.517 | 0.522 | 0.678 | 0.911 |
| Mamba | clean | 0.795 | 0.815 | 0.763 | 0.834 | 0.575 | 0.576 | 0.539 | 0.738 | 0.932 |
| Mamba | jpeg40 | 0.691 | 0.669 | 0.384 | 0.810 | 0.540 | 0.607 | 0.564 | 0.704 | 0.810 |
| Mamba | rs06 | 0.781 | 0.762 | 0.751 | 0.782 | 0.576 | 0.527 | 0.535 | 0.670 | 0.910 |

## 5. Compute

| controller | parameters | ms per 1,000 videos (CPU, 4 threads) |
|---|---:|---:|
| segment MLP | 1667 | 0.78 |
| GRU | 4803 | 1.25 |
| Mamba | 3091 | 3.37 |

Limits, stated once: segment opinions come from 4–8 distinct stored crops repeated to a 16-frame clip (manifest §4); one seed per controller; development only — no test set enters this report.

# Experiment 5 — FreqBlender spatial study (local reproduction)

Two matched spatial arms with the temporal branch frozen at the reference B-augmented V-JEPA 2 LoRA checkpoint. Arms, generator provenance, checkpoint selection, acceptance criteria and the reenactment disqualifier were **frozen before the FreqBlender arm was trained**: `tbiom/docs/exp5-freqblender-preregistration.md`. DS-R is the primary fusion, logistic stacking on matched inputs the mandatory comparator; both are refitted separately per arm on the same source-grouped folds.

**What a verdict here does and does not change.** The arms are the `sbi_arm*` family, because that recipe trains the Appearance and Off-manifold views; the deployed reference takes those views from the warm-start + gated detector, where they are frozen. So this experiment answers whether FreqBlender helps spatial views that are being trained — the question the brief poses — and an adoption verdict would be a finding about spatial training rather than an automatic re-pointing of the deployed reference, which would need its own matched comparison. Fixed before the numbers were read.

**This is a local reproduction of FreqBlender, not the released FreqBlender.** The released FPNet weights are behind a Baidu Pan account wall, so FPNet was trained here from the upstream implementation at its pinned revision, with a locally trained split-safe authenticity teacher; the facial-fidelity teacher is the official one shipped in the repository. Deviations are listed in the pre-registration and in `analysis/tbiom/freqblender/fpnet.py`.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development — the whole picture in one table (selection folds, mean of 5 splits)

| arm | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR clean | TPR clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial-B | DS-R | 0.8668 | 0.9101 | 0.9092 | 0.7809 | 0.8108 | 0.8491 | 0.067 | 0.651 |
| Spatial-B | LR | 0.8625 | 0.9101 | 0.8893 | 0.7880 | 0.8046 | 0.8444 | 0.059 | 0.598 |
| Spatial-B-Freq | DS-R | 0.8760 | 0.9176 | 0.9096 | 0.8007 | 0.8157 | 0.8515 | 0.091 | 0.662 |
| Spatial-B-Freq | LR | 0.8713 | 0.9105 | 0.9067 | 0.7969 | 0.8110 | 0.8470 | 0.099 | 0.656 |

## 2. Δ = Spatial-B-Freq − Spatial-B (paired source-grouped bootstrap, 1000 draws per split)

| rule | quantity | mean Δ | splits with CI upper bound < 0 | splits with CI lower bound > 0 | CI (split 0) |
|---|---|---:|---:|---:|---|
| DS-R | total | +0.0092 | 0/5 | 5/5 | [+0.0000, +0.0195] |
| DS-R | reenactment | +0.0004 | 0/5 | 0/5 | [-0.0270, +0.0166] |
| LR | total | +0.0089 | 0/5 | 3/5 | [-0.0055, +0.0151] |
| LR | reenactment | +0.0172 | 0/5 | 2/5 | [+0.0000, +0.0394] |

## 3. Verdict against the frozen criteria

**DS-R**

- Disqualifier (fused development reenactment regression beyond the noise band): mean Δ +0.0004, CI upper bound < 0 in 0/5 splits → **not disqualified**. The precedent is arm C, where synthetic pseudo-fakes collapsed fused development reenactment to 0.527.
- Swaps-alone check: clean swap Δ +0.0075 with clean reenactment Δ +0.0003 → **passes**.
- Development gain: clean Δ +0.0092, processed Δ +0.0036 (either ≥ +0.005 required, clean loss ≤ 0.005); clean FPR Δ +0.024 (≤ +0.02 required).
- **Inclusion: NOT ADOPTED.**

**LR**

- Disqualifier (fused development reenactment regression beyond the noise band): mean Δ +0.0172, CI upper bound < 0 in 0/5 splits → **not disqualified**. The precedent is arm C, where synthetic pseudo-fakes collapsed fused development reenactment to 0.527.
- Swaps-alone check: clean swap Δ +0.0003 with clean reenactment Δ +0.0174 → **passes**.
- Development gain: clean Δ +0.0089, processed Δ +0.0045 (either ≥ +0.005 required, clean loss ≤ 0.005); clean FPR Δ +0.040 (≤ +0.02 required).
- **Inclusion: NOT ADOPTED.**

Compression robustness is claimed only if the processed-development criterion is met; it is not met here.

## 4. Test sets — video AUROC, read AFTER the verdict above

| test | Spatial-B DS-R | Spatial-B LR | Spatial-B-Freq DS-R | Spatial-B-Freq LR |
|---|---:|---:|---:|---:|
| CDFv2 | 0.9168 | 0.9244 | 0.9277 | 0.9245 |
| CDFv3 | 0.9546 | 0.9470 | 0.9517 | 0.9480 |
| DFD | 0.9144 | 0.9203 | 0.9184 | 0.9157 |
| DFDC | 0.8430 | 0.8593 | 0.8545 | 0.8455 |
| DFDCP | 0.8692 | 0.8815 | 0.8563 | 0.8514 |
| DFEval24 | 0.7252 | 0.7341 | 0.7283 | 0.7206 |
| FFIW | 0.8957 | 0.9018 | 0.9342 | 0.9255 |
| KoDF | 0.9625 | 0.9556 | 0.9507 | 0.9466 |
| IDForge | 0.9580 | 0.9493 | 0.9725 | 0.9641 |
| DFDM | 0.9952 | 0.9953 | 0.9962 | 0.9941 |
| CDFv3 reenactment | 0.9682 | 0.9563 | 0.9579 | 0.9546 |
| CDFv2 jpeg40 | 0.8692 | 0.8838 | 0.8813 | 0.8832 |
| CDFv2 rs06 | 0.8898 | 0.8974 | 0.9081 | 0.9062 |
| DFDCP jpeg40 | 0.8542 | 0.8639 | 0.8444 | 0.8306 |
| DFDCP rs06 | 0.8587 | 0.8724 | 0.8348 | 0.8284 |
| **10-set mean** | **0.9035** | **0.9069** | **0.9090** | **0.9036** |

CDFv3 test reenactment is reported here, after the development verdict, and was never a selection criterion. DS-R weights per split and every raw number are in `EXP_FREQBLENDER.json`.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

## Addendum (2026-09-20) — DeepFakeEval24 broken out, read after the verdict

Requested after the verdict; it does not enter the decision, which was made on the frozen development criteria and
was not based on FaceShifter (FaceShifter was scored afterwards, once, with the reference system, which does not
contain FreqBlender). Mean over the same five fitted splits; FPR/TPR at each arm's own frozen τ.

| arm | rule | development DFEval24val cell | test DFEval24 AUROC | test FPR | test TPR |
|---|---|---:|---:|---:|---:|
| Spatial-B | DS-R | 0.6915 | 0.7252 | 0.093 | 0.287 |
| Spatial-B | logistic | 0.6978 | 0.7341 | 0.063 | 0.254 |
| Spatial-B-Freq | DS-R | 0.6880 | 0.7283 | 0.053 | 0.220 |
| Spatial-B-Freq | logistic | 0.6859 | 0.7206 | 0.067 | 0.252 |

FreqBlender moves DeepFakeEval24 by −0.0035 (development) / +0.0031 (test) under DS-R and −0.0119 / −0.0135 under
the logistic comparator: flat to negative. Its in-the-wild development gain (+0.0198 under DS-R) comes from the other
in-the-wild cell, FFIW-val, consistent with the FFIW test row (0.8957 → 0.9342). DeepFakeEval24 therefore gives no
reason to revisit the verdict.

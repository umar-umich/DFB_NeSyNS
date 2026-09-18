# Experiment 5 — FreqBlender spatial study (local reproduction)

Two matched spatial arms with the temporal branch frozen at the reference B-augmented V-JEPA 2 LoRA checkpoint. Arms, generator provenance, checkpoint selection, acceptance criteria and the reenactment disqualifier were **frozen before the FreqBlender arm was trained**: `tbiom/docs/exp5-freqblender-preregistration.md`. DS-R is the primary fusion, logistic stacking on matched inputs the mandatory comparator; both are refitted separately per arm on the same source-grouped folds.

**This is a local reproduction of FreqBlender, not the released FreqBlender.** The released FPNet weights are behind a Baidu Pan account wall, so FPNet was trained here from the upstream implementation at its pinned revision, with a locally trained split-safe authenticity teacher; the facial-fidelity teacher is the official one shipped in the repository. Deviations are listed in the pre-registration and in `analysis/tbiom/freqblender/fpnet.py`.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development — the whole picture in one table (selection folds, mean of 5 splits)

| arm | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 | FPR clean | TPR clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial-B | DS-R | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 |
| Spatial-B | LR | 0.8736 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 | 0.057 | 0.648 |
| Spatial-B-Freq | DS-R | 0.8808 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 | 0.107 | 0.737 |
| Spatial-B-Freq | LR | 0.8736 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 | 0.057 | 0.648 |

## 2. Δ = Spatial-B-Freq − Spatial-B (paired source-grouped bootstrap, 1000 draws per split)

| rule | quantity | mean Δ | splits with CI upper bound < 0 | splits with CI lower bound > 0 | CI (split 0) |
|---|---|---:|---:|---:|---|
| DS-R | total | +0.0000 | 0/5 | 0/5 | [+0.0000, +0.0000] |
| DS-R | reenactment | +0.0000 | 0/5 | 0/5 | [+0.0000, +0.0000] |
| LR | total | +0.0000 | 0/5 | 0/5 | [+0.0000, +0.0000] |
| LR | reenactment | +0.0000 | 0/5 | 0/5 | [+0.0000, +0.0000] |

## 3. Verdict against the frozen criteria

**DS-R**

- Disqualifier (fused development reenactment regression beyond the noise band): mean Δ +0.0000, CI upper bound < 0 in 0/5 splits → **not disqualified**. The precedent is arm C, where synthetic pseudo-fakes collapsed fused development reenactment to 0.527.
- Swaps-alone check: clean swap Δ +0.0000 with clean reenactment Δ +0.0000 → **passes**.
- Development gain: clean Δ +0.0000, processed Δ +0.0000 (either ≥ +0.005 required, clean loss ≤ 0.005); clean FPR Δ +0.000 (≤ +0.02 required).
- **Inclusion: NOT ADOPTED.**

**LR**

- Disqualifier (fused development reenactment regression beyond the noise band): mean Δ +0.0000, CI upper bound < 0 in 0/5 splits → **not disqualified**. The precedent is arm C, where synthetic pseudo-fakes collapsed fused development reenactment to 0.527.
- Swaps-alone check: clean swap Δ +0.0000 with clean reenactment Δ +0.0000 → **passes**.
- Development gain: clean Δ +0.0000, processed Δ +0.0000 (either ≥ +0.005 required, clean loss ≤ 0.005); clean FPR Δ +0.000 (≤ +0.02 required).
- **Inclusion: NOT ADOPTED.**

Compression robustness is claimed only if the processed-development criterion is met; it is not met here.

## 4. Test sets — video AUROC, read AFTER the verdict above

| test | Spatial-B DS-R | Spatial-B LR | Spatial-B-Freq DS-R | Spatial-B-Freq LR |
|---|---:|---:|---:|---:|
| CDFv2 | 0.9598 | 0.9697 | 0.9598 | 0.9697 |
| CDFv3 | 0.9616 | 0.9675 | 0.9616 | 0.9675 |
| DFD | 0.9338 | 0.9300 | 0.9338 | 0.9300 |
| DFDC | 0.8626 | 0.8602 | 0.8626 | 0.8602 |
| DFDCP | 0.8656 | 0.8557 | 0.8656 | 0.8557 |
| DFEval24 | 0.7252 | 0.6865 | 0.7252 | 0.6865 |
| FFIW | 0.9282 | 0.9262 | 0.9282 | 0.9262 |
| KoDF | 0.9574 | 0.9630 | 0.9574 | 0.9630 |
| IDForge | 0.9656 | 0.9622 | 0.9656 | 0.9622 |
| DFDM | 0.9956 | 0.9963 | 0.9956 | 0.9963 |
| CDFv3 reenactment | 0.9642 | 0.9709 | 0.9642 | 0.9709 |
| CDFv2 jpeg40 | 0.8695 | 0.7199 | 0.8695 | 0.7199 |
| CDFv2 rs06 | 0.9418 | 0.9470 | 0.9418 | 0.9470 |
| DFDCP jpeg40 | 0.8198 | 0.7299 | 0.8198 | 0.7299 |
| DFDCP rs06 | 0.8084 | 0.7935 | 0.8084 | 0.7935 |
| **10-set mean** | **0.9155** | **0.9117** | **0.9155** | **0.9117** |

CDFv3 test reenactment is reported here, after the development verdict, and was never a selection criterion. DS-R weights per split and every raw number are in `EXP_FREQBLENDER.json`.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

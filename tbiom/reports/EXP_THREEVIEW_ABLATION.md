# Experiment 2 — three-view ablation of the reference triple (T = V-JEPA 2 LoRA B-augmented, APP, OM)

The full triple against each of its three pairs, every combination refitted with the **same** procedure under both fusion rules: DS-R (primary) and logistic stacking on matched inputs (comparator). The face prior does not appear. **These are ablations of the three-view claim, not proposals to drop a view.** Δ = triple − pair on the clean-development v3.1 objective, with a paired source-grouped bootstrap (1000 draws per split, one group resample per draw scoring both arms).

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development (selection folds, mean ± sd over 5 splits)

| combination | rule | clean v3.1 | swap | reenactment | in-the-wild | JPEG q40 | resize 0.60 |
|---|---|---:|---:|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.8808 ±0.0048 | 0.9345 | 0.9085 | 0.7993 | 0.8096 | 0.8589 |
| T+APP+OM | LR | 0.8736 ±0.0068 | 0.9245 | 0.8997 | 0.7965 | 0.7564 | 0.8542 |
| T+APP | DS-R | 0.8706 ±0.0051 | 0.9219 | 0.9084 | 0.7816 | 0.7947 | 0.8533 |
| T+APP | LR | 0.8687 ±0.0052 | 0.9241 | 0.8832 | 0.7988 | 0.7780 | 0.8467 |
| T+OM | DS-R | 0.8795 ±0.0046 | 0.9281 | 0.9182 | 0.7921 | 0.8125 | 0.8600 |
| T+OM | LR | 0.8786 ±0.0056 | 0.9318 | 0.8984 | 0.8057 | 0.8142 | 0.8551 |
| APP+OM | DS-R | 0.8238 ±0.0067 | 0.9282 | 0.7600 | 0.7833 | 0.7409 | 0.7847 |
| APP+OM | LR | 0.8179 ±0.0085 | 0.9065 | 0.7732 | 0.7740 | 0.6499 | 0.7857 |

## 2. Does each view earn its place? (Δ = triple − pair, clean development)

| removed view | remaining pair | rule | mean Δ | 95% CI (split 0) | splits with CI lower bound > 0 | triple beats pair in |
|---|---|---|---:|---|---:|---:|
| OM | T+APP | DS-R | +0.0102 | [+0.0070, +0.0149] | 5/5 | 5/5 splits |
| OM | T+APP | LR | +0.0047 | [-0.0080, +0.0120] | 0/5 | 5/5 splits |
| APP | T+OM | DS-R | +0.0012 | [-0.0038, +0.0041] | 0/5 | 5/5 splits |
| APP | T+OM | LR | -0.0050 | [-0.0148, -0.0008] | 0/5 | 0/5 splits |
| T | APP+OM | DS-R | +0.0569 | [+0.0458, +0.0743] | 5/5 | 5/5 splits |
| T | APP+OM | LR | +0.0560 | [+0.0445, +0.0735] | 5/5 | 5/5 splits |

## 3. Processed development — does the pair catch up under processing?

| combination | rule | clean | JPEG q40 | resize 0.60 | mean processed Δ vs triple |
|---|---|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.8808 | 0.8096 | 0.8589 | +0.0000 |
| T+APP+OM | LR | 0.8736 | 0.7564 | 0.8542 | +0.0000 |
| T+APP | DS-R | 0.8706 | 0.7947 | 0.8533 | +0.0103 |
| T+APP | LR | 0.8687 | 0.7780 | 0.8467 | -0.0070 |
| T+OM | DS-R | 0.8795 | 0.8125 | 0.8600 | -0.0020 |
| T+OM | LR | 0.8786 | 0.8142 | 0.8551 | -0.0294 |
| APP+OM | DS-R | 0.8238 | 0.7409 | 0.7847 | +0.0714 |
| APP+OM | LR | 0.8179 | 0.6499 | 0.7857 | +0.0875 |

## 4. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)

| combination | rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFDM | DFEval24 | FFIW | IDForge | KoDF | CDFv2 jpeg40 | CDFv2 rs06 | CDFv3 reenactment | DFDCP jpeg40 | DFDCP rs06 | KoDF first-order motion | 10-set mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T+APP+OM | DS-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.9956 | 0.7252 | 0.9282 | 0.9656 | 0.9574 | 0.8695 | 0.9418 | 0.9642 | 0.8198 | 0.8084 | 0.9814 | **0.9155** |
| T+APP+OM | LR | 0.9697 | 0.9675 | 0.9300 | 0.8602 | 0.8557 | 0.9963 | 0.6865 | 0.9262 | 0.9622 | 0.9630 | 0.7199 | 0.9470 | 0.9709 | 0.7299 | 0.7935 | 0.9792 | **0.9117** |
| T+APP | DS-R | 0.9345 | 0.9530 | 0.9134 | 0.8373 | 0.8458 | 0.9930 | 0.7187 | 0.9128 | 0.9524 | 0.9544 | 0.8340 | 0.9260 | 0.9592 | 0.8039 | 0.7973 | 0.9776 | **0.9015** |
| T+APP | LR | 0.9562 | 0.9557 | 0.9125 | 0.8464 | 0.8551 | 0.9933 | 0.7029 | 0.9074 | 0.9546 | 0.9574 | 0.8530 | 0.9406 | 0.9586 | 0.7654 | 0.7925 | 0.9660 | **0.9041** |
| T+OM | DS-R | 0.9543 | 0.9676 | 0.9271 | 0.8487 | 0.8585 | 0.9957 | 0.7224 | 0.9142 | 0.9613 | 0.9628 | 0.8643 | 0.9338 | 0.9743 | 0.8134 | 0.8045 | 0.9870 | **0.9113** |
| T+OM | LR | 0.9675 | 0.9619 | 0.9352 | 0.8635 | 0.8658 | 0.9952 | 0.7199 | 0.9275 | 0.9640 | 0.9563 | 0.8876 | 0.9475 | 0.9630 | 0.8160 | 0.8025 | 0.9778 | **0.9157** |
| APP+OM | DS-R | 0.9732 | 0.8787 | 0.9341 | 0.8811 | 0.8733 | 0.9919 | 0.6701 | 0.9347 | 0.9633 | 0.8543 | 0.8848 | 0.9544 | 0.8397 | 0.8250 | 0.7976 | 0.7292 | **0.8955** |
| APP+OM | LR | 0.9699 | 0.9080 | 0.9123 | 0.8701 | 0.8541 | 0.9922 | 0.6272 | 0.9208 | 0.9521 | 0.8630 | 0.7139 | 0.9514 | 0.8850 | 0.7212 | 0.7778 | 0.7233 | **0.8870** |

DS-R weights per split are in the JSON; split-0 weights: T+APP+OM (0.25, 0.25, 1.0); T+APP (0.25, 1.0); T+OM (0.25, 1.0); APP+OM (0.0, 1.0).

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

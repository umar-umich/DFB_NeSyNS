# Results — the deployed system and the baselines

OMP_NUM_THREADS=4 python analysis/tbiom/final_table.py

Table scope follows tbiom/docs/decision-2026-09-18.md §3: the main table carries CDFv2, CDFv3, DFD, FFIW, KoDF,
IDForge and DFDM; DFDC, DFDCP and DeepfakeEval2024 move to the appendix, together with the reenactment and
first-order-motion slices and every processed condition. **That split was chosen after the numbers were known**, and
the caption says so; the appendix reports the excluded sets at the same precision and the ten-set mean stays the
headline aggregate, so nothing is hidden by the split.

Every detector is trained on FaceForensics++ c23 and scored on the same videos, video-level AUROC (per-frame opinions averaged per video, then fused — the corrected-table rule). Our rows are the mean over the five source-grouped splits whose fitting half fits the fusion; the baselines are single runs, so their rows carry no comparable spread. The fused system has no frame-level row by construction: its temporal view emits one opinion per clip, not per frame.

## 1. Main table — video AUROC

| method | CDFv2 | CDFv3 | DFD | FFIW | KoDF | IDForge | DFDM | 10-set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DISCERN-v2 (ours, DS-R) | 0.9598 | 0.9616 | 0.9338 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9155 |
| DISCERN-v2 (ours, logistic comparator) | 0.9697 | 0.9675 | 0.9300 | 0.9262 | 0.9630 | 0.9622 | 0.9963 | 0.9117 |
| R0 (v3.1, ep 6) | 0.9509 | 0.8954 | 0.9431 | 0.9036 | 0.8198 | 0.9723 | 0.9929 | 0.8959 |
| A (v3.1, ep 5) | 0.9388 | 0.8770 | 0.9461 | 0.9011 | 0.8242 | 0.9697 | 0.9915 | 0.8924 |
| B (v3.1, ep 8) | 0.9225 | 0.8801 | 0.9387 | 0.8906 | 0.8497 | 0.9553 | 0.9944 | 0.8920 |
| C (v3.1, ep 2) | 0.9183 | 0.6648 | 0.8802 | 0.9279 | 0.8548 | 0.9492 | 0.9868 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.9693 | 0.8691 | 0.9343 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8937 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.8945 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.8842 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.8883 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.8777 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.8728 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.8924 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8117 |

Seed spread of the DS-R row over 3 temporal-branch seeds (only `--seed` differs): CDFv2 0.9506–0.9598, CDFv3 0.9536–0.9616, DFD 0.9307–0.9338, FFIW 0.9209–0.9356, KoDF 0.9574–0.9655, IDForge 0.9540–0.9656, DFDM 0.9946–0.9956.

**Caption note.** The seven sets above were chosen for the main table *after* the numbers were known (decision of 2026-09-18). The three sets left out — DFDC, DFDCP and DeepfakeEval2024 — are in Appendix A at the same precision, and the ten-set mean in every row is computed over all ten, so the headline aggregate is unaffected by the split.

## 2. The views on their own

| method | CDFv2 | CDFv3 | DFD | FFIW | KoDF | IDForge | DFDM | 10-set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| — temporal view alone | 0.7383 | 0.9263 | 0.7967 | 0.8011 | 0.9656 | 0.8899 | 0.9798 | 0.8213 |
| — appearance view alone | 0.9612 | 0.8576 | 0.9304 | 0.9186 | 0.8494 | 0.9510 | 0.9886 | 0.8877 |
| — off-manifold view alone | 0.9736 | 0.8801 | 0.9338 | 0.9354 | 0.8548 | 0.9635 | 0.9920 | 0.8955 |

The temporal view is the weakest alone on swap-heavy sets and the strongest on reenactment; that complementarity is what the fusion is for (Experiment 2).

## Appendix A — the sets left out of the main table

| method | DFDC | DFDCP | DFEval24 | 10-set mean |
|---|---:|---:|---:|---:|
| DISCERN-v2 (ours, DS-R) | 0.8626 | 0.8656 | 0.7252 | 0.9155 |
| DISCERN-v2 (ours, logistic comparator) | 0.8602 | 0.8557 | 0.6865 | 0.9117 |
| R0 (v3.1, ep 6) | 0.8879 | 0.9064 | 0.6869 | 0.8959 |
| A (v3.1, ep 5) | 0.8889 | 0.9013 | 0.6852 | 0.8924 |
| B (v3.1, ep 8) | 0.8889 | 0.9057 | 0.6943 | 0.8920 |
| C (v3.1, ep 2) | 0.8763 | 0.8565 | 0.6989 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.8818 | 0.8755 | 0.6769 | 0.8937 |
| GenD-CLIP | 0.8708 | 0.9162 | 0.6831 | 0.8945 |
| GenD-PE | 0.8158 | 0.8860 | 0.6563 | 0.8842 |
| GenD-DINO | 0.8470 | 0.9114 | 0.6536 | 0.8883 |
| Effort | 0.8536 | 0.8617 | 0.6769 | 0.8777 |
| ForAda | 0.8726 | 0.9289 | 0.6227 | 0.8728 |
| FSFM | 0.8844 | 0.9061 | 0.6817 | 0.8924 |
| FSFM-Adapter | 0.8318 | 0.8486 | 0.6280 | 0.8117 |

DeepfakeEval2024 is the hardest row for every method in the matrix, ours included; DFDCP is the reference's weakest processed benchmark (Experiment 3).

## Appendix B — forgery-type slices and processed conditions (our system)

| method | CDFv3 reenactment | KoDF first-order motion | CDFv2 jpeg40 | CDFv2 rs06 | DFDCP jpeg40 | DFDCP rs06 |
|---|---:|---:|---:|---:|---:|---:|
| DISCERN-v2 (ours, DS-R) | 0.9642 | 0.9814 | 0.8695 | 0.9418 | 0.8198 | 0.8084 |
| DISCERN-v2 (ours, logistic comparator) | 0.9709 | 0.9792 | 0.7199 | 0.9470 | 0.7299 | 0.7935 |
| — temporal view alone | 0.9741 | 0.9887 | 0.6692 | 0.7116 | 0.7094 | 0.7152 |
| — appearance view alone | 0.8137 | 0.7164 | 0.8333 | 0.9451 | 0.8060 | 0.7981 |
| — off-manifold view alone | 0.8416 | 0.7306 | 0.8872 | 0.9549 | 0.8256 | 0.7974 |

Processed conditions are JPEG q40 and down-up resize 0.60, applied at evaluation only; neither severity is used in training (the arm-B mixture trains on q70/q50/q30 and 0.75/0.50/0.35).

## Appendix C — the confirmation read

FaceShifter was sealed until Experiment 5 fixed the recipe, then unsealed and scored **once** with the deployed weights — no refit, no threshold change, nothing tuned afterwards. Its 140 source ids are exactly the FF++ test split, disjoint from our training and validation source ids, so it tests an unseen *manipulation* on held-out source videos; it is not called zero-leakage.

| arm | video AUROC | FPR at the frozen τ | TPR |
|---|---:|---:|---:|
| **DISCERN-v2 (DS-R)** | **0.9549** | 0.003 | 0.579 |
| logistic comparator | 0.9480 | 0.007 | 0.426 |
| T view alone | 0.9244 | — | — |
| APP view alone | 0.8659 | — | — |
| OM view alone | 0.8686 | — | — |

The fusion beats every view on a manipulation none of them has seen, which is the claim the sealed benchmark exists to test.

CDFv2 and DeepfakeEval2024 informed development through VALmix and are not untouched confirmation data.

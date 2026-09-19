# FaceShifter — the confirmation read

OMP_NUM_THREADS=4 python analysis/tbiom/faceshifter_score.py

The fusion is not re-tuned here: DS-R's reliability weights and the logistic comparator come from the same
source-grouped fitting folds every other experiment uses, and tau is the fitted fold's FF++ val EER. FaceShifter
contributes one number per arm per split and nothing else; no threshold, weight or checkpoint is revisited after this
file is written.

What the benchmark is, from the audit of 2026-09-17 (metadata only, recorded in tbiom/docs/SEALED_BENCHMARKS.md):
the release's 140 source ids are exactly the FF++ TEST split, disjoint from our 720 training and 140 validation
source ids. It therefore tests an unseen MANIPULATION on held-out source videos. It is not called "zero-leakage":
that property belongs to this split pairing, not to the release.

140 genuine and 140 manipulated videos.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

| arm | video AUROC | FPR at the frozen τ | TPR |
|---|---:|---:|---:|
| **DISCERN-v2 (DS-R)** | **0.9549** | 0.003 | 0.579 |
| logistic comparator | 0.9480 | 0.007 | 0.426 |
| T view alone | 0.9244 | — | — |
| APP view alone | 0.8659 | — | — |
| OM view alone | 0.8686 | — | — |

Mean over the five fitted splits; the spread across splits comes only from the fusion parameters, since the checkpoints are fixed. Read once: nothing was tuned after this file was written.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

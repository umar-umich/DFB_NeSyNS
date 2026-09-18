# Experiment 6 — joint training vs the staged reference

The deployed reference is staged: the Appearance and Off-manifold views are frozen (warm-started from our DiCoME run), the temporal branch was adapted separately, and DS-R combines them afterwards on development folds. This fine-tunes all of it together from those same checkpoints — temporal LoRA and head, spatial CLIP LoRA, β-VAE projector and view heads — with per-view EDL, the staged β-VAE term and an EDL loss on the **equal-trust** DS fusion (DS-R's weights are fitted on development, so they are kept out of the training objective). The face prior stays frozen and outside every loss. **The staged system remains the reference; this is the ablation.**

Budget fixed before launch: 5 epochs, batch 8, 4 of the clip's 16 frames to the spatial views, adapters 0.0002, heads 0.002, arm-B augmentation, 1.65M adapter + 0.14M head parameters trainable. Development-only selection chose epoch 4 (equal-trust v3.1 0.8752).

`staged 4-frame` restricts the frozen views to exactly the four frames the joint model's spatial views see, recomputed from the same cached per-frame opinions, so the comparison is not a frame-budget difference in disguise.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## 1. Development (selection folds, mean of 5 splits)

| arm | rule | v3.1 | swap | reenactment | in-the-wild |
|---|---|---:|---:|---:|---:|
| staged | DS-R | 0.8808 | 0.9345 | 0.9085 | 0.7993 |
| staged | LR | 0.8736 | 0.9245 | 0.8997 | 0.7965 |
| staged | DS equal | 0.8771 | 0.9290 | 0.9099 | 0.7925 |
| staged 4-frame | DS-R | 0.8780 | 0.9333 | 0.9084 | 0.7922 |
| staged 4-frame | LR | 0.8713 | 0.9228 | 0.8980 | 0.7933 |
| staged 4-frame | DS equal | 0.8744 | 0.9277 | 0.9092 | 0.7863 |
| joint | DS-R | 0.8768 | 0.9347 | 0.9073 | 0.7885 |
| joint | LR | 0.8772 | 0.9338 | 0.9054 | 0.7924 |
| joint | DS equal | 0.8746 | 0.9368 | 0.8950 | 0.7920 |

## 2. Per-branch contributions

Each view's own development v3.1, and what the triple loses when that view is removed (Δ = triple − pair, both refitted with the same procedure).

| arm | branch v3.1 T | APP | OM | Δ drop T (DS-R) | Δ drop APP | Δ drop OM |
|---|---:|---:|---:|---:|---:|---:|
| staged | 0.8094 | 0.8174 | 0.8241 | +0.0570 | +0.0013 | +0.0102 |
| staged 4-frame | 0.8094 | 0.8095 | 0.8181 | +0.0604 | +0.0019 | +0.0107 |
| joint | 0.8097 | 0.8227 | 0.8231 | +0.0539 | -0.0004 | +0.0003 |

## 3. Test sets — video AUROC (read after fitting; mean over the 5 fitted splits)

| test | staged DS-R | staged LR | staged 4-frame DS-R | staged 4-frame LR | joint DS-R | joint LR |
|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9598 | 0.9697 | 0.9454 | 0.9493 | 0.9435 | 0.9420 |
| CDFv3 | 0.9616 | 0.9675 | 0.9599 | 0.9634 | 0.9556 | 0.9546 |
| DFD | 0.9338 | 0.9300 | 0.9276 | 0.9255 | 0.9278 | 0.9232 |
| DFDC | 0.8626 | 0.8602 | 0.8600 | 0.8567 | 0.8493 | 0.8432 |
| DFDCP | 0.8656 | 0.8557 | 0.8579 | 0.8449 | 0.8732 | 0.8629 |
| DFEval24 | 0.7252 | 0.6865 | 0.7161 | 0.6796 | 0.7207 | 0.7084 |
| FFIW | 0.9282 | 0.9262 | 0.9187 | 0.9158 | 0.9154 | 0.9033 |
| KoDF | 0.9574 | 0.9630 | 0.9536 | 0.9567 | 0.9559 | 0.9588 |
| IDForge | 0.9656 | 0.9622 | 0.9570 | 0.9541 | 0.9778 | 0.9751 |
| DFDM | 0.9956 | 0.9963 | 0.9965 | 0.9966 | 0.9990 | 0.9983 |
| **10-set mean** | **0.9155** | **0.9117** | **0.9093** | **0.9043** | **0.9118** | **0.9070** |

| per-branch test AUROC (10-set mean) | staged T | staged APP | staged OM | staged 4-frame T | staged 4-frame APP | staged 4-frame OM | joint T | joint APP | joint OM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | 0.8213 | 0.8877 | 0.8955 | 0.8213 | 0.8728 | 0.8847 | 0.8197 | 0.8906 | 0.8895 |

Interpretability is not assumed to change: the per-branch columns above are the measurement. One seed per arm.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

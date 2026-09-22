# Grouped-fusion reference manifest (Stage 4)

Stage 4 records what is actually on disk -- commit, checkpoints and their hashes, branch order, the opinion algebra
as the caches really carry it, fold and grouping structure, and the exposure history. Stage 7 exercises every
operator on constructed opinions where the answer is known independently, so an implementation error shows up here
and not as a dataset result.

Repository commit `2b3f75226801f1995efd07b7428927e48cc8314f`; working tree DIRTY at audit time (5 paths).

## 1. Frozen checkpoints

| role | path | sha256 (16) |
|---|---|---|
| T (V-JEPA 2 LoRA, B-augmented, seed 42) | `logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt` | `6f0d9c63a6c052de` |
| T (V-JEPA 2 LoRA, B-augmented, seed 1337) | `logs/tbiom/temporal/branches/vjepa2_lora_B_seed1337/best.pt` | `cdb205184f70ceab` |
| APP + OM (warm+gate, epoch 19) — deployed reference | `/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | `fdc924a7cc9e1579` |
| second-seed spatial, armB42 (v3.1-selected epoch 8) | `discern_ext-best-epoch=08-val_auroc_video=0.9951.ckpt` | `bc2ca7e4d0b73513` |
| second-seed spatial, armB1337 (v3.1-selected epoch 3) | `discern_ext-best-epoch=03-val_auroc_video=0.9946.ckpt` | `1a64945f6677a350` |

The deployed warm+gate recipe exists at seed 42 only; no second warm+gate checkpoint was ever trained and this task forbids training one. The seed check therefore runs inside SBI arm B, the one three-branch family trained at several seeds, where the seed is the only difference between the members.

## 2. Branch order and the shared path

| view | encoder | input | shares a path with |
|---|---|---|---|
| T | V-JEPA 2 ViT-L, LoRA | 16-frame clip | — |
| APP | CLIP ViT-L, LoRA | stored crop | OM |
| OM | beta-VAE head on the CLIP ViT-L features | stored crop | APP |

Appearance and Off-manifold are two heads on one CLIP ViT-L feature path; the temporal view is a separate encoder on a separate input. That is the partition the primary operator uses, and §2 of the confirmation report tests it against the two wrong partitions.

Aggregation order, unchanged from every other report: per-frame opinions, mean per video, then fuse.

## 3. Opinion algebra, as the caches carry it

| view | rows | outside u/2 <= p <= 1-u/2 | max violation | roundoff only | u range | max |b+u-1| |
|---|---:|---:|---:|---|---|---:|
| T | 5814 | 0 | 0.00e+00 | yes | [0.0170, 0.7729] | 1.11e-16 |
| APP | 5814 | 0 | 0.00e+00 | yes | [0.1235, 0.5653] | 1.11e-16 |
| OM | 5814 | 0 | 0.00e+00 | yes | [0.0623, 0.8167] | 1.11e-16 |

Opinions are reconstructed from the cached (p, u) pairs. Nothing is projected into the admissible region: the table reports violations instead, and separates roundoff from a substantive inconsistency by magnitude. alpha_k = e_k + 1, u = 2/sum(alpha), b_k = e_k/sum(alpha), p = b_fake + u/2, and p is the Dirichlet mean; all four identities are exercised in the algebraic checks below.

Reliability enters as evidence scaling e'_k = r e_k. Belief-mass (Shafer) discounting is a different operation and is not used anywhere in this study.

## 4. Folds, grouping, and sample counts

| split | fitting rows | selection rows |
|---|---:|---:|
| 0 | 2911 | 2903 |
| 1 | 2971 | 2843 |
| 2 | 2942 | 2872 |
| 3 | 2965 | 2849 |
| 4 | 2846 | 2968 |

Five source-grouped folds, `GroupShuffleSplit` on the source id, seeds 0-4 — the existing v3.1 splits, unchanged. Development sets: CDFv2val, CDFv3valG, DF40train, DFDCPval, DFEval24val, DS11val, DS20val, FFIWval, FFpp_val.

Processed development conditions: jpeg40, rs06, each linked to its clean source and resampled with it.

Test sets (16): CDFv2, CDFv2 jpeg40, CDFv2 rs06, CDFv3, CDFv3 reenactment, DFD, DFDC, DFDCP, DFDCP jpeg40, DFDCP rs06, DFDM, DFEval24, FFIW, IDForge, KoDF, KoDF first-order motion.

DF40: 41 generator sets. Genuine pools are shared within a source — cdf: 21 generator sets draw on one real pool; ff: 20 generator sets draw on one real pool. Bootstrap resampling therefore groups by source video, never by generator set, or the reused real videos would be counted as independent.

## 5. Exposure history

| data | status |
|---|---|
| CDFv2 | informed development (selection and diagnosis) — not untouched |
| DeepfakeEval2024 | informed development — not untouched |
| DF40 | **has informed operator design**: the grouped operator was checked against it before this task |
| FaceShifter | read once after the recipe was frozen (`SEALED_BENCHMARKS.md`) |
| the other seven test sets | scored after each decision was recorded, never fitted on |

**There is no untouched confirmation set for this study.** DF40 informed the choice of the grouped operator, so a DF40 result cannot also confirm it; re-partitioning exposed examples would not change that. What the confirmation stage can establish is whether the effect survives a second detector seed and whether the wrong partitions lose — neither of which is a generalisation claim to unseen generators. Every finding here is labelled exploratory for that reason, and the exact deficit is: no generator family held out from operator design.


# Loader audit — arm A wrapper vs R0 path

R0 train dataset: `H5DeepfakeDataset`; arm A train dataset: `ArmDataset` wrapping `H5DeepfakeDataset`.

- lengths: R0 115,198 / A 115,198
- files lists identical: True
- labels identical: True

## 1 — identical IDs, augmentation off

256 sampled indices. Mismatches — key 0, label 0, image 0 (max abs diff 0.00e+00), other fields 0.

## 2 — augmentation on, random state reseeded identically

64 indices: image mismatches 0 (max abs diff 0.00e+00). A mismatch here would mean the wrapper consumes or perturbs the random stream the augmentations use.

## 3 — one frozen checkpoint (R0 epoch 5) on both loaders' inputs

| output | max abs difference |
|---|---:|
| fused evidence | 0.00e+00 |
| semantic evidence | 0.00e+00 |
| artifact evidence | 0.00e+00 |
| fsvfm evidence | 0.00e+00 |

## Verdict

**The wrapper is inert for arm A.** Same samples, same labels, same metadata, bit-identical tensors with and without augmentation, bit-identical evidence from a fixed checkpoint. The A − R0 gap is not an inference or preprocessing difference.

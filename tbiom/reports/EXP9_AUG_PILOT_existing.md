# Experiment 9 — matched augmentation pilot for the V-JEPA 2 LoRA branch

Warranted by the Experiment 8 diagnosis (cause b: V-JEPA's development v3.1 drops 0.087 under JPEG q40). The pilot is the SAME V-JEPA 2 LoRA recipe as the reference branch (same checkpoint, clips, sampling, LoRA rank 8 / alpha 16 / dropout 0.1, losses, 10 epochs, batch 32, seed, development-only epoch selection) plus arm B's real processing mix on training clips: none 40%, JPEG q70/q50/q30 and down-up resize 0.75/0.50/0.35 at 10% each, one draw per clip, label-independent, per-worker RNG. JPEG q40 / resize 0.60 are never used in training. Appearance and Off-manifold residual are the same fixed warm+gate epoch-19 branches; DS-R reliability weights are refitted on the same source-grouped folds. One seed per arm.

| run | selected epoch | branch dev v3.1 (train record) | DS-R weights per split (T, APP, OM) |
|---|---:|---:|---|
| vjepa2_lora_seed42 | 9 | 0.8214 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |
| vjepa2_lora_B_seed42 | 6 | 0.8092 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |

## Development (selection folds, v3.1, mean of 5 splits)

| condition | arm | vjepa2_lora_seed42 | vjepa2_lora_B_seed42 | spread (max − min) |
|---|---|---:|---:|---:|
| clean | branch T | 0.8232 | 0.8094 | 0.0139 |
| clean | DS-R | 0.8748 | 0.8808 | 0.0060 |
| clean | DS-R reenactment | 0.8970 | 0.9085 | 0.0115 |
| jpeg40 | branch T | 0.7365 | 0.7596 | 0.0231 |
| jpeg40 | DS-R | 0.7846 | 0.8096 | 0.0250 |
| jpeg40 | DS-R reenactment | 0.8091 | 0.8601 | 0.0510 |
| rs06 | branch T | 0.8142 | 0.8027 | 0.0115 |
| rs06 | DS-R | 0.8470 | 0.8589 | 0.0119 |
| rs06 | DS-R reenactment | 0.8396 | 0.8807 | 0.0410 |

## Test sets — video AUROC (read after)

| test | arm | vjepa2_lora_seed42 | vjepa2_lora_B_seed42 | spread |
|---|---|---:|---:|---:|
| CDFv2 | branch T | 0.7964 | 0.7383 | 0.0581 |
| CDFv2 | DS-R | 0.9556 | 0.9598 | 0.0043 |
| CDFv3 | branch T | 0.9306 | 0.9263 | 0.0044 |
| CDFv3 | DS-R | 0.9588 | 0.9616 | 0.0027 |
| DFD | branch T | 0.8199 | 0.7967 | 0.0232 |
| DFD | DS-R | 0.9285 | 0.9338 | 0.0053 |
| DFDC | branch T | 0.7191 | 0.7009 | 0.0182 |
| DFDC | DS-R | 0.8566 | 0.8626 | 0.0060 |
| DFDCP | branch T | 0.7762 | 0.7408 | 0.0354 |
| DFDCP | DS-R | 0.8671 | 0.8656 | 0.0014 |
| DFEval24 | branch T | 0.6606 | 0.6732 | 0.0126 |
| DFEval24 | DS-R | 0.7038 | 0.7252 | 0.0214 |
| FFIW | branch T | 0.8114 | 0.8011 | 0.0103 |
| FFIW | DS-R | 0.9236 | 0.9282 | 0.0046 |
| KoDF | branch T | 0.9660 | 0.9656 | 0.0004 |
| KoDF | DS-R | 0.9564 | 0.9574 | 0.0011 |
| IDForge | branch T | 0.8918 | 0.8899 | 0.0019 |
| IDForge | DS-R | 0.9599 | 0.9656 | 0.0057 |
| DFDM | branch T | 0.9792 | 0.9798 | 0.0007 |
| DFDM | DS-R | 0.9934 | 0.9956 | 0.0022 |
| CDFv3 reenactment | branch T | 0.9720 | 0.9741 | 0.0021 |
| CDFv3 reenactment | DS-R | 0.9627 | 0.9642 | 0.0015 |
| KoDF first-order motion | branch T | 0.9927 | 0.9887 | 0.0040 |
| KoDF first-order motion | DS-R | 0.9833 | 0.9814 | 0.0019 |
| CDFv2 jpeg40 | branch T | 0.7153 | 0.6692 | 0.0461 |
| CDFv2 jpeg40 | DS-R | 0.8797 | 0.8695 | 0.0102 |
| CDFv2 rs06 | branch T | 0.7727 | 0.7116 | 0.0611 |
| CDFv2 rs06 | DS-R | 0.9307 | 0.9418 | 0.0111 |
| DFDCP jpeg40 | branch T | 0.7191 | 0.7094 | 0.0096 |
| DFDCP jpeg40 | DS-R | 0.8244 | 0.8198 | 0.0046 |
| DFDCP rs06 | branch T | 0.7557 | 0.7152 | 0.0405 |
| DFDCP rs06 | DS-R | 0.8107 | 0.8084 | 0.0023 |
| **10-set mean** | branch T | **0.8351** | **0.8213** | 0.0139 |
| **10-set mean** | DS-R | **0.9104** | **0.9155** | 0.0052 |

Two temporal-branch seeds cannot give a standard deviation worth quoting; spreads are reported instead. FaceShifter sealed.

# Experiment 9 — matched augmentation pilot for the V-JEPA 2 LoRA branch

Warranted by the Experiment 8 diagnosis (cause b: V-JEPA's development v3.1 drops 0.087 under JPEG q40). The pilot is the SAME V-JEPA 2 LoRA recipe as the reference branch (same checkpoint, clips, sampling, LoRA rank 8 / alpha 16 / dropout 0.1, losses, 10 epochs, batch 32, seed, development-only epoch selection) plus arm B's real processing mix on training clips: none 40%, JPEG q70/q50/q30 and down-up resize 0.75/0.50/0.35 at 10% each, one draw per clip, label-independent, per-worker RNG. JPEG q40 / resize 0.60 are never used in training. Appearance and Off-manifold residual are the same fixed warm+gate epoch-19 branches; DS-R reliability weights are refitted on the same source-grouped folds. One seed per arm.

| run | selected epoch | branch dev v3.1 (train record) | DS-R weights per split (T, APP, OM) |
|---|---:|---:|---|
| vjepa2_lora_seed42 | 9 | 0.8214 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |
| vjepa2_lora_B_seed42 | 6 | 0.8092 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |
| vjepa2_lora_B_fresh_seed42 | 7 | 0.8112 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |

## Development (selection folds, v3.1, mean of 5 splits)

| condition | arm | vjepa2_lora_seed42 | vjepa2_lora_B_seed42 | vjepa2_lora_B_fresh_seed42 | spread (max − min) |
|---|---|---:|---:|---:|---:|
| clean | branch T | 0.8232 | 0.8094 | 0.8103 | 0.0139 |
| clean | DS-R | 0.8748 | 0.8808 | 0.8792 | 0.0060 |
| clean | DS-R reenactment | 0.8970 | 0.9085 | 0.9066 | 0.0115 |
| jpeg40 | branch T | 0.7365 | 0.7596 | 0.7556 | 0.0231 |
| jpeg40 | DS-R | 0.7846 | 0.8096 | 0.8052 | 0.0250 |
| jpeg40 | DS-R reenactment | 0.8091 | 0.8601 | 0.8536 | 0.0510 |
| rs06 | branch T | 0.8142 | 0.8027 | 0.8004 | 0.0138 |
| rs06 | DS-R | 0.8470 | 0.8589 | 0.8572 | 0.0119 |
| rs06 | DS-R reenactment | 0.8396 | 0.8807 | 0.8772 | 0.0410 |

## Test sets — video AUROC (read after)

| test | arm | vjepa2_lora_seed42 | vjepa2_lora_B_seed42 | vjepa2_lora_B_fresh_seed42 | spread |
|---|---|---:|---:|---:|---:|
| CDFv2 | branch T | 0.7964 | 0.7383 | 0.7514 | 0.0581 |
| CDFv2 | DS-R | 0.9556 | 0.9598 | 0.9583 | 0.0043 |
| CDFv3 | branch T | 0.9306 | 0.9263 | 0.9236 | 0.0070 |
| CDFv3 | DS-R | 0.9588 | 0.9616 | 0.9585 | 0.0030 |
| DFD | branch T | 0.8199 | 0.7967 | 0.8014 | 0.0232 |
| DFD | DS-R | 0.9285 | 0.9338 | 0.9341 | 0.0056 |
| DFDC | branch T | 0.7191 | 0.7009 | 0.7095 | 0.0182 |
| DFDC | DS-R | 0.8566 | 0.8626 | 0.8624 | 0.0060 |
| DFDCP | branch T | 0.7762 | 0.7408 | 0.7461 | 0.0354 |
| DFDCP | DS-R | 0.8671 | 0.8656 | 0.8643 | 0.0028 |
| DFEval24 | branch T | 0.6606 | 0.6732 | 0.6755 | 0.0150 |
| DFEval24 | DS-R | 0.7038 | 0.7252 | 0.7230 | 0.0214 |
| FFIW | branch T | 0.8114 | 0.8011 | 0.8023 | 0.0103 |
| FFIW | DS-R | 0.9236 | 0.9282 | 0.9268 | 0.0046 |
| KoDF | branch T | 0.9660 | 0.9656 | 0.9620 | 0.0040 |
| KoDF | DS-R | 0.9564 | 0.9574 | 0.9556 | 0.0018 |
| IDForge | branch T | 0.8918 | 0.8899 | 0.8971 | 0.0072 |
| IDForge | DS-R | 0.9599 | 0.9656 | 0.9662 | 0.0063 |
| DFDM | branch T | 0.9792 | 0.9798 | 0.9822 | 0.0031 |
| DFDM | DS-R | 0.9934 | 0.9956 | 0.9954 | 0.0022 |
| CDFv3 reenactment | branch T | 0.9720 | 0.9741 | 0.9684 | 0.0057 |
| CDFv3 reenactment | DS-R | 0.9627 | 0.9642 | 0.9601 | 0.0041 |
| KoDF first-order motion | branch T | 0.9927 | 0.9887 | 0.9841 | 0.0086 |
| KoDF first-order motion | DS-R | 0.9833 | 0.9814 | 0.9784 | 0.0049 |
| CDFv2 jpeg40 | branch T | 0.7153 | 0.6692 | 0.6743 | 0.0461 |
| CDFv2 jpeg40 | DS-R | 0.8797 | 0.8695 | 0.8739 | 0.0102 |
| CDFv2 rs06 | branch T | 0.7727 | 0.7116 | 0.7185 | 0.0611 |
| CDFv2 rs06 | DS-R | 0.9307 | 0.9418 | 0.9435 | 0.0128 |
| DFDCP jpeg40 | branch T | 0.7191 | 0.7094 | 0.7077 | 0.0114 |
| DFDCP jpeg40 | DS-R | 0.8244 | 0.8198 | 0.8178 | 0.0066 |
| DFDCP rs06 | branch T | 0.7557 | 0.7152 | 0.7186 | 0.0405 |
| DFDCP rs06 | DS-R | 0.8107 | 0.8084 | 0.8091 | 0.0023 |
| **10-set mean** | branch T | **0.8351** | **0.8213** | **0.8251** | 0.0139 |
| **10-set mean** | DS-R | **0.9104** | **0.9155** | **0.9145** | 0.0052 |

Two temporal-branch seeds cannot give a standard deviation worth quoting; spreads are reported instead. FaceShifter sealed.

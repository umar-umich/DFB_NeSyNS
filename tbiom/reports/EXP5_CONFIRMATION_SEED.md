# Experiment 5 — confirmation seed: temporal-branch training variability conditional on fixed branches

V-JEPA 2 LoRA branch retrained on the EXACT seed-42 recipe (same pretrained checkpoint, clips, sampling, preprocessing, LoRA rank 8 / alpha 16 / dropout 0.1, losses, 10 epochs, batch 32, development-only epoch selection; no augmentation) with another seed. Appearance and Off-manifold residual are the SAME fixed warm+gate epoch-19 branches for every seed, and the DS-R reliability weights are refitted on the same source-grouped folds. **This measures temporal-branch training variability conditional on fixed branches — it is not a full detector replication.** Script `analysis/tbiom/temporal/dsr_system.py`.

| seed run | selected epoch | branch dev v3.1 (train record) | DS-R weights per split (T, APP, OM) |
|---|---:|---:|---|
| vjepa2_lora_seed42 | 9 | 0.8214 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |
| vjepa2_lora_seed1337 | 8 | 0.8364 | (0.25, 0.5, 1.0); (0.25, 0.25, 1.0); (0.25, 0.0, 1.0); (0.25, 0.25, 1.0); (0.25, 0.0, 1.0) |

## Development (selection folds, v3.1, mean of 5 splits)

| condition | arm | vjepa2_lora_seed42 | vjepa2_lora_seed1337 | spread (max − min) |
|---|---|---:|---:|---:|
| clean | branch T | 0.8232 | 0.8379 | 0.0146 |
| clean | DS-R | 0.8748 | 0.8777 | 0.0029 |
| clean | DS-R reenactment | 0.8970 | 0.8925 | 0.0045 |
| jpeg40 | branch T | 0.7365 | 0.7066 | 0.0299 |
| jpeg40 | DS-R | 0.7846 | 0.7711 | 0.0135 |
| jpeg40 | DS-R reenactment | 0.8091 | 0.7852 | 0.0239 |
| rs06 | branch T | 0.8142 | 0.7994 | 0.0147 |
| rs06 | DS-R | 0.8470 | 0.8417 | 0.0053 |
| rs06 | DS-R reenactment | 0.8396 | 0.8337 | 0.0059 |

## Test sets — video AUROC (read after)

| test | arm | vjepa2_lora_seed42 | vjepa2_lora_seed1337 | spread |
|---|---|---:|---:|---:|
| CDFv2 | branch T | 0.7964 | 0.8324 | 0.0359 |
| CDFv2 | DS-R | 0.9556 | 0.9557 | 0.0001 |
| CDFv3 | branch T | 0.9306 | 0.9578 | 0.0271 |
| CDFv3 | DS-R | 0.9588 | 0.9726 | 0.0138 |
| DFD | branch T | 0.8199 | 0.8656 | 0.0457 |
| DFD | DS-R | 0.9285 | 0.9338 | 0.0053 |
| DFDC | branch T | 0.7191 | 0.7273 | 0.0082 |
| DFDC | DS-R | 0.8566 | 0.8525 | 0.0041 |
| DFDCP | branch T | 0.7762 | 0.7727 | 0.0035 |
| DFDCP | DS-R | 0.8671 | 0.8624 | 0.0046 |
| DFEval24 | branch T | 0.6606 | 0.6743 | 0.0138 |
| DFEval24 | DS-R | 0.7038 | 0.7091 | 0.0054 |
| FFIW | branch T | 0.8114 | 0.8427 | 0.0314 |
| FFIW | DS-R | 0.9236 | 0.9246 | 0.0010 |
| KoDF | branch T | 0.9660 | 0.9654 | 0.0006 |
| KoDF | DS-R | 0.9564 | 0.9581 | 0.0017 |
| IDForge | branch T | 0.8918 | 0.8754 | 0.0164 |
| IDForge | DS-R | 0.9599 | 0.9487 | 0.0112 |
| DFDM | branch T | 0.9792 | 0.9458 | 0.0334 |
| DFDM | DS-R | 0.9934 | 0.9851 | 0.0083 |
| CDFv3 reenactment | branch T | 0.9720 | 0.9867 | 0.0147 |
| CDFv3 reenactment | DS-R | 0.9627 | 0.9773 | 0.0146 |
| KoDF first-order motion | branch T | 0.9927 | 0.9946 | 0.0019 |
| KoDF first-order motion | DS-R | 0.9833 | 0.9885 | 0.0052 |
| CDFv2 jpeg40 | branch T | 0.7153 | 0.6656 | 0.0497 |
| CDFv2 jpeg40 | DS-R | 0.8797 | 0.8785 | 0.0013 |
| CDFv2 rs06 | branch T | 0.7727 | 0.8055 | 0.0328 |
| CDFv2 rs06 | DS-R | 0.9307 | 0.9288 | 0.0019 |
| DFDCP jpeg40 | branch T | 0.7191 | 0.6110 | 0.1081 |
| DFDCP jpeg40 | DS-R | 0.8244 | 0.8130 | 0.0114 |
| DFDCP rs06 | branch T | 0.7557 | 0.7769 | 0.0212 |
| DFDCP rs06 | DS-R | 0.8107 | 0.8173 | 0.0066 |
| **10-set mean** | branch T | **0.8351** | **0.8459** | 0.0108 |
| **10-set mean** | DS-R | **0.9104** | **0.9103** | 0.0001 |

Two temporal-branch seeds cannot give a standard deviation worth quoting; spreads are reported instead. FaceShifter sealed.

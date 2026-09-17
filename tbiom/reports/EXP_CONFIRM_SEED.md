# Experiment 4 — confirmation seed for the reference recipe

The reference temporal branch retrained with **seed 1337 and nothing else changed**: same V-JEPA 2 checkpoint initialisation, clips, sampling, preprocessing, LoRA configuration (r 8, α 16, dropout 0.1), losses, budget (10 epochs, batch 32) and the **same B augmentation**; the same fixed Appearance and Off-manifold checkpoints; development-only epoch selection; DS-R and the logistic comparator refitted on the same source-grouped folds. **This is temporal-branch training variability conditional on the fixed branches, not a full three-seed detector replication.**

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

| run | seed | augmentation | selected epoch | branch dev v3.1 (training record) | DS-R weights per split (T, APP, OM) |
|---|---:|---|---:|---:|---|
| `vjepa2_lora_B_seed42` | 42 | B | 6 | 0.8092 | (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) |
| `vjepa2_lora_B_seed1337` | 1337 | B | 6 | 0.8174 | (0.25, 0.25, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0) |

## Development (selection folds, v3.1, mean of 5 splits)

| condition | arm | vjepa2_lora_B_seed42 | vjepa2_lora_B_seed1337 | spread |
|---|---|---:|---:|---:|
| clean | branch T | 0.8094 | 0.8171 | 0.0077 |
| clean | DS-R | 0.8808 | 0.8801 | 0.0007 |
| clean | LR | 0.8736 | 0.8779 | 0.0043 |
| clean | DS-R reenactment | 0.9085 | 0.9054 | 0.0032 |
| clean | LR reenactment | 0.8997 | 0.8996 | 0.0001 |
| jpeg40 | branch T | 0.7596 | 0.7827 | 0.0232 |
| jpeg40 | DS-R | 0.8096 | 0.8226 | 0.0130 |
| jpeg40 | LR | 0.7564 | 0.7780 | 0.0216 |
| jpeg40 | DS-R reenactment | 0.8601 | 0.8734 | 0.0133 |
| jpeg40 | LR reenactment | 0.8232 | 0.8438 | 0.0206 |
| rs06 | branch T | 0.8027 | 0.7994 | 0.0032 |
| rs06 | DS-R | 0.8589 | 0.8555 | 0.0033 |
| rs06 | LR | 0.8542 | 0.8523 | 0.0019 |
| rs06 | DS-R reenactment | 0.8807 | 0.8715 | 0.0091 |
| rs06 | LR reenactment | 0.8739 | 0.8660 | 0.0079 |

## Test sets — video AUROC (read after fitting)

| test | arm | vjepa2_lora_B_seed42 | vjepa2_lora_B_seed1337 | spread |
|---|---|---:|---:|---:|
| CDFv2 | branch T | 0.7383 | 0.7649 | 0.0266 |
| CDFv2 | DS-R | 0.9598 | 0.9506 | 0.0092 |
| CDFv2 | LR | 0.9697 | 0.9622 | 0.0075 |
| CDFv3 | branch T | 0.9263 | 0.9238 | 0.0025 |
| CDFv3 | DS-R | 0.9616 | 0.9585 | 0.0031 |
| CDFv3 | LR | 0.9675 | 0.9636 | 0.0039 |
| DFD | branch T | 0.7967 | 0.8121 | 0.0154 |
| DFD | DS-R | 0.9338 | 0.9307 | 0.0030 |
| DFD | LR | 0.9300 | 0.9273 | 0.0027 |
| DFDC | branch T | 0.7009 | 0.7160 | 0.0150 |
| DFDC | DS-R | 0.8626 | 0.8595 | 0.0031 |
| DFDC | LR | 0.8602 | 0.8585 | 0.0017 |
| DFDCP | branch T | 0.7408 | 0.7822 | 0.0414 |
| DFDCP | DS-R | 0.8656 | 0.8774 | 0.0118 |
| DFDCP | LR | 0.8557 | 0.8626 | 0.0069 |
| DFEval24 | branch T | 0.6732 | 0.6792 | 0.0060 |
| DFEval24 | DS-R | 0.7252 | 0.7239 | 0.0013 |
| DFEval24 | LR | 0.6865 | 0.6943 | 0.0078 |
| FFIW | branch T | 0.8011 | 0.8114 | 0.0103 |
| FFIW | DS-R | 0.9282 | 0.9209 | 0.0073 |
| FFIW | LR | 0.9262 | 0.9213 | 0.0049 |
| KoDF | branch T | 0.9656 | 0.9691 | 0.0035 |
| KoDF | DS-R | 0.9574 | 0.9655 | 0.0080 |
| KoDF | LR | 0.9630 | 0.9672 | 0.0042 |
| IDForge | branch T | 0.8899 | 0.8716 | 0.0183 |
| IDForge | DS-R | 0.9656 | 0.9540 | 0.0116 |
| IDForge | LR | 0.9622 | 0.9560 | 0.0062 |
| DFDM | branch T | 0.9798 | 0.9776 | 0.0022 |
| DFDM | DS-R | 0.9956 | 0.9946 | 0.0010 |
| DFDM | LR | 0.9963 | 0.9954 | 0.0009 |
| CDFv3 reenactment | branch T | 0.9741 | 0.9628 | 0.0113 |
| CDFv3 reenactment | DS-R | 0.9642 | 0.9613 | 0.0029 |
| CDFv3 reenactment | LR | 0.9709 | 0.9662 | 0.0047 |
| KoDF first-order motion | branch T | 0.9887 | 0.9930 | 0.0043 |
| KoDF first-order motion | DS-R | 0.9814 | 0.9898 | 0.0084 |
| KoDF first-order motion | LR | 0.9792 | 0.9857 | 0.0064 |
| CDFv2 jpeg40 | branch T | 0.6692 | 0.7058 | 0.0366 |
| CDFv2 jpeg40 | DS-R | 0.8695 | 0.8641 | 0.0054 |
| CDFv2 jpeg40 | LR | 0.7199 | 0.7362 | 0.0164 |
| CDFv2 rs06 | branch T | 0.7116 | 0.7211 | 0.0095 |
| CDFv2 rs06 | DS-R | 0.9418 | 0.9327 | 0.0091 |
| CDFv2 rs06 | LR | 0.9470 | 0.9379 | 0.0091 |
| DFDCP jpeg40 | branch T | 0.7094 | 0.7527 | 0.0432 |
| DFDCP jpeg40 | DS-R | 0.8198 | 0.8340 | 0.0142 |
| DFDCP jpeg40 | LR | 0.7299 | 0.7522 | 0.0223 |
| DFDCP rs06 | branch T | 0.7152 | 0.7510 | 0.0358 |
| DFDCP rs06 | DS-R | 0.8084 | 0.8211 | 0.0127 |
| DFDCP rs06 | LR | 0.7935 | 0.7997 | 0.0062 |
| **10-set mean** | branch T | **0.8213** | **0.8308** | 0.0095 |
| **10-set mean** | DS-R | **0.9155** | **0.9136** | 0.0020 |
| **10-set mean** | LR | **0.9117** | **0.9108** | 0.0009 |

Two temporal-branch seeds cannot give a standard deviation worth quoting; spreads (max − min) are reported instead. Both runs were trained on 2026-09-17 with the identical command except `--seed`; their training records are the evidence that only the seed differs.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

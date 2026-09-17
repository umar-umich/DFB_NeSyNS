# Experiment 8 — DFDCP / processing weakness: diagnosis for the deployed DS-R system (`vjepa2_lora_seed42` + APP + OM)

CAUSE (b) V-JEPA collapse: V-JEPA's own v3.1 score on processed development (JPEG q40) falls >= 0.05 below its clean
     development score.  [DFDCP-JPEG test AUROC drop reported as confirmation, not used for the decision.]
  CAUSE (a) fusion deterioration: on processed development the DS-R fused score is below the best single branch's
     processed score, OR the fused drop (clean -> processed) exceeds the smallest single-branch drop by >= 0.02.
  REMEDY for (a), no encoder retraining: refit DS-R reliability weights on clean + JPEG q40 + resize 0.60 versions of the
     FITTING-fold development videos (mean v3.1 of the three), evaluate on the SELECTION folds. Adopt if processed
     development improves by >= 0.005 (mean of the two conditions) with clean development loss <= 0.005. Caveat: the
     processed development severities equal the processed test severities.
  An augmentation pilot is warranted only if (b) holds.

## 1. Development (selection folds, v3.1, mean ± sd over 5 splits)

| arm | clean | JPEG q40 | resize 0.60 |
|---|---:|---:|---:|
| branch T | 0.8232 ±0.0049 | 0.7365 ±0.0084 | 0.8142 ±0.0048 |
| branch APP | 0.8174 ±0.0062 | 0.7179 ±0.0066 | 0.7848 ±0.0115 |
| branch OM | 0.8241 ±0.0067 | 0.7429 ±0.0063 | 0.7845 ±0.0112 |
| DS equal | 0.8700 ±0.0042 | 0.7748 ±0.0065 | 0.8412 ±0.0073 |
| DS-R (clean fit) | 0.8748 ±0.0039 | 0.7846 ±0.0067 | 0.8470 ±0.0071 |
| DS-R (clean+processed fit) | 0.8749 ±0.0037 | 0.7940 ±0.0060 | 0.8482 ±0.0059 |

## 2. Verdicts

- **Cause (b) — V-JEPA collapse under compression:** V-JEPA's development v3.1 drop clean → JPEG q40 = **+0.0867** (criterion ≥ 0.05) → **HOLDS**.
- **Cause (a) — fusion deterioration, jpeg40:** fused below best single branch: False; fused drop +0.0902 vs smallest branch drop +0.0813 (excess +0.0089, criterion ≥ 0.02) → **does not hold**. Branch drops: T +0.0867, APP +0.0994, OM +0.0813
- **Cause (a) — fusion deterioration, rs06:** fused below best single branch: False; fused drop +0.0278 vs smallest branch drop +0.0091 (excess +0.0187, criterion ≥ 0.02) → **does not hold**. Branch drops: T +0.0091, APP +0.0326, OM +0.0396
- **Remedy for (a) — clean+processed reliability fit:** processed development change +0.0053 (criterion ≥ +0.005), clean development loss -0.0001 (criterion ≤ 0.005); numerical criteria met, but the remedy applies only when cause (a) holds, which it does not → **not adopted**. Weights (T, APP, OM) clean fit: (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) | clean+processed fit: (0.25, 0.0, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0); (0.25, 0.0, 1.0)
- **Augmentation pilot warranted:** YES (cause b holds).

## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)

| test | branch T | branch APP | branch OM | DS equal | DS-R (clean fit) | DS-R (clean+processed fit) |
|---|---:|---:|---:|---:|---:|---:|
| DFDCP | 0.7762 | 0.8698 | 0.8727 | 0.8587 | 0.8671 | 0.8617 |
| DFDCP jpeg40 | 0.7191 | 0.8060 | 0.8256 | 0.8161 | 0.8244 | 0.8276 |
| DFDCP rs06 | 0.7557 | 0.7981 | 0.7974 | 0.8069 | 0.8107 | 0.8072 |
| CDFv2 | 0.7964 | 0.9612 | 0.9736 | 0.9471 | 0.9556 | 0.9495 |
| CDFv2 jpeg40 | 0.7153 | 0.8333 | 0.8872 | 0.8676 | 0.8797 | 0.8869 |
| CDFv2 rs06 | 0.7727 | 0.9451 | 0.9549 | 0.9174 | 0.9307 | 0.9109 |
| CDFv3 | 0.9306 | 0.8576 | 0.8801 | 0.9562 | 0.9588 | 0.9629 |
| DFD | 0.8199 | 0.9304 | 0.9338 | 0.9176 | 0.9285 | 0.9198 |
| DFDC | 0.7191 | 0.8758 | 0.8808 | 0.8474 | 0.8566 | 0.8452 |
| DFEval24 | 0.6606 | 0.6741 | 0.6688 | 0.6995 | 0.7038 | 0.6998 |
| FFIW | 0.8114 | 0.9186 | 0.9354 | 0.9127 | 0.9236 | 0.9110 |
| KoDF | 0.9660 | 0.8494 | 0.8548 | 0.9547 | 0.9564 | 0.9614 |
| IDForge | 0.8918 | 0.9510 | 0.9635 | 0.9523 | 0.9599 | 0.9551 |
| DFDM | 0.9792 | 0.9886 | 0.9920 | 0.9920 | 0.9934 | 0.9930 |
| CDFv3 reenactment | 0.9720 | 0.8137 | 0.8416 | 0.9615 | 0.9627 | 0.9706 |
| KoDF first-order motion | 0.9927 | 0.7164 | 0.7306 | 0.9815 | 0.9833 | 0.9895 |
| **10-set mean** | **0.8351** | **0.8877** | **0.8955** | **0.9038** | **0.9104** | **0.9060** |

Processed development and processed tests share the JPEG q40 / resize 0.60 severities, so the clean+processed reliability fit is severity-matched to the processed tests (optimistic for those rows). One detector seed, one temporal-branch seed. FaceShifter sealed.

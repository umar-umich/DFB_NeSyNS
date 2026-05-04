# Ablation Results - DeFakeNet on CDFv2 (video-level)

## Component Ablation (Table 4)
| Setting                              | AUC   | ECE    | E-AURC x100 | CW@0.9 |
|--------------------------------------|-------|--------|-------------|--------|
| GenD-CLIP (visual + softmax)          | —     | —      | —           | —      |
| Visual stream + EDL only              | 93.77 | 0.0912 | 2.44        | 0.0000 |
| DeFakeNet - IBDC                      | 94.27 | 0.1070 | 2.63        | 0.0000 |
| DeFakeNet - Causal stream             | 93.85 | 0.0770 | 2.31        | 0.0000 |
| DeFakeNet - Symbolic stream           | 94.29 | 0.0676 | 2.33        | 0.0039 |
| Full DeFakeNet                        | —     | —      | —           | —      |

## Faithfulness (Table 5)
| Intervention             | k=1   | k=3   | k=5   | Flip rate (%) |
|--------------------------|-------|-------|-------|---------------|
| Random predicates        | 0.00  | 0.00  | 0.00  | 0.04          |
| Top-k firing (cited)     | 0.00  | 0.00  | 0.00  | 0.10          |

## Selective Prediction (10% abstention on CDFv2)
| Method                         | Full coverage AUC | At 90% coverage AUC |
|--------------------------------|-------------------|---------------------|
| Full DeFakeNet                 | —                 | —                   |
| GenD-CLIP softmax              | —                 | —                   |

## Run log
- `full_defakenet_18rules` ✓ metrics.json present
- `no_causal` ✓ metrics.json present
- `no_ibdc` ✓ metrics.json present
- `no_symbolic` ✓ metrics.json present
- `visual_edl_only` ✓ metrics.json present
- faithfulness  ✓ /data/umar/Repos/DFB_NeSyNS/results/faithfulness/CDFv2.json
- selective     ✓ /data/umar/Repos/DFB_NeSyNS/results/selective/CDFv2.json

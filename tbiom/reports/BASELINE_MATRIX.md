# Baseline matrix — frame- and video-level AUROC

    python analysis/tbiom/baselines/build_baseline_matrix.py --out tbiom/reports/BASELINE_MATRIX.md

Every measured row scores the SAME frames per dataset: CDFv2/CDFv3/DFD/DFDC/DFDCP/DFEval24 use our crops (the
baselines read PNGs dumped from our h5 files); FFIW/KoDF/IDForge/DFDM use the GenD-release crops our h5 files were
built from. Scope filters as protocol v3.1: CDFv3 without IP_LAP; KoDF without audio-driven and adversarial subsets.
Our models: DS replay of their own views (frame = per-frame opinion; video = mean opinion per video, then fused —
the corrected-table rule). Baselines: prob_fake per frame; video = mean over frames (GenD's own reduction).
A video is the directory containing the frame (full path). Published numbers are listed separately, never mixed in.

All detectors trained on FF++ c23. One seed per row. — = not yet available.

## Video-level AUROC

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean (complete rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (v3.1, ep 6) | 0.9509 | 0.8954 | 0.9431 | 0.8879 | 0.9064 | 0.6869 | 0.9036 | 0.8198 | 0.9723 | 0.9929 | 0.8959 |
| A (v3.1, ep 5) | 0.9388 | 0.8770 | 0.9461 | 0.8889 | 0.9013 | 0.6852 | 0.9011 | 0.8242 | 0.9697 | 0.9915 | 0.8924 |
| B (v3.1, ep 8) | 0.9225 | 0.8801 | 0.9387 | 0.8889 | 0.9057 | 0.6943 | 0.8906 | 0.8497 | 0.9553 | 0.9944 | 0.8920 |
| C (v3.1, ep 2) | 0.9183 | 0.6648 | 0.8802 | 0.8763 | 0.8565 | 0.6989 | 0.9279 | 0.8548 | 0.9492 | 0.9868 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.9693 | 0.8691 | 0.9343 | 0.8818 | 0.8755 | 0.6769 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8937 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.8708 | 0.9162 | 0.6831 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.8945 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.8158 | 0.8860 | 0.6563 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.8842 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.8470 | 0.9114 | 0.6536 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.8883 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.8536 | 0.8617 | 0.6769 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.8777 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.8726 | 0.9289 | 0.6227 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.8728 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.8844 | 0.9061 | 0.6817 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.8924 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8318 | 0.8486 | 0.6280 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8117 |

## Frame-level AUROC

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean (complete rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (v3.1, ep 6) | 0.9019 | 0.8545 | 0.9032 | 0.8693 | 0.8680 | 0.6606 | 0.8585 | 0.8109 | 0.9360 | 0.9797 | 0.8643 |
| A (v3.1, ep 5) | 0.8854 | 0.8321 | 0.9052 | 0.8703 | 0.8665 | 0.6564 | 0.8582 | 0.8184 | 0.9315 | 0.9756 | 0.8600 |
| B (v3.1, ep 8) | 0.8746 | 0.8428 | 0.8986 | 0.8695 | 0.8712 | 0.6637 | 0.8450 | 0.8483 | 0.9261 | 0.9854 | 0.8625 |
| C (v3.1, ep 2) | 0.8437 | 0.6638 | 0.8401 | 0.8578 | 0.8243 | 0.6506 | 0.8800 | 0.8373 | 0.9049 | 0.9700 | 0.8273 |
| DiCoME (our training, v3.1, ep 5) | 0.9183 | 0.8177 | 0.9025 | 0.8538 | 0.8482 | 0.6377 | 0.8841 | 0.8422 | 0.9120 | 0.9743 | 0.8591 |
| GenD-CLIP | 0.8942 | 0.7944 | 0.8931 | 0.8332 | 0.8637 | 0.6369 | 0.8706 | 0.8201 | 0.9494 | 0.9744 | 0.8530 |
| GenD-PE | 0.8986 | 0.8500 | 0.8801 | 0.7990 | 0.8418 | 0.6264 | 0.8906 | 0.8276 | 0.9543 | 0.9563 | 0.8524 |
| GenD-DINO | 0.8457 | 0.7653 | 0.8677 | 0.8093 | 0.8469 | 0.6206 | 0.8878 | 0.8869 | 0.9528 | 0.9595 | 0.8442 |
| Effort | 0.8874 | 0.7625 | 0.8915 | 0.8276 | 0.8405 | 0.6352 | 0.8866 | 0.8629 | 0.9204 | 0.9445 | 0.8459 |
| ForAda | 0.8955 | 0.7134 | 0.9011 | 0.8429 | 0.8901 | 0.5883 | 0.8532 | 0.8650 | 0.8507 | 0.9361 | 0.8336 |
| FSFM | 0.8998 | 0.8322 | 0.8810 | 0.8574 | 0.8761 | 0.6521 | 0.8742 | 0.8388 | 0.9184 | 0.9518 | 0.8582 |
| FSFM-Adapter | 0.8311 | 0.8118 | 0.7287 | 0.8159 | 0.8338 | 0.6042 | 0.8238 | 0.7371 | 0.8809 | 0.8114 | 0.7879 |

## Video counts (must agree across rows)

| dataset | R0 (v3.1, ep 6) | A (v3.1, ep 5) | B (v3.1, ep 8) | C (v3.1, ep 2) | DiCoME (our training, v3.1, ep 5) | GenD-CLIP | GenD-PE | GenD-DINO | Effort | ForAda | FSFM | FSFM-Adapter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 | 518 |
| CDFv3 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 | 5118 |
| DFD | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 | 3431 |
| DFDC | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 | 4704 |
| DFDCP | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 | 654 |
| DFEval24 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 |
| FFIW | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 | 3470 |
| KoDF | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 | 1395 |
| IDForge | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 | 5974 |
| DFDM | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 | 1896 |

**Count mismatches:** none

## Published numbers (as printed; not re-measured)

- **DiCoME** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.977, DFD 0.982, DFDC 0.882, DFo 0.993, WDF 0.911, CDFv3 0.886
- **Effort** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.956, DFD 0.965, DFDC 0.843, DFo 0.977, WDF 0.848, CDFv3 0.850
- **GenD (backbone unstated)** (video, https://arxiv.org/html/2606.01885v1 (Table 1)): CDFv2 0.960, DFD 0.970, DFDC 0.871, DFo 0.989, WDF 0.890, CDFv3 0.859
- **Effort** (video, https://arxiv.org/html/2411.15633v3 (Table 1, Protocol-1)): CDFv2 0.956, DFD 0.965, DFDC 0.843, DFDCP 0.909, DFo 0.977, WDF 0.848, FFIW 0.921
- **ForAda** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.994, DFD 0.972, DFDC 0.873, FSh 0.820, CDFv2 0.957, FFIW 0.906, KoDF 0.882, FAVC 0.931, DFDM 0.971, PGF 0.866, IDForge 0.908, DSv1.1 0.818, DSv2 0.728, CDFv3 0.756
- **Effort** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.974, DFD 0.952, DFDC 0.854, FSh 0.912, CDFv2 0.932, FFIW 0.925, KoDF 0.881, FAVC 0.924, DFDM 0.982, PGF 0.849, IDForge 0.960, DSv1.1 0.821, DSv2 0.644, CDFv3 0.787
- **GenD-CLIP** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.992, DFD 0.964, DFDC 0.864, FSh 0.866, CDFv2 0.946, FFIW 0.915, KoDF 0.849, FAVC 0.960, DFDM 0.996, PGF 0.896, IDForge 0.978, DSv1.1 0.901, DSv2 0.777, CDFv3 0.859
- **GenD-PE** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.977, DFD 0.968, DFDC 0.822, FSh 0.876, CDFv2 0.950, FFIW 0.937, KoDF 0.851, FAVC 0.973, DFDM 0.983, PGF 0.923, IDForge 0.979, DSv1.1 0.878, DSv2 0.786, CDFv3 0.895
- **GenD-DINO** (video, https://arxiv.org/html/2508.06248 (v4, Table 4, 'recalculated by us')): UADFV 0.986, DFD 0.962, DFDC 0.856, FSh 0.888, CDFv2 0.925, FFIW 0.929, KoDF 0.897, FAVC 0.984, DFDM 0.998, PGF 0.924, IDForge 0.982, DSv1.1 0.869, DSv2 0.794, CDFv3 0.835
- **ForAda** (frame, /data/umar/Repos/ForensicsAdapter/README.md (frame table + video table)): CDFv1 0.914, CDFv2 0.900, DFDC 0.843, DFDCP 0.890, DFD 0.933
- **ForAda** (video, /data/umar/Repos/ForensicsAdapter/README.md (frame table + video table)): CDFv2 0.957, DFDC 0.872, DFDCP 0.929
- **FS-VFM ViT-L** (frame, https://huggingface.co/Wolowolo/fsfm-3c FS-VFM ViT-L FT_on_FF++_c23_32frames experiments_test/test_results.txt): DFD 0.933, CDFv1 0.845, CDFv2 0.876, DFDC 0.836, DFDCP 0.883, WDF 0.903, CDFv3 0.864
- **FS-VFM ViT-L** (video, https://huggingface.co/Wolowolo/fsfm-3c FS-VFM ViT-L FT_on_FF++_c23_32frames experiments_test/test_results.txt): DFD 0.972, CDFv1 0.937, CDFv2 0.952, DFDC 0.877, DFDCP 0.933, WDF 0.916, CDFv3 0.911

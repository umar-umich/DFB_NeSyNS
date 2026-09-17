# Experiment 3 — frozen X-CLIP temporal probe and fusion check

Feasibility check inside the evidential framework, run on GPU 1 while the matched augmentation pilot trains. Frozen X-CLIP (`microsoft/xclip-base-patch16-16-frames`) video features from 16 of the stored face-crop frames per video; small evidential probe trained on FF++ train only; epoch chosen by the v3.1 development rule. Existing views come from the development-selected warm+gate checkpoint (epoch 19). Fusion is fitted only on the grouped fitting folds of the fusion audit (identical splits); test sets are read afterwards and choose nothing. Script `analysis/tbiom/xclip/probe_and_fuse.py`, features `analysis/tbiom/xclip/extract_xclip.py`.

Names: APP = Appearance, OM = Off-manifold residual, FP = Bona-fide face prior (raw), T = X-CLIP ordered, Tshuf = X-CLIP shuffled frame order, S = plain CLIP ViT-B/16 frame mean (spatial control). DS = equal Dempster–Shafer replay; LR = logistic stacking on matched inputs (p and vacuity of the three views).

## 1. Probes alone (development, full dev set used for epoch selection)

| probe | epoch | v3.1 | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|---:|
| xclip_temporal | 17 | 0.6752 | 0.7037 | 0.6598 | 0.6621 |
| xclip_shuffled | 13 | 0.6745 | 0.7029 | 0.6483 | 0.6723 |
| xclip_framemean | 8 | 0.6936 | 0.7256 | 0.6439 | 0.7112 |
| clip_framemean | 12 | 0.7051 | 0.7121 | 0.6641 | 0.7391 |

## 2. Clean development — selection folds (mean ± sd over 5 grouped splits)

| arm | v3.1 | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|
| probe xclip_temporal | 0.6728 ±0.0084 | 0.7098 ±0.0087 | 0.6453 ±0.0115 | 0.6632 ±0.0244 |
| probe xclip_shuffled | 0.6733 ±0.0109 | 0.7088 ±0.0099 | 0.6335 ±0.0149 | 0.6775 ±0.0220 |
| probe xclip_framemean | 0.6926 ±0.0091 | 0.7337 ±0.0070 | 0.6378 ±0.0146 | 0.7064 ±0.0250 |
| probe clip_framemean | 0.7049 ±0.0075 | 0.7143 ±0.0054 | 0.6603 ±0.0106 | 0.7400 ±0.0101 |
| gate (trained) | 0.8204 ±0.0090 | 0.9092 ±0.0109 | 0.7769 ±0.0172 | 0.7751 ±0.0078 |
| DS APP+OM+FP (g=1) | 0.8179 ±0.0094 | 0.9054 ±0.0112 | 0.7764 ±0.0173 | 0.7720 ±0.0082 |
| DS APP+OM (two-view) | 0.8226 ±0.0060 | 0.9286 ±0.0086 | 0.7583 ±0.0179 | 0.7810 ±0.0083 |
| DS T+OM+FP | 0.8127 ±0.0111 | 0.8930 ±0.0118 | 0.7670 ±0.0182 | 0.7781 ±0.0124 |
| LR T+OM+FP | 0.8270 ±0.0078 | 0.9256 ±0.0095 | 0.7495 ±0.0168 | 0.8061 ±0.0103 |
| DS T+APP+OM | 0.8260 ±0.0077 | 0.9221 ±0.0089 | 0.7508 ±0.0180 | 0.8051 ±0.0114 |
| LR T+APP+OM | 0.8218 ±0.0090 | 0.9124 ±0.0119 | 0.7617 ±0.0154 | 0.7913 ±0.0107 |
| DS Tshuf+OM+FP | 0.8159 ±0.0109 | 0.8940 ±0.0117 | 0.7687 ±0.0196 | 0.7849 ±0.0101 |
| LR Tshuf+OM+FP | 0.8282 ±0.0075 | 0.9278 ±0.0094 | 0.7540 ±0.0172 | 0.8029 ±0.0083 |
| DS Tshuf+APP+OM | 0.8289 ±0.0079 | 0.9254 ±0.0087 | 0.7519 ±0.0195 | 0.8093 ±0.0100 |
| LR Tshuf+APP+OM | 0.8226 ±0.0089 | 0.9127 ±0.0117 | 0.7660 ±0.0160 | 0.7891 ±0.0107 |
| DS S+OM+FP | 0.8138 ±0.0121 | 0.8870 ±0.0120 | 0.7698 ±0.0184 | 0.7847 ±0.0103 |
| LR S+OM+FP | 0.8258 ±0.0083 | 0.9208 ±0.0101 | 0.7550 ±0.0190 | 0.8018 ±0.0090 |
| DS S+APP+OM | 0.8249 ±0.0086 | 0.9198 ±0.0090 | 0.7579 ±0.0187 | 0.7971 ±0.0052 |
| LR S+APP+OM | 0.8174 ±0.0084 | 0.9074 ±0.0092 | 0.7569 ±0.0163 | 0.7879 ±0.0122 |
| LR APP+OM+FP (g=1) | 0.8197 ±0.0092 | 0.9022 ±0.0122 | 0.7783 ±0.0175 | 0.7786 ±0.0116 |

## 4. Error overlap on the selection folds (diagnostic, not a veto)

Errors at each opinion's FF++-val EER threshold. `recovered` = share of the reference's errors the probe gets right.

| pair | Jaccard of errors | probe error | reference error | recovered |
|---|---:|---:|---:|---:|
| probe xclip_temporal vs DS APP+OM+FP (g=1) | 0.304 | 0.384 | 0.255 | 0.416 |
| probe xclip_temporal vs gate (trained) | 0.300 | 0.384 | 0.252 | 0.417 |
| probe xclip_temporal vs DS APP+OM (two-view) | 0.290 | 0.384 | 0.242 | 0.419 |
| probe clip_framemean vs DS APP+OM+FP (g=1) | 0.279 | 0.350 | 0.255 | 0.482 |
| probe clip_framemean vs gate (trained) | 0.279 | 0.350 | 0.252 | 0.477 |
| probe clip_framemean vs DS APP+OM (two-view) | 0.275 | 0.350 | 0.242 | 0.473 |

## 5. Test sets — video AUROC (read after; LR arms averaged over the 5 fitted splits)

| arm | CDFv2 | CDFv3 | CDFv3 reenactment | CDFv3 swap | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | KoDF first-order motion | IDForge | DFDM | CDFv2 jpeg40 | CDFv2 rs06 | DFDCP jpeg40 | DFDCP rs06 | OOD mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| probe xclip_temporal | 0.7043 | 0.7754 | 0.7887 | 0.7509 | 0.7297 | 0.7063 | 0.6764 | 0.6462 | 0.7040 | 0.7466 | 0.5971 | 0.7696 | 0.6870 | 0.6437 | 0.7374 | 0.6139 | 0.6597 | 0.7145 |
| probe xclip_shuffled | 0.7097 | 0.7771 | 0.7852 | 0.7621 | 0.7204 | 0.7065 | 0.6736 | 0.6480 | 0.7122 | 0.7557 | 0.6129 | 0.7520 | 0.7083 | 0.6500 | 0.7445 | 0.6228 | 0.6594 | 0.7163 |
| probe xclip_framemean | 0.7151 | 0.7767 | 0.7900 | 0.7523 | 0.7551 | 0.7474 | 0.6411 | 0.6724 | 0.7262 | 0.7566 | 0.5842 | 0.8005 | 0.6951 | 0.6353 | 0.7410 | 0.5959 | 0.6101 | 0.7286 |
| probe clip_framemean | 0.7163 | 0.8210 | 0.8774 | 0.7172 | 0.8108 | 0.7180 | 0.7555 | 0.5649 | 0.7835 | 0.7704 | 0.5903 | 0.6759 | 0.6477 | 0.6546 | 0.6643 | 0.7196 | 0.6688 | 0.7264 |
| gate (trained) | 0.9539 | 0.8881 | 0.8612 | 0.9376 | 0.9462 | 0.8887 | 0.8986 | 0.6907 | 0.9093 | 0.8434 | 0.7068 | 0.9660 | 0.9918 | 0.8426 | 0.9386 | 0.8751 | 0.8344 | 0.8977 |
| DS APP+OM+FP (g=1) | 0.9515 | 0.8879 | 0.8621 | 0.9354 | 0.9459 | 0.8869 | 0.8975 | 0.6902 | 0.9062 | 0.8421 | 0.7031 | 0.9656 | 0.9911 | 0.8397 | 0.9363 | 0.8732 | 0.8349 | 0.8965 |
| DS APP+OM (two-view) | 0.9693 | 0.8691 | 0.8271 | 0.9463 | 0.9343 | 0.8818 | 0.8755 | 0.6769 | 0.9284 | 0.8509 | 0.7206 | 0.9598 | 0.9909 | 0.8625 | 0.9507 | 0.8185 | 0.7986 | 0.8937 |
| DS T+OM+FP | 0.9268 | 0.9003 | 0.8877 | 0.9235 | 0.9204 | 0.8822 | 0.8926 | 0.7110 | 0.8865 | 0.8362 | 0.6705 | 0.9592 | 0.9853 | 0.8075 | 0.9237 | 0.8652 | 0.8380 | 0.8901 |
| LR T+OM+FP | 0.9699 | 0.9013 | 0.8729 | 0.9537 | 0.9381 | 0.8856 | 0.8886 | 0.7062 | 0.9366 | 0.8573 | 0.7245 | 0.9707 | 0.9924 | 0.8745 | 0.9554 | 0.8622 | 0.8170 | 0.9047 |
| DS T+APP+OM | 0.9528 | 0.8888 | 0.8598 | 0.9422 | 0.9256 | 0.8752 | 0.8768 | 0.7058 | 0.9290 | 0.8528 | 0.7165 | 0.9607 | 0.9857 | 0.8443 | 0.9461 | 0.8185 | 0.8056 | 0.8953 |
| LR T+APP+OM | 0.9678 | 0.9189 | 0.9004 | 0.9529 | 0.9132 | 0.8729 | 0.8582 | 0.6500 | 0.9243 | 0.8587 | 0.7126 | 0.9572 | 0.9924 | 0.7151 | 0.9528 | 0.7232 | 0.7871 | 0.8914 |
| DS Tshuf+OM+FP | 0.9282 | 0.8992 | 0.8857 | 0.9239 | 0.9190 | 0.8842 | 0.8878 | 0.7122 | 0.8920 | 0.8410 | 0.6776 | 0.9597 | 0.9861 | 0.8116 | 0.9241 | 0.8620 | 0.8325 | 0.8909 |
| LR Tshuf+OM+FP | 0.9713 | 0.8970 | 0.8663 | 0.9535 | 0.9392 | 0.8871 | 0.8859 | 0.7009 | 0.9383 | 0.8611 | 0.7332 | 0.9699 | 0.9927 | 0.8819 | 0.9559 | 0.8601 | 0.8139 | 0.9044 |
| DS Tshuf+APP+OM | 0.9550 | 0.8867 | 0.8562 | 0.9427 | 0.9247 | 0.8789 | 0.8737 | 0.7062 | 0.9327 | 0.8570 | 0.7239 | 0.9616 | 0.9876 | 0.8521 | 0.9463 | 0.8197 | 0.8017 | 0.8964 |
| LR Tshuf+APP+OM | 0.9689 | 0.9165 | 0.8969 | 0.9525 | 0.9137 | 0.8741 | 0.8561 | 0.6435 | 0.9261 | 0.8657 | 0.7271 | 0.9557 | 0.9931 | 0.7139 | 0.9525 | 0.7220 | 0.7838 | 0.8913 |
| DS S+OM+FP | 0.9295 | 0.9069 | 0.8995 | 0.9205 | 0.9364 | 0.8776 | 0.8961 | 0.6788 | 0.8886 | 0.8299 | 0.6711 | 0.9417 | 0.9827 | 0.8112 | 0.9101 | 0.8840 | 0.8299 | 0.8868 |
| LR S+OM+FP | 0.9665 | 0.9060 | 0.8824 | 0.9495 | 0.9418 | 0.8830 | 0.8880 | 0.6810 | 0.9330 | 0.8499 | 0.7229 | 0.9564 | 0.9895 | 0.8689 | 0.9489 | 0.8882 | 0.8116 | 0.8995 |
| DS S+APP+OM | 0.9528 | 0.8996 | 0.8791 | 0.9373 | 0.9400 | 0.8704 | 0.8786 | 0.6644 | 0.9257 | 0.8510 | 0.7302 | 0.9433 | 0.9816 | 0.8555 | 0.9355 | 0.8433 | 0.8006 | 0.8907 |
| LR S+APP+OM | 0.9639 | 0.9223 | 0.9077 | 0.9493 | 0.9207 | 0.8710 | 0.8601 | 0.6292 | 0.9240 | 0.8521 | 0.7111 | 0.9439 | 0.9888 | 0.7094 | 0.9431 | 0.7552 | 0.7861 | 0.8876 |
| LR APP+OM+FP (g=1) | 0.9682 | 0.9141 | 0.8951 | 0.9489 | 0.9199 | 0.8769 | 0.8688 | 0.6414 | 0.9181 | 0.8587 | 0.7094 | 0.9570 | 0.9928 | 0.7031 | 0.9500 | 0.7612 | 0.7980 | 0.8916 |

## Limitations

- Frozen features and a small probe: a failure here is evidence against this configuration, not proof that bounded adaptation cannot help. One probe seed; one detector seed.
- Temporal sampling is coarse: 16 of up to 32 frames stored across the whole clip (~0.5–1 s apart), not dense tracked clips.
- X-CLIP's encoder exchanges cross-frame message tokens, so its frame mean is not purely spatial; the plain CLIP ViT-B/16 frame mean is the spatial control.
- The probe epoch uses the full development set (like detector checkpoints), so fold comparisons are conditional on it.
- CDFv2 and DeepfakeEval2024 have informed development; no test set here is untouched confirmation data.

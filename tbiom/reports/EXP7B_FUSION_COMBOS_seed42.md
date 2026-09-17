# Experiment 7b — view combinations × fusion rules beside the temporal branch (`vjepa2_lora_seed42`)

OMP_NUM_THREADS=8 python analysis/tbiom/temporal/fusion_combos.py [--branch vjepa2_lora_seed42] [--tag seed42]

Views: T = V-JEPA 2 LoRA branch; APP = Appearance; OM = Off-manifold residual; FP = Bona-fide face prior (raw)
(APP/OM/FP from the development-selected warm+gate detector, epoch 19).
Combinations: T+APP+OM, T+APP+OM+FP, T+OM+FP, T+OM, T+APP, T+FP, APP+OM (two-view base), OM+FP.
Fusion rules, all fitted ONLY on the source-grouped fitting folds of the fusion audit (5 splits):
  LR    logistic stacking on matched inputs (p, vacuity)
  DS    Dempster-Shafer combination of the branch Dirichlet opinions, equal trust
  DS-R  conflict-aware DS with Shafer reliability discounting: each branch's evidence scaled by a reliability weight
        w_b in {0, .25, .5, .75, 1} (max w = 1) chosen on the fitting fold by the v3.1 objective, then combined by DS
Evaluated on the selection folds (clean, JPEG q40, resize 0.60 — processed never used for fitting); tests read after.

NON-INFERIORITY RULE (fixed before the run, UMAR 2026-09-17: prefer the conflict-aware evidential fusion when it is close
to the simple alternative). For a combination, the evidential rule (best of DS / DS-R on clean development) is ADOPTED
over LR if mean Delta = LR - evidential < +0.005 on clean development AND the 95% paired source-group bootstrap CI upper
bound < +0.010 in >= 4/5 splits. LR is always reported alongside.

## 1. Development (selection folds, mean of 5 splits)

| combination | rule | clean v3.1 | swap | reenactment | in-the-wild | jpeg40 v3.1 | rs06 v3.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| T+APP+OM | LR | 0.8661 | 0.9229 | 0.8827 | 0.7927 | 0.6809 | 0.8266 |
| T+APP+OM | DS | 0.8700 | 0.9310 | 0.8955 | 0.7835 | 0.7748 | 0.8412 |
| T+APP+OM | DS-R | 0.8748 | 0.9369 | 0.8970 | 0.7905 | 0.7846 | 0.8470 |
| T+APP+OM+FP | LR | 0.8655 | 0.9218 | 0.8772 | 0.7975 | 0.6796 | 0.8266 |
| T+APP+OM+FP | DS | 0.8585 | 0.9218 | 0.8753 | 0.7784 | 0.7577 | 0.8320 |
| T+APP+OM+FP | DS-R | 0.8748 | 0.9369 | 0.8970 | 0.7905 | 0.7846 | 0.8470 |
| T+OM+FP | LR | 0.8687 | 0.9323 | 0.8695 | 0.8043 | 0.7635 | 0.8251 |
| T+OM+FP | DS | 0.8547 | 0.9106 | 0.8823 | 0.7712 | 0.7533 | 0.8294 |
| T+OM+FP | DS-R | 0.8749 | 0.9322 | 0.9061 | 0.7863 | 0.7940 | 0.8482 |
| T+OM | LR | 0.8717 | 0.9341 | 0.8836 | 0.7973 | 0.7854 | 0.8275 |
| T+OM | DS | 0.8677 | 0.9192 | 0.9079 | 0.7759 | 0.7925 | 0.8434 |
| T+OM | DS-R | 0.8749 | 0.9322 | 0.9061 | 0.7863 | 0.7940 | 0.8482 |
| T+APP | LR | 0.8573 | 0.9236 | 0.8588 | 0.7894 | 0.7206 | 0.8153 |
| T+APP | DS | 0.8600 | 0.9146 | 0.9013 | 0.7641 | 0.7663 | 0.8406 |
| T+APP | DS-R | 0.8640 | 0.9251 | 0.8958 | 0.7712 | 0.7659 | 0.8441 |
| T+FP | LR | 0.8405 | 0.8726 | 0.8881 | 0.7609 | 0.7189 | 0.7979 |
| T+FP | DS | 0.8378 | 0.8828 | 0.8839 | 0.7467 | 0.7352 | 0.8145 |
| T+FP | DS-R | 0.8399 | 0.8824 | 0.8951 | 0.7423 | 0.7371 | 0.8187 |
| APP+OM | LR | 0.8179 | 0.9065 | 0.7732 | 0.7740 | 0.6499 | 0.7857 |
| APP+OM | DS | 0.8226 | 0.9286 | 0.7583 | 0.7810 | 0.7293 | 0.7864 |
| APP+OM | DS-R | 0.8238 | 0.9282 | 0.7600 | 0.7833 | 0.7409 | 0.7847 |
| OM+FP | LR | 0.8240 | 0.9250 | 0.7572 | 0.7897 | 0.7404 | 0.7889 |
| OM+FP | DS | 0.8064 | 0.8854 | 0.7722 | 0.7617 | 0.7137 | 0.7793 |
| OM+FP | DS-R | 0.8241 | 0.9280 | 0.7603 | 0.7841 | 0.7429 | 0.7845 |

## 2. Non-inferiority of the evidential rule vs logistic stacking (clean development)

| combination | evidential rule (per split) | mean Δ = LR − evidential | CI upper < +0.010 splits | DS-R weights (split 0) | adopt evidential? |
|---|---|---:|---:|---|---|
| T+APP+OM | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0087 | 4/5 | {'T': 0.25, 'APP': 0.25, 'OM': 1.0} | **YES** |
| T+APP+OM+FP | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0093 | 5/5 | {'T': 0.25, 'APP': 0.25, 'OM': 1.0, 'FP': 0.0} | **YES** |
| T+OM+FP | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0061 | 5/5 | {'T': 0.25, 'OM': 1.0, 'FP': 0.0} | **YES** |
| T+OM | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0032 | 5/5 | {'T': 0.25, 'OM': 1.0} | **YES** |
| T+APP | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0067 | 4/5 | {'T': 0.25, 'APP': 1.0} | **YES** |
| T+FP | DS-R,DS-R,DS-R,DS-R,DS-R | +0.0006 | 4/5 | {'T': 1.0, 'FP': 0.25} | **YES** |
| APP+OM | DS,DS-R,DS-R,DS-R,DS-R | -0.0059 | 4/5 | {'APP': 0.0, 'OM': 1.0} | **YES** |
| OM+FP | DS-R,DS-R,DS-R,DS-R,DS-R | -0.0002 | 5/5 | {'OM': 1.0, 'FP': 0.0} | **YES** |

## 3. Test sets — video AUROC (read after; mean over the 5 fitted splits)

| combination | rule | CDFv2 | CDFv3 | CDFv3 reenactment | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | CDFv2 jpeg40 | CDFv2 rs06 | DFDCP jpeg40 | DFDCP rs06 | 10-set mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T+APP+OM | LR | 0.9636 | 0.9641 | 0.9685 | 0.9226 | 0.8549 | 0.8523 | 0.6670 | 0.9172 | 0.9646 | 0.9546 | 0.9941 | 0.6910 | 0.9314 | 0.7068 | 0.7981 | 0.9055 |
| T+APP+OM | DS | 0.9471 | 0.9562 | 0.9615 | 0.9176 | 0.8474 | 0.8587 | 0.6995 | 0.9127 | 0.9547 | 0.9523 | 0.9920 | 0.8676 | 0.9174 | 0.8161 | 0.8069 | 0.9038 |
| T+APP+OM | DS-R | 0.9556 | 0.9588 | 0.9627 | 0.9285 | 0.8566 | 0.8671 | 0.7038 | 0.9236 | 0.9564 | 0.9599 | 0.9934 | 0.8797 | 0.9307 | 0.8244 | 0.8107 | 0.9104 |
| T+APP+OM+FP | LR | 0.9602 | 0.9633 | 0.9686 | 0.9195 | 0.8565 | 0.8640 | 0.6764 | 0.9174 | 0.9649 | 0.9547 | 0.9934 | 0.6861 | 0.9290 | 0.7365 | 0.8096 | 0.9070 |
| T+APP+OM+FP | DS | 0.9478 | 0.9504 | 0.9530 | 0.9250 | 0.8691 | 0.8876 | 0.7108 | 0.9098 | 0.9465 | 0.9640 | 0.9920 | 0.8467 | 0.9287 | 0.8737 | 0.8382 | 0.9103 |
| T+APP+OM+FP | DS-R | 0.9556 | 0.9588 | 0.9627 | 0.9285 | 0.8566 | 0.8671 | 0.7038 | 0.9236 | 0.9564 | 0.9599 | 0.9934 | 0.8797 | 0.9307 | 0.8244 | 0.8107 | 0.9104 |
| T+OM+FP | LR | 0.9590 | 0.9569 | 0.9590 | 0.9257 | 0.8581 | 0.8759 | 0.7111 | 0.9220 | 0.9575 | 0.9568 | 0.9920 | 0.8808 | 0.9270 | 0.8030 | 0.8168 | 0.9115 |
| T+OM+FP | DS | 0.9355 | 0.9536 | 0.9614 | 0.9114 | 0.8596 | 0.8816 | 0.7107 | 0.8918 | 0.9532 | 0.9590 | 0.9909 | 0.8271 | 0.9076 | 0.8694 | 0.8357 | 0.9047 |
| T+OM+FP | DS-R | 0.9495 | 0.9629 | 0.9706 | 0.9198 | 0.8452 | 0.8617 | 0.6998 | 0.9110 | 0.9614 | 0.9551 | 0.9930 | 0.8869 | 0.9109 | 0.8276 | 0.8072 | 0.9060 |
| T+OM | LR | 0.9638 | 0.9584 | 0.9592 | 0.9312 | 0.8608 | 0.8668 | 0.7037 | 0.9218 | 0.9560 | 0.9591 | 0.9932 | 0.8953 | 0.9314 | 0.8238 | 0.8099 | 0.9115 |
| T+OM | DS | 0.9346 | 0.9604 | 0.9714 | 0.8983 | 0.8285 | 0.8510 | 0.6925 | 0.8898 | 0.9620 | 0.9427 | 0.9907 | 0.8836 | 0.8817 | 0.8232 | 0.7999 | 0.8951 |
| T+OM | DS-R | 0.9495 | 0.9629 | 0.9706 | 0.9198 | 0.8452 | 0.8617 | 0.6998 | 0.9110 | 0.9614 | 0.9551 | 0.9930 | 0.8869 | 0.9109 | 0.8276 | 0.8072 | 0.9060 |
| T+APP | LR | 0.9518 | 0.9529 | 0.9567 | 0.9041 | 0.8395 | 0.8526 | 0.6816 | 0.8875 | 0.9554 | 0.9463 | 0.9908 | 0.8452 | 0.9194 | 0.7486 | 0.8024 | 0.8962 |
| T+APP | DS | 0.9211 | 0.9520 | 0.9635 | 0.8904 | 0.8202 | 0.8374 | 0.6903 | 0.8874 | 0.9563 | 0.9353 | 0.9884 | 0.8457 | 0.8836 | 0.8001 | 0.7955 | 0.8879 |
| T+APP | DS-R | 0.9338 | 0.9510 | 0.9586 | 0.9097 | 0.8333 | 0.8430 | 0.6957 | 0.9080 | 0.9538 | 0.9459 | 0.9907 | 0.8458 | 0.9130 | 0.8019 | 0.8025 | 0.8965 |
| T+FP | LR | 0.8605 | 0.9395 | 0.9644 | 0.8668 | 0.8126 | 0.8278 | 0.7176 | 0.8545 | 0.9649 | 0.9287 | 0.9821 | 0.6836 | 0.8411 | 0.7576 | 0.8045 | 0.8755 |
| T+FP | DS | 0.8864 | 0.9429 | 0.9605 | 0.8737 | 0.8336 | 0.8478 | 0.7034 | 0.8546 | 0.9556 | 0.9395 | 0.9854 | 0.7726 | 0.8612 | 0.8302 | 0.8196 | 0.8823 |
| T+FP | DS-R | 0.8775 | 0.9425 | 0.9625 | 0.8641 | 0.8213 | 0.8424 | 0.6961 | 0.8474 | 0.9574 | 0.9327 | 0.9838 | 0.7717 | 0.8424 | 0.8350 | 0.8114 | 0.8765 |
| APP+OM | LR | 0.9699 | 0.9080 | 0.8850 | 0.9123 | 0.8701 | 0.8541 | 0.6272 | 0.9208 | 0.8630 | 0.9521 | 0.9922 | 0.7139 | 0.9514 | 0.7212 | 0.7778 | 0.8870 |
| APP+OM | DS | 0.9693 | 0.8691 | 0.8271 | 0.9343 | 0.8818 | 0.8755 | 0.6769 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8625 | 0.9507 | 0.8185 | 0.7986 | 0.8937 |
| APP+OM | DS-R | 0.9732 | 0.8787 | 0.8397 | 0.9341 | 0.8811 | 0.8733 | 0.6701 | 0.9347 | 0.8543 | 0.9633 | 0.9919 | 0.8848 | 0.9544 | 0.8250 | 0.7976 | 0.8955 |
| OM+FP | LR | 0.9722 | 0.8863 | 0.8508 | 0.9373 | 0.8848 | 0.8862 | 0.6849 | 0.9340 | 0.8549 | 0.9674 | 0.9925 | 0.8861 | 0.9542 | 0.8608 | 0.8110 | 0.9000 |
| OM+FP | DS | 0.9378 | 0.8886 | 0.8693 | 0.9415 | 0.8824 | 0.8908 | 0.6929 | 0.8851 | 0.8334 | 0.9621 | 0.9899 | 0.8136 | 0.9181 | 0.8639 | 0.8326 | 0.8904 |
| OM+FP | DS-R | 0.9736 | 0.8801 | 0.8416 | 0.9338 | 0.8808 | 0.8727 | 0.6688 | 0.9354 | 0.8548 | 0.9635 | 0.9920 | 0.8872 | 0.9549 | 0.8256 | 0.7974 | 0.8955 |

One detector seed and one temporal-branch seed. CDFv2 and DeepfakeEval2024 informed development. FaceShifter is sealed.

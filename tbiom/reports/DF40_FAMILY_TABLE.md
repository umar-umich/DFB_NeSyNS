# DF40 four-family results table

    python tbiom/tables/build_table.py --render

Generated from `analysis/tables/per_generator.csv`; edit the builder, not this file.
## What the numbers are

**Protocol.** Every detector is trained on FaceForensics++ c23 and evaluated, without adaptation, on
DF40's test sets. A generator is scored on its own against the real set of its own domain --
`DF40-cdf-real` for a `-cdf` variant, `DF40-ff-real` for `-ff` -- so a family column is the mean and
standard deviation ACROSS the generators in that family, never a single pooled AUROC. The spread is
part of the result: it separates a family a method actually handles from a family carried by one
easy generator.

**Families.** FS / FR / EFS / FE are DF40's own published taxonomy, used unchanged so the columns
line up with published DF40 tables. Two consequences are ours to disclose rather than to fix.
DF40 files `wav2lip` under reenactment while our v3.1 objective excludes it as audio-driven
lip-sync; it is included here and flagged in the per-generator appendix. And `stargan`, `starganv2`
and `styleclip` ship their own paired reals rather than DF40's shared pools -- DF40's own split
between 31 "known" methods that share the FF++/Celeb-DF reals and 9 "unknown" ones that bundle
theirs. That is packaging, not taxonomy: all four editors are face editing and sit in ONE Editing
column. Averaging across different real pools is already what every family does, since `-cdf` and
`-ff` cells are scored against different pools; the appendix names the pool used for every cell.

**The `-ff` half is included.** DF40's FF++-domain sets are built over the 1000 FaceForensics++
source videos, 720 of which are in our training split. Those halves are therefore not
identity-disjoint from training for ANY method in the table -- every row here trains on FF++ c23, so
the advantage is shared rather than ours. They are included because excluding them would drop half
of DF40 and break comparability with published DF40 numbers; `-cdf` and `-ff` are reported
separately in the appendix so the two can always be read apart.

**Image-level sets.** DF40's EFS and FE sets are image collections rather than videos:
one image is one "video", and video level equals frame level for them by construction. Our temporal
view still runs on those sets -- it receives the single frame repeated across its clip window -- so
it contributes a real but uninformative opinion rather than being switched off. No row is given a
different fusion on those sets than on the video sets.

**Scoring.** Our rows and both DiCoME rows replay the model's own Dempster-Shafer fusion over the
views their export carries: two for DiCoME (semantic, artifact), three for ours (semantic,
artifact, temporal). Video level means aggregating frames to videos first and then scoring, with
video identity from the rule in `analysis/tbiom/video_id.py`. Third-party rows use their own
`prob_class_1` with GenD's reduction (mean over the frames of a video). Every method sees exactly
the same frames: the baselines read PNGs dumped from the same h5 files our exports read.

**DiCoME appears twice.** `DiCoME (our training)` is the comparison of record -- the same FF++ c23
recipe, seed and schedule as ours, so the difference is the method rather than the training budget.
`DiCoME (released weights)` is kept as a reproduction check on our retraining, not as the baseline.

## Sets whose two halves ship at different native resolutions

These are separable on resampling alone, by any detector, independently of the
manipulation. The confound is DF40's, not a method's -- every row in the table is
affected equally -- but a number from one of these sets does not measure what the
column claims to measure, so they are named here and should be read as an upper bound.

| set | fake | real |
|---|---|---|
| DF40_e4e_cdf | 1024x1024 | 256x256 |
| DF40_e4e_ff | 1024x1024 | 256x256 |
| DF40_rddm_cdf | 259x259 | 256x256 |
| DF40_rddm_ff | 259x259 | 256x256 |
| DF40_sd2.1_cdf | 512x512 | 256x256 |
| DF40_sd2.1_ff | 512x512 | 256x256 |
| DF40_styleclip | 1024x1024 | 178x218 |

Every other set in the table is matched at 256x256 on both halves.

### DF40 video-level AUROC — mean ± sd across the generators in each family

| Method | Face swap (18) | Reenactment (24) | Synthesis (18) | Editing (5) | Macro |
|---|---:|---:|---:|---:|---:|
| DISCERN (ours, DS-R) | 0.980 ± 0.016 | 0.932 ± 0.055 | 0.975 ± 0.037 | 0.933 ± 0.115 | 0.955 |
| DISCERN (ours, PCR6+-R) | 0.978 ± 0.016 | 0.935 ± 0.053 | 0.971 ± 0.036 | 0.929 ± 0.127 | 0.953 |
| DiCoME (our training) | 0.970 ± 0.019 | 0.787 ± 0.160 | 0.980 ± 0.029 | 0.916 ± 0.101 | 0.913 |
| DiCoME (released weights) | 0.975 ± 0.017 | 0.815 ± 0.151 | 0.992 ± 0.015 | 0.943 ± 0.077 | 0.931 |
| GenD-CLIP | 0.958 ± 0.035 | 0.881 ± 0.105 | 0.957 ± 0.076 | 0.932 ± 0.087 | 0.932 |
| Effort | 0.939 ± 0.046 | 0.771 ± 0.109 | 0.978 ± 0.036 | 0.951 ± 0.054 | 0.910 |
| ForAda | 0.919 ± 0.062 | 0.741 ± 0.133 | 0.916 ± 0.125 | 0.758 ± 0.133 | 0.833 |
| FSFM | 0.962 ± 0.024 | 0.826 ± 0.154 | 0.944 ± 0.092 | 0.882 ± 0.124 | 0.903 |

### DF40 frame-level AUROC — mean ± sd across the generators in each family

| Method | Face swap (18) | Reenactment (24) | Synthesis (18) | Editing (5) | Macro |
|---|---:|---:|---:|---:|---:|
| DISCERN (ours, DS-R) | — | — | — | — | — |
| DISCERN (ours, PCR6+-R) | — | — | — | — | — |
| DiCoME (our training) | 0.938 ± 0.033 | 0.756 ± 0.147 | 0.936 ± 0.077 | 0.909 ± 0.096 | 0.885 |
| DiCoME (released weights) | 0.941 ± 0.033 | 0.775 ± 0.138 | 0.951 ± 0.063 | 0.938 ± 0.074 | 0.901 |
| GenD-CLIP | 0.915 ± 0.051 | 0.833 ± 0.103 | 0.912 ± 0.109 | 0.921 ± 0.079 | 0.895 |
| Effort | 0.906 ± 0.059 | 0.737 ± 0.100 | 0.938 ± 0.077 | 0.947 ± 0.050 | 0.882 |
| ForAda | 0.873 ± 0.073 | 0.713 ± 0.114 | 0.842 ± 0.143 | 0.747 ± 0.125 | 0.794 |
| FSFM | 0.926 ± 0.037 | 0.792 ± 0.142 | 0.901 ± 0.102 | 0.873 ± 0.116 | 0.873 |

### DF40 video-level AUROC per generator

`*` marks audio-driven lip-sync, which DF40 files under reenactment and our v3.1 objective excludes; it is kept here because the taxonomy is DF40's.

| family | generator | real | DISCERN (ours, DS-R) | DISCERN (ours, PCR6+-R) | DiCoME (our training) | DiCoME (released weights) | GenD-CLIP | Effort | ForAda | FSFM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Face swap | blendface | cdf | 0.9801 | 0.9790 | 0.9502 | 0.9552 | 0.9260 | 0.8925 | 0.8278 | 0.9676 |
| Face swap | blendface | ff | 0.9858 | 0.9846 | 0.9505 | 0.9573 | 0.9281 | 0.8777 | 0.8809 | 0.9652 |
| Face swap | e4s | cdf | 0.9892 | 0.9841 | 0.9949 | 0.9953 | 0.9937 | 0.9952 | 0.8864 | 0.9350 |
| Face swap | e4s | ff | 0.9927 | 0.9910 | 0.9891 | 0.9922 | 0.9883 | 0.9837 | 0.9687 | 0.9822 |
| Face swap | facedancer | cdf | 0.9714 | 0.9710 | 0.9657 | 0.9781 | 0.9740 | 0.9803 | 0.8957 | 0.9126 |
| Face swap | facedancer | ff | 0.9713 | 0.9690 | 0.9632 | 0.9720 | 0.9518 | 0.9287 | 0.9462 | 0.9726 |
| Face swap | faceswap | cdf | 0.9995 | 0.9993 | 0.9991 | 0.9998 | 0.9986 | 0.9962 | 0.9962 | 0.9855 |
| Face swap | faceswap | ff | 0.9948 | 0.9949 | 0.9795 | 0.9807 | 0.9761 | 0.9772 | 0.9737 | 0.9782 |
| Face swap | fsgan | cdf | 0.9906 | 0.9912 | 0.9892 | 0.9947 | 0.9804 | 0.9786 | 0.9781 | 0.9857 |
| Face swap | fsgan | ff | 0.9935 | 0.9934 | 0.9879 | 0.9902 | 0.9814 | 0.9642 | 0.9748 | 0.9790 |
| Face swap | inswap | cdf | 0.9551 | 0.9523 | 0.9433 | 0.9529 | 0.8791 | 0.8618 | 0.8927 | 0.9530 |
| Face swap | inswap | ff | 0.9783 | 0.9739 | 0.9696 | 0.9709 | 0.9625 | 0.9381 | 0.9492 | 0.9648 |
| Face swap | mobileswap | cdf | 0.9584 | 0.9571 | 0.9571 | 0.9582 | 0.9322 | 0.9114 | 0.9027 | 0.9192 |
| Face swap | mobileswap | ff | 0.9848 | 0.9824 | 0.9745 | 0.9785 | 0.9792 | 0.9550 | 0.9791 | 0.9682 |
| Face swap | simswap | cdf | 0.9416 | 0.9396 | 0.9339 | 0.9392 | 0.8849 | 0.8486 | 0.7588 | 0.9222 |
| Face swap | simswap | ff | 0.9749 | 0.9708 | 0.9539 | 0.9659 | 0.9522 | 0.9210 | 0.9131 | 0.9504 |
| Face swap | uniface | cdf | 0.9837 | 0.9813 | 0.9758 | 0.9810 | 0.9655 | 0.9229 | 0.8604 | 0.9872 |
| Face swap | uniface | ff | 0.9928 | 0.9911 | 0.9803 | 0.9831 | 0.9817 | 0.9653 | 0.9515 | 0.9819 |
| Reenactment | MRAA | cdf | 0.9934 | 0.9927 | 0.9770 | 0.9857 | 0.9563 | 0.8530 | 0.8945 | 0.9903 |
| Reenactment | MRAA | ff | 0.9993 | 0.9987 | 0.9937 | 0.9949 | 0.9936 | 0.9483 | 0.9654 | 0.9983 |
| Reenactment | danet | cdf | 0.8645 | 0.8852 | 0.4821 | 0.5131 | 0.7352 | 0.6344 | 0.4993 | 0.6032 |
| Reenactment | danet | ff | 0.9546 | 0.9592 | 0.6764 | 0.7183 | 0.9096 | 0.7490 | 0.7453 | 0.7460 |
| Reenactment | facevid2vid | cdf | 0.8403 | 0.8512 | 0.6748 | 0.7410 | 0.8482 | 0.7368 | 0.6272 | 0.7771 |
| Reenactment | facevid2vid | ff | 0.9058 | 0.9071 | 0.7776 | 0.8264 | 0.9259 | 0.8092 | 0.7784 | 0.8402 |
| Reenactment | fomm | cdf | 0.9341 | 0.9337 | 0.8793 | 0.9086 | 0.9352 | 0.8211 | 0.7342 | 0.9214 |
| Reenactment | fomm | ff | 0.9792 | 0.9756 | 0.9609 | 0.9684 | 0.9838 | 0.9124 | 0.8844 | 0.9613 |
| Reenactment | hyperreenact | cdf | 0.9708 | 0.9759 | 0.7421 | 0.8123 | 0.9141 | 0.8665 | 0.7456 | 0.9671 |
| Reenactment | hyperreenact | ff | 0.9827 | 0.9839 | 0.8483 | 0.8899 | 0.9765 | 0.9452 | 0.8955 | 0.9875 |
| Reenactment | lia | cdf | 0.9977 | 0.9974 | 0.9854 | 0.9823 | 0.8970 | 0.7910 | 0.7966 | 0.9343 |
| Reenactment | lia | ff | 0.9999 | 0.9998 | 0.9921 | 0.9894 | 0.9898 | 0.9146 | 0.9272 | 0.9666 |
| Reenactment | mcnet | cdf | 0.8662 | 0.8819 | 0.5594 | 0.6049 | 0.8042 | 0.6649 | 0.5405 | 0.6529 |
| Reenactment | mcnet | ff | 0.9566 | 0.9602 | 0.7680 | 0.8026 | 0.9430 | 0.7752 | 0.7785 | 0.8077 |
| Reenactment | one_shot_free | cdf | 0.9102 | 0.9096 | 0.8506 | 0.8889 | 0.9029 | 0.7677 | 0.6739 | 0.8822 |
| Reenactment | one_shot_free | ff | 0.9442 | 0.9438 | 0.8815 | 0.9080 | 0.9448 | 0.8222 | 0.8143 | 0.9069 |
| Reenactment | pirender | cdf | 0.8626 | 0.8652 | 0.8066 | 0.8574 | 0.8672 | 0.7412 | 0.6198 | 0.9070 |
| Reenactment | pirender | ff | 0.9274 | 0.9222 | 0.8833 | 0.9057 | 0.9436 | 0.8172 | 0.8274 | 0.9286 |
| Reenactment | sadtalker | cdf | 0.8857 | 0.9020 | 0.5402 | 0.5251 | 0.5788 | 0.5437 | 0.5035 | 0.4851 |
| Reenactment | sadtalker | ff | 0.9503 | 0.9570 | 0.6071 | 0.6193 | 0.7584 | 0.6064 | 0.7304 | 0.6653 |
| Reenactment | tpsm | cdf | 0.9661 | 0.9751 | 0.5097 | 0.5705 | 0.6980 | 0.6508 | 0.6267 | 0.4812 |
| Reenactment | tpsm | ff | 0.9677 | 0.9705 | 0.7458 | 0.7791 | 0.9333 | 0.7652 | 0.7748 | 0.7757 |
| Reenactment | wav2lip* | cdf | 0.8104 | 0.8056 | 0.8361 | 0.8414 | 0.7772 | 0.6370 | 0.5757 | 0.7578 |
| Reenactment | wav2lip* | ff | 0.8910 | 0.8786 | 0.9127 | 0.9178 | 0.9216 | 0.7271 | 0.8231 | 0.8776 |
| Synthesis | DiT | cdf | 0.8735 | 0.8759 | 0.8958 | 0.9498 | 0.7230 | 0.8926 | 0.7490 | 0.8898 |
| Synthesis | DiT | ff | 0.9442 | 0.9370 | 0.9546 | 0.9615 | 0.9384 | 0.9766 | 0.9302 | 0.9610 |
| Synthesis | SiT | cdf | 0.9231 | 0.9235 | 0.9462 | 0.9827 | 0.8383 | 0.9452 | 0.8260 | 0.9241 |
| Synthesis | SiT | ff | 0.9693 | 0.9646 | 0.9680 | 0.9784 | 0.9652 | 0.9841 | 0.9558 | 0.9732 |
| Synthesis | StyleGAN2 | cdf | 1.0000 | 0.9977 | 1.0000 | 1.0000 | 1.0000 | 0.9971 | 1.0000 | 0.9957 |
| Synthesis | StyleGAN2 | ff | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 |
| Synthesis | StyleGAN3 | cdf | 0.9999 | 0.9866 | 1.0000 | 1.0000 | 1.0000 | 0.9965 | 0.9932 | 0.9925 |
| Synthesis | StyleGAN3 | ff | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 |
| Synthesis | StyleGANXL | cdf | 0.9859 | 0.9782 | 0.9997 | 0.9976 | 0.9992 | 0.9973 | 0.9819 | 0.9957 |
| Synthesis | StyleGANXL | ff | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | VQGAN | cdf | 0.9999 | 0.9929 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 0.9594 | 0.9887 |
| Synthesis | VQGAN | ff | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9981 | 1.0000 |
| Synthesis | ddim | cdf | 0.9348 | 0.9373 | 0.9418 | 0.9931 | 0.8813 | 0.9711 | 0.6062 | 0.8629 |
| Synthesis | ddim | ff | 0.9741 | 0.9701 | 0.9746 | 0.9930 | 0.9746 | 0.9857 | 0.8807 | 0.9361 |
| Synthesis | rddm | cdf | 0.9437 | 0.9252 | 0.9682 | 0.9944 | 0.9090 | 0.8815 | 0.6510 | 0.6399 |
| Synthesis | rddm | ff | 0.9984 | 0.9954 | 0.9893 | 0.9970 | 0.9925 | 0.9858 | 0.9825 | 0.8299 |
| Synthesis | sd2.1 | cdf | 0.9997 | 0.9865 | 1.0000 | 1.0000 | 0.9983 | 0.9949 | 0.9740 | 0.9974 |
| Synthesis | sd2.1 | ff | 1.0000 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 0.9989 | 0.9982 | 1.0000 |
| Editing | e4e | cdf | 0.9991 | 0.9993 | 0.9822 | 0.9928 | 0.9858 | 0.9894 | 0.5987 | 0.9719 |
| Editing | e4e | ff | 0.9958 | 0.9942 | 0.9903 | 0.9973 | 0.9945 | 0.9918 | 0.8111 | 0.9797 |
| Editing | stargan | own | 0.9882 | 0.9825 | 0.9902 | 0.9946 | 0.9712 | 0.9851 | 0.9525 | 0.9031 |
| Editing | starganv2 | own | 0.7300 | 0.7024 | 0.7758 | 0.8229 | 0.7842 | 0.8747 | 0.7298 | 0.6734 |
| Editing | styleclip | own | 0.9525 | 0.9656 | 0.8395 | 0.9057 | 0.9244 | 0.9133 | 0.6992 | 0.8834 |

### Video counts per set (must agree across methods)

| set | DISCERN (ours, DS-R) | DISCERN (ours, PCR6+-R) | DiCoME (our training) | DiCoME (released weights) | Effort | FSFM | ForAda | GenD-CLIP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DF40_DiT_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_DiT_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_MRAA_cdf | 827 | 827 | 827 | 827 | 827 | 827 | 827 | 827 |
| DF40_MRAA_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_SiT_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_SiT_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGAN2_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGAN2_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGAN3_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGAN3_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGANXL_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGANXL_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_VQGAN_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_VQGAN_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_blendface_cdf | 818 | 818 | 818 | 818 | 818 | 818 | 818 | 818 |
| DF40_blendface_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_danet_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_danet_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_ddim_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_ddim_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_e4e_cdf | 18828 | 18828 | 18828 | 18828 | 18828 | 18828 | 18828 | 18828 |
| DF40_e4e_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_e4s_cdf | 470 | 470 | 470 | 470 | 470 | 470 | 470 | 470 |
| DF40_e4s_ff | 255 | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| DF40_facedancer_cdf | 826 | 826 | 826 | 826 | 826 | 826 | 826 | 826 |
| DF40_facedancer_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_faceswap_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_faceswap_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_facevid2vid_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_facevid2vid_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_fomm_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_fomm_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_fsgan_cdf | 794 | 794 | 794 | 794 | 794 | 794 | 794 | 794 |
| DF40_fsgan_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_hyperreenact_cdf | 819 | 819 | 819 | 819 | 819 | 819 | 819 | 819 |
| DF40_hyperreenact_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_inswap_cdf | 602 | 602 | 602 | 602 | 602 | 602 | 602 | 602 |
| DF40_inswap_ff | 258 | 258 | 258 | 258 | 258 | 258 | 258 | 258 |
| DF40_lia_cdf | 824 | 824 | 824 | 824 | 824 | 824 | 824 | 824 |
| DF40_lia_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_mcnet_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_mcnet_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_mobileswap_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_mobileswap_ff | 998 | 998 | 998 | 998 | 998 | 998 | 998 | 998 |
| DF40_one_shot_free_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_one_shot_free_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_pirender_cdf | 814 | 814 | 814 | 814 | 814 | 814 | 814 | 814 |
| DF40_pirender_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_rddm_cdf | 622 | 622 | 622 | 622 | 622 | 622 | 622 | 622 |
| DF40_rddm_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_sadtalker_cdf | 838 | 838 | 838 | 838 | 838 | 838 | 838 | 838 |
| DF40_sadtalker_ff | 278 | 278 | 278 | 278 | 278 | 278 | 278 | 278 |
| DF40_sd2.1_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_sd2.1_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_simswap_cdf | 819 | 819 | 819 | 819 | 819 | 819 | 819 | 819 |
| DF40_simswap_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_stargan | 3968 | 3968 | 3968 | 3968 | 3968 | 3968 | 3968 | 3968 |
| DF40_starganv2 | 3996 | 3996 | 3996 | 3996 | 3996 | 3996 | 3996 | 3996 |
| DF40_styleclip | 3824 | 3824 | 3824 | 3824 | 3824 | 3824 | 3824 | 3824 |
| DF40_tpsm_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_tpsm_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_uniface_cdf | 804 | 804 | 804 | 804 | 804 | 804 | 804 | 804 |
| DF40_uniface_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_wav2lip_cdf | 844 | 844 | 844 | 844 | 844 | 844 | 844 | 844 |
| DF40_wav2lip_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 280 |

**Count mismatches:** none


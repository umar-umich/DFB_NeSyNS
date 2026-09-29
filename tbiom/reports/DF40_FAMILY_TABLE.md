# DF40 four-family results table

    python tbiom/tables/build_table.py --render

Generated from `analysis/tables/per_generator.csv`; edit the builder, not this file.
## What the numbers are

**Protocol.** Our rows and both DiCoME rows are trained on FaceForensics++ c23 by us, on the same
recipe, seed and schedule. The four third-party rows are the authors' RELEASED weights, and their
training data is taken on the authors' word rather than verified here: Effort ships as
`effort_clip_L14_trainOn_FaceForensic`, ForAda and FSFM are the published checkpoints, and
`yermandy/GenD_CLIP_L_14` carries only an architecture config, so nothing on this machine records
what it saw. FSFM in particular is a self-supervised face model pretrained well beyond FF++ before
any detection fine-tuning. Treat the third-party rows as "published detector, our evaluation", not
as a matched-training comparison; the DiCoME row is the one where training is matched.
Every detector is evaluated, without adaptation, on
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

**The DiCoME row is our own retraining, not the released checkpoint.** It is the comparison of
record: the same FF++ c23 recipe, seed and schedule as ours, so the difference between the rows is
the method rather than the training budget. The authors' released DiCoME weights were also exported
over all 65 sets and are not printed here; they remain on disk as a reproduction check on that
retraining.

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
| ForAda | 0.919 ± 0.062 | 0.741 ± 0.133 | 0.916 ± 0.125 | 0.758 ± 0.133 | 0.833 |
| Effort | 0.939 ± 0.046 | 0.771 ± 0.109 | 0.978 ± 0.036 | 0.951 ± 0.054 | 0.910 |
| FSFM | 0.962 ± 0.024 | 0.826 ± 0.154 | 0.944 ± 0.092 | 0.882 ± 0.124 | 0.903 |
| GenD-CLIP | 0.958 ± 0.035 | 0.881 ± 0.105 | 0.957 ± 0.076 | 0.932 ± 0.087 | 0.932 |
| DiCoME | 0.970 ± 0.019 | 0.787 ± 0.160 | 0.980 ± 0.029 | 0.916 ± 0.101 | 0.913 |
| DISCERN (ours, Dempster) | 0.977 ± 0.018 | 0.930 ± 0.057 | 0.967 ± 0.041 | 0.931 ± 0.113 | 0.951 |
| DISCERN (ours, PCR6+-R) | 0.978 ± 0.016 | 0.935 ± 0.053 | 0.971 ± 0.036 | 0.929 ± 0.127 | 0.953 |

### DF40 frame-level AUROC — mean ± sd across the generators in each family

| Method | Face swap (18) | Reenactment (24) | Synthesis (18) | Editing (5) | Macro |
|---|---:|---:|---:|---:|---:|
| ForAda | 0.873 ± 0.073 | 0.713 ± 0.114 | 0.842 ± 0.143 | 0.747 ± 0.125 | 0.794 |
| Effort | 0.906 ± 0.059 | 0.737 ± 0.100 | 0.938 ± 0.077 | 0.947 ± 0.050 | 0.882 |
| FSFM | 0.926 ± 0.037 | 0.792 ± 0.142 | 0.901 ± 0.102 | 0.873 ± 0.116 | 0.873 |
| GenD-CLIP | 0.915 ± 0.051 | 0.833 ± 0.103 | 0.912 ± 0.109 | 0.921 ± 0.079 | 0.895 |
| DiCoME | 0.938 ± 0.033 | 0.756 ± 0.147 | 0.936 ± 0.077 | 0.909 ± 0.096 | 0.885 |
| DISCERN (ours, Dempster) | — | — | — | — | — |
| DISCERN (ours, PCR6+-R) | — | — | — | — | — |

### DF40 video-level AUROC per generator

`*` marks audio-driven lip-sync, which DF40 files under reenactment and our v3.1 objective excludes; it is kept here because the taxonomy is DF40's.

| family | generator | real | ForAda | Effort | FSFM | GenD-CLIP | DiCoME | DISCERN (ours, Dempster) | DISCERN (ours, PCR6+-R) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Face swap | blendface | cdf | 0.8278 | 0.8925 | 0.9676 | 0.9260 | 0.9502 | 0.9771 | 0.9790 |
| Face swap | blendface | ff | 0.8809 | 0.8777 | 0.9652 | 0.9281 | 0.9505 | 0.9846 | 0.9846 |
| Face swap | e4s | cdf | 0.8864 | 0.9952 | 0.9350 | 0.9937 | 0.9949 | 0.9808 | 0.9841 |
| Face swap | e4s | ff | 0.9687 | 0.9837 | 0.9822 | 0.9883 | 0.9891 | 0.9921 | 0.9910 |
| Face swap | facedancer | cdf | 0.8957 | 0.9803 | 0.9126 | 0.9740 | 0.9657 | 0.9635 | 0.9710 |
| Face swap | facedancer | ff | 0.9462 | 0.9287 | 0.9726 | 0.9518 | 0.9632 | 0.9693 | 0.9690 |
| Face swap | faceswap | cdf | 0.9962 | 0.9962 | 0.9855 | 0.9986 | 0.9991 | 0.9995 | 0.9993 |
| Face swap | faceswap | ff | 0.9737 | 0.9772 | 0.9782 | 0.9761 | 0.9795 | 0.9947 | 0.9949 |
| Face swap | fsgan | cdf | 0.9781 | 0.9786 | 0.9857 | 0.9804 | 0.9892 | 0.9864 | 0.9912 |
| Face swap | fsgan | ff | 0.9748 | 0.9642 | 0.9790 | 0.9814 | 0.9879 | 0.9928 | 0.9934 |
| Face swap | inswap | cdf | 0.8927 | 0.8618 | 0.9530 | 0.8791 | 0.9433 | 0.9505 | 0.9523 |
| Face swap | inswap | ff | 0.9492 | 0.9381 | 0.9648 | 0.9625 | 0.9696 | 0.9778 | 0.9739 |
| Face swap | mobileswap | cdf | 0.9027 | 0.9114 | 0.9192 | 0.9322 | 0.9571 | 0.9496 | 0.9571 |
| Face swap | mobileswap | ff | 0.9791 | 0.9550 | 0.9682 | 0.9792 | 0.9745 | 0.9837 | 0.9824 |
| Face swap | simswap | cdf | 0.7588 | 0.8486 | 0.9222 | 0.8849 | 0.9339 | 0.9346 | 0.9396 |
| Face swap | simswap | ff | 0.9131 | 0.9210 | 0.9504 | 0.9522 | 0.9539 | 0.9738 | 0.9708 |
| Face swap | uniface | cdf | 0.8604 | 0.9229 | 0.9872 | 0.9655 | 0.9758 | 0.9792 | 0.9813 |
| Face swap | uniface | ff | 0.9515 | 0.9653 | 0.9819 | 0.9817 | 0.9803 | 0.9924 | 0.9911 |
| Reenactment | MRAA | cdf | 0.8945 | 0.8530 | 0.9903 | 0.9563 | 0.9770 | 0.9915 | 0.9927 |
| Reenactment | MRAA | ff | 0.9654 | 0.9483 | 0.9983 | 0.9936 | 0.9937 | 0.9989 | 0.9987 |
| Reenactment | danet | cdf | 0.4993 | 0.6344 | 0.6032 | 0.7352 | 0.4821 | 0.8616 | 0.8852 |
| Reenactment | danet | ff | 0.7453 | 0.7490 | 0.7460 | 0.9096 | 0.6764 | 0.9538 | 0.9592 |
| Reenactment | facevid2vid | cdf | 0.6272 | 0.7368 | 0.7771 | 0.8482 | 0.6748 | 0.8316 | 0.8512 |
| Reenactment | facevid2vid | ff | 0.7784 | 0.8092 | 0.8402 | 0.9259 | 0.7776 | 0.9049 | 0.9071 |
| Reenactment | fomm | cdf | 0.7342 | 0.8211 | 0.9214 | 0.9352 | 0.8793 | 0.9293 | 0.9337 |
| Reenactment | fomm | ff | 0.8844 | 0.9124 | 0.9613 | 0.9838 | 0.9609 | 0.9798 | 0.9756 |
| Reenactment | hyperreenact | cdf | 0.7456 | 0.8665 | 0.9671 | 0.9141 | 0.7421 | 0.9716 | 0.9759 |
| Reenactment | hyperreenact | ff | 0.8955 | 0.9452 | 0.9875 | 0.9765 | 0.8483 | 0.9838 | 0.9839 |
| Reenactment | lia | cdf | 0.7966 | 0.7910 | 0.9343 | 0.8970 | 0.9854 | 0.9965 | 0.9974 |
| Reenactment | lia | ff | 0.9272 | 0.9146 | 0.9666 | 0.9898 | 0.9921 | 0.9998 | 0.9998 |
| Reenactment | mcnet | cdf | 0.5405 | 0.6649 | 0.6529 | 0.8042 | 0.5594 | 0.8629 | 0.8819 |
| Reenactment | mcnet | ff | 0.7785 | 0.7752 | 0.8077 | 0.9430 | 0.7680 | 0.9563 | 0.9602 |
| Reenactment | one_shot_free | cdf | 0.6739 | 0.7677 | 0.8822 | 0.9029 | 0.8506 | 0.9048 | 0.9096 |
| Reenactment | one_shot_free | ff | 0.8143 | 0.8222 | 0.9069 | 0.9448 | 0.8815 | 0.9432 | 0.9438 |
| Reenactment | pirender | cdf | 0.6198 | 0.7412 | 0.9070 | 0.8672 | 0.8066 | 0.8539 | 0.8652 |
| Reenactment | pirender | ff | 0.8274 | 0.8172 | 0.9286 | 0.9436 | 0.8833 | 0.9248 | 0.9222 |
| Reenactment | sadtalker | cdf | 0.5035 | 0.5437 | 0.4851 | 0.5788 | 0.5402 | 0.8894 | 0.9020 |
| Reenactment | sadtalker | ff | 0.7304 | 0.6064 | 0.6653 | 0.7584 | 0.6071 | 0.9556 | 0.9570 |
| Reenactment | tpsm | cdf | 0.6267 | 0.6508 | 0.4812 | 0.6980 | 0.5097 | 0.9708 | 0.9751 |
| Reenactment | tpsm | ff | 0.7748 | 0.7652 | 0.7757 | 0.9333 | 0.7458 | 0.9688 | 0.9705 |
| Reenactment | wav2lip* | cdf | 0.5757 | 0.6370 | 0.7578 | 0.7772 | 0.8361 | 0.8067 | 0.8056 |
| Reenactment | wav2lip* | ff | 0.8231 | 0.7271 | 0.8776 | 0.9216 | 0.9127 | 0.8908 | 0.8786 |
| Synthesis | DiT | cdf | 0.7490 | 0.8926 | 0.8898 | 0.7230 | 0.8958 | 0.8553 | 0.8759 |
| Synthesis | DiT | ff | 0.9302 | 0.9766 | 0.9610 | 0.9384 | 0.9546 | 0.9359 | 0.9370 |
| Synthesis | SiT | cdf | 0.8260 | 0.9452 | 0.9241 | 0.8383 | 0.9462 | 0.9077 | 0.9235 |
| Synthesis | SiT | ff | 0.9558 | 0.9841 | 0.9732 | 0.9652 | 0.9680 | 0.9613 | 0.9646 |
| Synthesis | StyleGAN2 | cdf | 1.0000 | 0.9971 | 0.9957 | 1.0000 | 1.0000 | 0.9969 | 0.9977 |
| Synthesis | StyleGAN2 | ff | 1.0000 | 0.9991 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGAN3 | cdf | 0.9932 | 0.9965 | 0.9925 | 1.0000 | 1.0000 | 0.9846 | 0.9866 |
| Synthesis | StyleGAN3 | ff | 1.0000 | 0.9991 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGANXL | cdf | 0.9819 | 0.9973 | 0.9957 | 0.9992 | 0.9997 | 0.9687 | 0.9782 |
| Synthesis | StyleGANXL | ff | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | VQGAN | cdf | 0.9594 | 0.9999 | 0.9887 | 0.9999 | 1.0000 | 0.9894 | 0.9929 |
| Synthesis | VQGAN | ff | 0.9981 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | ddim | cdf | 0.6062 | 0.9711 | 0.8629 | 0.8813 | 0.9418 | 0.9208 | 0.9373 |
| Synthesis | ddim | ff | 0.8807 | 0.9857 | 0.9361 | 0.9746 | 0.9746 | 0.9655 | 0.9701 |
| Synthesis | rddm | cdf | 0.6510 | 0.8815 | 0.6399 | 0.9090 | 0.9682 | 0.9429 | 0.9252 |
| Synthesis | rddm | ff | 0.9825 | 0.9858 | 0.8299 | 0.9925 | 0.9893 | 0.9963 | 0.9954 |
| Synthesis | sd2.1 | cdf | 0.9740 | 0.9949 | 0.9974 | 0.9983 | 1.0000 | 0.9828 | 0.9865 |
| Synthesis | sd2.1 | ff | 0.9982 | 0.9989 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 |
| Editing | e4e | cdf | 0.5987 | 0.9894 | 0.9719 | 0.9858 | 0.9822 | 0.9993 | 0.9993 |
| Editing | e4e | ff | 0.8111 | 0.9918 | 0.9797 | 0.9945 | 0.9903 | 0.9948 | 0.9942 |
| Editing | stargan | own | 0.9525 | 0.9851 | 0.9031 | 0.9712 | 0.9902 | 0.9878 | 0.9825 |
| Editing | starganv2 | own | 0.7298 | 0.8747 | 0.6734 | 0.7842 | 0.7758 | 0.7331 | 0.7024 |
| Editing | styleclip | own | 0.6992 | 0.9133 | 0.8834 | 0.9244 | 0.8395 | 0.9398 | 0.9656 |

### Video counts per set (must agree across methods)

| set | DISCERN (ours, Dempster) | DISCERN (ours, PCR6+-R) | DiCoME | Effort | FSFM | ForAda | GenD-CLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| DF40_DiT_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_DiT_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_MRAA_cdf | 827 | 827 | 827 | 827 | 827 | 827 | 827 |
| DF40_MRAA_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_SiT_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_SiT_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGAN2_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGAN2_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGAN3_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGAN3_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_StyleGANXL_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_StyleGANXL_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_VQGAN_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_VQGAN_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_blendface_cdf | 818 | 818 | 818 | 818 | 818 | 818 | 818 |
| DF40_blendface_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_danet_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_danet_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_ddim_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_ddim_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_e4e_cdf | 18828 | 18828 | 18828 | 18828 | 18828 | 18828 | 18828 |
| DF40_e4e_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_e4s_cdf | 470 | 470 | 470 | 470 | 470 | 470 | 470 |
| DF40_e4s_ff | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| DF40_facedancer_cdf | 826 | 826 | 826 | 826 | 826 | 826 | 826 |
| DF40_facedancer_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_faceswap_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_faceswap_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_facevid2vid_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_facevid2vid_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_fomm_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_fomm_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_fsgan_cdf | 794 | 794 | 794 | 794 | 794 | 794 | 794 |
| DF40_fsgan_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_hyperreenact_cdf | 819 | 819 | 819 | 819 | 819 | 819 | 819 |
| DF40_hyperreenact_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_inswap_cdf | 602 | 602 | 602 | 602 | 602 | 602 | 602 |
| DF40_inswap_ff | 258 | 258 | 258 | 258 | 258 | 258 | 258 |
| DF40_lia_cdf | 824 | 824 | 824 | 824 | 824 | 824 | 824 |
| DF40_lia_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_mcnet_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_mcnet_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_mobileswap_cdf | 815 | 815 | 815 | 815 | 815 | 815 | 815 |
| DF40_mobileswap_ff | 998 | 998 | 998 | 998 | 998 | 998 | 998 |
| DF40_one_shot_free_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_one_shot_free_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_pirender_cdf | 814 | 814 | 814 | 814 | 814 | 814 | 814 |
| DF40_pirender_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_rddm_cdf | 622 | 622 | 622 | 622 | 622 | 622 | 622 |
| DF40_rddm_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_sadtalker_cdf | 838 | 838 | 838 | 838 | 838 | 838 | 838 |
| DF40_sadtalker_ff | 278 | 278 | 278 | 278 | 278 | 278 | 278 |
| DF40_sd2.1_cdf | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 | 1066 |
| DF40_sd2.1_ff | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| DF40_simswap_cdf | 819 | 819 | 819 | 819 | 819 | 819 | 819 |
| DF40_simswap_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_stargan | 3968 | 3968 | 3968 | 3968 | 3968 | 3968 | 3968 |
| DF40_starganv2 | 3996 | 3996 | 3996 | 3996 | 3996 | 3996 | 3996 |
| DF40_styleclip | 3824 | 3824 | 3824 | 3824 | 3824 | 3824 | 3824 |
| DF40_tpsm_cdf | 825 | 825 | 825 | 825 | 825 | 825 | 825 |
| DF40_tpsm_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_uniface_cdf | 804 | 804 | 804 | 804 | 804 | 804 | 804 |
| DF40_uniface_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |
| DF40_wav2lip_cdf | 844 | 844 | 844 | 844 | 844 | 844 | 844 |
| DF40_wav2lip_ff | 280 | 280 | 280 | 280 | 280 | 280 | 280 |

**Count mismatches:** none


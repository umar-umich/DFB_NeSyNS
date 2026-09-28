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
lip-sync; it is included here and flagged in the per-generator appendix. And the three standalone
editors (`stargan`, `starganv2`, `styleclip`) ship their own paired reals instead of DF40's shared
real sets, so they are reported as their own column and are never averaged into FE.

**The `-ff` half is included.** DF40's FF++-domain sets are built over the 1000 FaceForensics++
source videos, 720 of which are in our training split. Those halves are therefore not
identity-disjoint from training for ANY method in the table -- every row here trains on FF++ c23, so
the advantage is shared rather than ours. They are included because excluding them would drop half
of DF40 and break comparability with published DF40 numbers; `-cdf` and `-ff` are reported
separately in the appendix so the two can always be read apart.

**Image-level sets.** DF40's EFS, FE and standalone sets are image collections rather than videos:
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

### DF40 video-level AUROC — mean ± sd across the generators in each family

| Method | Face swap (18) | Reenactment (22) | Macro |
|---|---:|---:|---:|
| DISCERN (ours) | 0.968 ± 0.024 | 0.834 ± 0.143 | 0.901 |
| DiCoME (our training) | 0.970 ± 0.019 | 0.779 ± 0.164 | 0.875 |
| DiCoME (released weights) | 0.975 ± 0.017 | 0.809 ± 0.157 | 0.892 |

### DF40 frame-level AUROC — mean ± sd across the generators in each family

| Method | Face swap (18) | Reenactment (22) | Macro |
|---|---:|---:|---:|
| DISCERN (ours) | 0.939 ± 0.035 | 0.808 ± 0.135 | 0.874 |
| DiCoME (our training) | 0.938 ± 0.033 | 0.750 ± 0.152 | 0.844 |
| DiCoME (released weights) | 0.941 ± 0.033 | 0.772 ± 0.143 | 0.856 |

### DF40 video-level AUROC per generator

`*` marks audio-driven lip-sync, which DF40 files under reenactment and our v3.1 objective excludes; it is kept here because the taxonomy is DF40's.

| family | generator | real | DISCERN (ours) | DiCoME (our training) | DiCoME (released weights) |
|---|---|---|---:|---:|---:|
| Face swap | blendface | cdf | 0.9599 | 0.9502 | 0.9552 |
| Face swap | blendface | ff | 0.9576 | 0.9505 | 0.9573 |
| Face swap | e4s | cdf | 0.9936 | 0.9949 | 0.9953 |
| Face swap | e4s | ff | 0.9891 | 0.9891 | 0.9922 |
| Face swap | facedancer | cdf | 0.9551 | 0.9657 | 0.9781 |
| Face swap | facedancer | ff | 0.9720 | 0.9632 | 0.9720 |
| Face swap | faceswap | cdf | 0.9995 | 0.9991 | 0.9998 |
| Face swap | faceswap | ff | 0.9823 | 0.9795 | 0.9807 |
| Face swap | fsgan | cdf | 0.9935 | 0.9892 | 0.9947 |
| Face swap | fsgan | ff | 0.9878 | 0.9879 | 0.9902 |
| Face swap | inswap | cdf | 0.9358 | 0.9433 | 0.9529 |
| Face swap | inswap | ff | 0.9502 | 0.9696 | 0.9709 |
| Face swap | mobileswap | cdf | 0.9624 | 0.9571 | 0.9582 |
| Face swap | mobileswap | ff | 0.9832 | 0.9745 | 0.9785 |
| Face swap | simswap | cdf | 0.9103 | 0.9339 | 0.9392 |
| Face swap | simswap | ff | 0.9368 | 0.9539 | 0.9659 |
| Face swap | uniface | cdf | 0.9676 | 0.9758 | 0.9810 |
| Face swap | uniface | ff | 0.9792 | 0.9803 | 0.9831 |
| Reenactment | MRAA | cdf | 0.9887 | 0.9770 | 0.9857 |
| Reenactment | MRAA | ff | 0.9972 | 0.9937 | 0.9949 |
| Reenactment | danet | cdf | 0.6467 | 0.4821 | 0.5131 |
| Reenactment | danet | ff | 0.7469 | 0.6764 | 0.7183 |
| Reenactment | facevid2vid | cdf | 0.7885 | 0.6748 | 0.7410 |
| Reenactment | facevid2vid | ff | 0.7971 | 0.7776 | 0.8264 |
| Reenactment | fomm | cdf | 0.9393 | 0.8793 | 0.9086 |
| Reenactment | fomm | ff | 0.9670 | 0.9609 | 0.9684 |
| Reenactment | hyperreenact | cdf | 0.8690 | 0.7421 | 0.8123 |
| Reenactment | hyperreenact | ff | 0.9279 | 0.8483 | 0.8899 |
| Reenactment | lia | cdf | 0.9789 | 0.9854 | 0.9823 |
| Reenactment | lia | ff | 0.9920 | 0.9921 | 0.9894 |
| Reenactment | mcnet | cdf | 0.6951 | 0.5594 | 0.6049 |
| Reenactment | mcnet | ff | 0.8045 | 0.7680 | 0.8026 |
| Reenactment | one_shot_free | cdf | 0.9354 | 0.8506 | 0.8889 |
| Reenactment | one_shot_free | ff | 0.9179 | 0.8815 | 0.9080 |
| Reenactment | pirender | cdf | 0.9120 | 0.8066 | 0.8574 |
| Reenactment | pirender | ff | 0.9204 | 0.8833 | 0.9057 |
| Reenactment | sadtalker | cdf | 0.5707 | 0.5402 | 0.5251 |
| Reenactment | sadtalker | ff | 0.6391 | 0.6071 | 0.6193 |
| Reenactment | tpsm | cdf | 0.5409 | 0.5097 | 0.5705 |
| Reenactment | tpsm | ff | 0.7696 | 0.7458 | 0.7791 |

### Video counts per set (must agree across methods)

| set | DISCERN (ours) | DiCoME (our training) | DiCoME (released weights) |
|---|---:|---:|---:|
| DF40_MRAA_cdf | 827 | 827 | 827 |
| DF40_MRAA_ff | 280 | 280 | 280 |
| DF40_blendface_cdf | 818 | 818 | 818 |
| DF40_blendface_ff | 280 | 280 | 280 |
| DF40_danet_cdf | 825 | 825 | 825 |
| DF40_danet_ff | 280 | 280 | 280 |
| DF40_e4s_cdf | 470 | 470 | 470 |
| DF40_e4s_ff | 255 | 255 | 255 |
| DF40_facedancer_cdf | 826 | 826 | 826 |
| DF40_facedancer_ff | 280 | 280 | 280 |
| DF40_faceswap_cdf | 815 | 815 | 815 |
| DF40_faceswap_ff | 280 | 280 | 280 |
| DF40_facevid2vid_cdf | 815 | 815 | 815 |
| DF40_facevid2vid_ff | 280 | 280 | 280 |
| DF40_fomm_cdf | 825 | 825 | 825 |
| DF40_fomm_ff | 280 | 280 | 280 |
| DF40_fsgan_cdf | 794 | 794 | 794 |
| DF40_fsgan_ff | 280 | 280 | 280 |
| DF40_hyperreenact_cdf | 819 | 819 | 819 |
| DF40_hyperreenact_ff | 280 | 280 | 280 |
| DF40_inswap_cdf | 602 | 602 | 602 |
| DF40_inswap_ff | 258 | 258 | 258 |
| DF40_lia_cdf | 824 | 824 | 824 |
| DF40_lia_ff | 280 | 280 | 280 |
| DF40_mcnet_cdf | 825 | 825 | 825 |
| DF40_mcnet_ff | 280 | 280 | 280 |
| DF40_mobileswap_cdf | 815 | 815 | 815 |
| DF40_mobileswap_ff | 998 | 998 | 998 |
| DF40_one_shot_free_cdf | 825 | 825 | 825 |
| DF40_one_shot_free_ff | 280 | 280 | 280 |
| DF40_pirender_cdf | 814 | 814 | 814 |
| DF40_pirender_ff | 280 | 280 | 280 |
| DF40_sadtalker_cdf | 838 | 838 | 838 |
| DF40_sadtalker_ff | 278 | 278 | 278 |
| DF40_simswap_cdf | 819 | 819 | 819 |
| DF40_simswap_ff | 280 | 280 | 280 |
| DF40_tpsm_cdf | 825 | 825 | 825 |
| DF40_tpsm_ff | 280 | 280 | 280 |
| DF40_uniface_cdf | 804 | 804 | 804 |
| DF40_uniface_ff | 280 | 280 | 280 |

**Count mismatches:** none


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
| Reenactment | wav2lip* | cdf | 0.8347 | 0.8361 | 0.8414 |
| Reenactment | wav2lip* | ff | 0.9025 | 0.9127 | 0.9178 |
| Synthesis | DiT | cdf | 0.9328 | 0.8958 | 0.9498 |
| Synthesis | DiT | ff | 0.9663 | 0.9546 | 0.9615 |
| Synthesis | SiT | cdf | 0.9675 | 0.9462 | 0.9827 |
| Synthesis | SiT | ff | 0.9813 | 0.9680 | 0.9784 |
| Synthesis | StyleGAN2 | cdf | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGAN2 | ff | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGAN3 | cdf | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGAN3 | ff | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | StyleGANXL | cdf | 1.0000 | 0.9997 | 0.9976 |
| Synthesis | StyleGANXL | ff | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | VQGAN | cdf | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | VQGAN | ff | 1.0000 | 1.0000 | 1.0000 |
| Synthesis | ddim | cdf | 0.9719 | 0.9418 | 0.9931 |
| Synthesis | ddim | ff | 0.9903 | 0.9746 | 0.9930 |
| Synthesis | rddm | cdf | 0.9558 | 0.9682 | 0.9944 |
| Synthesis | rddm | ff | 0.9873 | 0.9893 | 0.9970 |
| Synthesis | sd2.1 | cdf | 0.9996 | 1.0000 | 1.0000 |
| Synthesis | sd2.1 | ff | 1.0000 | 1.0000 | 1.0000 |
| Editing | e4e | cdf | 0.9826 | 0.9822 | 0.9928 |
| Editing | e4e | ff | 0.9915 | 0.9903 | 0.9973 |
| Editing (standalone) | stargan | own | 0.9904 | 0.9902 | 0.9946 |
| Editing (standalone) | starganv2 | own | 0.7790 | 0.7758 | 0.8229 |
| Editing (standalone) | styleclip | own | 0.8953 | 0.8395 | 0.9057 |

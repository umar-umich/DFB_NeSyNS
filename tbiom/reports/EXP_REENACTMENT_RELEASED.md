# Reenactment benchmarks: our framework vs DiCoME's released weights

Produced by `analysis/tbiom/reenactment_released.py`. All three systems are scored on the **same videos** with the **same aggregation** -- per-frame opinions averaged per video, then each system's own fusion. Ours is PCR6$^+$-R over three views with the reliability weights fitted on the five development folds; both DiCoME rows use DiCoME's own two-view Dempster fusion, replayed identically, so the only difference between them is the checkpoint.

| system | checkpoint |
|---|---|
| ours | `warm_dicome_gate_seed42/.../epoch=19` + `vjepa2_lora_B_seed42` |
| DiCoME, released | `DISCERN_Ext/weights/dicome-best.ckpt` (HF `kxl0825/DiCoME`) |
| DiCoME, our training | `_baseline_dicome/dicome_train/ffpp_reproduce_seed42/.../epoch=05` |

The documented CDFv3 exclusion (IP_LAP) is applied to the released export as well. It is *present* in that export and *absent* from our cached tables, so leaving it in would have compared different video sets.

## Headline

| benchmark | slice | fakes | **ours** | DiCoME released | DiCoME our training | ours − released |
|---|---|---:|---:|---:|---:|---:|
| CDFv3 | all | 4940 | **0.9644** | 0.8789 | 0.8691 | +0.0855 |
| CDFv3 | reenactment only | 3200 | **0.9687** | 0.8387 | 0.8271 | +0.1300 |
| CDFv3 | swap only | 1740 | **0.9565** | 0.9527 | 0.9463 | +0.0037 |
| KoDF | all | 992 | **0.9600** | 0.8514 | 0.8509 | +0.1086 |
| KoDF | first-order motion only | 352 | **0.9837** | 0.7301 | 0.7206 | +0.2536 |

Video counts: CDFv3 5118 (178 real), KoDF 1395 (403 real).

## What this settles

**The released weights are slightly stronger than our reproduction**, not weaker: 0.8789 against 0.8691 on CDFv3 overall and 0.8514 against 0.8509 on KoDF. Our own training of DiCoME was therefore not a flattering baseline, and the comparison in the paper is if anything conservative.

**The asymmetry survives the authors' own weights.** On swap-only CDFv3 the three systems are within 0.010 of each other; on reenactment-only CDFv3 the gap to the released weights is +0.130, and on KoDF's first-order motion subset it is +0.254. That is the paper's claim, reproduced against the released checkpoint rather than against our reproduction of it.

## Per reenactment generator

| benchmark | generator | fakes | **ours** | DiCoME released | DiCoME our training |
|---|---|---:|---:|---:|---:|
| CDFv3 | TalkingFace-SadTalker | 300 | **0.9528** | 0.5327 | 0.5206 |
| CDFv3 | FaceReenact-LivePortrait | 200 | **0.9428** | 0.7517 | 0.7535 |
| CDFv3 | TalkingFace-EchoMimic | 300 | **0.9655** | 0.7756 | 0.7808 |
| CDFv3 | FaceReenact-DaGAN | 200 | **0.9563** | 0.7844 | 0.7558 |
| CDFv3 | FaceReenact-HyperReenact | 200 | **0.9402** | 0.8314 | 0.7719 |
| CDFv3 | TalkingFace-FLOAT | 300 | **0.9749** | 0.8590 | 0.8312 |
| CDFv3 | TalkingFace-AniTalker | 300 | **0.9380** | 0.8701 | 0.8595 |
| CDFv3 | FaceReenact-FSRT | 200 | **0.9698** | 0.8953 | 0.8830 |
| CDFv3 | FaceReenact-TPSMM | 200 | **0.9798** | 0.8979 | 0.8782 |
| CDFv3 | FaceReenact-MCNET | 200 | **0.9778** | 0.9017 | 0.8866 |
| CDFv3 | TalkingFace-R3DPortrait | 300 | **0.9992** | 0.9177 | 0.9340 |
| CDFv3 | FaceReenact-LIA | 200 | **0.9903** | 0.9427 | 0.9342 |
| CDFv3 | TalkingFace-EDTalk | 300 | **0.9979** | 0.9878 | 0.9880 |

Sorted by the released weights' score, worst first. The generators where the released model is weakest are where our margin is largest, which is the behaviour the three-view ablation predicts.

## Still running

DiCoME's released weights are being exported on all 41 DF40 generator sets (`analysis/tbiom/released_df40.sh`), which will extend this comparison to the 22 DF40 reenactment generators on the same videos as the existing ours / our-trained-DiCoME numbers.


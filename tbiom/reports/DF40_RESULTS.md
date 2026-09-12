# DF40 results — the frozen arm on 21 face-manipulation generators

Frozen configuration (arm B, three-view, DS fusion, per-branch EDL, seed 42). `tau` is its EER on its own FF++ val (0.3971); probabilities are the DS replay of the three branch opinions — the same convention as every other table in the writeup.

**FS and FR are reported as one dataset.** Our six existing OOD sets contain face swaps and reenactments only, so FS + FR is the like-for-like seventh dataset, with 21 generators against the handful the current sets cover. EFS (synthesis from no source face) is a different question and stays separate.

| dataset | videos | real / fake | AUROC | FPR_real @ τ |
|---|---:|---|---:|---:|
| DF40-FSFR (21 methods) | 13,109 | 178 / 12,931 | **0.8691** | 0.084 |
| DF40-EFS | 7,726 | 178 / 7,548 | **0.9795** | 0.084 |
| DF40-FE | 18,828 | 178 / 18,650 | **0.9710** | 0.084 |

## Integrity check on the shared real pool

DF40 ships one real pool for all families, so the FS and FR exports both contained it: 356 real rows collapsing to 178 unique videos. Reals are deduplicated on video id — counting them twice would not move AUROC much but would halve the effective real:fake ratio and shift FPR_real.

The duplication doubles as a determinism check: the same real video, same checkpoint, two separate export runs. Maximum disagreement in `p_fused` is **0.00e+00** — consistent, so the export is deterministic and the two runs agree.

## Per-generator breakdown

Within the combined set, so a family average cannot hide one generator the framework simply cannot see.

| generator | videos | AUROC |
|---|---:|---:|
| blendface-cdf | 640 | 0.9575 |
| e4s-cdf | 292 | 0.9917 |
| facedancer-cdf | 648 | 0.9634 |
| faceswap-cdf | 637 | 0.9995 |
| fsgan-cdf | 616 | 0.9938 |
| inswap-cdf | 424 | 0.9310 |
| mobileswap-cdf | 637 | 0.9630 |
| simswap-cdf | 641 | 0.9031 |
| uniface-cdf | 626 | 0.9688 |
| MRAA-cdf | 649 | 0.9895 |
| danet-cdf | 647 | 0.6467 |
| facevid2vid-cdf | 637 | 0.8009 |
| fomm-cdf | 647 | 0.9410 |
| hyperreenact-cdf | 641 | 0.8635 |
| lia-cdf | 646 | 0.9762 |
| mcnet-cdf | 647 | 0.7038 |
| one_shot_free-cdf | 647 | 0.9415 |
| pirender-cdf | 636 | 0.9133 |
| sadtalker-cdf | 660 | 0.5515 |
| tpsm-cdf | 647 | 0.5419 |
| wav2lip-cdf | 666 | 0.8199 |

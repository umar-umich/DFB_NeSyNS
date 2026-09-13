# Evaluation manifests — CDFv3 and DF40 (2026-09-13)

Written because generator-level reenactment results were reported under **two different datasets**:
`DF40_RESULTS.md` (DF40 `-cdf` subsets) and the CDFv3 per-subset analysis in
`decision-2026-09-13.md` (CDFv3 `synthesis-*` subsets). They are separate test sets with separate
fake videos; several generator **names** occur in both. Every number must name its dataset and subset.

Video id = the key's parent folder. Label: `Celeb-real`, `YouTube-real`, `*-real` → 0; everything else → 1.

## Celeb-DF v3 (`splits/CDFv3/`, h5 `CDFv3.h5`)

| subset | label | videos | frames | split file sha256[:16] |
|---|---:|---:|---:|---|
| Celeb-real | 0 | 108 | 3,452 | `90e8646d56a4bcd7` |
| YouTube-real | 0 | 70 | 2,240 | `73ee26b8aa4c57fc` |
| synthesis-FaceReenact-DaGAN | 1 | 200 | 6,385 | `bcbbc4450b39063f` |
| synthesis-FaceReenact-FSRT | 1 | 200 | 6,391 | `cc9eaf2ca22d9d9f` |
| synthesis-FaceReenact-HyperReenact | 1 | 200 | 6,400 | `bef4f7c368a5b10e` |
| synthesis-FaceReenact-LIA | 1 | 200 | 6,400 | `8449a39c9613f687` |
| synthesis-FaceReenact-LivePortrait | 1 | 200 | 6,367 | `bdc7b45f4b563aaa` |
| synthesis-FaceReenact-MCNET | 1 | 200 | 6,394 | `e51c8af7977adb1c` |
| synthesis-FaceReenact-TPSMM | 1 | 200 | 6,400 | `a647a60785f524e6` |
| synthesis-FaceSwap-BlendFace | 1 | 200 | 6,398 | `a8e6233c14c431b7` |
| synthesis-FaceSwap-Celeb-DF-v2 | 1 | 340 | 10,880 | `3b109313deb30198` |
| synthesis-FaceSwap-GHOST | 1 | 200 | 6,398 | `fa7c8ee7d64bac3a` |
| synthesis-FaceSwap-HifiFace | 1 | 200 | 6,398 | `1ffb9cbe39e08349` |
| synthesis-FaceSwap-InSwapper | 1 | 200 | 6,398 | `fa5de918b57dfb69` |
| synthesis-FaceSwap-MobileFaceSwap | 1 | 200 | 6,398 | `0c38eb15e3048304` |
| synthesis-FaceSwap-SimSwap | 1 | 200 | 6,398 | `3bee777d545f8208` |
| synthesis-FaceSwap-UniFace | 1 | 200 | 6,398 | `aa54b9bdc4afed26` |
| synthesis-TalkingFace-AniTalker | 1 | 300 | 9,600 | `ee501c4ff7e6d6dc` |
| synthesis-TalkingFace-EDTalk | 1 | 300 | 9,600 | `d3dc91e30c90f24a` |
| synthesis-TalkingFace-EchoMimic | 1 | 300 | 9,600 | `345e0ddff10a3a35` |
| synthesis-TalkingFace-FLOAT | 1 | 300 | 9,600 | `79163006f8bd870a` |
| synthesis-TalkingFace-IP_LAP | 1 | 300 | 9,600 | `a34928869e6f3e6f` |
| synthesis-TalkingFace-R3DPortrait | 1 | 300 | 9,600 | `b2f1262cfaf5c903` |
| synthesis-TalkingFace-SadTalker | 1 | 300 | 9,600 | `c1bd699ef2e53e24` |

Genuine pool: **178 videos** (Celeb-real + YouTube-real).

## DF40 `-cdf` families (`splits/DF40/`, h5 `DF40.h5`)

| family | subset | label | videos | frames | split file sha256[:16] |
|---|---|---:|---:|---:|---|
| real pool | DF40-cdf-real | 0 | 178 | 5,620 | `42e1d4b3dc116aa1` |
| DF40_FS | blendface-cdf | 1 | 640 | 20,323 | `f41dd4646593e5f8` |
| DF40_FS | e4s-cdf | 1 | 292 | 9,198 | `462fb7a7dfaf17d6` |
| DF40_FS | facedancer-cdf | 1 | 648 | 20,252 | `33f3f73dbdf795f5` |
| DF40_FS | faceswap-cdf | 1 | 637 | 20,173 | `46c73295ebe23433` |
| DF40_FS | fsgan-cdf | 1 | 616 | 19,601 | `39353d1f5cc7d417` |
| DF40_FS | inswap-cdf | 1 | 424 | 13,433 | `da1e912703ecf591` |
| DF40_FS | mobileswap-cdf | 1 | 637 | 20,243 | `8c66ceafa21c9bea` |
| DF40_FS | simswap-cdf | 1 | 641 | 20,322 | `fe86aa15c179a0ac` |
| DF40_FS | uniface-cdf | 1 | 626 | 19,881 | `93c12a6e7e63a17b` |
| DF40_FR | danet-cdf | 1 | 647 | 19,284 | `78aa9747b19d686e` |
| DF40_FR | facevid2vid-cdf | 1 | 637 | 19,890 | `b3534459943f7aba` |
| DF40_FR | fomm-cdf | 1 | 647 | 20,278 | `84ff7ae33459767e` |
| DF40_FR | hyperreenact-cdf | 1 | 641 | 20,503 | `3e2a71acba06a539` |
| DF40_FR | lia-cdf | 1 | 646 | 19,806 | `135f9b220d7491c1` |
| DF40_FR | mcnet-cdf | 1 | 647 | 19,201 | `8a82bf024b0b7e32` |
| DF40_FR | MRAA-cdf | 1 | 649 | 20,273 | `d4468f55829d2f87` |
| DF40_FR | one_shot_free-cdf | 1 | 647 | 20,625 | `bda53faab497fef3` |
| DF40_FR | pirender-cdf | 1 | 636 | 20,185 | `8b27e505eb85588f` |
| DF40_FR | sadtalker-cdf | 1 | 660 | 20,986 | `5007ce8caa2d26a6` |
| DF40_FR | tpsm-cdf | 1 | 647 | 20,587 | `7d7ea48a96143e1e` |
| DF40_FR | wav2lip-cdf | 1 | 666 | 20,814 | `1437fba0be79a657` |
| DF40_EFS | ddim-cdf | 1 | 888 | 28,174 | `af41f82ea3f439e0` |
| DF40_EFS | DiT-cdf | 1 | 888 | 28,174 | `3bf309e4e41e7e31` |
| DF40_EFS | rddm-cdf | 1 | 444 | 14,027 | `5c6353d6804c9339` |
| DF40_EFS | sd2.1-cdf | 1 | 888 | 28,174 | `fe2ffd9b2d34045c` |
| DF40_EFS | SiT-cdf | 1 | 888 | 28,174 | `caeed7fa110b4820` |
| DF40_EFS | StyleGAN2-cdf | 1 | 888 | 28,174 | `3ebe2cac4c8923d0` |
| DF40_EFS | StyleGAN3-cdf | 1 | 888 | 28,174 | `818a460e7560d1b1` |
| DF40_EFS | StyleGANXL-cdf | 1 | 888 | 28,174 | `ae415d05b2268de1` |
| DF40_EFS | VQGAN-cdf | 1 | 888 | 28,174 | `222ca2ae791d6352` |
| DF40_FE | e4e-cdf | 1 | 18650 | 18,650 | `d24b137e6316bdac` |

Genuine pool: **178 videos** (`DF40-cdf-real`), shared by every DF40 family.

## Generator names that occur in both datasets

Same method name, **different videos, different real pools, different reports**:

| CDFv3 subset | DF40 subset |
|---|---|
| synthesis-FaceReenact-HyperReenact | hyperreenact-cdf |
| synthesis-FaceReenact-LIA | lia-cdf |
| synthesis-FaceReenact-MCNET | mcnet-cdf |
| synthesis-FaceReenact-TPSMM | tpsm-cdf |
| synthesis-FaceSwap-BlendFace | blendface-cdf |
| synthesis-FaceSwap-InSwapper | inswap-cdf |
| synthesis-FaceSwap-MobileFaceSwap | mobileswap-cdf |
| synthesis-FaceSwap-SimSwap | simswap-cdf |
| synthesis-FaceSwap-UniFace | uniface-cdf |
| synthesis-TalkingFace-SadTalker | sadtalker-cdf |

## Genuine-pool relation

CDFv3 genuine videos: 178; DF40 `-cdf` genuine videos: 178.
DF40 re-indexes file names, so identity of the two pools cannot be confirmed from keys alone; both are 178 videos and DF40 `-cdf` is built from Celeb-DF sources, so they should be treated as **possibly the same genuine videos** until verified by pixel matching. Results on the two datasets are therefore not assumed independent.

## Where each reported number came from

| report | dataset | subsets | genuine pool |
|---|---|---|---|
| `reports/DF40_RESULTS.md` (frozen R0) | DF40 `-cdf` | FS 9, FR 12, EFS 9, FE 1 | DF40-cdf-real (178) |
| `docs/decision-2026-09-13.md`, CDFv3 per-subset (R0, A, B, C) | Celeb-DF v3 | 22 synthesis subsets | Celeb-real + YouTube-real (178) |
| all six-dataset OOD tables | CDFv3 as one set | all CDFv3 subsets pooled | same 178 |

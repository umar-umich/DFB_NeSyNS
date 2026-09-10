# Decision traces — frozen detector, symbolic overlay

Sampled at random within each rule, across all six OOD test sets, so the set is not selected for the traces that read best. Thresholds are the dev-fitted ones in SYMBOLIC_OVERLAY.md.

### CDFv2 · video `CDFv2/Celeb-synthesis/id6_id16_0005`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.356 | 0.056 |
| Off-manifold residual | fake (decisive) | +0.464 | 0.047 |
| Bona-fide face prior | fake (marginal) | +0.396 | 0.074 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id38_id33_0005`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.347 | 0.073 |
| Off-manifold residual | fake (decisive) | +0.452 | 0.070 |
| Bona-fide face prior | fake (marginal) | +0.446 | 0.023 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-real/id39_0005`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.135, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.242 | 0.284 |
| Off-manifold residual | real (marginal) | -0.179 | 0.306 |
| Bona-fide face prior | real (marginal) | -0.287 | 0.264 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-real/id36_0001`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.554 | 0.124 |
| Off-manifold residual | real (decisive) | -0.439 | 0.146 |
| Bona-fide face prior | real (decisive) | -0.525 | 0.028 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id10_id12_0004`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.995, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.349 | 0.069 |
| Off-manifold residual | fake (decisive) | +0.454 | 0.067 |
| Bona-fide face prior | real (marginal) | -0.129 | 0.258 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.928, decision weakens by 0.067

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id12_id10_0002`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.355 | 0.058 |
| Off-manifold residual | fake (decisive) | +0.462 | 0.050 |
| Bona-fide face prior | real (marginal) | -0.013 | 0.226 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.958, decision UNCHANGED

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id56_id57_0007`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.854, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.213 | 0.256 |
| Off-manifold residual | fake (marginal) | +0.295 | 0.270 |
| Bona-fide face prior | real (marginal) | -0.197 | 0.201 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id38_id34_0004`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.614, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.098 | 0.241 |
| Off-manifold residual | fake (marginal) | +0.156 | 0.234 |
| Bona-fide face prior | real (marginal) | -0.288 | 0.264 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id31_id2_0004`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.305 | 0.120 |
| Off-manifold residual | fake (marginal) | +0.404 | 0.122 |
| Bona-fide face prior | fake (marginal) | +0.442 | 0.030 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id35_id32_0007`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.753, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.015 | 0.422 |
| Off-manifold residual | fake (marginal) | +0.066 | 0.414 |
| Bona-fide face prior | fake (marginal) | +0.095 | 0.282 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-IP_LAP/IP_LAP__id20_0007_test_id04657_k03gXUbmDPM`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.308 | 0.130 |
| Off-manifold residual | fake (marginal) | +0.404 | 0.122 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EDTalk/EDTalk__id23_0009_test_id08456_gxRgzwlKUIY`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.357 | 0.054 |
| Off-manifold residual | fake (decisive) | +0.465 | 0.045 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-LivePortrait/LivePortrait__id34_id35_0006`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.025, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.306 | 0.367 |
| Off-manifold residual | real (marginal) | -0.231 | 0.333 |
| Bona-fide face prior | real (decisive) | -0.476 | 0.118 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-LivePortrait/LivePortrait__id16_id21_0009`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.003, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.507 | 0.210 |
| Off-manifold residual | real (decisive) | -0.425 | 0.172 |
| Bona-fide face prior | real (decisive) | -0.453 | 0.046 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-FSRT/FSRT__id21_id4_0006`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.990, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.333 | 0.100 |
| Off-manifold residual | fake (decisive) | +0.434 | 0.105 |
| Bona-fide face prior | real (marginal) | -0.068 | 0.219 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.914, decision weakens by 0.076

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-TPSMM/TPSMM__id42_id39_0001`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.920, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.315 | 0.133 |
| Off-manifold residual | fake (decisive) | +0.416 | 0.135 |
| Bona-fide face prior | real (marginal) | -0.418 | 0.145 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.605, decision weakens by 0.315

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-HyperReenact/HyperReenact__id43_id45_0003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.967, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.293 | 0.345 |
| Off-manifold residual | real (marginal) | -0.221 | 0.312 |
| Bona-fide face prior | fake (decisive) | +0.455 | 0.011 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.199, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-GHOST/GHOST__id1_id3_0006`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.642, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.530 | 0.161 |
| Off-manifold residual | real (decisive) | -0.420 | 0.178 |
| Bona-fide face prior | fake (decisive) | +0.452 | 0.017 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.016, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-InSwapper/InSwapper__id47_id40_0000`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.048, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.064 | 0.374 |
| Off-manifold residual | fake (marginal) | +0.007 | 0.350 |
| Bona-fide face prior | real (decisive) | -0.516 | 0.046 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-UniFace/UniFace__id54_id50_0004`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.718, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.131 | 0.285 |
| Off-manifold residual | fake (marginal) | +0.211 | 0.262 |
| Bona-fide face prior | real (marginal) | -0.242 | 0.271 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EDTalk/EDTalk__id29_0009_test_id02548_ECjj4AL9Njw`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.966, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.299 | 0.166 |
| Off-manifold residual | fake (marginal) | +0.385 | 0.200 |
| Bona-fide face prior | real (marginal) | -0.118 | 0.286 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-SimSwap/SimSwap__id43_id45_0003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.820, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.074 | 0.461 |
| Off-manifold residual | fake (marginal) | +0.109 | 0.483 |
| Bona-fide face prior | fake (marginal) | +0.117 | 0.184 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/23_24__walking_down_street_outside_angry__YR5OVD4S`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.317 | 0.084 |
| Off-manifold residual | fake (decisive) | +0.423 | 0.077 |
| Bona-fide face prior | fake (marginal) | +0.432 | 0.031 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/18_03__podium_speech_happy__UEC1BI8E`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.989, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.321 | 0.122 |
| Off-manifold residual | fake (decisive) | +0.420 | 0.128 |
| Bona-fide face prior | fake (marginal) | +0.034 | 0.089 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DFD-real/19__podium_speech_happy`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.132, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.189 | 0.322 |
| Off-manifold residual | real (marginal) | -0.108 | 0.342 |
| Bona-fide face prior | real (marginal) | -0.353 | 0.087 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DFD-real/16__walking_down_indoor_hall_disgust`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.030, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.430 | 0.227 |
| Off-manifold residual | real (marginal) | -0.340 | 0.241 |
| Bona-fide face prior | real (marginal) | -0.311 | 0.130 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/01_04__exit_phone_room__0XUW13RW`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.989, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.335 | 0.095 |
| Off-manifold residual | fake (decisive) | +0.436 | 0.097 |
| Bona-fide face prior | real (marginal) | -0.101 | 0.122 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.900, decision weakens by 0.089

---

### DFD · video `DFD/DeepFakeDetection/02_07__talking_against_wall__1ZE4HC06`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.997, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.355 | 0.058 |
| Off-manifold residual | fake (decisive) | +0.464 | 0.047 |
| Bona-fide face prior | real (marginal) | -0.067 | 0.193 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.948, decision UNCHANGED

---

### DFD · video `DFD/DeepFakeDetection/25_12__walking_outside_cafe_disgusted__G26LG5N7`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.992, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.150 | 0.242 |
| Off-manifold residual | real (marginal) | -0.143 | 0.321 |
| Bona-fide face prior | fake (decisive) | +0.458 | 0.005 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.363, decision FLIPS

---

### DFD · video `DFD/DeepFakeDetection/26_11__outside_talking_still_laughing__ALJ0OHNM`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.985, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.045 | 0.216 |
| Off-manifold residual | real (marginal) | -0.014 | 0.271 |
| Bona-fide face prior | fake (decisive) | +0.450 | 0.019 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.562, decision FLIPS

---

### DFD · video `DFD/DeepFakeDetection/10_22__walk_down_hall_angry__C1U7OY5M`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.760, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.198 | 0.275 |
| Off-manifold residual | fake (marginal) | +0.284 | 0.269 |
| Bona-fide face prior | real (marginal) | -0.302 | 0.155 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/14_07__secret_conversation__P9QFO50U`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.253, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.442 | 0.325 |
| Off-manifold residual | real (marginal) | -0.362 | 0.292 |
| Bona-fide face prior | fake (marginal) | +0.319 | 0.262 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/21_02__kitchen_pan__Z0XHPQAR`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.971, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.160 | 0.384 |
| Off-manifold residual | fake (marginal) | +0.209 | 0.431 |
| Bona-fide face prior | fake (marginal) | +0.351 | 0.199 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/18_14__podium_speech_happy__0YRBHIKG`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.991, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.314 | 0.126 |
| Off-manifold residual | fake (marginal) | +0.412 | 0.129 |
| Bona-fide face prior | fake (marginal) | +0.128 | 0.101 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/xlrrixwlax`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.353 | 0.061 |
| Off-manifold residual | fake (decisive) | +0.463 | 0.049 |
| Bona-fide face prior | fake (decisive) | +0.458 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/ruwgqzaxyd`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.357 | 0.053 |
| Off-manifold residual | fake (decisive) | +0.464 | 0.046 |
| Bona-fide face prior | fake (marginal) | +0.303 | 0.136 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/eykkgxsaef`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.003, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.491 | 0.187 |
| Off-manifold residual | real (marginal) | -0.390 | 0.200 |
| Bona-fide face prior | real (decisive) | -0.492 | 0.064 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/drqqfwbmeq`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.007, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.519 | 0.181 |
| Off-manifold residual | real (marginal) | -0.408 | 0.193 |
| Bona-fide face prior | real (marginal) | -0.366 | 0.184 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/aaoqanfmgd`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.984, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.355 | 0.057 |
| Off-manifold residual | fake (decisive) | +0.465 | 0.046 |
| Bona-fide face prior | real (marginal) | -0.444 | 0.096 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.732, decision weakens by 0.252

---

### DFDC · video `DFDC/DFDC-fake/bcjatefpva`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.358 | 0.052 |
| Off-manifold residual | fake (decisive) | +0.464 | 0.046 |
| Bona-fide face prior | real (marginal) | -0.080 | 0.264 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.954, decision UNCHANGED

---

### DFDC · video `DFDC/DFDC-fake/ofdkfmtnaj`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.803, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.475 | 0.235 |
| Off-manifold residual | real (marginal) | -0.373 | 0.242 |
| Bona-fide face prior | fake (decisive) | +0.452 | 0.017 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.040, decision FLIPS

---

### DFDC · video `DFDC/DFDC-real/wxemonzdds`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.991, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.033 | 0.171 |
| Off-manifold residual | real (marginal) | -0.051 | 0.165 |
| Bona-fide face prior | fake (decisive) | +0.455 | 0.012 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.541, decision FLIPS

---

### DFDC · video `DFDC/DFDC-real/eoacfrppkd`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.200, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.375 | 0.263 |
| Off-manifold residual | real (marginal) | -0.289 | 0.271 |
| Bona-fide face prior | fake (marginal) | +0.097 | 0.182 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/jefoqtmqis`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.656, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.108 | 0.314 |
| Off-manifold residual | fake (marginal) | +0.187 | 0.324 |
| Bona-fide face prior | real (marginal) | -0.229 | 0.114 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/euugqjlsoq`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.227, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.055 | 0.237 |
| Off-manifold residual | fake (marginal) | +0.043 | 0.236 |
| Bona-fide face prior | real (marginal) | -0.405 | 0.112 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/ycunrbmlpj`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.959, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.237 | 0.204 |
| Off-manifold residual | fake (marginal) | +0.309 | 0.242 |
| Bona-fide face prior | fake (marginal) | +0.110 | 0.256 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2090100_2022094_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.353 | 0.061 |
| Off-manifold residual | fake (decisive) | +0.458 | 0.059 |
| Bona-fide face prior | fake (marginal) | +0.007 | 0.320 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2022094_2090100_B_000`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.355 | 0.057 |
| Off-manifold residual | fake (decisive) | +0.463 | 0.048 |
| Bona-fide face prior | fake (marginal) | +0.418 | 0.049 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/DFDCP-real/1848521_C_001`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.003, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.425 | 0.230 |
| Off-manifold residual | real (marginal) | -0.336 | 0.238 |
| Bona-fide face prior | real (decisive) | -0.517 | 0.044 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/DFDCP-real/1782722_A_001`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.008, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.529 | 0.164 |
| Off-manifold residual | real (marginal) | -0.411 | 0.187 |
| Bona-fide face prior | real (marginal) | -0.335 | 0.138 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2090100_2022094_E_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.357 | 0.053 |
| Off-manifold residual | fake (decisive) | +0.465 | 0.045 |
| Bona-fide face prior | real (marginal) | -0.055 | 0.331 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.959, decision UNCHANGED

---

### DFDCP · video `DFDCP/method_A/method_A__1296203_2060058_A_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.982, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.328 | 0.104 |
| Off-manifold residual | fake (decisive) | +0.433 | 0.099 |
| Bona-fide face prior | real (marginal) | -0.227 | 0.239 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.849, decision weakens by 0.133

---

### DFDCP · video `DFDCP/method_A/method_A__660491_1390015_A_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.951, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.421 | 0.236 |
| Off-manifold residual | real (marginal) | -0.373 | 0.262 |
| Bona-fide face prior | fake (decisive) | +0.458 | 0.005 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.058, decision FLIPS

---

### DFDCP · video `DFDCP/method_A/method_A__2040724_1390015_A_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.960, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.336 | 0.203 |
| Off-manifold residual | real (marginal) | -0.396 | 0.228 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.076, decision FLIPS

---

### DFDCP · video `DFDCP/method_A/method_A__1320815_2031720_A_002`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.395, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.157 | 0.522 |
| Off-manifold residual | real (marginal) | -0.191 | 0.508 |
| Bona-fide face prior | fake (marginal) | +0.028 | 0.293 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__1782722_2090100_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.649, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.163 | 0.351 |
| Off-manifold residual | real (marginal) | -0.080 | 0.344 |
| Bona-fide face prior | fake (marginal) | +0.207 | 0.136 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2022094_2090100_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.986, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.253 | 0.150 |
| Off-manifold residual | fake (marginal) | +0.351 | 0.161 |
| Bona-fide face prior | fake (marginal) | +0.256 | 0.165 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__1968328_1390015_A_002`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.792, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.337 | 0.301 |
| Off-manifold residual | real (marginal) | -0.290 | 0.310 |
| Bona-fide face prior | fake (marginal) | +0.415 | 0.009 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/unQG2EqIG6m19LLidkUJVkwwrg0`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.327 | 0.096 |
| Off-manifold residual | fake (decisive) | +0.422 | 0.093 |
| Bona-fide face prior | fake (marginal) | +0.311 | 0.093 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/izFC8qs7h4Fjb9ojxTncDZVeA_Q`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.357 | 0.053 |
| Off-manifold residual | fake (decisive) | +0.465 | 0.046 |
| Bona-fide face prior | fake (marginal) | +0.445 | 0.030 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/vilmozXUFX_inlHr1lEz8DC9bko`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.001, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.561 | 0.111 |
| Off-manifold residual | real (decisive) | -0.440 | 0.145 |
| Bona-fide face prior | real (decisive) | -0.489 | 0.078 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/kovkZuADfjaWgyv0psuKWoKIQsI`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.545 | 0.115 |
| Off-manifold residual | real (decisive) | -0.440 | 0.144 |
| Bona-fide face prior | real (decisive) | -0.519 | 0.040 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/5yrsmdD1cKWNQrpeElkcKDYD7Ic`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.961, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.329 | 0.102 |
| Off-manifold residual | fake (decisive) | +0.426 | 0.113 |
| Bona-fide face prior | real (marginal) | -0.336 | 0.088 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.742, decision weakens by 0.219

---

### DFEval24 · video `DFEval24/DFEval24-fake/8-DdOmO_PogqI0TzR6ezNWrET5o`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.997, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.029 | 0.356 |
| Off-manifold residual | fake (marginal) | +0.055 | 0.317 |
| Bona-fide face prior | fake (decisive) | +0.458 | 0.005 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.630, decision weakens by 0.367

---

### DFEval24 · video `DFEval24/DFEval24-real/K7xk-DWvzKYlq-2m67ismximLNc`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.987, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.101 | 0.306 |
| Off-manifold residual | real (marginal) | -0.015 | 0.294 |
| Bona-fide face prior | fake (decisive) | +0.453 | 0.014 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.512, decision FLIPS

---

### DFEval24 · video `DFEval24/DFEval24-fake/I7VZWcVvY-vi30vpk1mrGVnTfqU`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.645, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.189 | 0.468 |
| Off-manifold residual | real (marginal) | -0.135 | 0.420 |
| Bona-fide face prior | fake (marginal) | +0.282 | 0.302 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/_jB8v1mKu-G-YtCcsbCDz-Xqy7I`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.621, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.043 | 0.284 |
| Off-manifold residual | real (marginal) | -0.015 | 0.335 |
| Bona-fide face prior | fake (marginal) | +0.039 | 0.270 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/_T1hNQTY3RW2qtijR5Hjtn-6Oyg`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.883, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.211 | 0.254 |
| Off-manifold residual | fake (marginal) | +0.096 | 0.468 |
| Bona-fide face prior | fake (marginal) | +0.077 | 0.580 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/lq1_uEcGCpVUWbe62SCCBvMVA0U`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.889, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.247 | 0.249 |
| Off-manifold residual | fake (marginal) | +0.103 | 0.501 |
| Bona-fide face prior | fake (marginal) | +0.018 | 0.588 |

**Relation:** all three concur · evidence insufficient
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

# Decision traces — frozen detector, symbolic overlay

Sampled at random within each rule, across all six OOD test sets, so the set is not selected for the traces that read best. Thresholds are the dev-fitted ones in SYMBOLIC_OVERLAY.md.

### CDFv2 · video `CDFv2/Celeb-synthesis/id57_id51_0000`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.999, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.292 | 0.166 |
| Off-manifold residual | fake (decisive) | +0.376 | 0.173 |
| Bona-fide face prior | fake (decisive) | +0.427 | 0.056 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id37_id3_0006`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.329 | 0.102 |
| Off-manifold residual | fake (decisive) | +0.429 | 0.099 |
| Bona-fide face prior | fake (decisive) | +0.455 | 0.011 |

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
| Bona-fide face prior | real (decisive) | -0.287 | 0.264 |

**Relation:** all three concur
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

### CDFv2 · video `CDFv2/Celeb-real/id22_0009`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.773, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.262 | 0.201 |
| Off-manifold residual | fake (decisive) | +0.276 | 0.291 |
| Bona-fide face prior | real (decisive) | -0.348 | 0.113 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.560, decision FLIPS

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id10_id7_0008`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.990, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.331 | 0.100 |
| Off-manifold residual | fake (decisive) | +0.419 | 0.101 |
| Bona-fide face prior | real (marginal) | -0.019 | 0.261 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.927, decision weakens by 0.063

---

### CDFv2 · video `CDFv2/YouTube-real/00227`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.148, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.036 | 0.346 |
| Off-manifold residual | fake (marginal) | +0.033 | 0.356 |
| Bona-fide face prior | real (decisive) | -0.463 | 0.099 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.607, decision FLIPS

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id4_id20_0000`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.257, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.027 | 0.310 |
| Off-manifold residual | fake (marginal) | +0.042 | 0.317 |
| Bona-fide face prior | real (decisive) | -0.385 | 0.104 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.622, decision FLIPS

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id16_id3_0010`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.918, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.122 | 0.376 |
| Off-manifold residual | real (marginal) | -0.162 | 0.394 |
| Bona-fide face prior | fake (decisive) | +0.425 | 0.053 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.369, decision FLIPS

---

### CDFv2 · video `CDFv2/YouTube-real/00048`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.835, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.080 | 0.344 |
| Off-manifold residual | real (marginal) | -0.073 | 0.358 |
| Bona-fide face prior | fake (decisive) | +0.337 | 0.126 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.480, decision FLIPS

---

### CDFv2 · video `CDFv2/YouTube-real/00265`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.428, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.050 | 0.424 |
| Off-manifold residual | fake (marginal) | +0.033 | 0.397 |
| Bona-fide face prior | real (marginal) | -0.214 | 0.230 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id4_id21_0009`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.520, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.041 | 0.358 |
| Off-manifold residual | fake (marginal) | +0.026 | 0.373 |
| Bona-fide face prior | real (marginal) | -0.119 | 0.238 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id32_id31_0008`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.537, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.083 | 0.255 |
| Off-manifold residual | fake (marginal) | +0.137 | 0.277 |
| Bona-fide face prior | real (decisive) | -0.304 | 0.152 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv2 · video `CDFv2/Celeb-synthesis/id35_id28_0006`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.579, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.091 | 0.180 |
| Off-manifold residual | fake (marginal) | +0.011 | 0.194 |
| Bona-fide face prior | fake (marginal) | +0.005 | 0.182 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-IP_LAP/IP_LAP__id22_0004_test_id05999_6x6ZSpvDezc`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.180 | 0.195 |
| Off-manifold residual | fake (decisive) | +0.264 | 0.225 |
| Bona-fide face prior | fake (decisive) | +0.451 | 0.020 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EDTalk/EDTalk__id0_0007_test_id08456_57jM1IfuGag`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.358 | 0.052 |
| Off-manifold residual | fake (decisive) | +0.465 | 0.046 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-LivePortrait/LivePortrait__id34_id35_0006`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.025, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.306 | 0.367 |
| Off-manifold residual | real (marginal) | -0.231 | 0.333 |
| Bona-fide face prior | real (decisive) | -0.476 | 0.118 |

**Relation:** all three concur
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

### CDFv3 · video `CDFv3/synthesis-FaceReenact-DaGAN/DaGAN__id30_id31_0009`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.650, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.249 | 0.229 |
| Off-manifold residual | fake (decisive) | +0.354 | 0.207 |
| Bona-fide face prior | real (decisive) | -0.477 | 0.099 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.295, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-FaceReenact-LIA/LIA__id30_id31_0009`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.831, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.285 | 0.187 |
| Off-manifold residual | fake (decisive) | +0.388 | 0.187 |
| Bona-fide face prior | real (decisive) | -0.421 | 0.095 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.478, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-IP_LAP/IP_LAP__id4_0003_test_id07621__8tsx2kPqNY`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.133, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.415 | 0.200 |
| Off-manifold residual | real (decisive) | -0.307 | 0.221 |
| Bona-fide face prior | fake (marginal) | +0.051 | 0.151 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.315, decision weakens by 0.181

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EchoMimic/EchoMimic__id37_0002_test_id01822_GK2R7qog3Hc`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.372, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.354 | 0.360 |
| Off-manifold residual | real (marginal) | -0.249 | 0.340 |
| Bona-fide face prior | fake (marginal) | +0.225 | 0.174 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.563, decision weakens by 0.191

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-Celeb-DF-v2/Celeb-DF-v2__id16_id3_0010`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.918, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.122 | 0.376 |
| Off-manifold residual | real (marginal) | -0.162 | 0.394 |
| Bona-fide face prior | fake (decisive) | +0.425 | 0.053 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.369, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EchoMimic/EchoMimic__id29_0008_test_id00081_cYXXH379T0c`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.974, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.134 | 0.351 |
| Off-manifold residual | real (marginal) | -0.051 | 0.371 |
| Bona-fide face prior | fake (decisive) | +0.447 | 0.023 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.453, decision FLIPS

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EchoMimic/EchoMimic__id54_0001_test_id03178_cmwo6RhAeX4`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.880, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.082 | 0.541 |
| Off-manifold residual | real (marginal) | -0.008 | 0.494 |
| Bona-fide face prior | fake (decisive) | +0.358 | 0.151 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-EDTalk/EDTalk__id29_0006_test_id04006_pWN2l9Kfc8Y`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.708, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.016 | 0.456 |
| Off-manifold residual | fake (marginal) | +0.065 | 0.428 |
| Bona-fide face prior | fake (marginal) | +0.061 | 0.386 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-TalkingFace-IP_LAP/IP_LAP__id48_0008_test_id06913_CFoP6f5lk7M`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.901, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.146 | 0.266 |
| Off-manifold residual | fake (marginal) | +0.231 | 0.253 |
| Bona-fide face prior | fake (marginal) | +0.082 | 0.367 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### CDFv3 · video `CDFv3/synthesis-FaceSwap-InSwapper/InSwapper__id34_id30_0002`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.262, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.074 | 0.263 |
| Off-manifold residual | fake (marginal) | +0.163 | 0.242 |
| Bona-fide face prior | real (decisive) | -0.475 | 0.087 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/23_24__kitchen_still__MDSDXSNU`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.358 | 0.051 |
| Off-manifold residual | fake (decisive) | +0.463 | 0.048 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/18_11__secret_conversation__PT4YKO2R`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.302 | 0.061 |
| Off-manifold residual | fake (decisive) | +0.409 | 0.055 |
| Bona-fide face prior | fake (decisive) | +0.441 | 0.037 |

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
| Bona-fide face prior | real (decisive) | -0.353 | 0.087 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DFD-real/16__walking_down_indoor_hall_disgust`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.030, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.430 | 0.227 |
| Off-manifold residual | real (decisive) | -0.340 | 0.241 |
| Bona-fide face prior | real (decisive) | -0.311 | 0.130 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/01_03__exit_phone_room__480LQD1C`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.949, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.272 | 0.134 |
| Off-manifold residual | fake (decisive) | +0.368 | 0.116 |
| Bona-fide face prior | real (marginal) | -0.167 | 0.134 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.781, decision weakens by 0.169

---

### DFD · video `DFD/DeepFakeDetection/02_06__kitchen_pan__3J3BHSHI`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.856, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.050 | 0.140 |
| Off-manifold residual | fake (marginal) | +0.052 | 0.134 |
| Bona-fide face prior | fake (decisive) | +0.298 | 0.178 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.617, decision weakens by 0.239

---

### DFD · video `DFD/DeepFakeDetection/14_26__talking_angry_couch__SU8SAHTT`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.239, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.093 | 0.238 |
| Off-manifold residual | real (decisive) | -0.433 | 0.158 |
| Bona-fide face prior | fake (marginal) | +0.194 | 0.446 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.702, decision FLIPS

---

### DFD · video `DFD/DeepFakeDetection/12_26__secret_conversation__LI63E4TS`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.145, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.458 | 0.298 |
| Off-manifold residual | real (decisive) | -0.372 | 0.271 |
| Bona-fide face prior | fake (marginal) | +0.212 | 0.298 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.369, decision weakens by 0.225

---

### DFD · video `DFD/DeepFakeDetection/11_21__kitchen_still__T7DK03O1`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.655, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.159 | 0.153 |
| Off-manifold residual | real (marginal) | -0.173 | 0.141 |
| Bona-fide face prior | fake (decisive) | +0.327 | 0.240 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.320, decision FLIPS

---

### DFD · video `DFD/DeepFakeDetection/12_15__talking_angry_couch__RK17XFJ9`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.853, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.018 | 0.161 |
| Off-manifold residual | real (marginal) | -0.015 | 0.103 |
| Bona-fide face prior | fake (decisive) | +0.330 | 0.222 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.585, decision FLIPS

---

### DFD · video `DFD/DeepFakeDetection/10_22__exit_phone_room__C1U7OY5M`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.689, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.146 | 0.442 |
| Off-manifold residual | real (marginal) | -0.100 | 0.437 |
| Bona-fide face prior | fake (marginal) | +0.246 | 0.162 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/07_21__kitchen_still__K7KXUHMU`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.847, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.097 | 0.397 |
| Off-manifold residual | real (marginal) | -0.002 | 0.477 |
| Bona-fide face prior | fake (marginal) | +0.232 | 0.387 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/27_02__kitchen_pan__78M8S6M6`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.650, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.154 | 0.165 |
| Off-manifold residual | fake (marginal) | +0.035 | 0.200 |
| Bona-fide face prior | real (marginal) | -0.195 | 0.156 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFD · video `DFD/DeepFakeDetection/15_27__outside_talking_still_laughing__WZ4MLUAV`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.753, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.111 | 0.292 |
| Off-manifold residual | fake (marginal) | +0.157 | 0.308 |
| Bona-fide face prior | real (marginal) | -0.090 | 0.114 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/yzmkvwdghk`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.988, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.320 | 0.125 |
| Off-manifold residual | fake (decisive) | +0.381 | 0.198 |
| Bona-fide face prior | fake (marginal) | +0.106 | 0.211 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-fake/soeyasjsvt`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 1.000, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.358 | 0.052 |
| Off-manifold residual | fake (decisive) | +0.464 | 0.046 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/eykkgxsaef`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.003, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.491 | 0.187 |
| Off-manifold residual | real (decisive) | -0.390 | 0.200 |
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
| Off-manifold residual | real (decisive) | -0.408 | 0.193 |
| Bona-fide face prior | real (decisive) | -0.366 | 0.184 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

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

### DFDC · video `DFDC/DFDC-fake/ebfqmkeyrz`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.947, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.310 | 0.141 |
| Off-manifold residual | fake (decisive) | +0.345 | 0.254 |
| Bona-fide face prior | real (marginal) | -0.210 | 0.313 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.826, decision weakens by 0.120

---

### DFDC · video `DFDC/DFDC-real/rfkokmqztd`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.030, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.541 | 0.151 |
| Off-manifold residual | real (decisive) | -0.425 | 0.175 |
| Bona-fide face prior | fake (marginal) | +0.067 | 0.246 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.177, decision weakens by 0.146

---

### DFDC · video `DFDC/DFDC-real/mxbbtolkvz`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.363, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.300 | 0.218 |
| Off-manifold residual | real (marginal) | -0.199 | 0.231 |
| Bona-fide face prior | fake (marginal) | +0.150 | 0.130 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.519, decision weakens by 0.156

---

### DFDC · video `DFDC/DFDC-fake/vtqenptvrq`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.923, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.209 | 0.305 |
| Off-manifold residual | real (marginal) | -0.126 | 0.314 |
| Bona-fide face prior | fake (decisive) | +0.435 | 0.043 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.328, decision FLIPS

---

### DFDC · video `DFDC/DFDC-real/amdxcpeuvi`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.909, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.058 | 0.202 |
| Off-manifold residual | real (marginal) | -0.011 | 0.200 |
| Bona-fide face prior | fake (decisive) | +0.378 | 0.078 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.553, decision FLIPS

---

### DFDC · video `DFDC/DFDC-real/otccvflwam`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.812, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.038 | 0.488 |
| Off-manifold residual | fake (marginal) | +0.020 | 0.460 |
| Bona-fide face prior | fake (marginal) | +0.246 | 0.190 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/jkjlnfsevl`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.767, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.109 | 0.437 |
| Off-manifold residual | real (marginal) | -0.036 | 0.449 |
| Bona-fide face prior | fake (marginal) | +0.262 | 0.148 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/vsvyhnzpxf`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.834, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.026 | 0.401 |
| Off-manifold residual | fake (marginal) | +0.095 | 0.382 |
| Bona-fide face prior | fake (marginal) | +0.173 | 0.045 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDC · video `DFDC/DFDC-real/dbzukyenis`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.852, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.144 | 0.285 |
| Off-manifold residual | fake (marginal) | +0.225 | 0.279 |
| Bona-fide face prior | real (marginal) | -0.025 | 0.155 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2090100_2022094_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.998, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.353 | 0.061 |
| Off-manifold residual | fake (decisive) | +0.458 | 0.059 |
| Bona-fide face prior | fake (marginal) | +0.007 | 0.320 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__2022094_2084869_B_002`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.999, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (decisive) | +0.332 | 0.088 |
| Off-manifold residual | fake (decisive) | +0.432 | 0.083 |
| Bona-fide face prior | fake (decisive) | +0.404 | 0.032 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/DFDCP-real/1848521_C_001`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.003, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.425 | 0.230 |
| Off-manifold residual | real (decisive) | -0.336 | 0.238 |
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
| Off-manifold residual | real (decisive) | -0.411 | 0.187 |
| Bona-fide face prior | real (decisive) | -0.335 | 0.138 |

**Relation:** all three concur
**Rule:** `R2_all_agree_real` → **Real**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/DFDCP-real/1501828_A_003`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.924, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.219 | 0.229 |
| Off-manifold residual | fake (decisive) | +0.279 | 0.283 |
| Bona-fide face prior | real (marginal) | -0.008 | 0.406 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Off-manifold residual — suppressing it gives p_fused 0.821, decision weakens by 0.102

---

### DFDCP · video `DFDCP/DFDCP-real/1869846_D_002`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.790, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.005 | 0.213 |
| Off-manifold residual | real (marginal) | -0.213 | 0.464 |
| Bona-fide face prior | fake (decisive) | +0.331 | 0.224 |

**Relation:** views conflict
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.456, decision FLIPS

---

### DFDCP · video `DFDCP/method_A/method_A__1883888_2040724_A_002`
*ground truth* **FAKE** · *model* **REAL** (p_fused 0.393, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.003 | 0.274 |
| Off-manifold residual | fake (marginal) | +0.060 | 0.323 |
| Bona-fide face prior | real (decisive) | -0.306 | 0.155 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.657, decision FLIPS

---

### DFDCP · video `DFDCP/DFDCP-real/2031720_A_001`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.127, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.396 | 0.342 |
| Off-manifold residual | real (decisive) | -0.362 | 0.281 |
| Bona-fide face prior | fake (marginal) | +0.092 | 0.404 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.267, decision weakens by 0.140

---

### DFDCP · video `DFDCP/method_A/method_A__1848521_1390015_A_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.996, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.048 | 0.197 |
| Off-manifold residual | real (marginal) | -0.079 | 0.250 |
| Bona-fide face prior | fake (decisive) | +0.459 | 0.005 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.504, decision FLIPS

---

### DFDCP · video `DFDCP/method_A/method_A__1883888_1390015_A_002`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.905, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.233 | 0.258 |
| Off-manifold residual | real (marginal) | -0.250 | 0.323 |
| Bona-fide face prior | fake (decisive) | +0.442 | 0.033 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.218, decision FLIPS

---

### DFDCP · video `DFDCP/method_A/method_A__2040724_1869846_A_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.856, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.012 | 0.488 |
| Off-manifold residual | fake (marginal) | +0.055 | 0.512 |
| Bona-fide face prior | fake (marginal) | +0.270 | 0.193 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__1828891_2022094_C_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.687, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.044 | 0.461 |
| Off-manifold residual | fake (marginal) | +0.080 | 0.504 |
| Bona-fide face prior | real (marginal) | -0.043 | 0.424 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__643049_2090100_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.751, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.112 | 0.356 |
| Off-manifold residual | real (marginal) | -0.039 | 0.362 |
| Bona-fide face prior | fake (marginal) | +0.259 | 0.184 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFDCP · video `DFDCP/method_A/method_A__1782722_2090100_B_003`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.649, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.163 | 0.351 |
| Off-manifold residual | real (marginal) | -0.080 | 0.344 |
| Bona-fide face prior | fake (marginal) | +0.207 | 0.136 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/FmOrztdwN0T36TSeiWcuUJwLbw0`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.992, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.247 | 0.189 |
| Off-manifold residual | fake (decisive) | +0.317 | 0.233 |
| Bona-fide face prior | fake (decisive) | +0.373 | 0.104 |

**Relation:** all three concur
**Rule:** `R1_consensus_fake` → **Fake**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/nzAQYrJgUMUsu_IqW-jaX4TCr8A`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.997, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.150 | 0.227 |
| Off-manifold residual | fake (marginal) | +0.203 | 0.256 |
| Bona-fide face prior | fake (decisive) | +0.451 | 0.020 |

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

### DFEval24 · video `DFEval24/DFEval24-fake/6SA9lhAup_UWxmnw7JjMs3RXKXU`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.863, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.042 | 0.432 |
| Off-manifold residual | real (marginal) | -0.059 | 0.604 |
| Bona-fide face prior | fake (decisive) | +0.315 | 0.272 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R3_majority_fake` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.607, decision weakens by 0.256

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

### DFEval24 · video `DFEval24/DFEval24-real/XZjkNI_O1JjFcO_arW1CgHKTetA`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.534, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (decisive) | -0.263 | 0.329 |
| Off-manifold residual | real (marginal) | -0.189 | 0.351 |
| Bona-fide face prior | fake (marginal) | +0.256 | 0.168 |

**Relation:** views conflict
**Rule:** `R6_majority_real` → **Real**
**Driver:** Appearance — suppressing it gives p_fused 0.654, decision FLIPS

---

### DFEval24 · video `DFEval24/DFEval24-real/TCAB77z9k9eFVVyc4HzO1DXmGsc`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.272, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.015 | 0.384 |
| Off-manifold residual | real (marginal) | -0.112 | 0.483 |
| Bona-fide face prior | real (decisive) | -0.317 | 0.163 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R6_majority_real` → **Real**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.535, decision weakens by 0.263

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

### DFEval24 · video `DFEval24/DFEval24-real/KvGwMvhknNk_6_XRtoLxUMKLMhk`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.941, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.018 | 0.335 |
| Off-manifold residual | real (marginal) | -0.109 | 0.403 |
| Bona-fide face prior | fake (decisive) | +0.421 | 0.068 |

**Relation:** views conflict
**Rule:** `R5_single_branch_drives` → **Fake**
**Driver:** Bona-fide face prior — suppressing it gives p_fused 0.506, decision FLIPS

---

### DFEval24 · video `DFEval24/DFEval24-real/SGPwpuC3Pv7Lg198Y8WYfsuVYQs`
*ground truth* **REAL** · *model* **FAKE** (p_fused 0.768, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.082 | 0.316 |
| Off-manifold residual | real (marginal) | -0.004 | 0.460 |
| Bona-fide face prior | fake (marginal) | +0.113 | 0.471 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/DtE_ZtU_z3fU4DSIvU4h9COtvno`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.476, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.087 | 0.312 |
| Off-manifold residual | real (marginal) | -0.179 | 0.358 |
| Bona-fide face prior | fake (marginal) | +0.079 | 0.483 |

**Relation:** views conflict · evidence insufficient
**Rule:** `R4_conflict_insufficient` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-real/rZjeG6OjF8IKvNjPzzErWU2YTSU`
*ground truth* **REAL** · *model* **REAL** (p_fused 0.420, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | real (marginal) | -0.143 | 0.336 |
| Off-manifold residual | real (marginal) | -0.237 | 0.388 |
| Bona-fide face prior | fake (marginal) | +0.086 | 0.149 |

**Relation:** views conflict
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

---

### DFEval24 · video `DFEval24/DFEval24-fake/wxkBV4Iu7u7tK0MnQe8FdVCOoq4`
*ground truth* **FAKE** · *model* **FAKE** (p_fused 0.930, τ 0.596)

| view | vote | margin past τ | vacuity |
|---|---|---:|---:|
| Appearance | fake (marginal) | +0.107 | 0.220 |
| Off-manifold residual | fake (marginal) | +0.172 | 0.262 |
| Bona-fide face prior | fake (marginal) | +0.266 | 0.221 |

**Relation:** all three concur
**Rule:** `R0_fallback` → **Inconclusive**
**Driver:** none named — the claim is agreement, not a single view

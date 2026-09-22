# Consolidated results — DISCERN-v2

Assembled by `analysis/tbiom/consolidate.py` from the committed result files; each section names the report it comes from. Development numbers are means over the five source-grouped splits used everywhere; test rows were read only after the corresponding decision was recorded.

## 1. The deployed system

| view | checkpoint | sha256 |
|---|---|---|
| T (V-JEPA 2 LoRA, B-augmented) | `vjepa2_lora_B_seed42/best.pt` | `6f0d9c63a6c052de` |
| APP + OM (warm+gate, epoch 19) | `checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | `fdc924a7cc9e1579` |

Fusion: Dempster's rule over the three views after **evidence scaling**, α' = 1 + w(α − 1), which is what the code implements — not belief-mass (Shafer) discounting; the two differ by up to 0.29 in p at w = 0.25 (`REFERENCE_MANIFEST.md` §2). Reliability weights are fitted per fold on development; logistic stacking on the same inputs is the matched comparator everywhere.

### Main test table — video AUROC

| method | CDFv2 | CDFv3 | DFD | FFIW | KoDF | IDForge | DFDM | 10-set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DISCERN-v2 (DS-R)** | 0.9598 | 0.9616 | 0.9338 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | **0.9155** |
| DISCERN-v2 (logistic comparator) | 0.9697 | 0.9675 | 0.9300 | 0.9262 | 0.9630 | 0.9622 | 0.9963 | 0.9117 |
| R0 (v3.1, ep 6) | 0.9509 | 0.8954 | 0.9431 | 0.9036 | 0.8198 | 0.9723 | 0.9929 | 0.8959 |
| A (v3.1, ep 5) | 0.9388 | 0.8770 | 0.9461 | 0.9011 | 0.8242 | 0.9697 | 0.9915 | 0.8924 |
| B (v3.1, ep 8) | 0.9225 | 0.8801 | 0.9387 | 0.8906 | 0.8497 | 0.9553 | 0.9944 | 0.8920 |
| C (v3.1, ep 2) | 0.9183 | 0.6648 | 0.8802 | 0.9279 | 0.8548 | 0.9492 | 0.9868 | 0.8614 |
| DiCoME (our training, v3.1, ep 5) | 0.9693 | 0.8691 | 0.9343 | 0.9284 | 0.8509 | 0.9598 | 0.9909 | 0.8937 |
| GenD-CLIP | 0.9534 | 0.8458 | 0.9350 | 0.9278 | 0.8329 | 0.9839 | 0.9959 | 0.8945 |
| GenD-PE | 0.9548 | 0.8952 | 0.9203 | 0.9330 | 0.8189 | 0.9808 | 0.9815 | 0.8842 |
| GenD-DINO | 0.9177 | 0.8172 | 0.9165 | 0.9450 | 0.8984 | 0.9860 | 0.9903 | 0.8883 |
| Effort | 0.9312 | 0.7975 | 0.9276 | 0.9252 | 0.8725 | 0.9583 | 0.9729 | 0.8777 |
| ForAda | 0.9544 | 0.7494 | 0.9395 | 0.9055 | 0.8739 | 0.9101 | 0.9713 | 0.8728 |
| FSFM | 0.9554 | 0.8751 | 0.9219 | 0.9168 | 0.8467 | 0.9566 | 0.9790 | 0.8924 |
| FSFM-Adapter | 0.8689 | 0.8381 | 0.7516 | 0.8565 | 0.7405 | 0.9077 | 0.8452 | 0.8117 |

The seven sets shown were chosen for the main table after the numbers were known (decision of 2026-09-18); DFDC, DFDCP and DeepfakeEval2024 are in `FINAL_TABLE.md` Appendix A at the same precision, and the ten-set mean covers all ten. Sources: `EXP_DFDCP_DIAGNOSIS.json`, `BASELINE_MATRIX.json`.

**FaceShifter**, read once after the recipe was fixed, never tuned against: DS-R 0.9549, logistic 0.9480. Its 140 source ids are exactly the FF++ test split, disjoint from our training and validation ids, so it is unseen manipulation on held-out source videos (`FACESHIFTER.md`).

## 2. What the system is made of

**Three views, and what each earns** (`EXP_THREEVIEW_ABLATION.md`). Δ = triple − pair on clean development, paired source-grouped bootstrap.

| removed view | remaining pair | Δ (DS-R) | splits with CI above 0 | Δ (logistic) |
|---|---|---:|---:|---:|
| temporal | APP+OM | +0.0569 | 5/5 | +0.0560 |
| appearance | T+OM | +0.0012 | 0/5 | -0.0050 |
| off-manifold | T+APP | +0.0102 | 5/5 | +0.0047 |

The temporal view earns its place decisively; off-manifold earns it under DS-R only; appearance is marginal and is reported as such rather than dropped.

**Uncertainty** (`EXP_UNCERTAINTY.md`). Each fusion is scored against its own decision with its own signals.

| condition | best error-ranking signal, DS-R | error AUROC | best signal, logistic | error AUROC |
|---|---|---:|---|---:|
| clean development | u_fused | 0.781 | conflict_pair_max | 0.745 |
| jpeg40 development | u_fused | 0.715 | u_mean | 0.717 |
| rs06 development | u_fused | 0.756 | conflict_pair_max | 0.735 |
| clean tests (10 sets pooled) | u_fused | 0.807 | conflict_pair_max | 0.779 |

The fused Dirichlet vacuity is the best error ranker for DS-R in every condition. The logistic comparator has no fused vacuity, and what ranks its errors best is the largest pairwise cross-view conflict (or, under JPEG, the mean per-branch vacuity).

**Seeds** (`EXP_CONFIRM_SEED.md`). Temporal-branch seeds 42, 1337, 7: fused 10-set mean 0.9155, 0.9136, 0.9155 (spread 0.0020) while the branch alone moves 0.0231. Fusion absorbs branch-level seed variation.

**The known weakness** (`EXP_DFDCP_DIAGNOSIS.md`). Under JPEG q40 the development score falls by T 0.0498, APP 0.0994, OM 0.0813; the fused DS-R system drops 0.0712. Refitting the fusion on clean plus processed development recovers nothing for DS-R (+0.0008 at JPEG) but helps the logistic comparator (+0.0496), which is a statement about the comparator rather than a fix for the reference.

## 3. What was tried and not adopted

Every row had its acceptance rule written and committed before the experiment ran. A stop means insufficient demonstrated benefit under the tested configurations within the allocated budget — not that the idea cannot work.

| # | question | frozen criterion | measured | outcome |
|---|---|---|---|---|
| 1 | joint training of all branches vs the staged system | beat the staged system | 0.8768, 0.8804 vs staged 0.8808; off-manifold's leave-one-out contribution falls to +0.0003 | not adopted; it also flattens the views |
| 2 | FreqBlender frequency pseudo-fakes (local reproduction) | development gain with FPR ≤ +0.02 | clean +0.0092, reenactment +0.0004, FPR +0.024 | not adopted (false positives) |
| 3 | acquisition policy, then PPO | learned selector beats the best fixed rule by ≥ +0.002 | no acquisition 0.9068, supervised one-step 0.8952, null oracle 0.9855 above the real oracle 0.9382 (aggregation `conf`) | no-go; no RL trained |
| 4 | adaptive reliability controllers (MLP / GRU / Mamba) | beat constant DS-R by ≥ +0.005 | constant 0.8235; MLP 0.8199, GRU 0.8215, Mamba 0.8055 | fails; quality-conditioned discounting does not transfer |
| 5 | dense vs sparse sampling, small temporal head | +0.005 on a held-out confirmation part, CI excluding zero, plus robustness and reenactment limits | best candidate +0.0056 with the CI including zero; robustness and reenactment both fail | stop; Mamba trigger never fired (sparse -0.0093, dense -0.0005) |
| 6 | composed processing (resize→JPEG) vs the existing mixture | processed ≥ +0.005, significant in ≥ 3/5 | processed +0.0081, clean +0.0090 | accepted **against arm B** |
| 7 | …but does that winner beat the DEPLOYED views? | same criteria | clean -0.0051, processed +0.0037, reenactment -0.0087 | no |
| 8 | composition applied to the deployed recipe (views unfrozen) | same criteria, against the deployed checkpoint | adoption -0.0074 clean / -0.0023 processed; unfreezing alone -0.0105 clean | not adopted |
| 9 | arm E adopted anyway (override), confirmed across 3 seeds | none — reported as measured | clean -0.0052, -0.0094, -0.0056 | not supported by its own seeds |

**The one positive worth keeping.** Under the logistic comparator the composed-augmentation processed gain is consistent across seeds (+0.0215, +0.0207, +0.0178, significant in 5/5 splits) and absent under DS-R. Composed augmentation does buy processing robustness — for the fusion that needs it; the evidential fusion already extracts it from the unchanged views (`EXP_COMBINE_CONFIRM.md`).

## 4. Standing caveats, stated once

- **Fitting access.** The encoders are trained on FF++ only, but the fusion (DS-R weights, logistic stacker, thresholds) is fitted on labelled external development data. This is not an FF++-only complete system, and every competing controller in these reports has the same access.
- **Development contamination.** CDFv2 and DeepfakeEval2024 informed development through VALmix; they are not untouched confirmation data. FaceShifter was read once, after the recipe was fixed.
- **Table scope.** The main-table subset was chosen after seeing the numbers; the appendix carries the rest at the same precision and the ten-set mean covers all ten.
- **Segment granularity.** Stored crops are 16–32 stills per video, so segment-level results rest on 4–8 distinct crops per segment (`REFERENCE_MANIFEST.md` §3–4).
- **Seeds.** Three temporal-branch seeds and three arm-E seeds; one seed elsewhere, stated per report.
- **Withdrawn work is kept, not hidden:** `tbiom/reports/void/` (a comparison invalidated by frozen views) and `logs/tbiom/freqblender/v1_facecut_confounded/` (a generator withdrawn for a crop-geometry confound).

## 5. Where each number lives

| report | what it settles |
|---|---|
| `REFERENCE_MANIFEST.md`, `STAGE0_AUDIT.json` | checkpoints, discount rule, segment granularity |
| `FINAL_TABLE.md`, `BASELINE_MATRIX.md` | test tables and baselines |
| `FACESHIFTER.md` | the confirmation read |
| `EXP_UNCERTAINTY.md`, `EXP_THREEVIEW_ABLATION.md`, `EXP_CONFIRM_SEED.md` | what the system is made of |
| `EXP_DFDCP_DIAGNOSIS.md` | the processing weakness and its cause |
| `EXP_JOINT_ABLATION.md`, `EXP_FREQBLENDER.md`, `EXP_ACQUISITION_PILOT.md` | earlier pilots |
| `EXP_RELIABILITY_CONTROLLER.md`, `EXP_AUGMENTATION_EXTENSION.md`, `EXP_TEMPORAL_SAMPLING.md` | executable-brief stages 1A, 1B, 2 |
| `STAGE3_PRECHECK.md`, `EXP_COMBINE_CONFIRM.md` | stage 3, both routes |

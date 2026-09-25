# I-JEPA branch report

Brief: `tbiom/docs/discern-v2-ijepa-branch-brief.md`. Code in `branches/ijepa/`, outputs in `analysis/ijepa/`.
The deployed configuration under test is **`DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml`**
(held in the **DISCERN_Ext** repository); the temporal branch and the fusion live in **DFB_NeSyNS**. No deployed
config or checkpoint was modified.

## Gate verdicts

| gate | verdict | evidence | one line |
|---|---|---|---|
| G0 predictor weights | **PASS** | `analysis/ijepa/G0_output.txt` | All three weight sets load with `strict=True` and the trained predictor beats all three controls on both blocks. |
| block alignment (pre-extraction stop check) | **PASS, with a caveat** | `analysis/ijepa/block_alignment.json`, `analysis/ijepa/block_overlays/` | Every block is majority-face on both FF++ and CDFv2; the two cheek blocks carry about a quarter background. |
| extraction budget | **RESOLVED** | `analysis/ijepa/budget_estimate.json`, `analysis/ijepa/logs/` | The 24-hour ceiling was a fp32 estimate. bf16 at batch 64 is 4.9x faster on the same card, so the non-DF40 half plus the processed cells is ~3.7 hours of wall clock. DF40 is still not queued. |
| G1 compression confound | TODO(run) | -- | Blocked by the budget stop. |
| G2 single-view signal | TODO(run) | -- | Blocked. |
| G3 fusion | TODO(run) | -- | Blocked. |

## G0, full output (tail)

```
)
encoder: loaded with strict=True
predictor: loaded with strict=True
target_encoder: loaded with strict=True

upper face
  trained predictor                  0.0928
  trained, another face's context    0.2238
  random predictor                   0.3743
  mean of visible patches            0.2597
  PASS: trained predictor beats all three controls

lower face
  trained predictor                  0.1177
  trained, another face's context    0.2341
  random predictor                   0.3739
  mean of visible patches            0.2623
  PASS: trained predictor beats all three controls

G0 PASSED
```

The checkpoint `IN1K-vit.h.14-300e.pth.tar` is **CC BY-NC 4.0**, i.e. non-commercial. That licence applies to anything derived from these features.

## Block alignment

The brief says to stop if the blocks do not land on the intended face areas. They do. Coverage is the fraction of each block's own area falling inside the 81-landmark face convex hull, over 20 real crops per dataset.

| block | FF++ c23 mean | FF++ worst crop | CDFv2 mean | CDFv2 worst crop |
|---|---:|---:|---:|---:|
| upper face | 0.94 | 0.71 | 0.92 | 0.61 |
| lower face | 0.85 | 0.68 | 0.85 | 0.63 |
| left cheek | 0.76 | 0.47 | 0.78 | 0.56 |
| right cheek | 0.75 | 0.46 | 0.71 | 0.31 |
| centre | 1.00 | 0.96 | 0.97 | 0.81 |

Centre, upper and lower land squarely on the face (0.85-1.00). The two cheek blocks sit at 0.71-0.78, because the aligned crop leaves background at the left and right edges, so roughly a quarter of each cheek block's residual will be driven by background rather than by face. That is worth carrying into the interpretation of any per-block result; it is not a reason to stop.

Two measurement caveats, stated because they affect how much weight the table can carry. The detector refuses most of these already-cropped faces -- it found a face on only 6 of 20 FF++ and 5 of 20 CDFv2 crops -- so the shape predictor is given the whole crop as its face box, which is the standard handling for pre-cropped faces and lets all 40 crops contribute. On the detector's own smaller sample the cheek numbers were lower (0.57-0.58 on FF++, 0.42 for the right cheek on CDFv2), so the cheek coverage above is the optimistic end of the range.

## The budget stop

Measured throughput on an idle GPU: **22.47 frames/s** (100 videos, 3,200 frames, 32 frames/video, batch 16, fp32 forward). Per frame the pass is one target-encoder forward plus, for each of the five blocks, one context-encoder forward and one predictor forward.

| group | frames | GPU hours |
|---|---:|---:|
| FF++ c23 train (B mixture) | 115,198 | 1.42 |
| FF++ c23 val (clean) | 22,396 | 0.28 |
| development pool VALmix | 40,142 | 0.50 |
| development pool VALmix-v2 | 77,394 | 0.96 |
| ten test sets (clean) | 904,675 | 11.18 |
| DF40 evaluation, 41 sets | 858,166 | 10.61 |
| processed tests (CDFv2+DFDCP x jpeg40,rs06) | 67,588 | 0.84 |
| **total** | **2,085,559** | **25.78** |

Cache size at 15,376 bytes per frame: **32.1 GB** fp16.

The brief allows continuing without asking at up to 24 GPU hours. The estimate is **25.78**, over by 1.8 hours, so extraction was not started.

Note that this is a GPU-*hours* figure, not wall clock. I-JEPA ViT-H needs about 10 GB, so six lanes fit on one 143 GB card and the wall clock would be roughly four hours -- but the budget in the brief is stated in GPU hours, and parallelism does not reduce those.

### Three ways forward, for the authors to choose

| option | GPU hours | what it costs |
|---|---:|---|
| approve the overrun | 25.78 | 7.4% over the stated ceiling, nothing else |
| drop DF40 from extraction | 15.17 | the DF40 column of the report's fusion table cannot be filled; the ten test sets and processed cells are unaffected |
| halve frames per video on the two largest test sets | ~22.5 | CDFv3 and IDForge get 16 frames instead of 32, which breaks the brief's rule that the frames match the Appearance view |

The middle option is the only one that keeps every remaining rule intact, and DF40 is the part of the report that the I-JEPA view is least likely to change, since the question is whether it can replace the Off-manifold view at all.

## Amendment 1 (2026-09-25)

`tbiom/docs/discern-v2-ijepa-branch-amendment1.md`. Adds R2b (regional feature control), R5
(cross-block cosine with validity gating), R6 (predictor LoRA on real faces), diagnostics D1/D2, and a
three-part keep rule. **Section D (R7) was truncated in transmission and is not planned.**

R5's features are implemented and unit-tested in `branches/ijepa/features.py`: the two disjoint pairs
(upper/lower, left/right cheek), the frozen 5th-percentile norm floor tau_k, the suppression-and-flag
rule, and video-level frame means giving 4 inputs. The centre block is excluded because it overlaps all
four others; disjointness is asserted in code rather than assumed.

R2b's s_bar_k is now produced by the extraction itself rather than by a second target-encoder pass:
inside the loop `y = apply_masks(h, [hid])` already is the layer-normalised target feature at the
block's 56 patches, so the block mean is free and cannot drift from the frame's processing draw.

## Extraction as actually launched

| item | value |
|---|---|
| precision | bf16 autocast, batch 64. Measured 18.76 f/s per lane against 3.82 fp32 at batch 16 |
| bf16 vs fp32 agreement | median relative difference in eps 0.006, max 0.069, Spearman 0.991-0.999 per block |
| lanes | 5 on GPU 1, each process launch-bound, so aggregate ~94 f/s |
| FF++ train | **B mixture, one draw per clip**, applied to the native crop before the 224 resize |
| processed cells | CDFv2 and DFDCP at `jpeg:40` and `resize:0.60`, applied after the 224 resize as in `export_views.apply_processing` |
| queued | 37 jobs, 1,258,741 frames, ~3.7 hours |
| not queued | DF40 (858,166 frames), pending the budget decision |
| cache | 28,176 bytes/frame including `sbar`, ~35 GB total |

An earlier claim in this report that the two processed test cells could not be extracted was wrong.
They can: the evaluation severities are JPEG q40 and resize 0.60, and CDFv2 + DFDCP at both is 67,588
frames, matching the budget line exactly.

## Settings recorded so far

| item | value |
|---|---|
| I-JEPA | ViT-H/14, 224x224, 16x16 patch grid, ImageNet mean/std |
| context weights | `encoder`; target weights `target_encoder` + layer norm over the feature dimension |
| frames per video | 32, matching the Appearance view (measured from the manifests) |
| blocks | upper (2-9, 4-12), lower (9-16, 4-12), left cheek (5-12, 0-8), right cheek (5-12, 8-16), centre (4-11, 4-12); 56 patches each, 22% of the image |
| quality descriptors | variance of the Laplacian, an 8-pixel-grid blockiness ratio, crop size. Yaw is not cached and is not computed. |
| cache | `analysis/ijepa/cache/<dataset>.npz`, fp16 |
| PCR6+-R commit | to be recorded when the fusion rows run |

## Caveats

- I-JEPA was pretrained on ImageNet-1K, not on faces.
- The checkpoint is CC BY-NC 4.0, non-commercial.
- The cheek blocks carry roughly a quarter background, as measured above.


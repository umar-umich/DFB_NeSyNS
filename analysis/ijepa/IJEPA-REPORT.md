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


## Amendment 2 precondition (2026-09-25)

`tbiom/docs/discern-v2-ijepa-branch-amendment2.md`. Sequence heads T0-T3 on the frozen residual cache,
to run after G2 and the R5 decision. Evidence: `analysis/ijepa/amendment2_cache_check.json`.

**The cache is per frame.** eps_k and r_bar_k are stored per frame with the key, from which the video
id and the original frame index both recover. a_k is derived rather than cached, deliberately: it is
log eps_k standardised by the frozen mu_k, sigma_k fitted on real FF++ training frames, and caching it
would freeze a fit the gates are entitled to refit.

**The cached frames are NOT consecutive.** The stored crop filenames carry the original video frame
index, so the stride is exact.

| dataset | videos | frames/video | median stride (source frames) | approx. seconds |
|---|---:|---:|---:|---|
| FFpp_train | 3600 | 32 | 13 | 0.43-0.52 |
| FFpp_val | 700 | 32 | 13 | 0.43-0.52 |
| VALmix | 1350 | 32 | 15 | 0.50-0.60 |
| **VALmixV2** | 4852 | **16** | 21 | 0.70-0.84 |
| CDFv2 | 518 | 32 | 12 | 0.40-0.48 |
| CDFv3 | 5418 | 32 | 10 | 0.33-0.40 |
| DFD | 3431 | 32 | 24 | 0.80-0.96 |
| DFDC | 4704 | 32 | 10 | 0.33-0.40 |
| DFDCP | 654 | 32 | 14 | 0.47-0.56 |
| DFDM | 1896 | 32 | 12 | 0.40-0.48 |
| DFEval24 | 814 | 31 | 37 | 1.23-1.48 |
| FaceShifter | 280 | 32 | 15 | 0.50-0.60 |
| FFIW | 3470 | 32 | 7 | 0.23-0.28 |
| IDForge | 5974 | 32 | 5 | 0.17-0.20 |
| KoDF | 2239 | 32 | 19 | 0.63-0.76 |

Seconds are **assumed**, at 25-30 fps. No native frame rate is recorded anywhere in the preprocessing
metadata, which holds resolutions only, so the stride in source frames is the measured quantity and
the seconds column is a stated conversion.

> **Wording constraint, binding.** Consecutive cached frames are 0.17 to 1.48 seconds apart. Every
> T0-T3 result is **aggregation over sampled frames** and must never be described as motion or as
> temporal dynamics.

**Two further consequences the amendment's tables have to carry.**

- **VALmix-v2 holds 16 frames per video, not 32.** Checkpoint selection therefore runs on sequences
  half the length of the training sequences, for every head.
- **DF40 is 13 manifests of single-frame entries.** 36,316 of its 62,156 videos are flat whole-image
  generators (CollabDiff, MidJourney, e4e, stargan, starganv2, styleclip, whichisr3al) where the
  canonical video rule makes each frame its own video, so a sequence head sees length 1 and T0-T3 are
  degenerate by construction. The remaining 67 manifests carry 25,840 videos at a median of 32 frames.
  Sequence results on DF40 must be reported over the multi-frame subset, with the single-frame subset
  named and excluded rather than silently averaged in.

**T3 will use the reference implementation.** `mamba_ssm` is not installed in any environment on this
machine (`discern_ext`, `discern_ext_timm`, `temporal_vjepa2` all fail to import it). The amendment
permits the reference PyTorch selective scan in that case, and it is recorded here as the one used.
Installing the package would require building CUDA kernels in an environment whose peft/diffusers pins
must not move.

**Training realisations.** FF++ c23 training residuals are extracted with the B mixture drawn **once
per clip**, so each training frame has exactly **one fixed augmented realisation**, not several cached
variants. The heads apply no image augmentation, so no result here may be described as having fresh
augmentation during head training.

### Amendment 2: heads built, parameter counts and the padding test

Code: `branches/ijepa/seq_heads.py` (heads), `branches/ijepa/train_seq.py` (training and shuffle
check), `branches/ijepa/test_seq_heads.py` (unit tests). Backend for T3:
**`reference_pytorch_selective_scan`**.

| head | sequence + aggregation | input | classifier | total trainable |
|---|---:|---:|---:|---:|
| T0 masked mean | **0** | 102,464 | 17,538 | 120,002 |
| T1 residual MLP + attention pooling | 64,661 | 102,464 | 17,538 | 184,663 |
| T2 two residual 1-D convolutions | 64,808 | 102,464 | 17,538 | 184,810 |
| T3 two Mamba layers, d_state 16 | 65,536 | 102,464 | 17,538 | 185,538 |

T1/T2/T3 sequence-module spread is **1.35%** against the amendment's 15% limit. Widths were solved
against T3, whose shapes are fixed by d_model and d_state: T1 hidden 468 with a 64-wide attention
scorer, T2 84 channels, T3 expand 2. T0 has no sequence parameters by construction.

Padding unit test output: `analysis/ijepa/amendment2_padding_tests.txt`, **15 passed**.

| check | result |
|---|---|
| padding equivalence, video alone vs in a padded batch | T0 5.96e-08, T1/T2/T3 exactly 0.0 (limit 1e-5) |
| overwriting the padded tail with garbage | no change, all four heads |
| T0, T1 order-invariant | exact, 0.0 |
| T2, T3 order-sensitive | 1.1e-03, 1.2e-03 |

T2 zeroes padded positions before each convolution, so a padded column cannot reach a valid one
through its symmetric kernel. T3's depthwise convolution is causal and its scan runs strictly
forward from a zero state per video, so a right-padded tail cannot reach an earlier position.

Frames are ordered by the ORIGINAL frame index recovered from the crop filename, not by their order
in the cache; `train_seq.py --dry-run` asserts that shuffling the cache rows leaves the built
sequences unchanged.

Single-view, fusion, shuffle-check and keep-rule tables: TODO(run). They follow G2 and the R5
decision, as the amendment directs.

### The fused-system harness (shared by G3 and both keep rules)

`branches/ijepa/fusion_eval.py`. A candidate row is substituted into the ARTIFACT slot beside the
deployed Appearance and V-JEPA 2 views, trust weights are refitted on the five source-grouped
development folds and the threshold at FF++ validation EER, exactly as deployed. Leave-one-out uses
PCR6+'s neutrality property: a view at r = 0 is vacuous and a vacuous view is exactly a removed one,
so a contribution is the fused AUROC minus the fused AUROC with that view's weight zeroed, with no
refitting in between.

**Self-check PASS.** Substituting the deployed Off-manifold view back into its own slot reproduces
the deployed PCR6+-R development mean AUROC to **0.00e+00** (0.907002 both ways) with all five
per-fold weight triples identical (`analysis/ijepa/fusion_eval_selfcheck.json`). Any candidate
measured through this harness is measured on the same instrument as the reference.

Baseline leave-one-out on the deployed system, per fold:

| split | APP weight | LOO(T) | LOO(APP) | LOO(OM) |
|---|---:|---:|---:|---:|
| 0 | 0.25 | +0.0653 | −0.0051 | +0.0128 |
| 1 | 0.25 | +0.0691 | −0.0063 | +0.0134 |
| 2 | 0 | +0.0614 | 0 | +0.0691 |
| 3 | 0 | +0.0659 | 0 | +0.0621 |
| 4 | 0 | +0.0576 | 0 | +0.0634 |

**Read the Appearance row carefully.** Its five-fold mean of −0.0023 is not a measurement of
Appearance: three of the five folds contribute an exact zero because the discount had already set its
weight to zero, so removing it there cannot change anything. Over the two folds where it actually
participates the mean is −0.0057. Note also that Off-manifold's own contribution is four to five
times larger in the folds where Appearance is absent (+0.062 to +0.069) than where it is present
(+0.013), which says the two spatial views overlap and each looks smaller while the other is there —
not that either is idle. An aggregate mean of this quantity has misled once in this project already;
it needs the family-resolved form before any claim is attached to it, and the earlier study found
Appearance helping on most source groups once resolved that way.

The video view is the steadiest contributor, +0.058 to +0.069 in every fold.

### The frozen standardiser, fitted on the completed FF++ train cache

`analysis/ijepa/standardizer_FFpp_train_B.json`, fitted on **23,039 real** FF++ c23 training frames
under the B mixture (115,198 frames, 3,600 videos, 92,159 fake).

| block | sigma_k | tau_k | mean a_k, real | mean a_k, fake |
|---|---:|---:|---:|---:|
| upper | 0.1669 | 4.64 | 0.0000 | **+0.1215** |
| lower | 0.1411 | 4.82 | 0.0000 | +0.0846 |
| left cheek | 0.1235 | 4.02 | 0.0000 | +0.0942 |
| right cheek | 0.1108 | 4.12 | 0.0000 | +0.0960 |
| centre | 0.1623 | 4.49 | 0.0000 | **+0.3098** |

Real frames sit at mean 0 and sd 1 by construction, so the fake column is the whole of the signal
before any head is trained: the predictor is measurably worse at reconstructing every hidden block of
a manipulated face, and worst by a factor of three at the CENTRE block, which is the one covering the
region most manipulations actually edit. The R5 validity flag fires on 7.59% of (frame, pair) cells,
and the cross-block cosine separates real from fake by +0.0053 (upper/lower) and +0.0033 (cheeks).

**Two defects found while fitting this, both fixed.**

*The fp16 accumulator.* `q` is cached as fp16 and numpy reduces in the input dtype, so summing
115,198 values overflowed the accumulator and shifted the frozen `mu_q` that the whole study then
applies. No cached value is near the fp16 ceiling -- the maximum blur is 1,737 against a limit of
65,504 -- so the data was never the problem, only the reduction. Reductions now run in float64. It
was not cosmetic: the fake-side means moved by about 10% relative, centre from +0.291 to +0.3098.

*The third quality descriptor is inert.* The brief asks for blur, blockiness and the face box size.
**Every stored crop in this corpus is 256x256** -- checked on FFpp_train, CDFv2 and IDForge -- so the
size descriptor is constant whether measured before or after the 224 resize. Its z-scored column and
its square are identically zero and the least-squares fit ignores them. Standardisation therefore
runs on blur and blockiness alone, which are the two that bear on the compression confound G1 tests.
Recorded rather than quietly dropped, and no re-extraction would change it.

### Early read on FF++ validation, before any head is trained

Video-level AUROC on FF++ c23 validation, clean, using the frozen training-fitted standardiser.

| block | AUROC of a_k | AUROC of log eps_k |
|---|---:|---:|
| upper | 0.5431 | 0.5400 |
| lower | 0.5243 | 0.5193 |
| left cheek | 0.5373 | 0.5371 |
| right cheek | 0.5391 | 0.5357 |
| **centre** | **0.6142** | 0.6102 |
| five a_k together, 5-fold logistic | **0.6190** | |

Two things follow.

**The standardiser behaves as intended.** a_k and log eps_k rank almost identically on clean data,
which is what should happen: standardisation exists to remove a quality confound, not to manufacture
separation. G1 is the test of whether it removes that confound under compression; this only confirms
it does not distort the clean ranking.

**G2 looks demanding, and this is a floor rather than a forecast.** G2 requires R4 to reach
AUROC >= .80 on FF++ validation. The five scalars alone reach 0.619. The R4 head additionally
receives the five 1280-dimensional residual DIRECTIONS through a shared 1280->64 projection, which is
where most of its capacity is, so the trained row can land well above this. But the gap from 0.62 to
0.80 has to come almost entirely from those directions, and that is worth knowing before the run
rather than after it. Reported as `TODO(run)` until R4 is actually trained.

### R1 needs an extraction that does not exist yet

The brief's row table includes R0h (the Off-manifold head retrained on r_perp) and R1 (retrained on
r, without the orthogonal projection). From `DISCERN_Ext/src/model/core_model.py:330`:

    r        = semantic_feature - manifold_feature          (f_s - f_c)
    r_perp   = r - proj_{f_s}(r)                            (the deployed artifact feature)

**R0h is buildable from what is already cached**, because r_perp is exactly the `artifact` block that
`export_views.py --features` stores and that the sweep is writing.

**R1 is not.** Recovering r from f_s and r_perp needs the projection coefficient
`<r, f_s> / ||f_s||^2`, and that scalar is destroyed by the projection: r_perp is orthogonal to f_s
by construction, so it carries no record of how much was removed. R1 therefore requires
`manifold_feature` (out[4] of the model's forward) to be cached, which no current export does. The
value is available in the forward return, so this is an extra pass over the relevant splits rather
than anything harder -- but it is a new extraction, and it is flagged rather than quietly dropped.

Both rows concern the Off-manifold view, not the I-JEPA view, so neither blocks G1 or G2. R0h is the
fair comparator the brief wants for R1 ("so that R1 is compared against a head retrained in the same
way, not against the deployed head"), which means running R0h without R1 answers no question the
brief asked. Status: **R0h TODO(run)**, **R1 blocked on an extraction not yet performed**.

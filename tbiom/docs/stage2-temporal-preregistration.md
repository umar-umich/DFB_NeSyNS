# Stage 2 — bounded temporal experiment: frozen before launch

Written 2026-09-20, before any dense frame was extracted or any configuration trained. Governs
`tbiom/reports/EXP_TEMPORAL_SAMPLING.md`. Cites `tbiom/reports/REFERENCE_MANIFEST.md`. After launch there is no
further architecture sweep: the configurations, budget, partition, margin and constraints below are final.

## Bounded development subset (sources with raw video)

| source | raw video | videos used |
|---|---|---|
| FF++ val (real + four manipulations) | `FaceForensics++/{original,manipulated}_sequences/*/c23/videos` | all 700 |
| CDFv2-val (VALmix) | `Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}` | all |
| DeepfakeEval2024-val (VALmix) | `video_eval24/<id>.mp4` | all |
| CDFv3-val (VALmix-v2), in-scope methods only | `Celeb-DF-v3/Celeb-synthesis/{FaceSwap,FaceReenact,TalkingFace}/<method>/`, `Celeb-DF-v3/Celeb-real` | at most 600, stratified by method, seed 2026 |

DFDCP-val, DS11/DS20-val, DF40-train and FFIW-val have no raw video on this machine and are outside the subset. The
subset's v3.1 therefore has four cells — swap/CDFv2, swap/CDFv3, reenactment/CDFv3, in-the-wild/DFEval24 — and FF++
val supplies τ (EER, as everywhere) rather than a scored cell.

**Partition, source-grouped (`groups_for`), seed 2026:** fitting 40% · selection 30% · **confirmation 30%**. Fusion
parameters are fitted on the fitting part, the candidate configuration is chosen on the selection part, and the
stopping rule is assessed only on the confirmation part, which nothing else touches. Temporal-branch epochs for every
new configuration are selected on fitting + selection only. The existing reference branch selected its epoch on the
whole development pool, confirmation part included; that favours the reference, which makes the rule harder, not
easier, for a new configuration to pass.

## Physical windows and the configurations

Frames are cut with the pipeline that made the stored crops (`preprocessing/preprocess_current.py::
extract_aligned_face_dlib`: dlib HOG detector, 81 landmarks, aligned 256-px crop at scale 1.3), so crop geometry is
identical between sparse and dense inputs. Failed detections are recorded per frame, never silently filled.

| window | frames the branch sees | physical span |
|---|---|---|
| **sparse** (current) | 16 frames at `linspace` over the stored stills of the whole video | the whole video (reported per source) |
| **dense** | 16 frames at stride 2 from the 32 consecutive source frames centred on the video's midpoint | 32 source frames ≈ 1.1–1.3 s |
| **dense-subsampled** (same window) | 4 frames at stride 8 from that same 32-frame window, each repeated 4× | the same 32 source frames |

| | mean pooling (existing) | small temporal head |
|---|---|---|
| sparse | **reference** (`vjepa2_lora_B_seed42`, no retraining) | head effect at sparse |
| dense | sampling effect | sampling-plus-head effect |

`dense-subsampled × pooling` is trained as well and reported as a diagnostic: it isolates frame density at a fixed
duration, since changing stride at a fixed frame count also changes duration.

* **Small temporal head:** V-JEPA 2's tokens are averaged over space within each temporal index (8 steps for 16
  frames), a one-layer GRU (hidden 256) runs over the 8 steps, its outputs are averaged, and the same evidential head
  follows. One head, fixed now; no alternative heads are tried.
* **Budget per configuration, identical to the reference recipe:** V-JEPA 2 ViT-L, LoRA r8 α16 dropout 0.1 on
  query/value of every layer, 10 epochs, batch 32, lr 2e-4 adapters / 2e-3 head (and the temporal head), arm-B
  processing on training clips, seed 42, FF++ train videos. Frame count and inference compute are reported for each.
* **Mamba** enters as a temporal head only if this comparison shows a small temporal head is useful, and then only
  within this same budget.

## The stopping rule

Each configuration's opinions replace the temporal view in the reference triple (Appearance and Off-manifold
unchanged, frozen); DS-R and the logistic comparator are refitted on the fitting part. The candidate is the
configuration with the best fused DS-R v3.1 on the selection part. Against the reference, on the **confirmation part**:

1. **Margin:** fused DS-R subset-v3.1 Δ ≥ **+0.005**, with the paired source-grouped bootstrap 95% CI (1000 draws)
   excluding zero.
2. **Robustness:** mean fused Δ over JPEG q40 and resize 0.60 ≥ **−0.005**.
3. **Reenactment:** fused Δ on the reenactment cell ≥ **−0.005**.
4. **Compute:** temporal-branch inference ≤ **2×** the reference per video. Frame decoding and face cropping for dense
   windows are reported separately, since the reference reads stored crops.

All four hold → temporal-architecture development may continue (Stage 3 may take it). Otherwise it stops for this paper
with the report "insufficient demonstrated benefit under the tested configurations within the allocated budget" — not
"temporal information is absent".

## Diagnostics, never gates

For the candidate and the reference: ordered, reversed, locally permuted (adjacent pairs swapped) and fully shuffled
inputs, with paired score changes and AUROC intervals. No shuffle-gap threshold gates anything; no linear order probe
is required.

## Amendment, 2026-09-21 — the Mamba trigger, defined before the fused comparison was computed

The brief lets Mamba enter as a temporal head "only if this comparison shows a small temporal head is useful" and
leaves *useful* undefined. Fixed now, while only branch-alone training logs exist and no fused selection or
confirmation number has been computed:

* **A small temporal head is useful** if, on the **selection part**, a GRU-head configuration beats the pooling
  configuration with the **same sampling** by **≥ +0.005** fused DS-R subset-v3.1. Checked separately for sparse and
  dense sampling.
* **If it is**, a Mamba head is trained for each sampling where the GRU head was useful, with the identical budget, and
  it joins the candidate set. The candidate is still chosen on the selection part and the stopping rule is still
  assessed only on the confirmation part. **If it is not**, no Mamba head is trained, and the report says so.
* **Mamba head:** the same place as the GRU — V-JEPA 2 tokens averaged over space within each of the 8 temporal
  steps, a linear map 1024 → 256, one selective state-space block (input-dependent step size, B and C; state size 8;
  causal depthwise convolution of width 2; gated output), mean over steps, the same evidential head. Pure PyTorch, the
  same block as in Stage 1A.

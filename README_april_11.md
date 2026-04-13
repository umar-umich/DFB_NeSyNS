# NeSyDeFake Changes — April 11, 2026

## Context

Cross-dataset CDF AUC plateau at ~88% (precomputed novlm baseline). Two suspected root causes investigated today:

1. **Train/test feature mismatch**: CLIP backbone sees an augmented frame every epoch, but precomputed `fast_semantic` (58-d) and `forensic_features` (83-d) are derived from the *original* unaugmented frames. Pixel-level forensic signals (SRM, noise residuals, FFT) are precisely what augmentation destroys.
2. **Backbone ceiling**: CLIP-ViT-L/14 is pretrained on global semantic contrast (InfoNCE), which is coarse for local artifact detection.

---

## 1. Spatial backbone ablations (SigLIP, EVA-02)

Two new detector configs replacing CLIP-L/14 with forensic-aligned spatial backbones. Everything else identical to `nesy_defake_ablation4_ccv_novlm.yaml` so the backbone is the only variable.

### 1a. SigLIP-SO400M spatial branch

**Config:** `training/config/detector/nesy_defake_ablation4_ccv_novlm_siglip.yaml`

- **Model:** `google/siglip-so400m-patch14-384` via HuggingFace `SiglipVisionModel`
- **Why SigLIP over CLIP:** CLIP's InfoNCE contrastive loss normalises per-pair scores across the whole batch — features are globally *comparative*. SigLIP uses sigmoid BCE: each image-text pair is independently scored, forcing locally self-sufficient features. Better fine-grained grounding (measured on MMVP). Training corpus WebLI (10B+ pairs) vs CLIP's LAION-400M.
- **Size:** 877M params (~3× CLIP-L), native 384×384 / patch14 → 729 tokens
- **Output dim:** 1152 (not 1024 — the rest of the pipeline adapts)
- **Batch:** 128 (down from 300; higher activation memory)
- **Normalization:** `[0.5, 0.5, 0.5] / [0.5, 0.5, 0.5]` (rescale to [-1, 1], not CLIP's norms)

### 1b. EVA-CLIP-02 ViT-L/14 spatial branch

**Config:** `training/config/detector/nesy_defake_ablation4_ccv_novlm_eva02.yaml`

- **Model:** `eva02_large_patch14_clip_224` via timm (`num_classes=0`)
- **Why EVA-02 over CLIP:** Dual pretraining — MIM (masked patch reconstruction, forces local pixel encoding) *plus* CLIP distillation (keeps semantic grounding). Inductive bias toward local pixel structure — exactly where deepfake artifacts (blending seams, noise residuals, unnatural textures) live.
- **Size:** 304M params (same as CLIP-L/14) → batch unchanged at 300
- **Output dim:** 1024 (drop-in)
- **Normalization:** CLIP-style (`[0.481, 0.458, 0.408] / [0.269, 0.261, 0.276]`)

### Spatial extractor wiring

**File:** `training/networks/nesy_defake/foundation_models/spatial_extractor.py`

- Added `_timm_model_name` field from `foundation_models.spatial.timm_model`.
- Extended `_get_required_input_size()` for SigLIP (384) and EVA-02 (224) based on the timm model name.
- New `name: siglip` and `name: eva_clip` dispatches in `_build_backbone()`.
- New `_build_eva_clip()` using `timm.create_model(..., num_classes=0)`.
- New `_build_siglip()` using `SiglipVisionModel.from_pretrained`.
- New `_forward_timm()` and `_forward_siglip()` forward paths.
- GenD-style LayerNorm-only training preserved (`_freeze_all_except_layernorms(self.backbone)` still runs on the new backbones).

**Env upgrade:** `transformers` bumped to 4.44.2 (was 4.31.0) so `SiglipVisionModel` and `Dinov2Model` are importable. Verified `peft 0.4.0` and existing CLIP/DINOv2 paths still work.

---

## 2. Removed learned projections between backbone and classifier

**Files:**
- `training/detectors/nesy_defake_detector.py`
- `training/networks/nesy_defake/fusion/multimodal_fusion.py`

### Rationale

The old `_make_projection` was `Linear(in, out) → LayerNorm → GELU` — a 3-layer non-linear transformation sitting between the frozen foundation-model features and the classifier. For a GenD linear-probe regime, this distorts the pretrained feature manifold before the classifier ever sees it. The GenD ideal is: frozen backbone → LayerNorm → classifier, with the backbone's own LayerNorms as the only trainable surface.

### Changes

- `_make_projection` → `_make_adapter`, now returns `nn.Identity()`. `spatial_proj` and `frequency_proj` apply zero learned transformation. `_zero_grad_anchor` still works correctly (no params to anchor).
- `MultiModalFusion._build_dim_adapter(in_dim, out_dim)` handles backbone→fusion dim bridging:
  - **Same dim** (`in == out`): `nn.LayerNorm(out)` only. Zero learned transformation.
  - **Different dim** (e.g., SigLIP 1152 → 1024, or concat of two 1024-d branches → 1024): bias-free `nn.Linear(in, out, bias=False) → LayerNorm(out)`. A bias-free Linear is a pure rotation/projection with no direction creation, and crucially, *no activation* — an activation would distort the pretrained manifold. ReLU/GELU is only appropriate when mixing features from independent encoders at high level, not for single-branch dim adaptation.
- `_build_concat_fusion` and `_build_attention_fusion` fallback now use `sum(self.active_dims)` (raw backbone dims) for the adapter input size, since `spatial_proj` is now `nn.Identity`.
- SigLIP config updated: `fusion.projection_dim: 1152`, `classifier.input_dim: 1152`, `causal_branch.backbone_dim: 1152`. End-to-end pure LayerNorm, no dim reduction anywhere. Even SigLIP's 1152-d features reach the classifier untransformed.

**Backbone LayerNorms remain trainable.** The `_freeze_all_except_layernorms` chain is untouched, and the `backbone_layernorms` optimizer param group continues to receive gradients. Only the projection heads on top were removed.

---

## 3. Training log folder now snapshots the detector config

**File:** `training/train.py` (rank 0 only, inside the existing DDP guard)

Added two artifacts alongside `training.log` and checkpoints:

- `detector_config.yaml` — `shutil.copy2` of `args.detector_path`. Preserves the exact as-authored YAML with comments.
- `detector_config_resolved.yaml` — `yaml.safe_dump(config, ...)` of the runtime-resolved config dict, which captures any mutations done by `train.py` itself (e.g., `config['fusion']['fused_dim']` recomputed from `active_branches`, CLI `--active_branches` overrides).

Wrapped in try/except so a filesystem hiccup never kills training. Import added: `shutil`.

### Reproducibility caveats

The YAML captures backbone name/path, output dim, freeze flags, fusion type, classifier shape, all branch configs, optimizer, and loss weights — enough to reconstruct the graph for the matching commit. It does **not** capture:

1. **Code version.** The same YAML + a different commit produces a different model (e.g., `_make_projection` → `_make_adapter` from this session). Recommendation: also dump `git rev-parse HEAD` to `git_commit.txt`.
2. **Package versions** (e.g., `transformers` 4.31 vs 4.44 affects whether `SiglipVisionModel` even imports). Recommendation: `pip freeze > pip_freeze.txt`.

Not added in this commit — flag for follow-up if we want bulletproof reproducibility.

---

## 4. OTF training — hybrid forensic feature extraction

**Goal:** Eliminate the pixel-level train/test mismatch between the augmented view the backbone sees and the precomputed forensic features derived from the original frame. Target: lift CDF AUC from ~88% toward 97%.

### Finding: full OTF in DataLoader workers is infeasible

Precompute pipelines require heavy GPU models:

- **`fast_semantic` (58-d)**: InsightFace (ONNX/GPU) + DeepFace (TF/CPU) + LibreFace (PyTorch/GPU) + MediaPipe. ~300 ms/image.
- **`forensic_features` (83-d)**: SegFormer face parser (PyTorch fp16/GPU) + InsightFace + MediaPipe. ~100 ms/image just for the parsing map.

In 12 forked DataLoader workers these would (a) duplicate multi-GB of GPU state and OOM, (b) break TensorFlow under fork, and (c) bottleneck the dataloader at 500+ ms/image effective, tanking throughput ~10–20×.

### Finding: `fast_semantic` is augmentation-robust in practice

Its 58 features are face-level semantic invariants — gender, age, pose, emotion, AUs, landmarks. A flipped/blurred/JPEG-60 face is still the same gender, age, emotion. The mismatch is small relative to the infeasibility cost. **Keep precomputed.**

### Finding: `forensic_features` splits cleanly at index 30

Read of `preprocessing/forensic_helpers.py:430` (`extract_forensic_features`) shows the 83 features break into two groups:

| Range | What | Cheap OTF? | Aug-sensitive? |
|---|---|---|---|
| 0–25 | Region boundary gradients, regional blur, symmetry, color, DCT by region | No — needs SegFormer parsing map | Moderate |
| 26–29 | antispoof, det_score, blendshape_sym, landmark_jitter | No — needs InsightFace/MediaPipe | Low |
| **30–37** | **PPNC — paired patch noise consistency** | **Yes — pure numpy** | **High** |
| **38–45** | **CCNC — cross-channel noise** | **Yes — pure numpy** | **High** |
| **46–60** | **SRM filter bank — 5 filters × (mean/std/kurt)** | **Yes — pure cv2/numpy** | **Very high** |
| **61–72** | **Multi-scale noise residuals** | **Yes — pure numpy** | **Very high** |
| **73–82** | **FFT spectral bins + slope + HF ratio** | **Yes — pure numpy** | **High** |

**53 of 83 features (indices 30..82)** are pure `cv2`/`numpy`/`scipy` — no GPU, no model state, measured at **15–20 ms/image** in-process. These are exactly the signals where blur/JPEG/brightness/noise augmentation actually affects the output.

### Implementation: `PixelForensicExtractor`

**New file:** `training/dataset/pixel_forensic_extractor.py`

Stateless extractor wrapping the 5 pure-numpy helpers from `preprocessing/forensic_helpers.py`:

- `extract_ppnc` → indices 30..37
- `extract_ccnc` → indices 38..45
- `extract_srm` → indices 46..60
- `extract_multiscale_noise` → indices 61..72
- `extract_spectral` → indices 73..82

`__slots__ = ()` — no instance state. Safe to construct per-worker or share across threads. `__call__(image_rgb)` returns a `(53,)` float32 array.

Robustness:
- `None` or wrong-shape input → zeros.
- Non-contiguous numpy slices → made contiguous via `np.ascontiguousarray`.
- cv2 color-conversion error → zeros.
- Any individual block raising → zeros for that block, other blocks still extracted.
- `NaN`/`Inf` from degenerate regions → `np.nan_to_num`.
- Module-level `sys.path.insert(preprocessing/)` so the dataset doesn't need to care about layout.

### Hybrid path in `nesy_defake_dataset.py`

**File:** `training/dataset/nesy_defake_dataset.py`

- `__init__`: new `pixel_forensic_otf` flag gated on `forensic_features.pixel_otf && enabled && output_dim == 83`. Lazy-imports the extractor only when the flag is on. Captures `PIXEL_SLICE = slice(30, 83)` from the module.
- `_load_single_frame`: after the existing `_load_forensic_features(frame_path)` call, when `pixel_forensic_otf=True`:
  1. Clone the 83-d tensor loaded from the `.pt` file (keeps indices 0..29 as-is).
  2. Call `self._pixel_forensic_extractor(image)` on the already-augmented numpy RGB (the same variable the backbone normalizes next).
  3. Overwrite `forensic_features[30:83]` with the OTF result.

The 0..29 regional features stay precomputed (augmentation-insensitive enough, and the SegFormer dependency makes OTF infeasible). This is the ~80/20 trade: fix the 53 pixel-level features at ~15 ms each, accept the structural slice staying stale.

### OTF config rewrite

**File:** `training/config/detector/nesy_defake_ablation4_ccv_novlm_otf.yaml`

Replaced "backbone + EDL only" plan with the full hybrid:

- Header rewritten to document the fast_semantic robustness / forensic hybrid rationale.
- `ablation_mode: causal_edl` (was `spatial_edl`).
- `concept_branch.enabled: true`, `causal_branch.enabled: true` — symbolic branches back on.
- `edl.nesy_fusion: true`, `aux_weight: 0.1`, `disagreement_weight: 0.05`.
- `fast_semantic.enabled: true`, `consistency_rules.enabled: true`, `forensic_features.enabled: true`.
- **New flag** `forensic_features.pixel_otf: true` — triggers the hybrid path.
- `use_refined_features: false` (still novlm).
- **`class_weights: [1.0, 1.0]`** — was `[1.8, 0.6]`. The 3:1 real weight was compensating for missing fake-specific signal in `spatial_edl`. With concept + CCV branches back on, the tilt overshoots (see §5 analysis below). User-set `train_batchSize / test_batchSize: 172` (after the `172` tuning for hybrid workload).

### Smoke test — hybrid OTF dataset

Constructed `NeSyDeFakeDataset` with the new config on the 184,315-frame FF++ split:

- **No-aug determinism**: back-to-back `ds[0]` calls produced byte-identical 83-d vectors — extractor is deterministic, splice is consistent.
- **Regional stays precomputed** across two seeds: `ff[:30]` identical, confirming the `.pt` cache path is reused.
- **Pixel slice changes under augmentation**: SRM `[46:51]` shifted from `[4.27, 11.97, 81.08, 4.12, 9.66]` (seed 1) to `[3.95, 9.58, 81.45, 3.90, 7.55]` (seed 42). The train/test-mismatch fix actually lands — forensic features now match the view the backbone sees.
- **All finite**, no NaN/Inf.
- **Single-worker timing**: 134.5 ms/sample end-to-end (disk read + aug + pixel forensic + two `.pt` loads + normalize). With 12 workers, ~11 ms effective per sample — well under the GPU forward time at batch 172.

---

## 5. Precomputed vs OTF-spatial_edl diagnostic (class-weight analysis)

Observed at epoch ~27–30 on `nesy_defake_ablation4_ccv_novlm` (precomputed) vs the earlier `nesy_defake_ablation4_ccv_novlm_otf` in `spatial_edl` mode:

| Run | FF++ real | FF++ fake | CDF real | CDF fake | CDF AUC |
|---|---|---|---|---|---|
| Precomputed (novlm, full causal_edl) | 0.79 | 0.95 | 0.80 | 0.81 | 0.88 |
| OTF spatial_edl (prev, no symbolic) | **0.93** | **0.74** | **0.93** | **0.52** | 0.86 |

Both used `class_weights: [1.8, 0.6]` (real : fake = 3:1).

**Diagnosis:** The 3:1 real weight was originally a "real bias fix" for the precomputed setup where concept + CCV branches inject fake-specific signals (forensic artifacts, consistency rules) that pushed the decision boundary toward fake. Removing those branches (spatial_edl) removed the fake signal, so the class-weight tilt dominated and over-predicted real — especially on CDF (fake acc collapsed to 0.52 while AUC stayed at 0.86, a classic threshold/bias decoupling).

**Implication for the hybrid OTF run:** with symbolic branches re-enabled *and* augmentation-matched pixel forensic features, the class-weight tilt should be dropped to neutral. Hence `[1.0, 1.0]` in the new OTF config.

---

## What to expect

| Ingredient | novlm (precomputed) | OTF-spatial_edl (prev) | OTF-hybrid (new) |
|---|---|---|---|
| Concept + CCV branches | on | off | **on** |
| Pixel forensic train/test match | stale | N/A | **matches augmented view** |
| Regional forensic (0..29) | stale | N/A | stale (SegFormer dep) |
| `fast_semantic` (58-d) | stale, low sensitivity | N/A | stale, low sensitivity |
| Class weights | [1.8, 0.6] | [1.8, 0.6] | **[1.0, 1.0]** |
| CDF AUC | ~0.88 | ~0.86 | target 0.96–0.97 |

Hypothesis being tested: *a meaningful chunk of the 88% CDF ceiling is pixel-level train/test mismatch on SRM / noise / FFT*. If this is right, CDF AUC should tick up noticeably and fake accuracy on CDF should recover. If it's not, we'll at least match the novlm baseline and will know the remaining ceiling is backbone capacity — which feeds into the SigLIP/EVA-02 ablations from §1.

## Known limitations

1. **Indices 0..29 still stale under augmentation.** Region boundary gradients, regional blur, DCT-by-region, and quality scores need SegFormer parsing maps that can't run in workers. Upper-bounds how much the hybrid fix alone can help. Path B for full closure: precompute features on 4–8 pre-augmented copies of each frame offline, sample one at train time.
2. **`fast_semantic` (58-d) still derived from the original frame.** Semantic attributes are augmentation-robust, so this matters less than the forensic slice, but it is not zero-impact.
3. **Augmentation settings in OTF config are still slightly stronger than the novlm baseline** (`blur_prob: 0.15`, `brightness/contrast: 0.15`, `quality_lower: 50`, `gaussian_noise_sigma: 0.01`). If you want a clean A/B isolating "hybrid forensic" as the only variable, match novlm's aug settings first.

## Files touched

**New:**
- `training/dataset/pixel_forensic_extractor.py`
- `training/config/detector/nesy_defake_ablation4_ccv_novlm_siglip.yaml`
- `training/config/detector/nesy_defake_ablation4_ccv_novlm_eva02.yaml`
- `README_april_11.md` (this file)

**Modified:**
- `training/config/detector/nesy_defake_ablation4_ccv_novlm_otf.yaml`
- `training/dataset/nesy_defake_dataset.py`
- `training/detectors/nesy_defake_detector.py`
- `training/networks/nesy_defake/fusion/multimodal_fusion.py`
- `training/networks/nesy_defake/foundation_models/spatial_extractor.py`
- `training/train.py`

**Environment:**
- `transformers`: 4.31.0 → 4.44.2 (dfb_nesy conda env)

## Run commands

```bash
# Hybrid OTF (primary experiment for CDF 97% target)
python training/train.py --detector_path \
  training/config/detector/nesy_defake_ablation4_ccv_novlm_otf.yaml

# Spatial backbone ablations
python training/train.py --detector_path \
  training/config/detector/nesy_defake_ablation4_ccv_novlm_eva02.yaml
python training/train.py --detector_path \
  training/config/detector/nesy_defake_ablation4_ccv_novlm_siglip.yaml
```

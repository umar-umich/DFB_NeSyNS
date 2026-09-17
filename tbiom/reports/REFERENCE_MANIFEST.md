# Reference manifest — DISCERN-v2 finish line (Stage 0)

Resolved from the live tree on 2026-09-17; repository commit `9cf8895`. Every finish-line experiment (1–4) cites this file and uses exactly these artefacts. Nothing here is from memory: paths, hashes and fitted parameters were read or recomputed by `analysis/tbiom/temporal/finish_line.py manifest`.

## 1. The three branches

| view | what it is | resolved checkpoint | sha256 (first 16) | size |
|---|---|---|---|---:|
| **T** | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), LoRA r=8 α=16 dropout 0.1 on q/v of every encoder layer + evidential head, **B-augmented** training | `logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt` | `6f0d9c63a6c052de` | 205 MB |
| **APP** | Appearance view of the warm+gate DISCERN detector: CLIP ViT-L/14 + LoRA r=8 α=16 (q_proj, v_proj), evidential head | `/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | `fdc924a7cc9e1579` | 2.4 GB |
| **OM** | Off-manifold residual view of the same checkpoint: β-VAE (latent 32, β_KLD 2.0, λ_align 1.0, λ_vae 0.7) on the shared CLIP feature path, evidential head | same file | `fdc924a7cc9e1579` | — |

The face prior (FSFM student, `logs/fpad/studentA_preserve_seed42/epoch_009.pth`) exists inside the warm+gate checkpoint but **is not part of the reference**: it is excluded from Experiments 1–4 and appears only in explicitly labelled historical comparisons (EXP7B).

## 2. Temporal-branch recipe (from its training record, not from memory)

```json
{
 "model": "vjepa2",
 "seed": 42,
 "best_epoch": 6,
 "best_dev_v31": 0.8091709420774639,
 "budget": {
  "augment": "B",
  "arm": "lora",
  "lora": {
   "r": 8,
   "alpha": 16,
   "dropout": 0.1,
   "lr": 0.0002
  },
  "lr_blocks": 0.0002,
  "lr_head": 0.002,
  "epochs": 10,
  "unfrozen_blocks": 2,
  "trainable_block_params": 786432,
  "head_params": 133506,
  "batch": 32,
  "frames": 16
 }
}
```

## 3. Preprocessing and aggregation

- **T**: one clip per video — 16 frames sampled uniformly (`np.linspace`, rounded) over the video's stored 256-px face crops, **in temporal order**; V-JEPA 2's own `AutoVideoProcessor` (256-px, its own mean/std). The branch emits ONE Dirichlet per video, so no frame aggregation is needed: `p = α₁/S`, `u = 2/S`.
- **APP/OM**: per-frame 224-px CLIP crops from the same H5 stores, each frame's evidential head giving (p, u); **per-video mean of every frame quantity first, then fuse** (the corrected-table rule). Video id = the directory containing the frame (`analysis/tbiom/video_id.py`).
- Processed conditions are applied to the stored crops before each branch's own preprocessing; JPEG q40 and resize 0.60 are evaluation-only.

## 4. DS-R, exactly

For view b with opinion (p_b, u_b): S_b = K/u_b with K = 2, α_b = [(1−p_b)S_b, p_b S_b]. Shafer discounting by a reliability weight w_b: α'_b = 1 + w_b(α_b − 1) (w_b = 1 leaves the opinion untouched, w_b = 0 makes it vacuous). The discounted opinions are combined **sequentially by Dempster's rule** — the identical operator DISCERN deploys (`analysis/tbiom/symbolic_overlay.py`; checked numerically, max |Δp| = 0.0e+00 at w = 1). The fused opinion carries p_f, vacuity u_f = K/S_f and accumulated conflict k_f = 1 − ∏(1 − k_pair).

Weights are searched over w ∈ {0, .25, .5, .75, 1}³ with max w = 1 (a global scale is unidentifiable), maximising the v3.1 objective on the fitting fold.

**Matched comparator (mandatory, everywhere):** logistic stacking on exactly the same inputs — the six numbers (p, u) of T, APP, OM — standardised, `LogisticRegression(C=1, class_weight='balanced')`, fitted on the same fitting fold.

## 5. Folds and fitted parameters

Development = VALmix + VALmix-v2 + FF++ val videos, in-scope cells only. Each split is a source-grouped 50/50 `GroupShuffleSplit(test_size=0.5, random_state=k)`, k = 0…4, over the union-find source groups of `exp1_fusion_audit.groups_for` (identical to every earlier audit). τ = EER of the fused score on the selection fold's FF++ val videos.

| split | fit videos | fit groups | selection videos | selection groups | DS-R w (T, APP, OM) | τ DS-R | τ LR |
|---|---:|---:|---:|---:|---|---:|---:|
| 0 | 2911 | 1364 | 2903 | 1364 | (0.25, 0.25, 1.0) | 0.5648 | 0.8244 |
| 1 | 2971 | 1364 | 2843 | 1364 | (0.25, 0.25, 1.0) | 0.5648 | 0.5209 |
| 2 | 2942 | 1364 | 2872 | 1364 | (0.25, 0.25, 1.0) | 0.3778 | 0.8235 |
| 3 | 2965 | 1364 | 2849 | 1364 | (0.25, 0.25, 1.0) | 0.5648 | 0.7373 |
| 4 | 2846 | 1364 | 2968 | 1364 | (0.25, 0.25, 1.0) | 0.6376 | 0.8188 |

Logistic coefficients per split (standardised inputs):

| split | p_T | u_T | p_APP | u_APP | p_OM | u_OM | intercept |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | +1.141 | +0.116 | -1.995 | +0.134 | +4.080 | +0.821 | +1.017 |
| 1 | +1.105 | +0.023 | -1.843 | +0.170 | +3.784 | +0.803 | +0.839 |
| 2 | +1.159 | +0.116 | -1.794 | +0.259 | +3.724 | +0.648 | +0.893 |
| 3 | +1.171 | +0.140 | -2.006 | +0.235 | +4.183 | +0.783 | +1.029 |
| 4 | +1.139 | +0.139 | -2.030 | +0.241 | +3.968 | +0.721 | +0.831 |

## 6. Evaluation sets

- Development (clean): VALmix, VALmixV2, FFpp_val; processed development: the same selection-fold videos under JPEG q40 and resize 0.60.
- Tests (read only after fitting): CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24, FFIW, KoDF, IDForge, DFDM, plus the CDFv3-reenactment and KoDF first-order-motion slices and the CDFv2 / DFDCP processed versions.
- Out of scope and dropped from every table: lip-sync-only (IP_LAP), audio-driven fakes, adversarial subsets.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).

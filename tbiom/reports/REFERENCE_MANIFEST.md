# Reference manifest — executable brief, Stage 0 audit

Resolved from the live tree on 2026-09-20, commit `baf16b6`, by `analysis/tbiom/stage0_audit.py`; every number below is recomputed, not retyped. Every later experiment in `tbiom/docs/discern-v2-executable-brief.md` cites this file. Raw numbers: `STAGE0_AUDIT.json`.

## 1. The three reference branches

| view | checkpoint | sha256 (first 16) |
|---|---|---|
| T (V-JEPA 2 LoRA, B-augmented) | `logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt` | `6f0d9c63a6c052de` |
| APP + OM (warm+gate, epoch 19) | `/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | `fdc924a7cc9e1579` |

APP and OM are the Appearance and Off-manifold views of one checkpoint (shared CLIP ViT-L/14 + LoRA r8 α16 path; β-VAE latent 32, β_KLD 2.0, λ_align 1.0, λ_vae 0.7). The face prior inside that checkpoint is outside the reference. The hashes match the previous manifest (yes).

**Preprocessing and aggregation order.**
- T: one clip per video, 16 frames at `linspace(0, n−1, 16)` over the video's stored crops, in temporal order, V-JEPA 2's own 256-px processor; the branch emits one Dirichlet per clip, so there is no frame aggregation.
- APP/OM: every stored crop scored separately at 224 px (CLIP preprocessing); per-frame (p, u) are **averaged per video first**, then the three views are discounted and fused.
- Order: per-frame spatial opinions → per-video mean → reconstruct α from (p, u) → discount each view → Dempster combination T ⊕ APP ⊕ OM (associative).

Temporal-branch training record:

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

## 2. The discount rule, read from code

`analysis/tbiom/temporal/finish_line.py::dsr` (identical in `dsr_system.py`):

```python
S = K / u                                   # K = 2 classes
alpha = [(1 - p) * S, p * S]                # reconstruct the Dirichlet from (p, u)
alpha' = 1 + w * (alpha - 1)                # discount
```

That is **evidence scaling**: e' = w·e, S' = w·Σe + K, b' = w·e / (w·Σe + K), u' = K / (w·Σe + K). It is **not** belief-mass (Shafer) discounting, b' = w·b, u' = 1 − w·(1 − u). The two coincide only at w = 1. The previous manifest called this rule "Shafer discounting"; that name was wrong and is corrected here — the equations it printed were right.

Numerical check on 2,000 opinions (p ∈ [0.05, 0.95], u ∈ [0.02, 0.9]):

| w | max \|Δp\| code vs belief-mass | max \|Δu\| | mean \|Δu\| |
|---:|---:|---:|---:|
| 0.25 | 0.2886 | 0.6788 | 0.1646 |
| 0.5 | 0.2053 | 0.4705 | 0.1453 |
| 0.75 | 0.1055 | 0.2384 | 0.0830 |

The combination after discounting is Dempster's rule on the discounted opinions, identical to DISCERN's in-model `DS_Combin` / `_combine_two_opinions` (max |Δp| 2.2e-16 at w = 1): with b_i = e_i / S_i and u_i = K / S_i, conflict C = Σ_{j≠k} b_1j b_2k, b = (b_1 b_2 + b_1 u_2 + b_2 u_1)/(1 − C), u = u_1 u_2 / (1 − C), α = b·K/u + 1. DISCERN's own face-prior gate uses the same evidence-scaling form (`1 + g·(α − 1)`).

**Consequence for Stage 1A:** every controller predicts w per view and applies α' = 1 + w(α − 1), unchanged. Changing to belief-mass discounting would be a separate experiment, never combined with a new predictor.

Fitted weights on the five source-grouped splits (from the previous manifest): (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0); (0.25, 0.25, 1.0) for (T, APP, OM).

## 3. Cache inventory and segment granularity

**What a 'frame' is here.** The stored crops are a sparse, even sample of each source video, not its frames:

| development source file | videos | stored crops per video (min / median / max) | median raw-frame step between stored crops |
|---|---:|---|---:|
| VALmix | 1350 | 1 / 32 / 32 | 11.0 |
| VALmixV2 | 4852 | 1 / 16 / 16 | 21.0 |
| FFpp_val | 700 | 31 / 32 / 32 | 14.0 |

So a video is 16–32 stills roughly 10–15 source frames apart (≈ 0.3–0.5 s at 30 fps). `/data/umar/Datasets/tbiom_frames` (29 GB) holds the same 32-still sampling for CDFv2, CDFv3, DFD, DFDC, DFDCP and DFEval24 — it is not denser.

**What exists per (view, granularity, condition):**

| view | granularity | clean development | JPEG q40 / resize 0.60 development | tests |
|---|---|---|---|---|
| APP, OM | per stored crop | yes (`warmgate_e19_*`) | yes (`p_dev_warmgate_e19_*`) | clean all ten; processed CDFv2, DFDCP |
| T | one opinion per video (full span) | yes | yes | clean all ten; processed CDFv2, DFDCP |
| T | per segment (4 contiguous quarters + full) | yes (`logs/tbiom/acquisition/`) | **no** | **no** |
| quality descriptors (blur, blockiness, contrast, motion, luma) | per segment | yes, same cache | **no** | **no** |

Segment cache detail (clean development only):

| source | videos | segments/video | distinct stored crops per segment (min / median / max) | segments whose 16-frame clip repeats crops | degenerate (empty) segments |
|---|---:|---:|---|---:|---:|
| VALmix | 1350 | 4 | 0 / 8 / 8 | 5388 of 5400 | 12 in 6 videos |
| VALmixV2 | 3764 | 4 | 0 / 4 / 4 | 15035 of 15056 | 21 in 10 videos |
| FFpp_val | 700 | 4 | 7 / 8 / 8 | 2800 of 2800 | 0 in 0 videos |

Timestamps: the stored-crop file names carry the source frame index (`000.png`, `014.png`, …; `frame_0010.png`), so segment order and approximate time are recoverable; source fps is not stored.

## 4. Gate

Aligned segment-level opinions from all three views **do not exist** in the form Stage 1A needs:

1. **Processed conditions have no segment-level temporal opinions and no quality descriptors.** The controllers are trained on clean *and* processed examples, so these must be generated by local clip inference before any controller is fitted. This adds GPU cost (the clean pass took ≈13 GPU-minutes for 29k segment clips; two processed conditions ≈ 30 more).
2. **At the stored density a segment holds 4–8 distinct crops**, so every per-segment V-JEPA clip is those crops repeated to 16 frames. Each segment is still a genuinely different observation — its frames are distinct from the other segments' — so this is not the forbidden repetition of one video-level opinion across timestamps; but the temporal branch sees a stuttered clip at segment granularity, and that is recorded as a limitation of every segment-level result.
3. **Degenerate segments** (videos with fewer crops than segments) were filled with the full-span opinion in the acquisition cache. That *is* the forbidden simulation; those segments are excluded, not filled, in every Stage 1A fit.
4. **Spatial segment opinions need no inference**: they are exact means of the cached per-frame opinions over the same crops the temporal segment used.

**Processing-sensitive branch for Stage 1B** (from Experiment 3 on the same reference): development v3.1 drop clean → JPEG q40 is T 0.0498, APP 0.0994, OM 0.0813. The spatial path is the more processing-sensitive one, and APP and OM share the adapted CLIP path, so 1B treats them as one coupled spatial experiment.

**Pass condition.** Three checkpoints named and hashed; the discount rule read from code and verified (evidence scaling); segment granularity recorded (4 segments of 4–8 distinct stored crops, clean development only). Stage 1A proceeds only after the missing processed segment opinions and descriptors are generated.

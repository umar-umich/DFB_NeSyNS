# V1_PARITY_REPORT — FS-VFM pipeline fidelity and the crop decision

Deliverable for build spec §26, covering §6's numerical parity requirement and §8's crop parity
control. Reproduce with:

```bash
/data/umar/miniconda3/envs/dfb_nesy/bin/python analysis/discern_v2/fsvfm_parity.py \
    --videos 40 --frames 8 --out analysis/discern_v2/V1_parity
```

---

## 1. §6 — do we run the authors' pipeline? **Yes, exactly.**

§6 requires a comparison against the official FS-VFM downstream path matching the *full* input
pipeline — crop convention, normalization, pooling rule, CLS vs mean-patch, final normalization —
not normalization alone.

| quantity | result |
|---|---|
| max absolute difference | **0.0** |
| mean absolute difference | **0.0** |
| minimum cosine similarity | **1.0000** (0.99999994) |
| pooling compared | `global_pool` |

The official side is built from the cloned `FSFM-CVPR25` (`fsvfm/models_vit.py`) and loaded with
the official recipe (`interpolate_pos_embed`, `strict=False`, shape-mismatched keys dropped), so
this compares two implementations rather than our code against itself.

Supporting facts, all verified rather than assumed:

- Normalization is **read** from the shipped `pretrain_ds_mean_std.txt`
  (mean `[0.54822075, 0.42340535, 0.36546516]`, std `[0.27891761, 0.24385408, 0.23493893]`) and
  checked, never hardcoded. It is **not** ImageNet, and it matches the constants in `GenD_NeSy`'s
  independent port exactly.
- All 296 encoder tensors load. Only `fc_norm.{weight,bias}` and `head.{weight,bias}` are missing,
  which is expected: the pretraining checkpoint has no downstream head.
- In `global_pool` mode the ViT deletes `norm` and creates `fc_norm`; frozen, that is LayerNorm at
  default init. `cls` pooling instead keeps the pretrained `norm`. This is a genuine fork in what
  `z_ref` means; `global_pool` is the V1 default because it is what the authors' downstream path
  and released fine-tuned checkpoints use.

---

## 2. §8 — what does the aligned crop cost?

**Decision in force (Umar, 2026-08-18):** the reference branch reads the **existing aligned
crop** with FS-VFM normalization, uniformly across all sources, rather than §8's native
DLIB+30% default. Rationale: it avoids generating and storing a second crop cache across every
split. Recorded as a deviation from §8; this control exists to measure its cost.

### 2.1 Crop convention, confirmed from the official repo

- `face_scale = 1.3` — the "30% additional cropping"
  (`datasets/finetune/preprocess/config/default.py:14`, identical in the pretrain config).
- The downstream deepfake-detection path calls `extract_and_save_face`, the **non-aligned**
  variant, so **no landmark predictor is involved** and no `shape_predictor_81_face_landmarks.dat`
  download is needed.
- `get_boundingbox` takes a square box around the DLIB detection, enlarges it by `face_scale`, and
  clips it to the frame.

### 2.2 Feature-side agreement — high

Measured on 64 paired FF++ frames (16 videos × 4 frames, real and Deepfakes), where each native
crop is re-extracted from the **source video at the same frame index** as its aligned counterpart,
so the pair describes one image rather than two similar ones.

| quantity | value |
|---|---|
| mean cosine(native, aligned) | **0.9942** |
| minimum cosine | 0.9885 |
| mean `1 − cosine` | 0.0058 |
| mean norm ratio (aligned / native) | 1.000 |
| DLIB detection failures | 0 of 64 |

Read: the frozen FS-VFM representation is nearly unchanged by the crop convention on this subset,
which is early support for the aligned-crop default.

### 2.3 Probe-side gap — `TODO(run)` at a usable subset size

The probe arm is **not yet interpretable**. At 16 videos (8 groups per class), both arms scored
AUROC ≈ 0.0 — consistently inverted across folds, not noisy. That is the probe separating
*identities* rather than manipulations: it fits the training identities and every held-out
identity lands on the wrong side. Reporting "native 0.016 vs aligned 0.000, gap +0.016" would be a
meaningless number with three decimal places, so the script refuses and says why.

`TODO(run)`: rerun with `--videos 40 --frames 8` (Umar). The script emits `NOT INTERPRETABLE`
whenever neither arm beats chance, and reports per-fold AUROCs and their spread otherwise, so the
result cannot be over-read.

### 2.4 Coverage consequence, stated up front

Native DLIB crops require source frames, which exist locally only for:

| source | source frames | native crop reachable |
|---|---|---|
| FaceForensics++ (incl. DeepFakeDetection) | videos, 77 GB | yes |
| Celeb-DF v1 / v2 / v3 | videos | yes |
| Deepfake-Eval-2024 | videos | yes |
| DFDC | 256×256 crops only | **no** |
| DFDCP | preprocessed archive only | **no** |
| UADFV | preprocessed only | **no** |
| DF40 | mixed (some 1024×1024 generated images) | partial |

So if the probe gap ever turns out to matter, switching to native crops is not only a
preprocessing change but a **coverage** change: three evaluation sources could not follow. That is
recorded now rather than discovered during the OOD sweep.

---

## 3. What would change the decision

The aligned-crop default should be revisited if the §2.3 run at 40+ videos shows a native-vs-aligned
probe gap that exceeds §22's significance floor (|ΔAUC| ≥ 0.01) **and** is consistent across
folds. In that case the options are, in order: (a) native crops for the sources that have source
frames, with the crop mode logged per sample and DFDC/DFDCP/UADFV explicitly marked; (b) fetching
raw DFDC/DFDCP video to restore uniformity. Both are 🟡 ASK-UMAR.

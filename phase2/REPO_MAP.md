# Stage -1 — Repository map

Resolved by inspection on 2026-08-20, branch `discern-v2-phase2`, repo `/data/umar/Repos/DFB_NeSyNS`.
Every path below was listed on disk or read out of a file; nothing here is inferred from a name.

Two items change downstream effort more than the rest and are surfaced first, as the brief asks.

---

## ⚠️ Surfaced up front

### Item 3 (DF40 staging) — **no gap.** All five brief-named methods are staged and loadable.

`danet`, `mcnet`, `tpsm`, `facevid2vid`, `heygen` all have both a frames directory under
`/data/umar/Datasets/df40/test/` and a per-method JSON under `dataset_json/`. The V1 harness
**already has a DF40 code path** (`training/eval_v1.py --df40`, backed by
`training/dataset/df40_paths.py`), added during V1 and measured at 60/81 methods resolving at
rate 1.000. No new loader is needed for Stage 0.

### Item 6 (diverse-real corpora) — **this is the real gap.**

The machine holds exactly **one** real-face corpus that is not itself an evaluation source:

| corpus | path | size | identity metadata | overlap risk |
|---|---|---:|---|---|
| FFHQ-256 subset | `/data/umar/Datasets/ffhq256_subset/` | 8,750 PNG, 927 MB | none (`000000.png` … flat) | Flickr-sourced; no known CDF/DFDC subject overlap |

Everything else under `/data/umar/Datasets/` is an evaluation corpus (Celeb-DF v1/v2/v3, DFDC,
DFDCP, UADFV, Deepfake-Eval-2024, FF++) or a subset of one. Specifically:

* `/data/umar/Datasets/df40/real/` contains **only** `FaceForensics++` and `Celeb-DF-v2` — DF40's
  "authentic" halves are borrowed from the corpora we test on, so it is not an independent source
  of diverse reals.
* `/data/umar/Datasets/ide_data/` is `Celeb-real` + `Celeb-synthesis`, i.e. Celeb-DF again.
* **No VGGFace2, no CelebA-HQ, no LAION-face** anywhere on the machine.

Consequences for Stage 1.1, recorded now so the choice there is made on evidence:

1. FFHQ-256 is identity-disjoint from Celeb-DF and DFDC (Flickr photographs vs. YouTube celebrity
   footage and paid crowdworkers), which satisfies the brief's first constraint cleanly.
2. It is **weak on the brief's second constraint**. 8,750 still images from one curated Flickr
   collection is not "coverage across capture conditions"; FF++ reals alone are 23,039 frames.
   FFHQ has no video compression, no motion blur, no broadcast pipeline.
3. **Crop convention differs.** FFHQ ships in its own alignment (eyes-level, ~1024 originally,
   here 256). V1's reference branch reads the DLIB-derived aligned crops in
   `/data/umar/Datasets/preprocessed/`. Fitting `P_R` on FFHQ crops and evaluating it on
   preprocessed crops makes `r_ref` partly a *crop-convention* residual — the exact failure mode
   Stage 1 is trying to remove. Re-cropping FFHQ through the same DLIB pipeline is a prerequisite,
   not an optimisation.
4. The realistic option is therefore **FF++ reals ∪ re-cropped FFHQ**, not FFHQ alone — breadth
   added to the existing fit rather than substituted for it. 🟡 **ASK-UMAR**: this is a corpus
   choice, and the alternative (fetch VGGFace2 or CelebA-HQ, ~40 GB and hours of preprocessing) is
   a 🔴 download decision only Umar can make.

---

## Item 1 — V1 evaluation entry point

`training/eval_v1.py` — produced every V1 per-source and per-branch video-AUROC number.

```
--checkpoint PATH             Stage-B epoch checkpoint
--detector-config PATH        default training/config/detector/nesy_defake_d1_v.yaml
--datasets NAME [NAME ...]    source list; names key into the shared test config
--output DIR
--df40                        treat --datasets as DF40 per-method names
--dataset-json-folder PATH    override the JSON folder (DF40 keeps its own)
--stage-de PATH               apply a frozen stage_de.pt (gates + risk + defer policy)
--batch-size / --workers / --max-batches / --device / --overwrite
```

Config of record for the run itself: `training/config/discern_v2/V1_CONFIG.yaml`.
Source lists are passed as bare `--datasets` names, resolved through the shared test config's
`label_dict`; DF40 injects its own per-method label dict at runtime
(`df40_paths.label_dict_for`, `eval_v1.py:355`).

Writes `results_epoch_N.json` (per-source AUROC for `fused_gated`, `fused_ungated`, `sem`, `ref`,
`proc`, `direct`) and `per_sample_epoch_N.parquet`.

## Item 2 — per-sample dump

`logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet` — **387,913 rows**, 81 MB.
This is what Stage 0.3 and 0.4 read; no re-forward pass is needed.

Schema (45 columns):

| group | columns |
|---|---|
| keys | `dataset`, `key`, `video_id`, `label` |
| fused | `prob_fused`, `u_fused`, `ds_conflict`, `ds_degenerate` |
| reliability | `V`, `C`, `A` |
| per branch (`b` ∈ sem/ref/proc) | `p_b`, `u_b`, `e_b_real`, `e_b_fake`, `valid_b` |
| §4.2 control | `p_direct`, `u_direct` |
| process diagnostics | `proc_mse_mean`, `proc_mse_std`, `proc_mse_max`, `proc_mse_p90`, `proc_lpips_proxy`, `proc_center_ratio` |
| reference diagnostics | `ref_residual_norm`, `ref_angle` |
| gated outputs | `prob_gated`, `u_gated`, `V_gated`, `C_gated`, `A_gated`, `risk`, `decision`, `q_ref`, `q_proc` |

Row counts per source: Celeb-DF-v3 173,295 · DFDC 132,116 · Deepfake-Eval-2024 23,209 ·
FaceForensics++ 22,400 · DFDCP 17,222 · Celeb-DF-v2 16,572 · UADFV 3,099.
**DF40 is absent from this export** — Stage 0.2 must produce it.

Reader: `pandas.read_parquet` needs pyarrow, which is in `dfb_nesy` and **not** in `base`.
Use `/data/umar/miniconda3/envs/dfb_nesy/bin/python` for everything in this phase.

## Item 3 — DF40 staging (detail)

Root `/data/umar/Datasets/df40`, 39 method directories under `test/`, 81 per-method JSONs under
`dataset_json/`.

* `danet`, `mcnet`, `tpsm`, `facevid2vid` each ship **two** JSONs, `_ff` and `_cdf`, i.e. the same
  generator driven on FF++ source footage and on Celeb-DF source footage. The brief names
  `facevid2vid (cdf)` specifically; the `_ff` and `_cdf` arms must be kept as separate rows.
* `heygen` ships a single `heygen.json` (no `_ff`/`_cdf` split).
* `DF40_all.json` is unusable — every entry sits under `train`, `test` is empty
  (`df40_paths.py:29`). Use per-method JSONs.
* Path resolution is candidate-based, not a single rewrite: video methods and image methods have
  different on-disk layouts, and a rule generalised from one resolved 21.6% of the other
  (`df40_paths.candidates`, `df40_paths.py:60`). Rates are measured per method by
  `df40_paths.verify()` and recorded, never assumed.

**No new loader required.** Recorded here per the brief's instruction to note additions: the DF40
code path was added during V1, not during this phase.

## Item 4 — FS-VFM reference branch

| what | where |
|---|---|
| encoder | `training/networks/discern_v2/fsvfm_encoder.py` — `weights/FS-VFM/checkpoint-599.pth`, sha256 `3fd99324…26bc87`, `fsfm_vit_large_patch16`, frozen |
| normalization of the encoder input | `weights/FS-VFM/pretrain_ds_mean_std.txt` — mean `[0.5482, 0.4234, 0.3655]`, std `[0.2789, 0.2439, 0.2349]`. **Not ImageNet**; read from file, never hardcoded |
| `P_R` fit script | `analysis/discern_v2/fit_reference.py` (`--features` .npz → `--out` dir) |
| `P_R` artifact | `configs/discern_v2/reference/reference_C3_ae_cosine.pt` |
| fit provenance | `configs/discern_v2/reference/fit_report.json` |
| branch wrapper | `training/networks/discern_v2/reference_branch.py`, core in `reference.py` |

**Which reals it is currently fit on — the thing Stage 1 exists to change:**

```
"features": "cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz"
"cache_slice": "FaceForensics++/train"      "cache_split": "train"
"n_real": 23039        feature_dim 1024   latent_dim 128   hidden_dim 256
arm C3_ae / objective cosine, adam lr=1e-3 epochs=100, 591,488 params
```

FF++ train reals only, exactly as the brief states. **Residual normalization statistics live in
the same `.pt`** as calibrator buffers written by `fit_one()` (`fit_reference.py:221` computes the
authentic residual norm; `fit_reference.py:281` saves it alongside the state dict) — there is no
separate stats file for the reference branch, so refitting the AE and refitting the statistics are
one operation, and Stage 1.2's "fit statistics on the same diverse-real population" is automatic
provided the same feature cache is used for both.

Feature caching: `analysis/discern_v2/cache_encoder_features.py --backbone fsvfm --encoder frozen
--datasets NAME --split SPLIT`. It builds a `NeSyDeFakeDataset`, so it currently reaches **only
corpora that have a dataset JSON**. FFHQ has none. → Stage 1 needs a small image-directory
caching path; recorded as an addition below.

## Item 5 — MR-VAE / P1d operator

| what | where |
|---|---|
| implementation | `training/networks/discern_v2/projectors.py::MRVAEProjector` (ported from DiCoME `experiments/pilot_P1d/ae_operator.py::MRVAE`) |
| branch wrapper | `training/networks/discern_v2/branches.py` (`projector: mr_vae`, `export_rate_response: true`) |
| rate-response method | `MRVAEProjector.rate_distortion_response()` → per-sample cosine distortion, `(B, K)` |
| config | `training/config/discern_v2/D3_full_p1d.yaml`, `_base.yaml` |

**Rate schedule it was trained with** (`projectors.py:38-40`):

```
BETA_RANGE = (0.1, 10.0)                          # log-uniform sampling during training
BETA_GRID  = (0.1, 0.32, 1.0, 3.16, 10.0)         # K = 5, log-spaced, the readout grid
```

β enters via FiLM modulation, initialised to identity (γ=δ=0) so the model starts as an
unconditioned VAE. `HIDDEN_DIM = 32` and `latent_dim = 32`, sized for DiCoME's 64-d `f_s`.

**Two trained checkpoints exist, and neither is directly attachable — this is a real blocker for
Stage 2.1:**

```
/data/umar/Repos/DiCoME/runs/pilots/P1d/checkpoints/dicome-best-epoch=01-val_auroc_video=0.9957.ckpt
/data/umar/Repos/DFB_NeSyNS/logs/train/nesy_defake_d3_p1d_2026-08-16-18-33-53/best_avg.pth
```

The MR-VAE is a **feature-space** operator, not an image-space one. It consumes a 64-d projected
semantic feature. The first checkpoint's 64-d space is DiCoME's LoRA-adapted CLIP feature; the
second's is DISCERN's NeSy backbone feature. **Neither is the V1 Branch-A feature space**, and
V1's Branch-A feature does not exist until Stage 4 trains its LoRA. So the brief's "attach the
frozen MR-VAE as Branch C" has no valid frozen checkpoint to attach, because "frozen before
detector training" and "operates on the detector's feature" cannot both hold for a feature-space
operator.

Three ways out, none free — 🟡 **ASK-UMAR**:

* **(a) Re-fit MR-VAE on a frozen encoder's features.** Fit it on frozen FS-VFM (or frozen CLIP,
  no LoRA) embeddings of FF++ reals. Genuinely frozen, genuinely independent of Stage 4, and the
  rate response then measures *image manifold* compressibility rather than *the anchor's own
  feature* compressibility. Cheapest and the only option that keeps Stage 2 before Stage 4.
* **(b) Two-pass.** Train Stage 4, freeze Branch A, fit MR-VAE on its features, retrain the heads.
  Costs a second training run and makes the "frozen mechanism" claim weaker.
* **(c) Reuse the DiCoME checkpoint as-is** on DiCoME's own feature. Only valid if Branch A is
  literally DiCoME's released LoRA rather than a freshly trained one — it is not.

Recommendation on the evidence: **(a)**, with FS-VFM as the host encoder, because FS-VFM is
already frozen, already cached, and already the branch whose residual we trust least — a rate
response over the same embedding is directly comparable to `r_ref`.

## Item 7 — training entry point, selection, sampling, LoRA

**Training:** `training/train_v1.py`

```
--config              default training/config/discern_v2/V1_CONFIG.yaml
--detector-config     default training/config/detector/nesy_defake_d1_v.yaml
--output DIR (required)  --batch-size 32  --workers  --epochs  --device
--max-batches 0       0 = full epoch, >0 smoke test
--overwrite           refuses to append to an existing run otherwise
```

Saves **every** epoch (`V1_CONFIG.yaml: save_every_epoch: true`); selection is a separate step so
no epoch window is hardcoded. Validation is restricted to VAL_select by `train_v1.py:402`, which
raises if the split file does not match — validating on all of FF++ val would select using
VAL_meta, which the gates are fit on.

**Checkpoint selection:** `analysis/discern_v2/select_checkpoint.py --run DIR`. Primary metric
VAL_select video AUROC, tie-break ECE then val loss. `assert_no_ood()` (`select_checkpoint.py:59`)
refuses to run if any OOD-looking key is present in `metrics.jsonl`. Also reports the noise-aware
§22 reading beside the literal one, because in-domain validation saturates — which is exactly the
caveat the brief's Stage 4 asks to be recorded.

**Source-paired sampling — partially exists.** `training/dataset/nesy_defake_dataset.py` has
`paired_training` (default `True`, `nesy_defake_dataset.py:101`), but it is GenD-style *pair
inclusion at data-preparation time*: the training list contains both members of an FF++
real/fake source pair, relying on shuffled batching to co-occur them. It does **not** place a pair
in the same batch and does **not** apply matched augmentation — `train_v1.py` augments per sample
(`torch.stack([self.transform(img) for img in raw])`), so the two members of a pair draw
independent flips, affines, blurs and jitters. Stage 4's "matched benign augmentation applied to
both members of a pair" therefore needs new code: a pair-aware batch sampler plus a
shared-parameter transform. Recorded as an addition below.

**LoRA config for the CLIP branch** — `training/networks/discern_v2/semantic_branch.py`, traced
from DiCoME `src/config.py:38-45`:

```python
DICOME_LORA = {"target_modules": ["q_proj", "v_proj"], "r": 8, "lora_alpha": 16,
               "lora_dropout": 0.1, "bias": "none", "task_type": "FEATURE_EXTRACTION"}
```

Backbone `openai/clip-vit-large-patch14`, `DICOME_FEATURE_DIM = 64`.
856,450 trainable (786,432 LoRA + 65,600 projection + 128 norm + 4,290 head) vs 303,179,776 frozen.

---

## Environment

`/data/umar/miniconda3/envs/dfb_nesy/bin/python` — torch 2.9.1, transformers 4.44.2, peft 0.14.0,
pyarrow present. `base` has **no** pyarrow and cannot read the per-sample parquet.

## Additions made in this phase

Listed as the brief requires, so nothing looks pre-existing that is not:

* `analysis/discern_v2/phase2/` — the Stage 0–8 analysis scripts.
* An image-directory feature-caching path for Stage 1 (FFHQ has no dataset JSON).
* A pair-aware sampler with matched augmentation for Stage 4.
* A multi-source CCF operator for Stage 6.

Nothing was added to the DF40 loader: it already existed.

---

## Conflicts between this brief and the shipped V1 state

Both are consequential and neither can be resolved by inspection.

1. **Calibration protocol.** Ground rule 2 says no calibration may see a non-FF++ label, and
   Stage 5 says FF++ validation only. V1's shipped Stage D/E artifact
   (`logs/v1/stage_de/epoch_007_diverse/`) was calibrated on the `diverse` protocol —
   `FaceForensics++:val` **and** `Celeb-DF-v2:val` — on Umar's explicit instruction, with the
   overlap computed and recorded. The `ffpp` protocol run also exists
   (`logs/v1/stage_de/epoch_007/`). Phase-2 code below defaults to **`ffpp`**, following this
   brief; the `diverse` protocol stays selectable and every artifact records which was used.
2. **`A` vs `U_sup`.** Stage 7 renames `A` to `U_sup` and redefines the conflict weights as
   `w_b = q_b(1 - u_b)`. V1's shipped `C` uses a different weighting and the parquet column is
   named `A`. Phase-2 recomputes both under the new definitions rather than reusing the V1
   columns, so V1 and Phase-2 reliability numbers are **not** column-comparable even though the
   brief asks the reports to mirror each other.

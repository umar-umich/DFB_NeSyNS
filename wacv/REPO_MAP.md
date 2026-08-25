# Stage 0 — Repository map (FPAD / WACV brief)

Resolved by inspection on 2026-08-21, branch `discern-v2-phase2`, repo `/data/umar/Repos/DFB_NeSyNS`.
Everything below was listed on disk, loaded, or read out of a file. Nothing is inferred from a name.

Env for everything: `/data/umar/miniconda3/envs/dfb_nesy/bin/python` (torch 2.9.1, peft 0.14.0,
pyarrow). `base` has no pyarrow and cannot read the per-sample parquets.

---

## ⚠️ The three scope-changing items, first

### Item 7 — FF++ manipulation masks: **STAGED.** Stage 5 gets genuine localization.

| manipulation | frame videos | mask videos |
|---|---:|---:|
| Deepfakes | 1000 | **1000** |
| Face2Face | 1000 | **1000** |
| FaceSwap | 1000 | **1000** |
| NeuralTextures | 1000 | **1000** |
| DeepFakeDetection | 3068 | **3068** |
| FaceShifter | 1000 | **0** |

`/data/umar/Datasets/preprocessed/FaceForensics++/manipulated_sequences/<type>/c23/masks/<video>/<frame>.png`

Masks are **already warped through the same crop transform as the frames** — single-channel
uint8, 256×256, matching the frame's 256×256×3 exactly (`preprocessing/preprocess.py` passes the
mask through `align_face(..., mask=mask)` with the same affine). So no re-alignment is needed and
the masks are directly comparable to a patch map. A sampled Deepfakes mask has 21.8% of pixels
non-zero, which is a sensible manipulated-region fraction.

Consequence: **Stage 5 reports quantitative localization on four of FF++'s manipulation types**
(the brief's tier 1), with pseudo-masks needed for nothing. FaceShifter is the one exception and
must be excluded from the localization table rather than pseudo-masked into it.

One limitation to design around: at 224 input with patch 16, the patch grid is **14×14 = 196**.
Each patch covers ≈18 px of the 256×256 mask. That is adequate for patch-level AUPRC (≈43 of 196
patches positive at a 22% mask fraction) but coarse for IoU, where the discretisation error is a
large fraction of the region. Prefer AUPRC as the headline and report IoU with the grid resolution
stated.

### Item 8 — FF++ compression variants: **ONLY c23 EXISTS.** This removes work from the brief.

```
/data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/  -> c23 only
/data/umar/Datasets/FaceForensics++/                                          -> c23 only
```

Neither preprocessed frames nor source videos exist for `c40` or `raw`. Two consequences:

* **Stage 6's compression-robustness check cannot run.** The brief calls it "high value for the
  thesis", so this is a real loss, not a tidy-up.
* **Stage 2's permitted OOD side shrinks.** The brief specifies "DF40-Dev reals and the FF++
  compression shift" as the two permitted sources of domain variation for the gate decision.
  Without c40, **the Stage 2 domain audit rests on DF40-Dev reals alone** — a single axis of
  domain shift, and one drawn from corpora (FF++ and Celeb-DF source footage) that partly overlap
  the training distribution.

🟡 **ASK-UMAR.** Either accept a single-axis Stage 2 audit and drop Stage 6's compression row, or
download FF++ c40 and re-run the crop pipeline. c40 is a 🔴 download decision plus preprocessing
over 1000+1000×4 videos.

### Item 1 — intermediate layers are **not** exposed; hooks are required (as the brief anticipated)

`VisionTransformer.forward_features` (`training/networks/discern_v2/fsvfm/models_vit.py:44`) runs
the block loop and returns only the pooled output. There is no `get_intermediate_layers`. Hooks on
`model.blocks[i]` are the route, and they work — verified:

```
blocks[11] forward-hook output: (B, 197, 1024)
   CLS      = out[:, 0]      (B, 1024)
   patches  = out[:, 1:]     (B, 196, 1024)   -> 14x14 grid
```

All 24 blocks share `embed_dim = 1024`, so one layer set applies at every depth. The brief's
`{4, 8, 12, 16, 20, 24}` maps to zero-based block indices `{3, 7, 11, 15, 19, 23}` — record which
convention the config uses, because off-by-one here silently shifts the whole depth profile.

**A design caveat that matters for the notation.** The brief defines `h^(l)` as CLS. This
checkpoint is loaded with `global_pool=True`, and that constructor **deletes `self.norm`** and adds
`fc_norm` (`models_vit.py:37-42`) — i.e. the representation the model was pretrained and evaluated
through is the **mean of the patch tokens**, not CLS. The CLS token is carried along but is not the
trained readout. So `d_l^TS` on CLS may be measuring a token the model has little reason to have
organised.

Recommendation: compute `d_l^TS` on **both** the CLS token and the mean-pooled patch tokens, make
mean-pool the primary, and report CLS beside it. Cosine distance is scale-invariant so the missing
per-layer `norm` is not itself a problem. 🟡 flagging rather than silently substituting, since it
changes the notation section of the paper.

---

## Item 1 (continued) — FS-VFM loading and the checkpoint

| what | where |
|---|---|
| wrapper | `training/networks/discern_v2/fsvfm_encoder.py::FrozenFSVFM` |
| architecture | `training/networks/discern_v2/fsvfm/models_vit.py::vit_large_patch16` |
| checkpoint | `weights/FS-VFM/checkpoint-599.pth`, sha256 `3fd99324c998ee06e4daa7f9d845e73f65f1628dee741b2c98664d566526bc87` |
| normalization | `weights/FS-VFM/pretrain_ds_mean_std.txt` — mean `[0.5482, 0.4234, 0.3655]`, std `[0.2789, 0.2439, 0.2349]`. **Not ImageNet**; read from file, never hardcoded |
| upstream clone | `/data/umar/Repos/FSFM-CVPR25`, branch `FSVFM-extension-R1`, commit `b84aafa` |

**Not encoder-only.** The checkpoint is the full pretrain state — 441 tensors:

```
blocks 288 (24 blocks)      patch_embed 2   cls_token/pos_embed/mask_token 3
projector 6   predictor 6                      <- the pretext heads
rep_decoder_* 29   decoder_* 103               <- 103 decoder tensors
```

That is more than the teacher-student design needs, which is fine — `FrozenFSVFM._load_pretrained`
already loads the encoder subset and **fails** on missing encoder tensors rather than warning.
Nothing new is required here.

## Item 2 — LoRA infrastructure

peft 0.14.0 is installed. The existing recipe (`training/networks/discern_v2/semantic_branch.py`,
traced from DiCoME `src/config.py:38-45`):

```python
DICOME_LORA = {"target_modules": ["q_proj", "v_proj"], "r": 8, "lora_alpha": 16,
               "lora_dropout": 0.1, "bias": "none", "task_type": "FEATURE_EXTRACTION"}
```

**`target_modules` does not transfer.** `q_proj`/`v_proj` are HuggingFace CLIP names. This timm-derived
ViT names its Linears differently — confirmed by walking `blocks[0]`:

```
attn.qkv    (3072, 1024)      <- FUSED q,k,v in one Linear
attn.proj   (1024, 1024)
mlp.fc1     (4096, 1024)
mlp.fc2     (1024, 4096)
```

So the brief's "LoRA on the attention and MLP projections" becomes
`target_modules = ["qkv", "proj", "fc1", "fc2"]`. One consequence to record rather than discover
later: **LoRA on the fused `qkv` adapts q, k and v together**, where DiCoME's recipe deliberately
adapted only q and v. Matching DiCoME's intent would need a per-slice adapter, which peft cannot
express on a fused Linear. Recommend adapting the fused `qkv` and recording the deviation; the
rank and alpha carry over unchanged. `task_type="FEATURE_EXTRACTION"` still applies.

The two Stage-2 students must share LoRA targets exactly, so this choice is made once, in config.

## Item 3 — FF++ c23 loaders and source-paired sampling

| what | where |
|---|---|
| dataset | `training/dataset/nesy_defake_dataset.py::NeSyDeFakeDataset` |
| split loader | `analysis/discern_v2/cache_encoder_features.py::load_split(config, name, split)` |
| paired sampler | `training/dataset/paired_sampler.py::PairedBatchSampler` |
| matched augmentation | `training/dataset/paired_sampler.py::MatchedAugment` |
| toggle | `training/train_v1.py --sampling {paired,random}` |

Built and committed in the Phase-2 work (`01d5702`), with 12 unit tests. Pairing is keyed on the
source-video id derived from the frame path, so it survives worker boundaries; pair members share
one augmentation draw and non-partners do not. `pair_fraction()` is logged per epoch.

**This is what Stage 6's `L_pair` needs** — the sampler already guarantees both members of a pair
are in the same batch, which a ranking loss requires and which shuffled batching does not provide.

## Item 4 — DF40 harness and the sealed split

| what | where |
|---|---|
| path resolution | `training/dataset/df40_paths.py` (77/81 methods resolve; see `2bddc0a`, `71d1317`) |
| eval harness | `training/eval_v1.py --df40 --datasets <method> ...` |
| **sealed split** | `configs/discern_v2/df40_split.json` — **38 Dev / 35 Holdout** |
| the claim it supports | `phase2/DF40_SPLIT.md` |

Reuse this split, as the brief instructs. Read `DF40_SPLIT.md` before writing any zero-shot
sentence: Phase 1 computed a per-generator table over 70 generators, so Holdout is *sealed before
Phase 2 and unread by any Phase-2 decision*, **not** *never seen*.

**Item 4's rescue targets — one correction.** The brief names "danet, mcnet, tpsm,
facevid2vid-cdf". The AUROC-inverted rows actually recorded in
`analysis/discern_v2/A1_complementarity/DF40/inversion_rows_detected.csv` are:

```
sadtalker-ff   0.2534      danet-cdf   0.2855      tpsm-ff        0.2903
danet-ff       0.4007      mcnet-cdf   0.4042      mcnet-ff       0.4320
facevid2vid-ff 0.4796      tpsm-cdf    0.4997
```

`facevid2vid-**cdf**` is **not** in the inverted set — `facevid2vid-**ff**` is. (`facevid2vid-cdf`
appears in `breakout_inversion_and_heygen.csv`, which is *accuracy*-based, not AUROC-based; that is
the likely origin of the mix-up.) Also, **`sadtalker-ff` is the most inverted row of all** at
0.2534 and the brief omits it. Recommend the rescue target set be the eight rows above, and all
eight are in Dev so they are free to use. 🟡 confirm.

These were measured on **DiCoME's P0-DS operator**, not on this paper's B1 student, so they
identify *where the anchor family fails*, not necessarily where B1 fails. B1's own inverted rows
must be recomputed before "rescue" means anything for this method.

## Item 5 — the §20 domain audit

`analysis/discern_v2/domain_audit.py`. Reusable, with one change: `BRANCH_QUANTITIES`
(`domain_audit.py:57`) is a hardcoded dict keyed by branch (`ref` → `ref_residual_norm`,
`ref_angle`, `p_ref`, `u_ref`). Adding the new signal means adding an entry for the trajectory
quantities (`d_4 … d_24`, plus the head's `p_traj`/`u_traj`), not editing the metric code.

The metric itself — forensic separability vs domain separability, `forensic_minus_domain`, and the
`DATASET DETECTOR` / `BORDERLINE` / `pass` verdicts — carries over verbatim, which is what the
brief asks for. It needs FF++-real, OOD-real and OOD-fake rows in one export, so the Stage 2 export
must include DF40-Dev reals.

## Item 6 — EDL head and evidence→opinion

| what | where |
|---|---|
| EDL loss | `training/train_v1.py::edl_loss` (ported from DiCoME `src/losses/evidential_loss.py`) |
| evidence → Dirichlet | `training/networks/discern_v2/dirichlet.py::to_dirichlet` |
| Dirichlet → opinion | `training/networks/discern_v2/ds_fusion.py::Opinion.from_evidence` |
| a head to copy | `training/networks/discern_v2/rate_branch.py::RateEvidenceBranch` |

`alpha = e + 1`, `S = sum alpha`, `u = K/S`, `belief = e/S`. `H_traj` reuses all of it; the rate
branch is the closest template — a frozen operator producing a K-vector, standardized, then a small
head emitting `softplus` evidence — and `D(x)` is structurally the same object as `R(x)`.

---

## What Stage 1 must build (nothing else is missing)

1. A **teacher-student wrapper**: two `FrozenFSVFM`-derived encoders from one checkpoint, LoRA on
   the student, forward hooks at the configured layer set, emitting `D(x)` and the patch map.
2. **`H_traj`** — trivial, copy `RateEvidenceBranch`.
3. The **two-stage trainer** (Stage A adapt with `L_preserve`, then freeze, then Stage B interpret).
4. A **localization evaluator** consuming the staged masks.
5. The **official FS-VFM linear-probe baseline** — see below.

## The external baseline is runnable

`/data/umar/Repos/FSFM-CVPR25/fsvfm/linearprobe/cross_dataset_DFD_and_DiFF/`:

```
main_linearprobe_DfD.py    engine_linearprobe.py    main_test_DfD.py
scripts_DFD/               scripts_DiFF/
```

and `fsvfm/finetune/` plus `fsvfm/finetune_fs-adapter/` alongside it, so both the linear probe and
the FS-Adapter are available. The authors' preprocessing is **DLIB with 30% additional crop**
(`README.md:500`), whereas our staged crops are RetinaFace + GenD template at `scale=1.3`
(≈30% margin) — close but not identical. Record it as a deviation on that row rather than claiming
an exact reproduction of the authors' protocol.

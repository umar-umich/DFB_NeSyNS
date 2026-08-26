# Step 0 — Recovery repo discovery

Answers every item the recovery brief's Step 0 asks for, from the tree rather than from memory.
Paths are real and were checked.

**Headline for Step 2: a retrainable DiCoME recipe exists, was already run on FF++ c23, and beat
our CLIP port by +0.042 AUROC on Celeb-DF-v2.** That changes Step 2 from a recipe search into a
choice between two existing chassis.

---

## 1. DiCoME reproduction code and checkpoints

| item | path |
|---|---|
| repo | `/data/umar/Repos/DiCoME` |
| env | **`discern_ext`** — renamed from `dicome` on 2026-08-12; the scripts still say `conda activate dicome`, which no longer exists |
| released checkpoint | `weights/dicome-best.ckpt` (HF `kxl0825/DiCoME`, epoch 4, 0.868 M trainable) |
| training entry point | `tools/train/train_dicome.py` (`train` / `test` subcommands) |
| eval driver (released) | `eval_adaptation/run_eval_all.sh` |
| eval driver (self-trained) | `eval_adaptation/run_eval_trained.sh` — same per-dataset configs, checkpoint is the only difference |
| model | `src/model/core_model.py`, `src/model/dicome_module.py` |
| reproduction results | `eval_adaptation/RESULTS.md` |

`peft` must stay at **0.14.0** and `diffusers` at 0.32.2 — every P0/P1 checkpoint was trained
under that LoRA implementation, and 0.39 demands `peft>=0.17`, which would silently invalidate
them.

### Is a DiCoME-direct / CLIP-only readout accessible? **Yes, without touching the model.**

`core_model.py:125` returns, in order:

```
fused_evidence, semantic_evidence, artifact_evidence, semantic_feature, ...
```

So the three configurations Step 1 needs are all readable from one forward pass:

- **DiCoME-full** — `fused_evidence`, i.e. `DS_Combin(semantic_alpha, artifact_alpha)`.
- **DiCoME-semantic-only** — `semantic_evidence`, the CLIP branch alone. This is the like-for-like
  comparator against our port.
- **DiCoME-artifact-only** — `artifact_evidence`, available free and worth logging.

What the artifact view actually is (`_geometric_view_purification`, `core_model.py:138`): the
β-VAE reconstructs the CLIP feature onto a learned manifold, the residual `f_r = f_s − f_c` is
taken, its component parallel to `f_s` is projected out, and the orthogonal remainder is the
artifact feature. **Both views live in the CLIP manifold** — DiCoME is not a multi-encoder model.
That matters for the fusion-in-fusion question in Step 8.

No existing script exports the per-view evidences; the eval path reports the fused head only. A
small exporter is needed for Step 1.

---

## 2. Does a retrained-on-FF++ DiCoME exist? **Yes — this is `P0-DS`.**

| item | path |
|---|---|
| run dir | `runs/dicome_train/ffpp_reproduce_seed42/` |
| checkpoints | `checkpoints/dicome-best-epoch={01,02,05}-val_auroc_video=0.99{60,50,53}.ckpt`, `last.ckpt` |
| hyperparameters | `csv_logs/version_0/hparams.yaml` |
| training data | `eval_adaptation/data/h5/FFpp_train.h5` + the five `data/splits/FFpp_train/*.txt` |
| results | `docs/DiCoME_eval/RESULTS_ALL_PILOTS.md`, where it is the `P0-DS` baseline column |

`RESULTS_ALL_PILOTS.md` describes P0-DS as the "self-trained FF++ c23 reproduction, seed 42" and
uses it as the baseline for all seven pilots. Its checkpoint selection follows the released
recipe: highest `val_auroc_video`, not the last epoch.

### The three-way anchor comparison that already exists on disk

| dataset | DiCoME released | **P0-DS (retrained FF++ c23)** | our CLIP port | port − P0-DS |
|---|---:|---:|---:|---:|
| FF++ (in-domain) | 0.9905 | 0.9929 | 0.9909 | −0.0020 |
| Celeb-DF-v2 | 0.9729 | **0.9644** | 0.9225 | **−0.0419** |
| CDFv3 | — | 0.9516 | — | — |
| DFD | 0.9392 | 0.9419 | 0.9222 | −0.0197 |
| DFDC | 0.8822 | 0.8825 | 0.8477 | −0.0348 |
| DFDCP | 0.8799 | 0.8576 | 0.8912 | **+0.0336** |
| Deepfake-Eval-2024 | — | 0.6918 | 0.6357 | −0.0561 |

Caveat that must be carried: P0-DS's numbers come from the DiCoME repo's own evaluation path and
our port's from `eval_v1`. They agree closely on the shared reference points (FF++ 0.9929 vs
0.9909, DFDC 0.8825 vs 0.8822 for released), but Step 1 must re-run all three through **one**
pipeline before any gap is treated as real.

---

## 3. The current CLIP-port anchor

`training/config/discern_v2/V1_CONFIG.yaml`, run `logs/v1/stage_b_seed42/`, selected
**epoch 7** (`SELECTED.json`: video AUROC 0.9986, frame 0.9805, ECE 0.0565 on FF++ VAL_select).

Side by side with DiCoME's own recipe:

| | our port | DiCoME (`hparams.yaml`) |
|---|---|---|
| backbone | `openai/clip-vit-large-patch14` | same |
| LoRA targets | `q_proj, v_proj` | same |
| rank / alpha / dropout / bias | 8 / 16 / 0.1 / none | same |
| feature dim | 64 | 64 |
| optimizer | AdamW, betas (0.9, 0.999), WD 0.01 | Adam-family, betas (0.9, 0.999) |
| LR / schedule | 1e-4, cosine, min 1e-6 | 1e-4, cosine, min 1e-6 |
| epochs | 20 | 20 |
| **batch size** | **32** (CLI) | **128** |
| **precision** | **fp32** | **bf16-mixed** |
| **architecture** | semantic head only | + β-VAE manifold projector, artifact view, internal DS fusion |
| **VAE losses** | none | `beta_kld 2.0`, `lambda_vae 0.7`, `lambda_align 1.0` |
| augmentation | ported DiCoME pipeline, matched within source group | DiCoME's own |

**The LoRA configuration is identical.** The differences that could carry a recipe gap are batch
size (32 vs 128), precision, and the VAE/alignment loss terms. The architectural difference — the
artifact view and internal DS fusion — is what Step 1 is designed to separate from those.

---

## 4. VALmix

`/data/umar/Datasets/preprocessed/dataset_json/VALmix.json`, built by
`preprocessing/build_valmix_manifest.py`, documented in `tbiom/VALMIX.md`.

1,350 videos / 40,142 frames, balanced 450/450/450 across CDFv2val (14,395 frames), DFDCPval
(12,725) and DFEval24val (13,022). Real/fake 675/675.

**Disjointness verified on the final manifest, not inherited:** zero overlap with Celeb-DF-v2's,
DFDCP's and Deepfake-Eval-2024's test lists, re-checked at build time with the builder refusing to
write otherwise.

Label keys `VALmix_Real` = 0, `VALmix_Fake` = 1, registered in `training/config/test_config.yaml`.
The `val` and `test` keys hold identical content because `abstract_dataset` supports only
`train`/`test` modes and raises on `val`; VALmix is a validation set and a number computed on it
is never a test result.

---

## 5. Stage-1 expert artifacts

| expert | run | VALmix export | DF40-Dev export | FF++ val (its own threshold) |
|---|---|---|---|---|
| FS-VFM preservation | `logs/fpad/studentA_preserve_seed42` (epoch 9) | `logs/tbiom/valmix/fsvfm_preserve/` | `logs/tbiom/score/fsvfm_preserve_df40dev/` | `logs/tbiom/score/fsvfm_preserve_ffppval/` |
| FS-VFM ordinary | `logs/fpad/studentA_ordinary_seed42` (epoch 9) | `logs/tbiom/valmix/fsvfm_ordinary/` | `logs/tbiom/score/fsvfm_ordinary_df40dev/` | `logs/tbiom/score/fsvfm_ordinary_ffppval/` |
| MR-VAE rate | operator `configs/discern_v2/rate/rate_operator_mrvae.pt`, probe `rate_probe.json` | `logs/tbiom/valmix/fsvfm_preserve_rate/` (`p_rate`) | `logs/tbiom/score/fsvfm_preserve_df40dev/` (`p_rate`) | same file |

Both students were trained with `sampling: paired`, seed 42, FF++ c23, LoRA on `qkv, proj, fc1,
fc2` r=8 — the fused-qkv deviation forced by the timm-derived ViT.

Gate: `analysis/tbiom/stage1_membership.py`; reports `tbiom/STAGE1_MEMBERSHIP.md` (DF40-Dev) and
`tbiom/valmix/STAGE1_MEMBERSHIP.md`. Both verdicts: nothing enters. The gate already supports
per-expert operating thresholds, margin-space fusion, and VALmix domain grouping.

`u_rate` is written all-NaN — the rate branch is a logistic probe, not a Dirichlet head, so it has
no evidential uncertainty. The gate drops that feature rather than imputing one. **Relevant to
Step 6's G3 rung**, which will want real rate statistics rather than a single probability.

---

## 6. §20 domain audit

`analysis/discern_v2/domain_audit.py`. Exists and is the tool Step 6's mandatory audit clamp
needs. It measures forensic-vs-domain separability and emits `DATASET DETECTOR` verdicts. It has
been run on branch representations before; **it has never been run on a gate's own inputs**, which
is what Step 6 requires — the clamp there is a new application of an existing tool, not new code.

---

## 7. Health dashboard — built, and validated against the known collapse

`eval_v1` reported `frame_auroc` and `video_auroc` only. EER, FPR_real@τ and Δ_RF were **missing**
and are now in `analysis/tbiom/health.py`, which reports all five columns from any scored parquet,
with the threshold frozen once on a named development source and applied unchanged to every row.

Validation — the dashboard must call the epoch-2 FF++ ⊕ DF40 model broken where AUROC did not.
τ = 0.3260, EER on FF++ val:

| model / domain | AUROC | EER | **FPR_real@τ** | Δ_RF | mean p on real |
|---|---:|---:|---:|---:|---:|
| anchor FF++ | 0.9909 | 0.0375 | 0.050 | +0.864 | 0.071 |
| anchor CDFv2 | 0.9225 | 0.1623 | 0.124 | +0.531 | 0.124 |
| anchor DFDCP | 0.8912 | 0.1878 | 0.239 | +0.491 | 0.215 |
| anchor UADFV | 0.9958 | 0.0408 | 0.041 | +0.899 | 0.053 |
| ep2 FF++ | 0.9926 | 0.0277 | 0.243 | +0.769 | 0.205 |
| ep2 CDFv2 | 0.7503 | 0.3046 | **1.000** | **+0.000** | 0.997 |
| ep2 DFDCP | 0.7247 | 0.3484 | **0.939** | +0.106 | 0.879 |
| ep2 UADFV | **0.9354** | 0.1020 | **0.959** | +0.091 | 0.906 |

**UADFV is the case that justifies the whole dashboard.** AUROC 0.9354 reads as a mild regression;
FPR_real@τ 0.959 says 96% of real videos are called fake. On CDFv2, FPR_real is exactly 1.000 with
Δ_RF exactly 0.000 — total operational failure at 0.75 AUROC.

Note the ep2 row on FF++ already shows FPR_real 0.243 against the anchor's 0.050 — **the damage
was visible on in-domain data**, at an epoch whose FF++ AUROC (0.9926) was the best number in the
run.

---

## 8. What is missing and must be built

1. **A per-view DiCoME exporter** — nothing writes `semantic_evidence` / `artifact_evidence` to a
   parquet. Required by Step 1.
2. **One shared evaluation pipeline for all three anchors.** P0-DS's numbers come from the DiCoME
   repo's path and the port's from `eval_v1`; a 0.042 gap cannot be attributed until both run the
   same way. The DiCoME side reads HDF5, our side reads PNG frames — `eval_adaptation/RESULTS.md`
   records that frames were verified byte-identical, so this is a plumbing job, not a data one.
3. **Rate-branch diagnostics beyond `p_rate`** for Step 6's G3 rung.

Nothing else in the brief's Step 0 list is absent.

---

## Pass condition

Met. Every item is resolved, and the decisive one is answered: **a retrainable DiCoME recipe
exists, its FF++ c23 checkpoints are on disk, and it outperforms the CLIP port cross-dataset.**
Step 2 is therefore not a recipe search — it is a choice between two existing chassis, and Step 1
decides which by separating the CLIP branch from DiCoME's extra machinery.

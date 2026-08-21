# Phase-2 execution plan — 🔴 every command here is Umar's to launch

Generated from `configs/discern_v2/df40_split.json`, so the method lists below are the sealed
split, not a copy of it that can drift. **Env for everything:**

```
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
```

`base` has no pyarrow and cannot read the per-sample parquets.

Run the stages in order. Each one's pass condition is in its own result file; do not start a stage
until the previous stage's artifacts are on disk. **A hard gate sits after Stage 3 and can end the
multi-specialist build** — read `phase2/STAGE3_GATE.md` before spending time on Stages 4-8.

---

## Stage 0 — current V1 on DF40-Dev

### 0.2 Evaluate the frozen V1 checkpoint on DF40-Dev

The nine methods the brief names as the minimum. All resolve at 1.000 after the path fix.

```
$PY training/eval_v1.py \
    --checkpoint logs/v1/stage_b_seed42/epoch_007.pth \
    --datasets danet_ff danet_cdf mcnet_ff mcnet_cdf tpsm_ff \
                tpsm_cdf facevid2vid_ff facevid2vid_cdf heygen \
    --df40 \
    --output logs/phase2/eval/epoch_007_df40dev \
    --device cuda:N
```

Add `--stage-de logs/v1/stage_de/epoch_007/stage_de.pt` if you also want the V1 frozen policy
applied. Stage 0 does not need it — it fits its own diagnostic gate.

For the full Dev set (38 methods) rather than the nine:

```
$PY training/eval_v1.py --checkpoint logs/v1/stage_b_seed42/epoch_007.pth --df40 \
    --output logs/phase2/eval/epoch_007_df40dev_full --device cuda:N \
    --datasets CollabDiff DiT_cdf DiT_ff MidJourney SiT_cdf SiT_ff \
                StyleGAN2_cdf StyleGAN2_ff VQGAN_cdf VQGAN_ff blendface_cdf blendface_ff \
                danet_cdf danet_ff deepfacelab faceswap_cdf faceswap_ff facevid2vid_cdf \
                facevid2vid_ff heygen mcnet_cdf mcnet_ff rddm_cdf rddm_ff \
                sadtalker_cdf sadtalker_ff simswap simswap_cdf simswap_ff stargan \
                starganv2 styleclip tpsm_cdf tpsm_ff uniface uniface_cdf \
                uniface_ff uniface_ori
```

### 0.3 + 0.4 Complementarity, rescue and harm

Needs the DF40 export **and** an export containing FF++ rows — the `protocol` arm fits its gate on
FF++ only, and refuses rather than falling back to something that is not protocol-valid.

```
$PY analysis/discern_v2/phase2/complementarity.py \
    --parquet logs/phase2/eval/epoch_007_df40dev/per_sample_epoch_7.parquet \
              logs/v1/eval/epoch_007_gated_diverse/per_sample_epoch_7.parquet \
    --out phase2/stage0 --protocol both
```

Writes `STAGE0_DF40DEV_protocol.md` (deployable) and `STAGE0_DF40DEV_insample.md` (ceiling).
**The gap between them is a finding, not noise** — read both.

---

## Stage 1 — diverse-real reference

🟡 **ASK-UMAR first.** The only non-evaluation real corpus on this machine is
`/data/umar/Datasets/ffhq256_subset` (8,750 images). No VGGFace2, no CelebA-HQ. Decide between
FF++ reals ∪ re-cropped FFHQ, or fetching a larger corpus, before running 1.1.

### 1.1 Re-crop through DISCERN's own pipeline, then measure identity overlap

FFHQ ships in its own alignment; without this step `r_ref` becomes partly a crop-convention
residual, which is the failure Stage 1 exists to remove.

```
$PY analysis/discern_v2/phase2/prepare_diverse_reals.py \
    --images /data/umar/Datasets/ffhq256_subset \
    --out /data/umar/Datasets/preprocessed/FFHQ-recrop/frames

$PY analysis/discern_v2/phase2/identity_overlap.py \
    --corpus /data/umar/Datasets/preprocessed/FFHQ-recrop/frames \
    --corpus-name FFHQ-recrop --out phase2/stage1
```

The overlap threshold is derived from the test reals' own different-identity distribution, not
quoted from a paper. Report the flagged fraction in `STAGE1_REFERENCE.md`.

### 1.2 Cache features, then refit and freeze `P_R`

```
$PY analysis/discern_v2/phase2/cache_image_features.py \
    --images /data/umar/Datasets/preprocessed/FFHQ-recrop/frames \
    --name FFHQ-recrop --out cache/discern_v2/fsvfm_frozen \
    --confirm-all-real --device cuda:N

$PY analysis/discern_v2/phase2/fit_diverse_reference.py \
    --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
               cache/discern_v2/fsvfm_frozen/FFHQ-recrop_real/features.npz \
    --balance --out configs/discern_v2/reference_diverse --device cuda:N
```

`--balance` matters: without it the population is 72% FF++ (23,039 vs 8,750) and calling the
result "diverse" would overstate what changed. The script warns if one source exceeds 80%.

### 1.3 Re-run the §20 domain audit

Needs an eval of the refit branch first (point `reference.artifact_path` at
`configs/discern_v2/reference_diverse/reference_C3_ae_cosine.pt`), then:

```
$PY analysis/discern_v2/domain_audit.py \
    --parquet logs/phase2/eval/<refit-run>/per_sample_epoch_7.parquet \
    --out analysis/discern_v2/phase2_domain_audit_diverse
```

It already audits `p_ref` and `u_ref` alongside `ref_residual_norm` and `ref_angle` — which is the
point, because V1's head partly laundered the provenance signal the raw residual carried.

---

## Stage 2 — MR-VAE (P1d)

### 2.1 Fit and freeze the operator

The Phase-1 checkpoints cannot be attached — the MR-VAE is a feature-space operator and both live
in feature spaces V1 does not have. It is re-fit on the frozen FS-VFM embedding instead; see
`phase2/REPO_MAP.md` item 5.

```
$PY analysis/discern_v2/phase2/fit_rate_operator.py \
    --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
    --out configs/discern_v2/rate --device cuda:N
```

Check the printed `monotone in beta` line. A non-monotone authentic rate-distortion curve means
something is wrong with the fit, not an interesting finding.

Then enable the branch in the config:

```yaml
rate:
  enabled: true
  artifact_path: configs/discern_v2/rate/rate_operator_mrvae.pt
  hidden_dim: 32
  use_slopes: true
```

### 2.2 + 2.3 Evaluate on DF40-Dev, then test for structure

```
$PY training/eval_v1.py --checkpoint <checkpoint-with-rate-branch> --df40 \
    --output logs/phase2/eval/rate_df40dev --device cuda:N \
    --datasets danet_ff danet_cdf mcnet_ff mcnet_cdf tpsm_ff \
                tpsm_cdf facevid2vid_ff facevid2vid_cdf heygen

$PY analysis/discern_v2/phase2/rate_response_analysis.py \
    --parquet logs/phase2/eval/rate_df40dev/per_sample_epoch_N.parquet \
    --out phase2/stage2
```

**P1d enters the architecture only if this shows conditional structure.** The number that decides
it is the *shape increment over level*, not the full-vector probe — a probe on the whole response
may only be re-reading its overall magnitude.

---

## Stage 3 — HARD GATE

```
$PY analysis/discern_v2/phase2/stage3_gate.py \
    --stage0 phase2/stage0/stage0_protocol.json \
    --stage0-ceiling phase2/stage0/stage0_insample.json \
    --stage2 phase2/stage2/stage2_rate_response.json \
    --audit analysis/discern_v2/phase2_domain_audit_diverse/domain_audit.json \
    --out phase2
```

Thresholds are fixed in the script (headroom 0.02, recovery 0.25, minimum 2 methods) so that
moving one is a diff rather than a judgement call made after seeing the numbers. **If this says
NO-GO, stop.** That is a legitimate result and the brief says so explicitly.

---

## Stage 4 — source-paired training  *(only on a GO)*

Both arms are required before any claim that pairing helped.

```
# PRIMARY
$PY training/train_v1.py --output logs/phase2/stage_b_paired_seed42 \
    --sampling paired --device cuda:N

# CONTROL
$PY training/train_v1.py --output logs/phase2/stage_b_random_seed42 \
    --sampling random --device cuda:N

$PY analysis/discern_v2/select_checkpoint.py --run logs/phase2/stage_b_paired_seed42
```

Watch the printed pair fraction. If under half the training samples have a source partner, most of
the epoch is not paired and that fraction belongs beside any provenance-shortcut claim. Do **not**
add SBI. Selection is close to arbitrary — V1's in-domain validation saturated at epoch 0 with a
between-model to within-model signal ratio near 2.

---

## Stages 5-7 — gates, fusion comparison, reliability

One pass, because all three read the same frozen expert opinions.

```
$PY training/stage567.py \
    --run logs/phase2/stage_b_paired_seed42 \
    --val-protocol ffpp \
    --output logs/phase2/stage567/paired_seed42 --device cuda:N
```

`ffpp` is the brief's ground rule 2. `--val-protocol diverse` reproduces the V1 run's calibration
(FF++ ∪ Celeb-DF-v2) and every artifact records which was used, with the resulting loss of
zero-shot status computed rather than described.

Read `STAGE567.md` for:

* whether each gate beats chance out of fold — if not, any gain that arm shows is not the gate's;
* **Equal-CCF vs Applicability-CCF**, which is the clean test of whether applicability earns its
  place;
* **`ccf_vs_ds_isolated`**, which is the operator comparison on a shared `q`. The brief's own
  three-arm comparison (`ccf_vs_ds_confounded`) differs in both the operator and the gate;
* Stage 7's sign check. A wrong coefficient sign means the defer policy abstains on the wrong
  samples, and that is a stop, not a footnote.

---

## Stage 8 — final evaluation, everything frozen

### The conventional and wild suite

```
$PY training/eval_v1.py \
    --checkpoint logs/phase2/stage_b_paired_seed42/epoch_NNN.pth \
    --stage-de logs/phase2/stage567/paired_seed42/stage567.pt \
    --arm applicability_ccf \
    --datasets FaceForensics++ Celeb-DF-v1 Celeb-DF-v2 Celeb-DF-v3 \
               DeepFakeDetection DFDCP DFDC UADFV Deepfake-Eval-2024 \
    --output logs/phase2/eval/final --device cuda:N
```

### DF40-Holdout — the only DF40 rows that carry a zero-shot claim

35 methods, sealed since before Phase 2 and read here for the first time.

```
$PY training/eval_v1.py \
    --checkpoint logs/phase2/stage_b_paired_seed42/epoch_NNN.pth \
    --stage-de logs/phase2/stage567/paired_seed42/stage567.pt \
    --arm applicability_ccf --df40 \
    --output logs/phase2/eval/final_df40holdout --device cuda:N \
    --datasets MRAA_cdf MRAA_ff StyleGAN3_cdf StyleGAN3_ff StyleGANXL_cdf StyleGANXL_ff \
                ddim_cdf ddim_ff e4e_cdf e4e_ff e4s_cdf e4s_ff \
                facedancer_cdf facedancer_ff fomm_cdf fomm_ff fsgan_cdf fsgan_ff \
                hyperreenact_cdf hyperreenact_ff inswap_cdf inswap_ff lia_cdf lia_ff \
                mobileswap_cdf mobileswap_ff one_shot_free_cdf one_shot_free_ff pirender_cdf pirender_ff \
                sd2.1_cdf sd2.1_ff wav2lip_cdf wav2lip_ff whichisreal
```

### The table

```
$PY analysis/discern_v2/phase2/stage8_final.py \
    --parquet logs/phase2/eval/final/per_sample_epoch_NNN.parquet \
              logs/phase2/eval/final_df40holdout/per_sample_epoch_NNN.parquet \
    --stage567 logs/phase2/stage567/paired_seed42/stage567.pt \
    --baseline docs/DiCoME_eval/reproduced_baseline.json \
    --out phase2
```

**Pass `--stage567`.** Without it the script cannot know which sources were calibrated and labels
every row `UNVERIFIED` rather than guessing — but it will still print a table, so it is easy to
run wrong and easy to misread afterwards.

`--baseline` takes a JSON of `{source: video_auroc}`. Sources absent from it stay `TODO(run)`
rather than being filled from a neighbouring column.

---

## §22 / ground rule 3 — the second seed

Anything inside ±0.01, and any effect that is inconsistent across families, needs a confirming
seed before it supports a promotion, a removal or an architecture change. That means rerunning
Stage 4 through Stage 8 with a different `meta.seed` in `V1_CONFIG.yaml` — 🟡 changing the seed is
itself an ASK-UMAR item, so decide it deliberately rather than as a side effect.


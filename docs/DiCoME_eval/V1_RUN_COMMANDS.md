# V1_RUN_COMMANDS — the V1 pipeline, in order

Deliverable for build spec §26. 🔴 **Umar launches every one of these. Nothing here has been run
beyond the smoke subsets noted in `V1_BUILD_REPORT.md`.**

All commands assume:

```bash
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
```

Steps are ordered by dependency. Each says what it writes and roughly what it costs; the cost
figures are extrapolated from CPU smoke runs and are indicative, not measured on a GPU.

> Every step below is runnable. Steps 4–9 are marked `TODO(run)` because they consume Stage B's
> checkpoints, which only exist after step 3 has actually been run.

---

## Step 0a — FS-VFM DLIB+30% crop cache — **NOT REQUIRED for V1**

§8 makes the native crop the default and puts this on the critical path. **Umar's 2026-08-18
decision replaces it with the existing aligned crop**, so no crop cache is generated for V1 and
the CLIP/VAE caches are untouched. Only the small parity subset in step 0b re-extracts native
crops, and it does so on the fly from the source videos.

If the parity result later reverses this decision, see `V1_PARITY_REPORT.md` §3.

## Step 0b — parity control (§6, §8)

```bash
$PY analysis/discern_v2/fsvfm_parity.py \
    --videos 40 --frames 8 \
    --out analysis/discern_v2/V1_parity
```

Writes `analysis/discern_v2/V1_parity/parity.json`. Reads the source videos directly; needs dlib
(present in `dfb_nesy`). Minutes on CPU. The §6 numerical check already passes exactly; this run
is to make the §8 probe gap interpretable at a usable subset size.

## Step 1a — cache frozen FS-VFM features for Stage A

```bash
$PY analysis/discern_v2/cache_encoder_features.py \
    --config training/config/detector/nesy_defake_d1_v.yaml \
    --backbone fsvfm --encoder frozen \
    --datasets FaceForensics++ --split train \
    --out cache/discern_v2/fsvfm_frozen \
    --batch-size 64 --workers 12
```

Writes `cache/discern_v2/fsvfm_frozen/FaceForensics++_train/{features.npz,labels.npy,metadata.csv}`
plus `manifest.json` with the encoder fingerprint. 115,198 frames × 1024 floats ≈ 0.5 GB.
GPU strongly preferred (CPU ran ~2.5 frames/s in the smoke test).

Optionally add `--split val` in a second invocation into the same `--out`; manifests accumulate,
and a different encoder state is refused at write time.

## Step 1b — Stage A: fit `P_R` on FF++ reals, then freeze (§4.1, §9 A)

```bash
$PY analysis/discern_v2/fit_reference.py \
    --features cache/discern_v2/fsvfm_frozen/FaceForensics++_train/features.npz \
    --feature-key f0 \
    --labels   cache/discern_v2/fsvfm_frozen/FaceForensics++_train/labels.npy \
    --arms C3_ae --objectives cosine \
    --latent-dim 128 --hidden-dim 256 --epochs 100 \
    --out configs/discern_v2/reference
```

Writes `configs/discern_v2/reference/reference_C3_ae_cosine.pt` (+ `fit_report.json`). Enforces
reals-only itself, refuses a cache whose encoder was not frozen, and stamps the encoder
fingerprint into the artifact. Minutes.

🟡 `--latent-dim 128 --hidden-dim 256` are the flagged choices — see `V1_BUILD_REPORT.md` §10.

## Step 1c — fit the process-branch standardisation (§5)

```bash
$PY analysis/discern_v2/fit_process_stats.py \
    --config training/config/detector/nesy_defake_d1_v.yaml \
    --vae-path /data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae \
    --dataset FaceForensics++ --split train \
    --batch-size 32 --workers 12 \
    --out configs/discern_v2/process/process_stats.pt
```

Writes the frozen calibrator (+ a JSON provenance file). Runs the frozen VAE over FF++ *real*
train frames only; on GPU this is the longest of the offline steps.

## Step 2 — the meta split (§12)

```bash
$PY analysis/discern_v2/meta_split.py \
    --out configs/discern_v2/meta_split.json
```

Seconds; reads only the dataset JSON. Already verified on the real FF++ val split: 700 videos →
70 identity groups → 420 / 280 (60.0%), folds 60/60/60/50/50, zero shared identities.

---

## Step 3 — Stage B: train the experts (§9 B, §10)

```bash
$PY training/train_v1.py \
    --config training/config/discern_v2/V1_CONFIG.yaml \
    --detector-config training/config/detector/nesy_defake_d1_v.yaml \
    --output logs/v1/stage_b_seed42 \
    --batch-size 32 --workers 12
```

Depends on steps 1b, 1c and 2 (it refuses to start without the Stage-A artifact, the process
statistics, or the meta split). Writes `epoch_XXX.pth` for **every** epoch (§11 forbids a fixed
window), plus `metrics.jsonl` and a config snapshot. Refuses to write into a directory that
already holds a run unless `--overwrite` is passed, so two runs cannot share one provenance.

What it does, and why each part is not the obvious alternative:

- **`model.assert_frozen_protocol()` runs before the first step** (§19), so FS-VFM, `P_R` and the
  VAE are checked frozen rather than assumed.
- **One augmented image feeds all three branches.** The dataset's own augmentation is switched
  off; the ported DiCoME pipeline is applied per sample to the raw [0,1] tensor, and the three
  views are derived from that single image. Augmenting after the dataset produced separately
  normalized tensors would give each branch a *differently* augmented image — the mismatch
  `process_residual.py` warns about, which is invisible during training.
- **Augmentation is applied per sample, not per batch.** torchvision transforms on a batched
  tensor draw their random parameters once and apply the same flip/affine/blur/jitter to the whole
  batch, which would make the ported recipe weaker than the one it reproduces.
- **The loss is per-branch EDL, masked by `branch_valid_b`** (§9 B), ported from DiCoME's
  `evidential_loss_dicome` with per-sample reduction added so masking is expressible. The §4.2
  control head is trained (so the comparison is fair) and reported separately.
- 🟡 `training.fused_loss_weight` defaults to **0.0** — §9 B's literal reading. Setting it above
  zero couples the branches through DS during Stage B, so Stage D's gate would then be choosing
  among experts that already co-adapted. Umar's call.

Validation each epoch runs on **VAL_select only** (13,436 frames, read from `meta_split.json`),
reporting video AUROC and ECE — the two quantities §11 selects on.

## Step 3b — epoch-wise OOD scoring (§11: **analysis only**)

```bash
$PY training/eval_v1.py \
    --checkpoint logs/v1/stage_b_seed42/epoch_001.pth \
    --datasets FaceForensics++ Celeb-DF-v2 Celeb-DF-v3 DFDC DFDCP \
    --output logs/v1/eval/epoch_001 \
    --batch-size 32 --workers 12 --device cuda:N
```

Can be run on any epoch's checkpoint, including while Stage B is still training. §11 explicitly
permits this — "save epoch-wise OOD scores for later analysis only" — and it must not select an
epoch, a hyperparameter or a threshold.

**These are not "DISCERN-v2 V1" numbers.** At Stage B there is no applicability gate (Stage D) and
no defer policy (Stage E), so fusion runs **ungated at q = 1**, which is *plain DS over three
ungated experts* — the baseline the applicability layer is later supposed to beat. The output JSON
says so in an `IMPORTANT` field so a number cannot be lifted into a results table without it.

Writes, per source: fused-ungated frame/video AUROC, the same for each branch separately
(`sem`, `ref`, `proc`), the §4.2 `direct_probe_control`, mean V/C/A, the DS degenerate-fusion
rate, and branch-validity rates. Plus `per_sample_epoch_N.parquet` — the §20 instrumentation, one
row per frame, which is the input to the domain-detector audit and the contribution diagnostics,
so neither needs another forward pass.

Pick the GPU explicitly. The script never chooses one.

## Step 3c — DF40 and Deepfake-Eval-2024

**Deepfake-Eval-2024 needs nothing special.** Its JSON is already in the configured folder and its
frame paths resolve: 428 real + 386 fake test videos, 32 frames each. Add it to `--datasets` in
step 3b.

**DF40 needs `--df40`**, because three things about it differ from every other source:

```bash
$PY training/eval_v1.py \
    --checkpoint logs/v1/stage_b_seed42/epoch_001.pth \
    --df40 \
    --datasets danet_cdf danet_ff fomm_cdf fomm_ff tpsm_cdf tpsm_ff \
               facedancer_cdf faceswap_cdf inswap_cdf simswap_cdf \
               blendface_cdf mcnet_cdf sadtalker_cdf wav2lip_cdf \
               stargan StyleGAN2_ff MidJourney CollabDiff \
    --output logs/v1/eval/epoch_001_df40 \
    --batch-size 16 --workers 4 --device cuda:N
```

1. Its JSONs live in `/data/umar/Datasets/df40/dataset_json/`, not the configured folder.
2. Its frame paths are relative to a root that does not exist here, and the rewrite is
   **not one rule**: authentic halves are borrowed from FF++/CDFv2 (`df40/real/…`), video methods
   sit at `df40/test/<m>/<subset>/frames/…`, and image methods at
   `df40/test/<m>/<half>/<half>/…` with a doubled directory. `--df40` resolves by trying
   candidates and reports the per-method resolution rate; a method missing frames is warned
   about, because an AUROC over a partial method otherwise looks like a complete one.
3. Its labels are per method (`danet_Real`, `stargan_Fake`, …) and the loader RAISES on a label
   absent from `config['label_dict']`. `--df40` injects them for that run only, so the shared
   `training/config/test_config.yaml` is untouched.

Check coverage before committing to a method list:

```bash
$PY training/dataset/df40_paths.py       # per-method resolution rates and families
```

Measured: **60 of 81 method JSONs resolve fully** (52 face-manipulation, 8 whole-image), 21 are
excluded with their rates recorded. `DF40_all.json` is unusable regardless — every entry sits
under `train` and its `test` split is empty, so use the per-method JSONs, which is also what
§21's family/method breakdown asks for.

**Report DF40's two families separately.** Its face-swap and reenactment methods manipulate real
footage and are comparable to FF++/CDF. Its whole-image generators (MidJourney, StyleGAN2/3/XL,
VQGAN, CollabDiff, DiT, SiT, RDDM) emit entire synthetic images up to 1024×1024 — not manipulated
face crops. One pooled "DF40 AUROC" averages over two different tasks.

Because DF40 uses its own JSON folder, run it as a separate invocation and pass **both** exports
to the audit, which accepts several: FF++ comes from the main run and supplies `p(r | R_FF++)`.

```bash
$PY analysis/discern_v2/domain_audit.py \
    --parquet logs/v1/eval/epoch_001/per_sample_epoch_1.parquet \
              logs/v1/eval/epoch_001_df40/per_sample_epoch_1.parquet \
    --out analysis/discern_v2/V1_domain_audit
```

## Step 4 — checkpoint selection (§11)

```bash
$PY analysis/discern_v2/select_checkpoint.py --run logs/v1/stage_b_seed42
```

Seconds, no GPU. Writes `SELECTED.json` + `SELECTION.md` into the run directory. Selects on
**VAL_select only** — video AUROC, tie-broken by ECE then val loss — and refuses to run if the
metrics file carries anything resembling an OOD source, or duplicate epochs from two runs sharing
a directory.

It also reports how many epochs fall inside §22's 0.01 noise floor. On the seed-42 run **all 20
do** (spread 0.0041), so the primary metric orders the epochs without ranking them, and the two
defensible readings disagree: literal §11 picks **epoch 7**, noise-aware §11+§22 picks **epoch 4**
(better ECE). `--noise-aware` switches. 🟡 The choice binds Stages D and E, because §13's target is
defined from the frozen selected experts.

## Steps 5 and 6 — Stages D and E (§12, §13, §18)

```bash
$PY training/stage_de.py \
    --run logs/v1/stage_b_seed42 \
    --output logs/v1/stage_de/epoch_007 \
    --batch-size 32 --workers 12 --device cuda:N
```

One script, because the two stages share one forward pass over VAL_meta and Stage E must consume
Stage D's **out-of-fold** `q`. It reads `SELECTED.json` rather than guessing an epoch, freezes the
experts (`set_stage("D")`, then asserts nothing is trainable), and touches no OOD source.

What it produces:

- **Stage D** — §13's fusion-utility target from undiscounted pairwise DS, gate features, 5-fold
  cross-fitting over `meta_split.json`'s folds, out-of-fold `q` per specialist, plus a deployment
  gate refit on all of VAL_meta. Logs each gate's out-of-fold AUROC and accuracy **against the
  always-admit baseline**, since a gate can score well on accuracy alone when the target is mostly 1.
- **plain DS vs DS + applicability** on VAL_meta, with the caveat printed: it is in-sample for the
  gates and on FF++ validation, so it is not the applicability layer's benefit — the OOD suite
  decides that.
- **Stage E** — the logistic risk model on `[V, C, A, fused_margin]`, the frozen `DeferPolicy`
  (EER decision threshold + abstention-budget quantile, both from FF++ validation), the
  risk-coverage curve, and selective risk at the fixed budget.

Artifacts: `stage_de.pt` (gates + risk model + policy), `stage_de.json`, and
`val_meta_per_sample.parquet`.

## Step 7 — FF++ test

`TODO(run)` — same command as step 3b on the **selected** checkpoint, with
`--datasets FaceForensics++`. Once Stages D and E exist, the gated/deferring system is scored by
applying the frozen gates and `DeferPolicy` to the same per-sample export rather than by a second
forward pass.

## Step 8 — the OOD suite (§21)

`TODO(run)` — step 3b's command on the selected checkpoint, across every source. Sources: Celeb-DF-v1/v2/v3 (CDFv3 with FS/FR/TF breakdown), DFD, DFDCP, DFDC, UADFV,
Deepfake-Eval-2024, DF40 by family/method. Video AUC is primary; report frame AUC where available.
No OOD result may change a checkpoint or hyperparameter after the fact.

Every table must keep §21's four baseline rows distinct: DiCoME paper / released checkpoint; **our
reproduced DiCoME**; DISCERN-v1; DISCERN-v2 V1. Compare against the reproduced numbers — for DFD
that means the in-house 0.9419, not the paper's 0.982.

## Step 9 — instrumentation and read-off (§20)

`TODO(run)`. Save per-frame and per-video: dataset, sample/video id, label, method/family,
`e/p/u` for each of sem/ref/proc, `e_direct` and its prediction, `branch_valid_*`, `q_*`,
reference residual diagnostics, process diagnostics, discounted opinions, DS conflict (including
the degenerate-fusion flag), fused `p/u`, `V/C/A`, risk, prediction, correctness.

Then the post-hoc **contribution diagnostics** (toggles on the trained model, not retrained
ablations): semantic only; +reference; +process; full; plain DS without applicability; DS +
applicability; residual reference vs `e_direct`. And the §20 domain-detector distribution audit
for both specialists — required output, not a build gate.

---

## Applying §22 to everything above

`|ΔAUC| < 0.01`, or inconsistent family-level effects, requires a second seed before any
promotion, removal or architecture change. Our prior controlled experiments already produced
several differences at the noise floor, so this governs every read-off decision.

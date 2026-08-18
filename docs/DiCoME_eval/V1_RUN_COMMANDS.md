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

## Step 4 — checkpoint selection (§11)

`TODO(run)` — needs step 3's checkpoints. Select on **VAL_select only**: primary video-level AUROC,
tie-break lower ECE then val loss. Never select using CDF/DFDC/DF40 or any OOD source; save
epoch-wise OOD scores for later analysis only.

## Step 5 — applicability gates (§13, §12 cross-fit)

`TODO(run)` — depends on step 4. With the selected checkpoint frozen, score VAL_meta, then:

```python
from networks.discern_v2.applicability_gate import cross_fit, fusion_utility_target, gate_features
target = fusion_utility_target(sem_opinion, spec_opinion, labels)["target"]
result = cross_fit(gate_features(sem_opinion, spec_opinion), target, folds)
# result["q_out_of_fold"] feeds step 6; result["gate"] is the deployment gate
```

Log gate AUROC and accuracy against the fusion-utility target, alongside the always-admit
baseline the metrics helper reports.

## Step 6 — risk / defer calibration (§18)

`TODO(run)` — depends on step 5, and must use the **out-of-fold** `q`:

```python
from networks.discern_v2.risk_model import fit_risk_model, freeze_thresholds, risk_features
features = risk_features(V, C, A, prob_fake)
model, info = fit_risk_model(features, wrong)
policy = freeze_thresholds(labels, prob_fake, risk, abstention_budget=0.10)
```

`policy` is immutable and carries its provenance; it is applied unchanged to every OOD source.

## Step 7 — FF++ test

`TODO(run)`.

## Step 8 — the OOD suite (§21)

`TODO(run)`. Sources: Celeb-DF-v1/v2/v3 (CDFv3 with FS/FR/TF breakdown), DFD, DFDCP, DFDC, UADFV,
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

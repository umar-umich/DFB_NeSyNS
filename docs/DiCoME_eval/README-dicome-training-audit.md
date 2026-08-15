# DiCoME Training Reproducibility Audit

Operational README for verifying whether DiCoME's paper claim — training on FaceForensics++ c23 only — is reproducible in-house, or whether the released checkpoint benefits from undisclosed data / pretraining / composite training. Companion to `README-dicome-reproduction.md`, which covered evaluation-side reproduction.

---

## Why this task

The eval-side reproduction concluded cleanly: on three independently reproducible datasets (CDFv2, DFDC, DFDCP where applicable), our numbers land within noise of DiCoME's paper. Video counts match DeepfakeBench splits exactly. The pipeline is trustworthy for reading their released checkpoint.

But that says nothing about **how** that checkpoint was trained. Deepfake papers sometimes:
- Initialize the β-VAE, LoRA, or evidential heads from weights pretrained on other data
- Run multi-stage / curriculum training with an earlier stage on non-FF++ data
- Use auxiliary supervision from out-of-domain datasets
- Apply synthetic data augmentation not disclosed as "training data"
- Load undisclosed prior checkpoints via config fields like `resume_from_checkpoint`

Before building DISCERN v2 on top of DiCoME as a comparison, we need to know: does the FF++-only claim hold? If yes, then a self-trained DiCoME checkpoint becomes the strongest possible baseline — reviewers can't attribute any DISCERN v2 gap to preprocessing, weight provenance, or hidden data. If no, we need to know what they actually did before designing around them.

---

## Established baselines (from eval reproduction task)

For the training reproduction comparison, use these in-house numbers, **not** the paper numbers:

| Dataset | Released ckpt AUC (our reproduction) | Δ vs paper | Status |
|---|---:|---:|---|
| CDFv2 | 0.9729 | −0.0041 | Clean reproduction |
| DFD | 0.9392 | −0.0428 | Unexplained gap, scene-concentrated failure mode |
| DFDC | 0.8822 | +0.0002 | Clean reproduction |
| DFDCP | 0.8799 | — | No paper counterpart |
| CDFv3 (face-swap only) | 0.9529 | +0.0669 | ⚠️ Not comparable — subset-only |
| FF++ (in-domain) | 0.9905 | — | In-domain, do not table cross-dataset |
| DFo | — | — | Not in our data setup |
| WDF | — | — | Not in our data setup |

The training-reproducibility check compares a self-trained checkpoint's AUC against **these** numbers, dataset-by-dataset. A clean training reproduction means the self-trained checkpoint lands within ~0.005 of the released-checkpoint numbers on the comparable datasets.

---

## Prerequisites

- **Env**: conda env `dicome` already set up from the previous task.
- **Repo**: `../DiCoME` with `weights/dicome-best.ckpt` already downloaded and verified.
- **Data**: FF++ c23 present in the project directory. All four manipulations (DF, FS, F2F, NT) plus pristine.
- **H5s**: test-side H5s from the previous task exist. **Train and val H5s for FF++ need to be created — this is new** and belongs to Phase 2.
- **Compute**: 2× H200 (148 GB). One full 20-epoch training run per their config.
- **Disk**: FF++ c23 at 32 frames per video ≈ 8000 videos × 32 = 256K frames → estimate 20–60 GB depending on frame resolution and encoding.
- **Time budget**: 2–6 hours for one full training run on a single H200, per their config's batch size 128 and 20 epochs.

---

## How to use

1. Paste the full prompt (below, "Claude Code prompt" section) into Claude Code.
2. Let **Phase 1 (static training-pipeline audit)** run to completion first. This alone may answer the reproducibility question without needing a training run.
3. Review `../DiCoME/eval_adaptation/training_audit.md`. If the audit surfaces any undisclosed data source or pretraining step, stop and re-plan — do not proceed to actual training on the assumption their claim holds.
4. If Phase 1 clears, allow **Phase 2 (FF++ train/val H5 preparation)** commands to be prepared. Launch the conversion manually.
5. **Phase 3 (training-run preparation)** produces the `fit` command. Review the config alignment and seed before launching.
6. Launch training manually (Phase 4). 2–6 hours.
7. **Phase 5 (post-training eval on our 6 datasets)** reuses the eval infrastructure from the previous task with the new checkpoint substituted in. Prepare then launch.
8. Phase 6 aggregates and interprets against the in-house baselines above.

Gate discipline is important here — the static audit alone can invalidate the whole plan cheaply, so do not skip straight to training.

---

## Claude Code prompt

Paste everything inside the code block:

````markdown
# Task: Verify DiCoME's training reproducibility on FF++ only

## Goal
Determine whether DiCoME's released `dicome-best.ckpt` was actually trained on FF++ c23 only (as their paper claims), or whether the training pipeline touches additional data — pretraining, curriculum stages, auxiliary supervision, or undisclosed dataset mixing. Then run their exact `fit` command on FF++ only, one seed, and compare the resulting checkpoint's eval numbers to the released checkpoint's numbers (already reproduced in the previous task).

## Context
- Repo at `../DiCoME`. Eval already reproduced cleanly on our 6 datasets — this investigation is upstream of that.
- Their paper claims training on FF++ c23 only (Section 4.1: "all experiments, the model is trained on FaceForensics++ (FF++) under the c23 compression setting").
- Released checkpoint has ~0.868M trainable params on frozen CLIP ViT-L/14 (LoRA + β-VAE + evidential heads).
- Compute: 2× H200 (148 GB). One full training run is 20 epochs per `src/config/dicome_default.yaml` (max_epochs: 20, dicome_epochs: 20, batch_size: 128, learning_rate: 1e-4).
- Env `dicome` already set up.
- Eval-side H5s exist under `../DiCoME/eval_adaptation/data/h5/`. Train and val H5s for FF++ do NOT yet exist — Phase 2 creates them.
- Baseline eval numbers on the released checkpoint (for Phase 5 comparison): CDFv2 0.9729, DFD 0.9392, DFDC 0.8822, DFDCP 0.8799, CDFv3 (face-swap only) 0.9529, FF++ 0.9905.

## Contract
- 🟢 YOU-RUN: all static code and config analysis, dataloader inspection, checkpoint provenance analysis, H5 preparation scripts, comparison scripting, aggregation.
- 🔴 UMAR-RUNS: FF++ train/val H5 conversion (potentially many GB), the actual FF++ training run (2–6 hours), Phase 5 evals.
- 🟡 ASK-UMAR: any decision that would change training config, seed, or data. Also ask if Phase 1 surfaces undisclosed data sources.
- No invented numbers. If a question can't be answered from the code, say so explicitly.
- Do NOT modify DiCoME's config or model code. Do NOT modify our project's data directories.

## Phase 1 — Static training-pipeline audit (YOU-RUN)

Investigate whether the training pipeline touches any data or weights beyond FF++ c23 + the public CLIP backbone.

1. **Config audit**. Read `../DiCoME/src/config/dicome_default.yaml` completely. Report every field that references:
   - Data paths (`trn_h5_path`, `val_h5_path`, any auxiliary data path)
   - Pretrained weight loading (`pretrained`, `resume_from_checkpoint`, `init_from`, `load_state_dict`, similar)
   - Model checkpoints beyond the CLIP backbone
   - Any auxiliary datasets or loss weights that imply extra supervision

2. **Training script trace**. Read `../DiCoME/tools/train/train_dicome.py` end-to-end. Trace what `fit` actually does:
   - What datasets does the trainer see? Just `trn_h5_path` and `val_h5_path`, or more?
   - Are there multiple training stages (pretraining → fine-tuning)?
   - Are any weights loaded before training begins beyond the CLIP backbone?
   - Are there conditional branches gated on env vars or CLI flags that could pull in extra data?
   - Note any callbacks or hooks that could touch external data.

3. **Model init trace**. Trace how each component is initialized:
   - CLIP ViT-L/14: loaded from `openai/clip-vit-large-patch14` via `transformers` (expected).
   - LoRA adapters: random init or loaded?
   - β-VAE: random init or loaded?
   - Evidential heads: random init or loaded?
   - Report file:line for each initialization.

4. **Dataloader inspection**. Look at the training data loader implementation. Report:
   - What data does the training loader actually sample from at train time?
   - Are there any additional labels beyond binary real/fake?
   - Is there any synthetic augmentation that generates content from a model?

5. **Checkpoint provenance**. Load `../DiCoME/weights/dicome-best.ckpt` in Python and inspect:
   - State-dict key list (save to a file, don't print all keys inline).
   - Any metadata keys (`epoch`, `global_step`, `optimizer_states`, `hyper_parameters`, `datamodule_hyper_parameters` — Lightning stores these).
   - Any hyperparameters embedded in the checkpoint that reveal the training data used.
   - Compare embedded hyperparameters against `src/config/dicome_default.yaml`. Note any differences.

6. **DeepfakeBench heritage check**. Their paper says they follow DeepfakeBench preprocessing. Check whether DeepfakeBench code in this repo imports or references any pretrained submodules, face detectors trained on external data, or auxiliary networks. A face detector for cropping is standard and not a concern; anything training-time-supervising is.

Write findings to `../DiCoME/eval_adaptation/training_audit.md`. Structure:
- Section per subtask above
- Direct quotes / file:line refs for each claim
- Concluding assessment: "based on static analysis, training touches only FF++ c23 + CLIP backbone" OR "training touches these additional sources: {...}" OR "insufficient evidence to determine from static analysis; the following requires a runtime check: {...}"

🟡 ASK-UMAR before Phase 2 if the static audit surfaces any undisclosed data or pretraining. That may change the plan entirely.

## Phase 2 — FF++ train/val H5 preparation (UMAR-RUNS, YOU prepare)

Only proceed if Phase 1 concludes the pipeline is FF++-only.

1. Check whether our project's FF++ frames cover all four manipulations (Deepfakes, FaceSwap, Face2Face, NeuralTextures) plus pristine (real). Report per-manipulation video counts. FF++ standard training split is ~720 videos per manipulation, ~720 pristine — verify our counts match or note the deviation.
2. Write `../DiCoME/eval_adaptation/convert_ffpp_train.sh` that runs `tools/data/folder_to_h5_dataset.py` to produce:
   - `../DiCoME/eval_adaptation/data/h5/FFpp_train.h5`
   - `../DiCoME/eval_adaptation/data/h5/FFpp_val.h5`
   Reuse the same label-inference conventions the eval task landed on — this pipeline had an id-collision issue during eval that required renaming. Preserve whatever fix was applied.
3. Write the corresponding split-txt generation commands using `tools/data/h5_to_split_txt.py` with the correct `--group_index` per subset structure. Store split txts under `../DiCoME/eval_adaptation/data/splits/train/FFpp/` and `.../val/FFpp/`.
4. Estimate H5 disk size before launching (num_videos × frames_per_video × avg_frame_bytes). Report per-file estimate. If total exceeds 100 GB, flag it.
5. Write `verify_train_loader.py` that instantiates DiCoME's training data loader against the new H5s + splits, pulls three batches, and confirms:
   - Label distribution is balanced-ish (real vs fake)
   - All four manipulations are sampled
   - Tensor shapes match what CLIP ViT-L/14 expects (typically 224×224 RGB)
6. Print the conversion script and disk estimate for review. I will launch it.

🟡 ASK-UMAR if any FF++ manipulation is missing or has fewer videos than expected — do not silently proceed with a subset.

## Phase 3 — Training run preparation (YOU-RUN, don't launch)

After train/val H5s exist and verification passes:

1. Write a training config `../DiCoME/eval_adaptation/configs/train_ffpp.yaml` that:
   - Copies `src/config/dicome_default.yaml` verbatim
   - Overrides ONLY `trn_h5_path`, `val_h5_path`, and their associated split files to point at our FF++ H5s
   - Preserves every other field (backbone, feature_dim, batch_size, max_epochs, dicome_epochs, learning rate, β_kld, λ_align, λ_vae)
   - Adds a unique `run_name` or output-dir suffix so this run doesn't overwrite anything
2. Write `../DiCoME/eval_adaptation/train_ffpp_reproduce.sh`:
```bash
cd ../DiCoME
python tools/train/train_dicome.py fit \
  --config_path eval_adaptation/configs/train_ffpp.yaml \
  2>&1 | tee eval_adaptation/logs/train_ffpp.log
```
Add seed pinning as a CLI flag if the training script supports one — verify from Phase 1's trace.
3. Confirm the config's `trn_h5_path` and `val_h5_path` point at ONLY FF++ (no accidental mixing).
4. Estimate training time: 20 epochs × (FF++ train set size ÷ batch 128) × per-step wall time. Report the estimate.
5. Print the full command for review.

## Phase 4 — Training run (UMAR-RUNS)

I launch the training manually. Monitor logs. Expected artifacts:
- `../DiCoME/eval_adaptation/logs/train_ffpp.log`
- New checkpoint(s) at wherever DiCoME's config directs (usually `runs/{run_name}/` — verify).

Do not proceed to Phase 5 until training completes all 20 epochs and a `last.ckpt` or `best.ckpt` equivalent exists.

## Phase 5 — Post-training evaluation (YOU-RUN scripts, UMAR-RUNS eval)

Evaluate the newly trained checkpoint on our 6 datasets (CDFv2, DFD, DFDC, DFDCP, CDFv3, FF++) using the exact same configs and infrastructure from the previous eval task.

1. Locate the new checkpoint. Write `../DiCoME/eval_adaptation/run_eval_trained.sh` that runs the same six-dataset eval loop from the previous task, but with `--config_path` pointing at each dataset's config AND overriding the checkpoint path to the new one. If DiCoME's test command takes the checkpoint as a positional arg (per README: `python tools/train/train_dicome.py test <ckpt> --config_path <cfg>`), just swap the checkpoint path.
2. Results go under `../DiCoME/eval_adaptation/results/trained_from_scratch/{dataset}.json`. Logs under `.../logs/trained_from_scratch/{dataset}.log`.
3. If frame-level CSV aggregation is needed (as in Phase 6 of the eval task), reuse `aggregate_video_auc.py`.
4. Print the eval script for review. I will launch it.

## Phase 6 — Reporting and interpretation (YOU-RUN)

Aggregate results into `../DiCoME/eval_adaptation/TRAINING_REPRODUCTION.md`. Columns:

| Dataset | Released ckpt AUC (in-house) | Our-trained ckpt AUC | Δ | Notes |

Reference numbers for the released-ckpt column (in-house, from the previous task):
- CDFv2: 0.9729
- DFD: 0.9392
- DFDC: 0.8822
- DFDCP: 0.8799
- CDFv3 (face-swap only): 0.9529
- FF++ (in-domain): 0.9905

Interpretation bands:
- **|Δ| ≤ 0.005 across all comparable datasets**: training reproduces cleanly. DiCoME's FF++-only claim holds. Self-trained checkpoint is a valid DiCoME baseline for DISCERN v2 comparisons.
- **0.005 < |Δ| ≤ 0.015**: within likely one-seed noise (paper reports no variance). Probably clean but note it.
- **|Δ| > 0.015 systematically negative (our-trained worse)**: released checkpoint benefited from something our training didn't reproduce. Candidates: (a) additional data, (b) cherry-picked seed, (c) longer training than the config specifies, (d) hyperparameter tuning not in the released config.
- **|Δ| > 0.015 systematically positive (our-trained better)**: also suspicious — likely means released checkpoint is undertrained, or we've accidentally trained on more data than they did (verify no leakage in Phase 2's H5).

Special-case rows:
- **CDFv3**: our data is face-swap-only. Both the released-ckpt AUC (0.9529) and our-trained AUC will be on the same subset, so the delta is meaningful even though neither matches the paper's 0.886.
- **FF++**: in-domain. Both checkpoints saw FF++ during training (assuming Phase 1 clears). This row measures training convergence quality, not generalization. A large negative Δ here is a training bug; a small Δ is expected.
- **DFD**: our released-ckpt already shows a −0.043 gap vs paper, scene-concentrated. If our-trained checkpoint's DFD number is close to 0.9392, that inherits the same pipeline behavior — expected. If it's much worse, that's a training issue on top of the pipeline issue.

Write a short interpretation section (15–30 lines) at the end covering: which datasets reproduced, which showed gaps, whether the training reproduces cleanly enough to declare DiCoME's FF++-only claim verified, and whether the self-trained checkpoint should be used as the DiCoME baseline for DISCERN v2 head-to-head comparisons.

## Explicit non-goals
- Do NOT modify DiCoME's config, model code, or training script.
- Do NOT modify our project's FF++ data directories.
- Do NOT report training-reproducibility conclusions from a partially-completed training run. Full 20 epochs or nothing.
- Do NOT compare against the paper's Table 1 numbers directly — the in-house released-ckpt numbers are the fair baseline.
- Do NOT skip Phase 1's static audit. It may resolve the question without needing training.

## Success criteria
- `training_audit.md` exists with clear conclusions about what the training pipeline actually touches.
- If Phase 1 clears: `TRAINING_REPRODUCTION.md` exists with the 6-dataset comparison table and interpretation.
- Final verdict: is DiCoME's FF++-only training claim reproducible in-house, and can the self-trained checkpoint serve as the DiCoME baseline for DISCERN v2 comparisons.
````

---

## Expected directory layout after completion

```text
../DiCoME/
├── weights/
│   └── dicome-best.ckpt                          # from previous task
├── eval_adaptation/
│   ├── training_audit.md                         # Phase 1 output
│   ├── convert_ffpp_train.sh                     # Phase 2 (I run)
│   ├── verify_train_loader.py                    # Phase 2 verification
│   ├── train_ffpp_reproduce.sh                   # Phase 3 (I run in Phase 4)
│   ├── run_eval_trained.sh                       # Phase 5 (I run)
│   ├── TRAINING_REPRODUCTION.md                  # Phase 6 output
│   ├── data/
│   │   ├── h5/
│   │   │   ├── FFpp_train.h5                     # Phase 2
│   │   │   ├── FFpp_val.h5                       # Phase 2
│   │   │   └── {6 test H5s from previous task}
│   │   └── splits/
│   │       ├── train/FFpp/
│   │       ├── val/FFpp/
│   │       └── test/{6 datasets, from previous task}
│   ├── configs/
│   │   ├── train_ffpp.yaml                       # Phase 3
│   │   └── {6 per-dataset test configs from previous task}
│   ├── logs/
│   │   ├── train_ffpp.log                        # Phase 4
│   │   └── trained_from_scratch/
│   │       └── {dataset}.log                     # Phase 5
│   └── results/
│       └── trained_from_scratch/
│           └── {dataset}.json                    # Phase 5
└── runs/
    └── {training run's checkpoints from Phase 4}
```

---

## Known limitations and caveats

- **DFD gap carries through**. Our released-ckpt DFD number is already 0.043 below the paper (scene-concentrated failure on `meeting_serious` and `secret_conversation` scenes). Any self-trained checkpoint will inherit this pipeline behavior. Interpret the DFD delta against 0.9392, not the paper's 0.982.
- **CDFv3 face-swap-only**. The training-reproduction delta on CDFv3 will be against the released-ckpt's 0.9529, which is on the face-swap subset only. This is fine for measuring training reproducibility but is not usable as a general-model comparison until the missing three families (reenactment, entire-face-synthesis, face-edit) are added to our data.
- **DFo and WDF absent**. Two of DiCoME's paper datasets are not in our data. The training-reproduction check runs on the six we do have. No claim of full six-paper-dataset reproduction is possible without acquiring DFo and WDF.
- **Single-seed training**. DiCoME's paper Table 1 numbers appear to be single-seed. Our reproduction is also single-seed. Δ interpretation bands assume seed noise of ~0.005 AUC, which is consistent with their post-rebuttal 5-seed std values (0.001–0.003).
- **FF++ subset choice**. If our FF++ data uses a different train/val split than DeepfakeBench's canonical split (e.g., different video-id allocation), this alone can produce a small systematic gap on all datasets. Verify the split matches in Phase 2.
- **Time cost**. One full training run is 2–6 hours. If Phase 1's static audit is inconclusive and multiple runs are needed to resolve the question (e.g., different seeds), the cost multiplies. Prioritize Phase 1 sharpness.

---

## Next steps after this task

Two outcome branches:

**If training reproduces cleanly (|Δ| ≤ 0.005 across comparable datasets)**:
1. Declare DiCoME's FF++-only claim verified. Use the self-trained checkpoint as the DiCoME baseline for all DISCERN v2 comparisons — this eliminates checkpoint-provenance as a confound in reviewer critique.
2. Move to the multi-backbone perception pilot (CLIP + DINOv2 + FSFM standalone probes on FF++ → CDFv2/DFDC held-out).
3. In parallel, expand data coverage: full CDFv3 splits, Deepfake-Eval-2024, and (if in scope) DFo + WDF.

**If training does not reproduce (systematic |Δ| > 0.015)**:
1. Do NOT use DiCoME as a baseline for DISCERN v2 until the gap is understood. Their reported numbers may not be achievable with the released code.
2. Investigate the specific direction of the gap: worse (they had more data or better tuning) or better (their released checkpoint is undertrained). Either finding is publishable on its own as a reproducibility note.
3. Consider reaching out to the DiCoME authors or filing a GitHub issue on the missing details.
4. Reframe the DISCERN v2 comparison target — if DiCoME isn't reproducible, GenD or Effort become the primary baselines instead.

Regardless of outcome, the static audit (Phase 1) is the most valuable single step. It's cheap, it may resolve the question without any training, and its conclusions should be preserved in the T-BIOM extension paper's related-work discussion — reproducibility findings are legitimate scholarly content.
claude --resume 8a628fa1-80f0-48a5-98c3-e432c79b3042
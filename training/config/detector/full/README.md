# Full-training configs — run manifest (Phase 2, Task 4)

Three 100-epoch configs generated from `configs/ablations/full_defakenet_18rules.yaml`
by `_generate_full.py`. Regenerate with:

```bash
cd training && python config/detector/full/_generate_full.py
```

## What each config isolates

| config | concept | causal | edl.ibdc | symbolic_reweight | isolates |
|---|---|---|---|---|---|
| `f1_lean`    | substrate_only, mlp head    | — (none)   | v1 | off | lean neuro-symbolic floor (no causal) |
| `f2_full`    | both, **rule_linear** head  | **CCV**    | **v2** | **on** | full stack (S1+S2+S9 + CCV) |
| `f3_ibdc_v1` | both, rule_linear head      | CCV        | v1 | off | f2 minus S2+S9 (isolates them jointly) |

All three: `nEpochs=100`, `early_stopping.patience=20`, `tta.enabled=false`,
`test_dataset` = all six, per-sample CSV logging on (automatic in `test.py`),
`consistency_rules_version=v8_retained` (18 rules), CLIP ViT-L/14 frozen.

## Prerequisites

- Env `dfb_nesy`: `PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python`
- **Run from the repo root** (dlib landmark file + `retained_predicates.yaml` are
  CWD-relative).
- Seeds are passed at launch via `--seed` (overrides `manualSeed`); run each of
  **3407, 42, 1024** → 9 training runs total.

## Launch (added 2026-08-01)

Each run writes to `logs/full/<config>/train/<config>_<timestamp>/`; the timestamp
keeps the three seeds of a config from colliding. Best checkpoint is `best_avg.pth`
there. Record which timestamp = which seed as you launch (the folder name has no
seed tag).

**Two-stage protocol (important).** Training evaluates + early-stops on a small
VALIDATION set only — `test_dataset: [FaceForensics++, Celeb-DF-v2]` — so the model
is selected for cross-dataset generalization (not just FF++) without letting the
other four sets leak into model selection, and without wasting each epoch on ~335k
OOD frames. The full six-dataset numbers are produced POST-HOC with `test.py
--test_dataset` (below). Checkpoints land in `logs/train/<config>_<timestamp>/`
(`best_avg.pth` = best on the validation average; per-dataset `best_*.pth` too).

```bash
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
DS="FaceForensics++ Celeb-DF-v2 Celeb-DF-v3 DeepFakeDetection DFDC DFDCP"

# ---- TRAIN: 4 configs x 3 seeds = 12 runs (add f2_noaug for the T1 ablation) ----
for CFG in f1_lean f2_full f2_noaug f3_ibdc_v1; do
  for SEED in 3407 42 1024; do
    $PY training/train.py \
        --detector_path training/config/detector/full/$CFG.yaml \
        --seed $SEED       # early-stops on FF++ + CDFv2 validation
  done
done

# ---- TEST post-hoc on ALL SIX (fill in the timestamped run dir per config/seed) ----
# RUN=logs/train/f2_full_2026-08-03-10-25-14
$PY training/test.py \
    --detector_path training/config/detector/full/f2_full.yaml \
    --weights_path  $RUN/best_avg.pth \
    --test_dataset  $DS \
    --output_dir    logs/test/f2_full_seed3407_full
# → metrics.csv + per_sample_*.csv per dataset under --output_dir
```

Runtime: early stopping (patience 20, on the FF++/CDFv2 validation average) usually
stops each run in ~25–35 epochs, not 100 — the seed-3407 batch stopped at epochs
25–31. Parallelize configs across GPUs (one `CUDA_VISIBLE_DEVICES` per config).

## Tracking

| config | seed 3407 | seed 42 | seed 1024 |
|---|---|---|---|
| f1_lean    | launched ____ / done ____ | ____ | ____ |
| f2_full    | launched ____ / done ____ | ____ | ____ |
| f3_ibdc_v1 | launched ____ / done ____ | ____ | ____ |

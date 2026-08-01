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

```bash
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python

# ---- TRAIN: 3 configs x 3 seeds = 9 runs ----
for CFG in f1_lean f2_full f3_ibdc_v1; do
  for SEED in 3407 42 1024; do
    $PY training/train.py \
        --detector_path training/config/detector/full/$CFG.yaml \
        --seed $SEED
  done
done

# ---- TEST one run (fill in the actual timestamped run dir per seed) ----
# e.g. RUN=logs/full/f2_full/train/f2_full_2026-08-01-12-00-00
$PY training/test.py \
    --detector_path training/config/detector/full/f2_full.yaml \
    --weights_path  $RUN/best_avg.pth \
    --output_dir    logs/test/f2_full_seed3407
# → writes metrics.csv + per_sample_*.csv per dataset under --output_dir
```

Runtime: ~100 epochs on FF++ c23 (frozen backbone) ≈ many hours per run; budget
accordingly and parallelize across GPUs. Early stopping (patience 20) will usually
cut this short.

## Tracking

| config | seed 3407 | seed 42 | seed 1024 |
|---|---|---|---|
| f1_lean    | launched ____ / done ____ | ____ | ____ |
| f2_full    | launched ____ / done ____ | ____ | ____ |
| f3_ibdc_v1 | launched ____ / done ____ | ____ | ____ |

#!/usr/bin/env bash
# DiCoME across DF40, one row per generator with both source halves pooled.
#
# POOLED, because every other row of the DF40 table is pooled from videos and `eval_df40.py`
# keeps no per-video scores -- averaging its two per-half AUROCs would make this row a different
# quantity from the rest. A config is only a file list, so listing both halves' fakes and both
# real pools gives the pooled number directly. 21 rows instead of 41, same frames.
#
# SAME CHECKPOINT AS THE TABLE 1 ROW: `ffpp_reproduce_seed42` epoch 1, which reproduces that
# row's CDFv2 0.9644 exactly.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="/data/umar/Repos/DiCoME/runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=01-val_auroc_video=0.9960.ckpt"
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40/dicome_repo_df40_combined.json
L=logs/scope_expand/dicome
M=logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== DiCoME over DF40, 21 pooled generator rows ==="
CUDA_VISIBLE_DEVICES=1 $PY analysis/tbiom/baselines/dicome_df40_run.py "$CK" --out "$OUT" \
    >> "$L/df40_combined.log" 2>&1 \
  && say "=== DiCoME DF40 pooled done ===" || say "=== DiCoME DF40 pooled FAILED ==="

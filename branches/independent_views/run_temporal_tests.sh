#!/usr/bin/env bash
# Temporal per-clip features on the test sets, needed because the J-runs train the video head and so
# cannot be evaluated through the deployed opinions.
set -uo pipefail
GPU=$1; shift
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
cd /data/umar/Repos/DFB_NeSyNS || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for DS in "$@"; do
  [ -f "logs/joint_ibdc/cache/${DS}_T_clean.npz" ] && { echo "[$(date '+%H:%M')] skip $DS"; continue; }
  echo "[$(date '+%H:%M')] temporal $DS"
  CUDA_VISIBLE_DEVICES=$GPU $PY training/joint_ibdc/extract_temporal_train.py \
      --dataset "$DS" --arm none --draws 1 --batch-size 32 --workers 6 \
      >> "logs/joint_ibdc/temporal_${DS}.log" 2>&1 \
    && echo "[$(date '+%H:%M')] $DS done" || echo "[$(date '+%H:%M')] $DS FAILED"
done
echo "[$(date '+%H:%M')] temporal test lane finished"

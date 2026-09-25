#!/usr/bin/env bash
# Part 3 extraction lane: one backbone over the development and training splits.
set -uo pipefail
GPU=$1; BB=$2
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for DS in FFpp_val VALmix VALmixV2 FFpp_train; do
  echo "[$(date '+%H:%M')] $BB $DS"
  CUDA_VISIBLE_DEVICES=$GPU $PY branches/independent_views/extract_backbone.py \
      --backbone "$BB" --dataset "$DS" >> "logs/independent_views/${BB}_${DS}.log" 2>&1 \
    && echo "[$(date '+%H:%M')] $BB $DS done" || echo "[$(date '+%H:%M')] $BB $DS FAILED"
done
echo "[$(date '+%H:%M')] $BB lane finished"

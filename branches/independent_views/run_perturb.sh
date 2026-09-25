#!/usr/bin/env bash
set -uo pipefail
GPU=$1; BB=$2
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# the perturbation is chosen on FF++ val, so all four run there; the dev sets follow
for DS in FFpp_val VALmix VALmixV2; do
  CUDA_VISIBLE_DEVICES=$GPU $PY branches/independent_views/perturb_stability.py \
      --backbone "$BB" --dataset "$DS" >> "logs/independent_views/pert_${BB}_${DS}.log" 2>&1 \
    && echo "[$(date '+%H:%M')] $BB $DS done" || echo "[$(date '+%H:%M')] $BB $DS FAILED"
done
# the compression check needs a JPEG q50 copy of FF++ validation
CUDA_VISIBLE_DEVICES=$GPU $PY branches/independent_views/perturb_stability.py \
    --backbone "$BB" --dataset FFpp_val --jpeg-check >> "logs/independent_views/pert_${BB}_jpeg.log" 2>&1 \
  && echo "[$(date '+%H:%M')] $BB jpeg50 done" || echo "[$(date '+%H:%M')] $BB jpeg50 FAILED"
echo "[$(date '+%H:%M')] $BB perturbation lane finished"

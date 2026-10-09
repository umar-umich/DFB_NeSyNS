#!/bin/bash
# End-to-end P-dense: the same architecture as branch P, retrained on stride-1 consecutive clips.
# GPU 0 -- third priority, used because GPU 1 and GPU 3 are both carrying LoRA adaptations.
# setsid-detached: survives SSH disconnection.
#
# DEVELOPMENT-ONLY PILOT. Dense crops exist for FF++ train and the development pool only, and they
# cover 821 of VALmix's 1350 videos, so the dense development score is measured over a SUBSET of the
# pool and is not directly comparable to a sparse development score. The comparison that IS valid is
# dense-P against sparse-P on the same 821 videos, and against the shuffled control of each.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "=== smoke $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/train_p_dense.py --smoke || { echo "SMOKE FAILED - not training"; exit 3; }
echo "=== train P-dense seed 42 $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=0 $PY upgrade/branchfusion/train_p_dense.py --train --seed 42 --epochs 10 --workers 6 \
  || echo FAILED_p_dense
echo "P_DENSE_DONE $(date -u +%FT%TZ)"

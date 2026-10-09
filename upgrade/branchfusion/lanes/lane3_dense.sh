#!/bin/bash
# Lane 3: dense stride-1 PE features, then the S/P dense variants and their order-loss arms.
# GPU priority: 1, then 3, then 0. Never GPU 2 (kutub).
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU="${1:-3}"
for part in ffpp_train dev2; do
  for depth in "" "--depth mid"; do
    echo "=== dense $part $depth $(date -u +%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/dense_features.py \
        --part $part $depth --batch-size 64 || echo "FAILED dense $part $depth"
  done
done
echo "LANE3 EXTRACT DONE $(date -u +%FT%TZ)"

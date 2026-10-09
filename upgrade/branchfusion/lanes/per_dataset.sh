#!/bin/bash
# The full per-dataset ablation grid: every subset of the six branches, every test set.
# EXPLORATORY by construction -- see the module docstring. CPU only. setsid-detached.
set -u
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "=== per-dataset grid $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/per_dataset.py \
    --branches A,S,V,O,M,P --sizes 1,2,3,4,5,6 || echo FAILED_per_dataset
echo "PER_DATASET_DONE $(date -u +%FT%TZ)"

#!/bin/bash
# EVERY combination of all seven branches on every test set: 127 subsets x 10 sets.
# A,S,V,O,M,P plus Y (off-manifold on the PE-Core trunk, arm 1).
# 201,811 weight evaluations in total -- CPU only, no GPU. setsid-detached.
#
# EXPLORATORY by construction: sweeping every subset over every test set is a full search on the
# test partitions, so the best cell is biased upward and is not a held-out estimate. The module
# stamps that on the JSON and on the markdown.
set -u
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "=== all-seven-branch per-dataset grid $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/per_dataset.py \
    --branches A,S,V,O,M,P,Y --sizes 1,2,3,4,5,6,7 --tag all7 || echo FAILED_all7
echo "ALL7_DONE $(date -u +%FT%TZ)"

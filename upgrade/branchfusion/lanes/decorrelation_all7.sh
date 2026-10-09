#!/bin/bash
# Full pairwise correlation structure over all seven branches, per video, plus every triple.
# UMAR: the point is three branches that are each strong AND mutually uncorrelated, so the whole
# matrix is needed -- Y's correlation with the CLIP-based OM and with V matters more than its
# correlation with A. CPU only. setsid-detached.
set -u
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "=== decorrelation, all seven branches $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/decorrelation.py \
    --branches A,S,V,O,M,P,Y --tag all7 || echo FAILED_decorr
echo "DECORR_DONE $(date -u +%FT%TZ)"

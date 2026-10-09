#!/bin/bash
# The unbiased nested estimate over the SAME subset space the screen now covers.
# Why: the screen was re-run with S and R included but capped at `--sizes 1,2,3`, so the selection
# rule only ever saw triples while the earlier five-branch screen had A+V+O+M+P at 0.9115. Comparing
# a triples-only nested estimate against a full-space screen is comparing two different searches.
# CPU only -- no GPU, 4 threads per the box's convention. setsid-detached.
set -u
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== nested, full subset space, 6 operators $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/subsets.py --nested \
    --branches A,S,V,O,M,P --nested-sizes 1,2,3,4,5,6 || echo FAILED_nested_full
echo "NESTED_FULL_DONE $(date -u +%FT%TZ)"

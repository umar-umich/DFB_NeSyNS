#!/bin/bash
# Lane 1: the section-4 subset screen and nested selection. CPU-only -- it reads cached opinions.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
echo "=== screen $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/subsets.py --screen --branches A,V,O,M,P --sizes 1,2,3,4,5 || echo FAILED_screen
echo "=== nested $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/subsets.py --nested --branches A,V,O,M,P || echo FAILED_nested
echo "LANE1 DONE $(date -u +%FT%TZ)"

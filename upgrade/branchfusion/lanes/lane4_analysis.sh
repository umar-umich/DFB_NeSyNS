#!/bin/bash
# Lane 4: sections 4, 5 and 7 on the branches that already have full coverage (A, V, O, M, P).
# CPU-only: it reads cached opinions. S and R join after their extractions finish.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
BR=A,V,O,M,P
SHORT=A+V+O,A+V+P,A+V+O+M+P,A+V

echo "=== section 7 temporal controls $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/temporal_controls.py --branches V,M,P --negative-control || echo FAILED_temporal

echo "=== section 4 diagnostics $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/subsets.py --branches $BR --diagnostics "$SHORT" || echo FAILED_diag

echo "=== section 4 contribution $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/subsets.py --branches $BR --contribution A+V+O,A+V+P,A+V+O+M+P || echo FAILED_contrib

echo "=== section 4 locked test evaluation $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/subsets.py --branches $BR --test --subsets "$SHORT" || echo FAILED_test

echo "=== section 5 fusion matrix $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/fusion_matrix.py --branches $BR --subsets "$SHORT" --ops all || echo FAILED_fusion

echo "=== tables $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/tables.py --interim || echo FAILED_tables
echo "LANE4 DONE $(date -u +%FT%TZ)"

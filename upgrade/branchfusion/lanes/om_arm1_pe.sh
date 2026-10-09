#!/bin/bash
# ARM 1 of the off-manifold-on-other-encoders study: OM on the PE-Core APP trunk.
#
# PREDICTED TO FAIL, and run because of that. A and S share this trunk and the fusion fit zeroes S in
# 4-5 of 5 folds whenever A is present (subset macro moves exactly 0.0000 in 11 of 13 cases). The
# deployed OM works partly BECAUSE it sits on CLIP while APP moved to PE -- even across two different
# backbones their errors correlate at phi ~ +0.41..+0.46, the highest pair in the system. If this arm
# earns non-zero weight beside A anyway, the shared-trunk explanation is wrong.
#
# The criterion is `folds_weight_zero` in the screen, NOT standalone dev v3.1.
# GPU 1 per UMAR's instruction. setsid-detached.
set -u
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== ARM 1: OM on pe_core_lLN $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/om_branch.py --backbone pe_core_lLN \
    --operator bvae --seed 0 --train --export --arm r || { echo FAILED_arm1; exit 1; }

echo "=== register and screen against the core branches $(date -u +%FT%TZ) ==="
$PY -c "
import sys; sys.path.insert(0,'upgrade/branchfusion')
import branches as B, common as C
B.register('Y', C.BRANCH_OUT/'OM-pe_core_lLN-bvae_s0',
           'ARM 1: off-manifold residual on the PE-Core APP trunk (shared-trunk falsification test)')
print('registered Y')
print(B.coverage().to_string())
"
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/subsets.py --screen \
    --branches A,V,O,Y --sizes 1,2,3,4 || echo FAILED_arm1_screen
echo "ARM1_DONE $(date -u +%FT%TZ)"

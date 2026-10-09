#!/bin/bash
# Relaunch the two order-loss S arms (fixed: the frame permutation shadowed the batch permutation)
# and the end-to-end P-dense run (fixed: dense coverage is floored, not asserted at 100%).
# setsid-detached. GPU 1 is carrying the LoRA sweep, GPU 3 is heavily used by kutub, so the small
# cached-feature arms go on GPU 1 and the expensive P-dense goes on the idle GPU 0.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== order-loss arms (cached features, minutes each) $(date -u +%FT%TZ) ==="
for v in dense adapt; do
  CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/seq_branch.py --variant $v --seed 0 \
      --order-loss 0.3 || echo "FAILED S-$v-ord"
done

echo "=== P-dense end-to-end, seed 42 $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=0 $PY upgrade/branchfusion/train_p_dense.py --train --seed 42 \
    --epochs 10 --workers 6 || echo FAILED_p_dense

echo "=== sequence controls on every S arm, including the new ones $(date -u +%FT%TZ) ==="
TAGS=$(ls logs/branchfusion/seq/S-*_s0.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | paste -sd, -)
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/temporal_controls.py --branches V,M,P \
    --seq "$TAGS" --negative-control || echo FAILED_controls

echo "=== refresh tables $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/tables.py --all || echo FAILED_tables
echo "RELAUNCH_DONE $(date -u +%FT%TZ)"

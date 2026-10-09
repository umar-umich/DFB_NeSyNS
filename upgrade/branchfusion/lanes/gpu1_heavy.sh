#!/bin/bash
# GPU 1 heavy queue, per UMAR's instruction that heavy tasks go on GPU 1.
# Sequential so two ~48 GB jobs never overlap with the LoRA cell already on this GPU.
# setsid-detached: survives SSH disconnection.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== P-dense end-to-end, seed 42, GPU 1 $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/train_p_dense.py --train --seed 42 \
    --epochs 10 --workers 6 || echo FAILED_p_dense

echo "=== refresh tables with whatever has landed $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/tables.py --all || echo FAILED_tables
echo "GPU1_HEAVY_DONE $(date -u +%FT%TZ)"

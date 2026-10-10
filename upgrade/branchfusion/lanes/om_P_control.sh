#!/usr/bin/env bash
# GATE 1's control: branch P with the block-16 tap and NO residual. GPU 2, medium load.
# Identical to the OM run on GPU 3 in every other respect, so the difference between the two is
# the reconstruction and nothing else. Without this number "P+OM scored X" says nothing about OM.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "=== P no-OM b16 control start $(date -Is)"
$PY upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
    --model pecore_l14 --lora --augment B --seed 42 --workers 6 \
    >> logs/branchfusion/lanes/om_P_b16_control.log 2>&1 \
  && echo "=== control done $(date -Is)" || echo "=== control FAILED $(date -Is)"
echo "OM_P_CONTROL_DONE $(date -Is)"

#!/bin/bash
# Section 8 LoRA sweep, shard A on GPU 1. setsid-detached: survives SSH disconnection.
# The centre cell (r8 a16 d0.1) IS B0, already trained, so it is in no shard.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run () { echo "=== r$1 a$2 d$3 $(date -u +%FT%TZ) ==="
  BF_LORA_R=$1 BF_LORA_ALPHA=$2 BF_LORA_DROPOUT=$3 CUDA_VISIBLE_DEVICES=1 \
    $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 \
    || echo "FAILED r$1a$2d$3"; }
run 16 32 0.1
run 4 8 0.1
run 2 4 0.1
run 8 16 0
run 8 16 0.2
echo "LORA_SHARD_A_DONE $(date -u +%FT%TZ)"

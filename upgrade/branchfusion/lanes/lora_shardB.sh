#!/bin/bash
# Section 8 LoRA sweep, shard B on GPU 3. setsid-detached.
# Ordered so the brief's nominated drop cells -- dropout 0.05, then alpha/r 0.5 -- come LAST and
# fall off first if the cap bites.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run () { echo "=== r$1 a$2 d$3 $(date -u +%FT%TZ) ==="
  BF_LORA_R=$1 BF_LORA_ALPHA=$2 BF_LORA_DROPOUT=$3 CUDA_VISIBLE_DEVICES=3 \
    $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 \
    || echo "FAILED r$1a$2d$3"; }
run 8 32 0.1
run 8 8 0.1
run 8 16 0.05
run 8 4 0.1
echo "LORA_SHARD_B_DONE $(date -u +%FT%TZ)"

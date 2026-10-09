#!/bin/bash
# Section 8 LoRA sweep on branch A. GPU priority 1, then 3, then 0 -- never GPU 2.
set -u
# Detached: survives SSH disconnection (launched under setsid + nohup).
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo '=== r8_a16_d0.1 (rank) =='; BF_LORA_R=8 BF_LORA_ALPHA=16 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a16_d0.1
echo '=== r16_a32_d0.1 (rank) =='; BF_LORA_R=16 BF_LORA_ALPHA=32 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r16_a32_d0.1
echo '=== r2_a4_d0.1 (rank) =='; BF_LORA_R=2 BF_LORA_ALPHA=4 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r2_a4_d0.1
echo '=== r4_a8_d0.1 (rank) =='; BF_LORA_R=4 BF_LORA_ALPHA=8 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r4_a8_d0.1
echo '=== r8_a16_d0 (dropout) =='; BF_LORA_R=8 BF_LORA_ALPHA=16 BF_LORA_DROPOUT=0 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a16_d0
echo '=== r8_a16_d0.05 (dropout) =='; BF_LORA_R=8 BF_LORA_ALPHA=16 BF_LORA_DROPOUT=0.05 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a16_d0.05
echo '=== r8_a16_d0.2 (dropout) =='; BF_LORA_R=8 BF_LORA_ALPHA=16 BF_LORA_DROPOUT=0.2 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a16_d0.2
echo '=== r8_a32_d0.1 (scaling) =='; BF_LORA_R=8 BF_LORA_ALPHA=32 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a32_d0.1
echo '=== r8_a4_d0.1 (scaling) =='; BF_LORA_R=8 BF_LORA_ALPHA=4 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a4_d0.1
echo '=== r8_a8_d0.1 (scaling) =='; BF_LORA_R=8 BF_LORA_ALPHA=8 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed 0 || echo FAILED r8_a8_d0.1
echo LORA_SWEEP_DONE $(date -u +%FT%TZ)

# Section 8's seed requirement: three ADAPTATION seeds for the CENTRE recipe (a seed means a new
# adaptation run, not a re-scored head). The selected recipe gets its three seeds after the sweep
# identifies it, which is a separate launch.
for s in 1 2; do
  echo "=== centre seed $s =="
  BF_LORA_R=8 BF_LORA_ALPHA=16 BF_LORA_DROPOUT=0.1 CUDA_VISIBLE_DEVICES=1 \
    $PY upgrade/adapt_backbone.py --backbone pe_core_l --recipe LN --seed $s || echo "FAILED centre s$s"
done
echo "LORA_SWEEP_AND_SEEDS_DONE $(date -u +%FT%TZ)"

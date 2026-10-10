#!/usr/bin/env bash
# Block 8 for the PE spatial in-loop OM, run CONCURRENTLY with block 12 rather than behind it.
# The adaptive lane would only reach b8 if b12 beat b16, which costs ~3h of serial wall clock to
# find out. GPU 1 has the memory for both, so both numbers arrive at once and the "improving?"
# question is answered by the pair instead of by a sequence.
set -u
cd /data/umar/Repos/DFB_NeSyNS
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "=== b8 start $(date -Is)"
/data/umar/miniconda3/envs/discern_ext_timm/bin/python upgrade/adapt_om.py \
    --backbone pe_core_l --recipe LN --operator bvae --seed 0 --readout b8 --select r \
    >> logs/branchfusion/lanes/om_eil_b8.log 2>&1 \
  && echo "=== b8 done $(date -Is)" || echo "=== b8 FAILED $(date -Is)"
echo "OM_DEPTH_B8_DONE $(date -Is)"

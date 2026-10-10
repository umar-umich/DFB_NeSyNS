#!/usr/bin/env bash
# THE NOISE FLOOR. A second seed of the block-16 in-loop OM, on GPU 1.
#
# Every depth claim made so far rests on single-seed differences of 0.005-0.007:
#   b16 0.8673  last 0.8664  b20 0.8613   (spatial, in-loop)
#   last 0.8643  b16 0.8624  b20 0.8550   (S, frozen control)
# and the two orderings already disagree about which of b16/last wins. Without a seed-to-seed
# spread there is no way to say whether ANY of that is signal. If seed 1 moves b16 by more than
# the gap to b20, the depth axis is noise and no further depth run is worth the GPU time --
# which is a more useful answer than another layer.
set -u
cd /data/umar/Repos/DFB_NeSyNS
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "=== b16 seed 1 start $(date -Is)"
/data/umar/miniconda3/envs/discern_ext_timm/bin/python upgrade/adapt_om.py \
    --backbone pe_core_l --recipe LN --operator bvae --seed 1 --readout b16 --select r \
    >> logs/branchfusion/lanes/om_eil_b16_s1.log 2>&1 \
  && echo "=== b16 seed 1 done $(date -Is)" || echo "=== b16 seed 1 FAILED $(date -Is)"
echo "OM_SEED_NOISE_DONE $(date -Is)"

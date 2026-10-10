#!/usr/bin/env bash
# The CHEAP decisive test for the encoder-in-the-loop OM branch, before committing 6 GPU-hours.
#
# Standalone dev says the in-loop runs did not beat the frozen one (b16 0.8673 vs frozen Y 0.8696,
# both under A's 0.8736). But standalone dev was never the criterion: the question is whether the
# branch's ERRORS left the PE cluster. That needs opinions, and opinions on the three DEVELOPMENT
# sets are enough to compute the dev phi against A -- 3 datasets instead of 13.
#
# If phi stays near the frozen arm's +0.789, OM on PE-Core is dead on both constructions and the
# ten test sets are not worth extracting. If it drops toward O's +0.40, the full export is earned.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=logs/branchfusion/lanes/om_eil_devcheck.log
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
: > "$LOG"
for DS in VALmix VALmixV2 FFpp_val; do
  echo "=== $DS $(date -Is)" >> "$LOG"
  $PY upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --readout b16 \
      --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt \
      --cache-tag omEIL_b16 --dataset "$DS" --batch-size 64 >> "$LOG" 2>&1 \
    || { echo "FAILED on $DS" >> "$LOG"; exit 1; }
done
echo "OM_EIL_DEVCHECK_EXTRACT_DONE $(date -Is)" >> "$LOG"

#!/usr/bin/env bash
# Branch S with an off-manifold FRONT END, on GPU 1.
#
# UMAR's architecture: PE-Core over all 16 frames -> an intermediate block -> reconstruction ->
# transformer. The reconstruction sits BEFORE the sequence stage and the transformer trains through
# it, so the residual is shaped by what the temporal stage can use -- unlike arms 1 and 2, which
# fitted a residual on a frozen pipeline and then froze it.
#
# Every cell trains on cached block features in about two minutes, so the matrix is run in full
# rather than sampled: three depths, two operators, and the NO-OM baseline at each depth, because
# "OM helped" is only meaningful against the same depth without it.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=logs/branchfusion/lanes/om_seq.log
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
: > "$LOG"

cell () {  # cell <variant> [--om OP]
  echo "=== $* $(date -Is)" >> "$LOG"
  $PY upgrade/branchfusion/seq_branch.py --variant "$@" --seed 0 >> "$LOG" 2>&1 \
    || echo "FAILED: $*" >> "$LOG"
}

for V in adapt-b16 adapt-b20 adapt; do
  cell "$V"                      # the baseline at this depth
  cell "$V" --om bvae
  cell "$V" --om bvae_real       # one-class manifold: the VAE sees genuine frames only
done
echo "OM_SEQ_DONE $(date -Is)" >> "$LOG"

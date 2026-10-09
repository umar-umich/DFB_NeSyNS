#!/usr/bin/env bash
# Off-manifold with the ENCODER IN THE LOOP, on GPU 1.
#
# WHY. Arms 1 and 2 fitted the beta-VAE on a CACHED feature, so the encoder was never in the graph
# and the residual could only ever be a function of a frozen embedding. The deployed OM view was not
# built that way -- bf_audit records its encoder as "CLIP ViT-L/14 + LoRA(q,v)", trained warm+gate
# to epoch 19 WITH the beta-VAE in the same loop. These runs give PE-Core the same treatment, which
# is the only version of the question that can come back positive.
#
# THREE RUNS, and the pair is the point:
#   b16   block 16, per UMAR: PE-Core's best visual embeddings are not at the output
#   last  the output layer -- the control that isolates whether depth is doing the work, and the
#         apples-to-apples against arm 1's frozen Y (dev 0.8696, test 0.8908, phi +0.79 with A)
#   b20   added once the first two report their memory, so the GPU is not over-committed blind
#
# GPU 1 only. Threads capped at 4 per job.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=logs/branchfusion/lanes
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

run () {  # run <readout> <logname>
  echo "=== $2 start $(date -Is)"
  $PY upgrade/adapt_om.py --backbone pe_core_l --recipe LN --operator bvae \
      --seed 0 --readout "$1" --select r >> "$LOG/$2.log" 2>&1 \
    && echo "=== $2 done $(date -Is)" || echo "=== $2 FAILED $(date -Is)"
}

run b16  om_eil_b16 &
P1=$!
sleep 90          # stagger the two data loads so they do not contend for the same h5 reads
run last om_eil_last &
P2=$!
wait $P1 $P2

# b20 only after the first two have freed their memory, and only if neither failed outright.
if [ -f analysis/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.json ]; then
  run b20 om_eil_b20
fi
echo "OM_ENCODER_IN_LOOP_DONE $(date -Is)"

#!/usr/bin/env bash
# Branch P with the off-manifold residual and the LoRA IN THE LOOP, block 16. GPU 3.
#
# PE-Core (adapters train) -> block 16 -> reconstruction -> TemporalMixer -> head, one graph, one
# loss: EDL + lambda*(alignment + KL). The adapters are therefore shaped by what the beta-VAE
# cannot reconstruct, which is the whole point and the thing every earlier arm failed to do.
#
# The NO-OM control at the same depth does NOT run here -- it runs beside this on GPU 2, as
# `om_P_b16_control`. An earlier version of this comment claimed the control ran in this lane when
# the script below only ever launched the OM arm; the claim is corrected rather than deleted,
# because gate 1 for this branch is "beats its own no-OM control at the same depth" and a reader
# needs to know where that number comes from.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
LOG=logs/branchfusion/lanes
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "=== P+OM b16 start $(date -Is)"
$PY upgrade/branchfusion/train_om_pecore.py --operator bvae --block 16 \
    --model pecore_l14 --lora --augment B --seed 42 --workers 8 \
    >> "$LOG/om_P_b16.log" 2>&1 && echo "=== P+OM b16 done $(date -Is)" || echo "=== P+OM b16 FAILED"
echo "OM_PECORE_DONE $(date -Is)"

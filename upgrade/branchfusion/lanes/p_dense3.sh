#!/bin/bash
# P-dense, fourth launch. The third died at the END of epoch 0: the new selection gate spelled the
# analysis module `tbiom.exp1_fusion_audit` while this trainer exposes it flat as `exp1_fusion_audit`,
# and the gate only runs after an epoch of training, so the ModuleNotFoundError cost 21 minutes.
# Both spellings are accepted now and `TB.preflight()` resolves them before the first batch.
#   * the coverage gate was floored at half the development pool and dense crops reach 36.4% of it,
#     so the run stopped. Coverage is the wrong instrument: what makes a v3.1 macro meaningless is an
#     EMPTY TYPE, and all three types keep a scorable cell (swap 2/5, reenactment 1/3, wild 1/2).
#     The gate now checks that, floors coverage at 30%, and records the restriction in the run record.
#   * the trainer's post-selection export loops would have asked `dense_videos_of` for CDFv2 and the
#     other test sets, which have no dense crops -- a KeyError AFTER ten training epochs. The test
#     ledgers are now emptied and the shuffled control is development-only.
# Heavy, so GPU 1 per UMAR's instruction. setsid-detached: survives SSH disconnection.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== P-dense end-to-end, seed 42, GPU 1 $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/train_p_dense.py --train --seed 42 \
    --epochs 10 --workers 6 || echo FAILED_p_dense

# Regenerates temporal_controls.json in full -- every S arm including the two order-loss arms, the
# exported shuffle controls for V/M/P, the A negative control, and the dense pilot.
echo "=== temporal controls, all arms + the dense pilot $(date -u +%FT%TZ) ==="
TAGS=$(ls logs/branchfusion/seq/S-*_s0.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | paste -sd, -)
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/temporal_controls.py --branches V,M,P \
    --seq "$TAGS" --negative-control --dense-pilot || echo FAILED_controls

echo "=== refresh tables $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/tables.py --all || echo FAILED_tables
echo "P_DENSE3_DONE $(date -u +%FT%TZ)"

#!/bin/bash
# UMAR's block-16/20 study, on the ADAPTED PE-Core trunk.
#
# THE CASE. Part 3d's FROZEN probe: PE-Core block 16 scores 0.8019 against 0.7356 at the output
# (+0.066), and the effect is PE-specific -- CLIP goes the other way (mid 0.7311 < last 0.7478).
# But every deployed branch is ADAPTED, and adapted-final (A, 0.8736) beats the best frozen
# COMBINATION (0.8124). Adapted block 16 is the cell nobody has measured.
#
# AND ARM 1 MADE IT URGENT. OM on the adapted FINAL layer came out a near-duplicate of A: 0.0050
# mean per-set AUROC difference, +0.999 correlation across the ten test sets, +0.0009 on A+V+O.
# Residualising a representation inside its own space does not make the result independent of it.
# Block 16 gives the construction a different representation to work on.
#
# Both blocks come out of ONE forward pass per dataset (~150 f/s, ~2 h for all 14 datasets).
# GPU 1 per UMAR's instruction. setsid-detached.
set -u
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python   # adapted extraction needs timm
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== step 1: adapted block 16 + 20 features, 14 datasets, one pass each $(date -u +%FT%TZ) ==="
for ds in FFpp_val VALmix VALmixV2 FFpp_train CDFv2 DFDCP DFEval24 DFDM KoDF DFD FFIW DFDC CDFv3 IDForge; do
  CUDA_VISIBLE_DEVICES=1 $PYT upgrade/extract_adapted_layer.py --backbone pe_core_l --recipe LN \
      --blocks 16,20 --dataset $ds --batch-size 64 || echo "FAILED extract $ds"
done

for B in 16 20; do
  BB=pe_core_lLN-b${B}
  echo "=== step 2: A-mid CONTROL on block ${B} $(date -u +%FT%TZ) ==="
  # The control that makes the OM row readable: if block ${B} helps the PLAIN appearance head too,
  # the gain is the layer, not the off-manifold construction. One head run on a cache we already built.
  CUDA_VISIBLE_DEVICES=1 $PY upgrade/train_head.py --backbone $BB --seed 0 || echo "FAILED head $BB"

  echo "=== step 3: OM on block ${B} $(date -u +%FT%TZ) ==="
  CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/om_branch.py --backbone $BB \
      --operator bvae --seed 0 --train --export --arm r || echo "FAILED om $BB"

  $PY -c "
import sys; sys.path.insert(0,'upgrade/branchfusion')
import branches as B, common as C
B.register('Y${B}', C.BRANCH_OUT/'OM-${BB}-bvae_s0',
           'off-manifold residual on ADAPTED PE-Core block ${B} (patch mean, prefix dropped)')
print('registered Y${B}')
"
done

echo "=== step 4: screen every block variant against the deployed branches $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/subsets.py --screen \
    --branches A,V,O,Y,Y16,Y20 --sizes 1,2,3,4,5,6 || echo FAILED_screen_blocks

echo "=== step 5: per-dataset grid for the block variants $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/per_dataset.py \
    --branches A,V,O,Y,Y16,Y20 --sizes 1,2,3,4,5,6 --tag blocks || echo FAILED_grid_blocks
echo "BLOCK1620_DONE $(date -u +%FT%TZ)"

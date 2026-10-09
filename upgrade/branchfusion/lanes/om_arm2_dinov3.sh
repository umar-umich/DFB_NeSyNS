#!/bin/bash
# ARM 2: OM on DINOv3 -- the MECHANISM test, and the reason arms 1 and 2 are run as a pair.
#
# OM's story is that the beta-VAE captures what a caption would name and the orthogonal residual
# keeps what text cannot describe. DINOv3 is the only text-free trunk available: self-distillation,
# no captions. If OM works here too, the image-text story is wrong and OM is a generic residual
# detector. If it works on CLIP and fails here, the contrastive image-text objective IS the mechanism
# -- which is a real claim and directly supports section 2 of PE_CORE_VS_CLIP_AND_THE_OM_TRUNK.md.
#
# DINOv3 is cached for the three development sets only, so the ten test sets are extracted first
# (~45 min total: 905k frames at 256px/16, roughly 0.45x PE-Core's 336px/14 cost).
# GPU 1 per UMAR's instruction. setsid-detached.
set -u
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python   # extraction needs timm
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== ARM 2 step 1: DINOv3 features for the ten test sets $(date -u +%FT%TZ) ==="
for ds in CDFv2 DFDCP DFEval24 DFDM KoDF DFD FFIW DFDC CDFv3 IDForge; do
  CUDA_VISIBLE_DEVICES=1 $PYT upgrade/extract_spatial.py --backbone dinov3 --dataset $ds \
      --batch-size 64 || echo "FAILED extract dinov3 $ds"
done

echo "=== ARM 2 step 2: train and export $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 $PY upgrade/branchfusion/om_branch.py --backbone dinov3 \
    --operator bvae --seed 0 --train --export --arm r || { echo FAILED_arm2; exit 1; }

echo "=== ARM 2 step 3: register and screen $(date -u +%FT%TZ) ==="
$PY -c "
import sys; sys.path.insert(0,'upgrade/branchfusion')
import branches as B, common as C
B.register('Z', C.BRANCH_OUT/'OM-dinov3-bvae_s0',
           'ARM 2: off-manifold residual on DINOv3, the text-free trunk (mechanism test)')
print('registered Z')
print(B.coverage().to_string())
"
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/subsets.py --screen \
    --branches A,V,O,Z --sizes 1,2,3,4 || echo FAILED_arm2_screen
echo "ARM2_DONE $(date -u +%FT%TZ)"

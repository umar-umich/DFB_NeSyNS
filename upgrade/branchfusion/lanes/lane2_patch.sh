#!/bin/bash
# Lane 2: R's patch-covariance extraction and the block-16 mean pool S-frozen needs.
# GPU priority is GPU 1, then GPU 3, then GPU 0 -- never GPU 2 (kutub).
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU="${1:-1}"
for ds in FFpp_train FFpp_val VALmix VALmixV2 CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
  echo "=== $ds $(date -u +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/extract_patch.py \
      --dataset "$ds" --depth mid --batch-size 32 || echo "FAILED $ds"
done
echo "LANE2 DONE $(date -u +%FT%TZ)"

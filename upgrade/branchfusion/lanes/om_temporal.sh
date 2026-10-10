#!/usr/bin/env bash
# Off-manifold on the temporal trunks, encoder in the loop. GPU 3.
#
# ORDER OF RUNS IS THE ARGUMENT, not convenience:
#   1. V with `order --null-perm` -- the NEGATIVE CONTROL, and it runs FIRST. The identity
#      permutation makes the "shuffled" clip the ordered clip, so the residual is zero by
#      construction and the branch MUST land at chance. Anything above chance means evidence is
#      reaching the residual arm by a path other than frame order, which would make run 2
#      uninterpretable. It gates the rest.
#   2. V with `order`  -- the real question: does what frame ORDER adds stand as its own branch?
#   3. V with `bvae`   -- the CLIP recipe transplanted, for comparison with the image-side arms.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
LOG=logs/branchfusion/lanes
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

run () {  # run <model> <operator> <logname> [extra flags...]
  local m=$1 op=$2 name=$3; shift 3
  echo "=== $name start $(date -Is)"
  $PY upgrade/branchfusion/train_om_temporal.py --operator "$op" "$@" \
      --model "$m" --lora --augment B --seed 42 --workers 8 >> "$LOG/$name.log" 2>&1 \
    && echo "=== $name done $(date -Is)" || echo "=== $name FAILED $(date -Is)"
}

run vjepa2 order om_temp_V_nullperm --null-perm
run vjepa2 order om_temp_V_order
run vjepa2 bvae  om_temp_V_bvae
echo "OM_TEMPORAL_DONE $(date -Is)"

#!/usr/bin/env bash
# Stage 2 — measure how ordinary processing changes the measurements.
#
# FIT severities run on DEV (FF++ val + VALmix), which carries both classes, so the reliability
# curves are fit under matched processing and the model cannot learn "compressed means fake".
# HELD-OUT severities are deliberately different values, on two small OOD sets, and exist for
# Stage 4's processing-discrimination test. Fitting and testing never share a severity.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage2
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="$EXT/runs/discern_ext_train/run1_auxedl_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt"
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$OUT"; cd "$EXT" || exit 2

run() {  # <dataset> <spec> <tag>
  local ds=$1 spec=$2 tag=$3
  local out="$OUT/p_${ds}_${tag}.csv"
  [ -f "$out" ] && { echo "skip $ds/$tag"; return 0; }
  local arg=""; [ "$spec" != "none" ] && arg="--process $spec"
  echo "[$(date +%H:%M)] $ds / $tag"
  $PY eval_adaptation/export_views.py --checkpoint "$CK" \
      --config "eval_adaptation/configs/run1auxedl_${ds}.yaml" --dataset "$ds" \
      --device cuda:0 --batch-size 64 $arg --out "$out" \
      >> "$OUT/${ds}_${tag}.log" 2>&1 || echo "  !! FAILED $ds/$tag"
}

# fit grid, dev only, both classes, matched processing
for ds in FFpp_val VALmix; do
  run "$ds" none          base
  for q in 70 50 30;   do run "$ds" "jpeg:$q"    "jpeg$q";   done
  for s in 0.75 0.5 0.35; do run "$ds" "resize:$s" "rs${s/./}"; done
done
# held-out severities, never used for fitting
for ds in CDFv2 DFDCP; do
  run "$ds" none        base
  run "$ds" "jpeg:40"   jpeg40
  run "$ds" "resize:0.6" rs06
done
echo "[$(date +%H:%M)] Stage 2 export done"

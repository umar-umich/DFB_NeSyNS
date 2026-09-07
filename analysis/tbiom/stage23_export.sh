#!/usr/bin/env bash
# Stage 2/3 export driver — two phases, so no final OOD test set is ever touched before the
# checkpoint has been chosen.
#
#   phase select : every saved checkpoint -> FF++ val + VALmix        (selection data only)
#   phase test   : the CHOSEN checkpoint  -> the six OOD test sets
#
# GPU 1 is ours. CUDA_VISIBLE_DEVICES pins it at the process level so a stray allocation cannot
# reach another card; the model is then addressed as cuda:0, which CUDA remaps to physical GPU 1.
# `--device` alone has been wrong before -- Lightning restores checkpoint tensors to the device
# they were SAVED on, leaving a context behind on GPU 0.
set -uo pipefail

EXT=/data/umar/Repos/DISCERN_Ext
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage23
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
case "${1:-}" in run1*|seed1337) PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python ;; esac
export CUDA_VISIBLE_DEVICES=1

arm=${1:?usage: stage23_export.sh <stage2|stage3|stage5> <select|test> [ckpt] [tag]}
phase=${2:?}
ckpt=${3:-}
# Optional output tag. Defaults to the arm, which is what the Stage-4 primary uses. A second
# checkpoint of the SAME arm (e.g. the e10 sensitivity check) needs a distinct tag or it would
# overwrite the primary's exports -- same architecture, same configs, different weights.
tag=${4:-$arm}

case "$arm" in
  stage2) RUN=stage2_mrvae_projonly_seed42 ;;
  stage3) RUN=stage3_bvae_rate_seed42 ;;
  stage5) RUN=stage5_three_branch_seed42 ;;
  run1fusedonly) RUN=run1_fusedonly_seed42 ;;
  run1auxedl)    RUN=run1_auxedl_seed42 ;;
  run1cft)       RUN=run1c_ft_seed42 ;;
  run1dce)       RUN=run1d_simplece_seed42 ;;
  seed1337)      RUN=run1_auxedl_seed1337 ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac

mkdir -p "$OUT"
cd "$EXT" || exit 2

run_one() {  # <tag> <dataset> <checkpoint>
  local tag=$1 ds=$2 ck=$3
  local out="$OUT/${tag}_${ds}.csv"
  [ -f "$out" ] && { echo "  skip $tag/$ds (exists)"; return 0; }
  echo "  == $tag / $ds =="
  $PY eval_adaptation/export_views.py \
      --checkpoint "$ck" \
      --config "eval_adaptation/configs/${arm}_${ds}.yaml" \
      --dataset "$ds" --device cuda:0 --batch-size 64 \
      --out "$out" >> "$OUT/${tag}_${ds}.log" 2>&1 \
    || echo "  !! FAILED $tag/$ds (see $OUT/${tag}_${ds}.log)"
}

if [ "$phase" = "select" ]; then
  # Every checkpoint on disk, on the two permitted development sources.
  for ck in "runs/discern_ext_train/$RUN/checkpoints/"*.ckpt; do
    [ -e "$ck" ] || { echo "no checkpoints under $RUN"; exit 1; }
    base=$(basename "$ck" .ckpt)
    # discern_ext-best-epoch=10-val_auroc_video=0.9964 -> e10 ; last -> last
    if [[ $base =~ epoch=([0-9]+) ]]; then tag="${arm}_e${BASH_REMATCH[1]}"; else tag="${arm}_last"; fi
    for ds in FFpp_val VALmix; do run_one "$tag" "$ds" "$ck"; done
  done
elif [ "$phase" = "test" ]; then
  [ -n "$ckpt" ] || { echo "phase test needs a checkpoint path"; exit 2; }
  for ds in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24; do run_one "$tag" "$ds" "$ckpt"; done
  # re-export the development sources under the same tag so the dashboard has all seven
  for ds in FFpp_val VALmix; do run_one "$tag" "$ds" "$ckpt"; done
else
  echo "unknown phase $phase"; exit 2
fi
echo "done: $arm / $phase"

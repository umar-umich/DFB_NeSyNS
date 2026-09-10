#!/usr/bin/env bash
# Export the frozen arm's per-branch opinions over the four DF40 families.
#
# GPU 1 is ours. CUDA_VISIBLE_DEVICES pins it per process; the model is then addressed as
# cuda:0, which CUDA remaps to physical GPU 1. `--device` alone has been wrong before --
# Lightning restores checkpoint tensors to the device they were SAVED on.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage23
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CKPT="$EXT/runs/discern_ext_train/run1_auxedl_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt"
ARM=${1:-run1auxedl}
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$OUT"; cd "$EXT" || exit 2
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 2; }

for fam in DF40_FS DF40_FR DF40_EFS DF40_FE; do
  out="$OUT/${ARM}_${fam}.csv"
  [ -f "$out" ] && { echo "skip $fam (exists)"; continue; }
  echo "[$(date +%H:%M)] == $fam =="
  $PY eval_adaptation/export_views.py \
      --checkpoint "$CKPT" \
      --config "eval_adaptation/configs/${ARM}_${fam}.yaml" \
      --dataset "$fam" --device cuda:0 --batch-size 64 \
      --out "$out" >> "$OUT/${ARM}_${fam}.log" 2>&1 \
    || echo "  !! FAILED $fam (see $OUT/${ARM}_${fam}.log)"
done
echo "[$(date +%H:%M)] DF40 export done"

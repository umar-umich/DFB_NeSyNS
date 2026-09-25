#!/usr/bin/env bash
# Second extraction pass: DiCoME's RELEASED weights on the three panels UMAR asked for, plus the DF40 panel for the
# two locally trained models. Waits for the first pass so the two never share GPU 1.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
OUT=$ROOT/logs/tsne; PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
OURS="$EXT/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
DIC="$EXT/runs/_baseline_dicome/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=05-val_auroc_video=0.9953.ckpt"
REL="$EXT/weights/dicome-best.ckpt"
cd "$EXT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while [ ! -f "$OUT/SPATIAL_DONE" ]; do sleep 60; done
say "first pass done; starting released-weights and DF40 extractions"
run() {  # tag dataset checkpoint configfam
  local TAG=$1 DS=$2 CK=$3 FAM=$4 F="$OUT/$1_$2.npz"
  [ -s "$F" ] && { say "skip $TAG $DS"; return; }
  [ -f "eval_adaptation/configs/${FAM}_${DS}.yaml" ] || { say "MISSING CONFIG ${FAM}_${DS}.yaml"; return; }
  say "$TAG $DS"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY eval_adaptation/export_views.py \
    --checkpoint "$CK" --config "eval_adaptation/configs/${FAM}_${DS}.yaml" --dataset "$DS" \
    --device cuda:0 --batch-size 64 --features "$F" --out "$OUT/${TAG}_${DS}_scores.csv" \
    > "$OUT/${TAG}_${DS}.log" 2>&1 && say "  ok $(du -h "$F" | cut -f1)" || say "  FAILED $TAG $DS"
}
for DS in CDFv3 KoDF DF40tsne; do run dicome_released "$DS" "$REL" dicome; done
run ours DF40tsne "$OURS" warmgate
run dicome DF40tsne "$DIC" dicome
touch "$OUT/PASS2_DONE"; say "second pass finished"

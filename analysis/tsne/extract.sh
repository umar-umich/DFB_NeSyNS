#!/usr/bin/env bash
# t-SNE feature extraction: per-view penultimate blocks for ours and for our trained DiCoME.
# Uses the EXISTING --features path of eval_adaptation/export_views.py; no model code is modified.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
OUT=$ROOT/logs/tsne; mkdir -p "$OUT"
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
OURS="$EXT/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
DIC="$EXT/runs/_baseline_dicome/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=05-val_auroc_video=0.9953.ckpt"
cd "$EXT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
for DS in FFpp_val CDFv2 CDFv3 KoDF DFDC; do
  for TAG in ours dicome; do
    case $TAG in ours) CK=$OURS; CFG=warmgate_${DS}.yaml ;; dicome) CK=$DIC; CFG=dicome_${DS}.yaml ;; esac
    F="$OUT/${TAG}_${DS}.npz"
    [ -s "$F" ] && { say "skip $TAG $DS (exists)"; continue; }
    [ -f "eval_adaptation/configs/$CFG" ] || { say "MISSING CONFIG $CFG -- skipped"; continue; }
    say "$TAG $DS"
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY eval_adaptation/export_views.py \
      --checkpoint "$CK" --config "eval_adaptation/configs/$CFG" --dataset "$DS" \
      --device cuda:0 --batch-size 64 --features "$F" \
      --out "$OUT/${TAG}_${DS}_scores.csv" > "$OUT/${TAG}_${DS}.log" 2>&1 \
      && say "  ok $(du -h "$F" | cut -f1)" || say "  FAILED $TAG $DS (see ${TAG}_${DS}.log)"
  done
done
touch "$OUT/SPATIAL_DONE"; say "spatial feature extraction finished"

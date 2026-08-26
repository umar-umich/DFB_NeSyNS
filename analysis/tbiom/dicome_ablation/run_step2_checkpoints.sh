#!/usr/bin/env bash
# Step 2 — re-select the P0-DS checkpoint on the health dashboard, NOT on val AUROC.
#
# P0-DS was originally picked at epoch 1 on the highest `val_auroc_video` (0.9960). That metric
# is saturated: every candidate epoch scores above 0.995 and the ranking among them is noise. It
# also says nothing about real-side health, and Step 1 measured this checkpoint calling 31.7% of
# OOD reals fake — worse than our own weaker CLIP port at 17.9%.
#
# SELECTION DATA ONLY. FFpp_val fixes the operating threshold; VALmix supplies the selection
# metrics. No final OOD test set is touched here — selecting an epoch on CDFv2/DFDC/DFD/DFDCP
# would be selecting on test, and those sets have to stay clean for the eventual comparison.
#
# The released checkpoint is scored on the same two sets as a reference point. It is NOT a
# candidate: it is not retrainable, which is the property Step 2 exists to preserve.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
OUT=${OUT:-/data/umar/Repos/DFB_NeSyNS/logs/tbiom/step2}
DEVICE=${DEVICE:-cuda:3}
BATCH=${BATCH:-64}
CKDIR=runs/dicome_train/ffpp_reproduce_seed42/checkpoints

mkdir -p "$OUT"
declare -A ARMS=(
  [p0ds_e01]="$CKDIR/dicome-best-epoch=01-val_auroc_video=0.9960.ckpt"
  [p0ds_e02]="$CKDIR/dicome-best-epoch=02-val_auroc_video=0.9950.ckpt"
  [p0ds_e05]="$CKDIR/dicome-best-epoch=05-val_auroc_video=0.9953.ckpt"
  [p0ds_last]="$CKDIR/last.ckpt"
  [released]="weights/dicome-best.ckpt"
)

for arm in p0ds_e01 p0ds_e02 p0ds_e05 p0ds_last released; do
  ck=${ARMS[$arm]}
  [ -f "$ck" ] || { echo "  missing $ck — skipping $arm"; continue; }
  for ds in FFpp_val VALmix; do
    dest="$OUT/${arm}_${ds}.csv"
    [ -f "$dest" ] && { echo "  skip ${arm}_${ds}"; continue; }
    echo "== $arm / $ds"
    $PY eval_adaptation/export_views.py --checkpoint "$ck" \
        --config "eval_adaptation/configs/$ds.yaml" --dataset "$ds" \
        --out "$dest" --device "$DEVICE" --batch-size "$BATCH" \
        > "$OUT/${arm}_${ds}.log" 2>&1 || echo "  FAILED — see $OUT/${arm}_${ds}.log"
  done
done
echo "done: $(ls "$OUT"/*.csv 2>/dev/null | wc -l) exports"

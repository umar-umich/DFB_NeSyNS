#!/usr/bin/env bash
# Step 1 of the recovery brief — export per-view evidence for both DiCoME anchors.
#
# Two checkpoints, one exporter, the same six datasets and the same configs, so the ONLY
# difference between the two runs is the checkpoint:
#   released  weights/dicome-best.ckpt                    (theirs, HF kxl0825/DiCoME)
#   P0-DS     runs/dicome_train/ffpp_reproduce_seed42/…   (ours, FF++ c23 seed 42, retrainable)
#
# Checkpoint selection for P0-DS follows the released recipe — highest val_auroc_video, not the
# last epoch — which is how run_eval_trained.sh auto-picks and how their epoch-4 release was
# chosen.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
OUT=${OUT:-/data/umar/Repos/DFB_NeSyNS/logs/tbiom/step1}
DEVICE=${DEVICE:-cuda:3}
BATCH=${BATCH:-64}

RELEASED=weights/dicome-best.ckpt
P0DS=$(ls -1 runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=*.ckpt \
       | sed 's/.*val_auroc_video=//; s/\.ckpt//' | paste -d' ' - <(ls -1 runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=*.ckpt) \
       | sort -rn | head -1 | awk '{print $2}')
echo "released : $RELEASED"
echo "P0-DS    : $P0DS"

mkdir -p "$OUT"
for ds in FFpp CDFv2 DFD DFDC DFDCP DFEval24; do
  cfg=eval_adaptation/configs/$ds.yaml
  [ -f "$cfg" ] || { echo "  no config for $ds — skipping"; continue; }
  for arm in released p0ds; do
    case $arm in released) ck=$RELEASED;; p0ds) ck=$P0DS;; esac
    dest="$OUT/${arm}_${ds}.csv"
    [ -f "$dest" ] && { echo "  skip ${arm}_${ds}"; continue; }
    echo "== $arm / $ds"
    $PY eval_adaptation/export_views.py --checkpoint "$ck" --config "$cfg" --dataset "$ds" \
        --out "$dest" --device "$DEVICE" --batch-size "$BATCH" \
        > "$OUT/${arm}_${ds}.log" 2>&1 || echo "  FAILED — see $OUT/${arm}_${ds}.log"
  done
done
echo "done"; ls -la "$OUT"/*.csv 2>/dev/null | wc -l

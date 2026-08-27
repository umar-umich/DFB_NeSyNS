#!/usr/bin/env bash
# Batch-size arm of the Step-1 recipe investigation — score and compare.
#
# Exports the batch-32 checkpoint on the same development sets P0-DS was scored on, then puts its
# SEMANTIC readout head to head with P0-DS's. Semantic is the readout that matters here: Step 1
# located the gap in the CLIP branch, so that is where a recipe cause has to show up.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FORK=/data/umar/Repos/DISCERN_Ext
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
OUT=logs/tbiom/step2c
R=$FORK/runs/discern_ext_train/ffpp_batch32_seed42/checkpoints
mkdir -p "$OUT"

# selection follows the released recipe: highest val_auroc_video, not the last epoch
BEST=$(ls -1 $R/*epoch=*.ckpt 2>/dev/null | sed 's/.*val_auroc_video=//; s/\.ckpt//' \
       | paste -d' ' - <(ls -1 $R/*epoch=*.ckpt 2>/dev/null) | sort -rn | head -1 | awk '{print $2}')
[ -n "$BEST" ] || { echo "no batch-32 checkpoint yet"; exit 1; }
echo "batch-32 checkpoint: $BEST"

cd "$FORK"
for ds in FFpp_val VALmix CDFv2; do
  dest="/data/umar/Repos/DFB_NeSyNS/$OUT/batch32_${ds}.csv"
  [ -f "$dest" ] && { echo "  skip $ds"; continue; }
  CUDA_VISIBLE_DEVICES=1 $PY eval_adaptation/export_views.py --checkpoint "$BEST" \
      --config "eval_adaptation/configs/$ds.yaml" --dataset "$ds" \
      --out "$dest" --device cuda:0 --batch-size 64 \
      > "/data/umar/Repos/DFB_NeSyNS/$OUT/${ds}.log" 2>&1 && echo "  $ds ok" || echo "  $ds FAILED"
done
echo done

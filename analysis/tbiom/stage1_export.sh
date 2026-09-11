#!/usr/bin/env bash
# Stage 1 re-export: the frozen arm's per-sample MEASUREMENT record.
#
# Adds availability (read BEFORE it is multiplied into the evidence), the raw measurements the
# heads consume, and leaves the existing p_*/u_* columns untouched so every current report keeps
# working. Frozen weights, no retraining.
#
# GPU 1 is ours. CUDA_VISIBLE_DEVICES pins it per process; the model is addressed as cuda:0,
# which CUDA remaps to physical GPU 1.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage1
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="$EXT/runs/discern_ext_train/run1_auxedl_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt"
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$OUT"; cd "$EXT" || exit 2

for ds in FFpp_val VALmix CDFv2 CDFv3 DFD DFDC DFDCP DFEval24; do
  out="$OUT/m_${ds}.csv"
  [ -f "$out" ] && { echo "skip $ds (exists)"; continue; }
  echo "[$(date +%H:%M)] == $ds =="
  $PY eval_adaptation/export_views.py --checkpoint "$CK" \
      --config "eval_adaptation/configs/run1auxedl_${ds}.yaml" \
      --dataset "$ds" --device cuda:0 --batch-size 64 --out "$out" \
      >> "$OUT/${ds}.log" 2>&1 || echo "  !! FAILED $ds (see $OUT/${ds}.log)"
done
echo "[$(date +%H:%M)] Stage 1 export done"

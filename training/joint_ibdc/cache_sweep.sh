#!/usr/bin/env bash
# Task 2 spatial feature cache: the per-frame blocks the Appearance and Off-manifold heads consume.
#
#   bash training/joint_ibdc/cache_sweep.sh <cuda index>
#
# Uses the EXISTING --features path of eval_adaptation/export_views.py; no model code is touched.
# Six datasets are NOT re-extracted because the t-SNE sweep already exported them from the SAME
# deployed checkpoint (warm_dicome_gate_seed42 epoch 19): FFpp_val, CDFv2, CDFv3, DFDC, KoDF and
# DF40tsne, in logs/tsne/ours_*.npz. verify_cache.py confirmed ours_FFpp_val reproduces the deployed
# per-view opinions to 1.4e-08, which is what licenses reusing them.
#
# FF++ c23 TRAIN is deliberately absent: it needs three independent arm-B draws per sample, which
# this exporter cannot do, and the brief puts that new code in training/joint_ibdc/ instead.
set -uo pipefail
DEV=${1:-3}
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
OUT=$ROOT/logs/joint_ibdc/cache; LOG=$ROOT/logs/joint_ibdc
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="$EXT/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
mkdir -p "$OUT"
cd "$EXT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

run() {  # <dataset> <tag> [process spec]
  local DS=$1 TAG=$2 PROC=${3:-}
  local F="$OUT/${TAG}.npz"
  [ -s "$F" ] && { say "skip $TAG (exists)"; return; }
  local CFG="eval_adaptation/configs/warmgate_${DS}.yaml"
  [ -f "$CFG" ] || { say "MISSING CONFIG warmgate_${DS}.yaml -- skipped"; return; }
  say "$TAG"
  # batch 64 keeps this near 8 GB, a small footprint on a card someone else is sharing
  CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY eval_adaptation/export_views.py \
      --checkpoint "$CK" --config "$CFG" --dataset "$DS" --device cuda \
      --batch-size 64 --features "$F" --out "$OUT/${TAG}_scores.csv" \
      ${PROC:+--process "$PROC"} >> "$LOG/cache_${TAG}.log" 2>&1 \
    && say "$TAG done" || say "$TAG FAILED (see cache_${TAG}.log)"
}

# development pool first: G0b and every run need it before anything else
for DS in VALmix VALmixV2; do run "$DS" "$DS"; done
# the test sets the t-SNE sweep did not already cover
for DS in DFD DFDCP DFDM DFEval24 FaceShifter FFIW IDForge; do run "$DS" "$DS"; done
# the four processed cells, at the evaluation severities
run CDFv2 CDFv2_jpeg40 "jpeg:40"; run CDFv2 CDFv2_rs06 "resize:0.60"
run DFDCP DFDCP_jpeg40 "jpeg:40"; run DFDCP DFDCP_rs06 "resize:0.60"
say "spatial cache sweep finished"

#!/usr/bin/env bash
# Phase T driver -- train arm B + coherence term at two lambdas, then export, then score.
#
# Chained deliberately: the last time a training run finished without its export queued behind
# it, GPU 1 sat idle overnight. Both lambdas run concurrently on GPU 1 (143 GB, ~50 GB each).
#
# GPU 1 is ours. CUDA_VISIBLE_DEVICES pins it per process; devices:[0] in the config then means
# physical GPU 1. Attribution is verified after launch, not assumed.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/phaseT
mkdir -p "$LOG"

one() {  # <tag> <config> <runname>
  local tag=$1 cfg=$2 run=$3
  cd "$EXT" || exit 2
  echo "[$(date +%H:%M)] train $tag"
  CUDA_VISIBLE_DEVICES=1 $PY tools/train/train_discern_ext.py fit \
      --config_path="eval_adaptation/configs/${cfg}.yaml" > "$LOG/${tag}_train.log" 2>&1
  local ck
  ck=$(ls -t "runs/discern_ext_train/${run}/checkpoints/"*best*.ckpt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "[$(date +%H:%M)] $tag: NO CHECKPOINT -- see $LOG/${tag}_train.log"; return 1; fi
  echo "[$(date +%H:%M)] export $tag from $(basename "$ck")"
  bash /data/umar/Repos/DFB_NeSyNS/analysis/tbiom/stage23_export.sh "$tag" test "$EXT/$ck" \
      > "$LOG/${tag}_export.log" 2>&1
  echo "[$(date +%H:%M)] done $tag"
}

one phaseTc01 phaseT_cons01 phaseT_cons01_seed42 &
one phaseTc05 phaseT_cons05 phaseT_cons05_seed42 &
wait
echo "[$(date +%H:%M)] Phase T complete"

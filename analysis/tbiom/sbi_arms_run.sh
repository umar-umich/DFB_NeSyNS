#!/usr/bin/env bash
# Arms A/B/C, matched budget, then export. Two at a time -- three parallel jobs put this box
# into memory pressure earlier today.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
LOG=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/sbi
mkdir -p "$LOG"; cd "$EXT" || exit 2
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1

one() {
  local A=$1
  echo "[$(date +%H:%M)] train arm $A"
  $PY tools/train/train_discern_ext.py fit \
      --config_path="eval_adaptation/configs/sbi_arm${A}.yaml" > "$LOG/arm${A}_train.log" 2>&1
  local ck
  ck=$(ls -t "runs/discern_ext_train/sbi_arm${A}_seed42/checkpoints/"*best*.ckpt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "[$(date +%H:%M)] arm $A: NO CHECKPOINT"; return 1; fi
  echo "[$(date +%H:%M)] done arm $A -> $(basename "$ck")"
}

one A & one B & wait
one C
echo "[$(date +%H:%M)] arms A/B/C complete"

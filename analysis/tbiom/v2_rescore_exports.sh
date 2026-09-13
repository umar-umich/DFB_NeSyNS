#!/usr/bin/env bash
# After rule-v2 selection: export each run's SELECTED checkpoint on the six OOD sets (+ FF++ val and
# VALmix) and on the held-out processing conditions. An existing export is reused only when it is
# known to come from the same checkpoint file; otherwise a new tag `<arm>_v2e<NN>` is written.
# Inference only. Run after select_checkpoint.py has produced <arm>_selection_v2.json for all arms.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS
EXT=/data/umar/Repos/DISCERN_Ext
S23=$ROOT/logs/tbiom/stage23; S2=$ROOT/logs/tbiom/stage2
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1
cd "$ROOT" || exit 2

# checkpoints already exported, verified from driver logs and the Phase F record
declare -A OOD_TAG=( [run1auxedl:5]=run1auxedl [sbiarmA:5]=sbiarmA_e05 [sbiarmA:14]=sbiarmA [sbiarmB:11]=sbiarmB [sbiarmC:4]=sbiarmC )
declare -A PROC_PREFIX=( [run1auxedl:5]="" [sbiarmA:14]="armA_" [sbiarmB:11]="armB_" [sbiarmC:4]="armC_" )

for arm in run1auxedl sbiarmA sbiarmB sbiarmC; do
  J=$S23/${arm}_selection_v2.json
  [ -f "$J" ] || { echo "no selection for $arm — run select_checkpoint.py --arm $arm"; continue; }
  read -r EPOCH CKPT < <(/data/umar/miniconda3/envs/discern_ext/bin/python -c "import json;s=json.load(open('$J'))['selected'];print(s['epoch'],s['checkpoint'])")
  key="$arm:$EPOCH"
  tag=${OOD_TAG[$key]:-${arm}_v2e$(printf %02d "$EPOCH")}
  echo "[$(date +%H:%M)] $arm: epoch $EPOCH -> OOD tag $tag"
  echo "$tag" > "$S23/${arm}_v2_tag.txt"
  bash analysis/tbiom/stage23_export.sh "$arm" test "$CKPT" "$tag" >> "$ROOT/logs/tbiom/sbi/v2_export_${arm}.log" 2>&1

  if [ -n "${PROC_PREFIX[$key]+set}" ]; then
    echo "${PROC_PREFIX[$key]}" > "$S23/${arm}_v2_proc_prefix.txt"; continue
  fi
  pre="v2_${arm}_"; echo "$pre" > "$S23/${arm}_v2_proc_prefix.txt"
  for ds in CDFv2 DFDCP; do
    for spec in "none:base" "jpeg:40:jpeg40" "resize:0.6:rs06"; do
      tagc=${spec##*:}; proc=${spec%:*}
      out="$S2/p_${pre}${ds}_${tagc}.csv"; [ -f "$out" ] && continue
      arg=""; [ "$proc" != "none" ] && arg="--process $proc"
      (cd "$EXT" && $PY eval_adaptation/export_views.py --checkpoint "$CKPT" \
         --config "eval_adaptation/configs/${arm}_${ds}.yaml" --dataset "$ds" \
         --device cuda:0 --batch-size 64 $arg --out "$out") >> "$S2/${pre}${ds}_${tagc}.log" 2>&1 \
         || echo "  !! FAILED $pre$ds/$tagc"
    done
  done
done
echo "[$(date +%H:%M)] v2 exports complete"

#!/usr/bin/env bash
# One spatial arm, end to end on GPU 1: train -> export every candidate on the development sources -> unchanged v3.1
# selection -> export the selected checkpoint on the ten test sets and the processed conditions.
#   bash analysis/tbiom/arm_pipeline.sh <tag> <run_name> <config yaml> [export-config prefix, default warmgate]
# The export prefix must name configs with the arm's architecture: warmgate_* for the warm+gate family, sbiarmB_* for
# the sbi_arm family.
# Tags: dev candidates <tag>_eNN / <tag>_last, tests <tag>_sel_<ds>, processed p_dev_<selected>_<ds>_<c>, p_v3_<tag>_<ds>_<c>.
set -uo pipefail
TAG=$1; RUNNAME=$2; CFG=$3; XP=${4:-warmgate}
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python; PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
LOG=$ROOT/logs/tbiom/stage3; S23=$ROOT/logs/tbiom/stage23; S2=$ROOT/logs/tbiom/stage2; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
RUN=$EXT/runs/discern_ext_train/$RUNNAME
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$LOG"; cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $TAG: $*"; }
lanes() { for L in 1 2 3 4; do BATCH=96 bash "$LANE" 1 "$1" >> "$LOG/${TAG}_lane$L.log" 2>&1 & done; wait; }

if ! grep -aq "Trainer.fit\` stopped" "$LOG/${TAG}_train.log" 2>/dev/null; then
  say "training ($CFG)"
  (cd "$EXT" && $PY tools/train/train_discern_ext.py fit --config_path="eval_adaptation/configs/$CFG" --num_workers=8) \
      > "$LOG/${TAG}_train.log" 2>&1
fi
grep -aq "Trainer.fit\` stopped" "$LOG/${TAG}_train.log" || { say "training FAILED"; exit 1; }
say "trained: $(ls "$RUN/checkpoints" | tr '\n' ' ')"

J=$LOG/${TAG}_jobs_dev.txt; : > "$J"
for ck in "$RUN/checkpoints/"*.ckpt; do
  b=$(basename "$ck" .ckpt)
  if [[ $b =~ epoch=([0-9]+) ]]; then t="${TAG}_e${BASH_REMATCH[1]}"; else t="${TAG}_last"; fi
  for ds in FFpp_val VALmix VALmixV2; do echo "$ck|${XP}_${ds}.yaml|${ds}|$S23/${t}_${ds}.csv|" >> "$J"; done
done
say "development exports ($(wc -l < "$J") jobs)"; lanes "$J"
$PYA analysis/tbiom/select_checkpoint_v3.py --arm "$TAG" > "$LOG/${TAG}_selection.log" 2>&1 || { say "selection FAILED"; exit 1; }
read -r SEL CK < <($PYA -c "import json;s=json.load(open('$S23/${TAG}_selection_v3.json'))['selected'];print(s['tag'],s['checkpoint'])")
say "selected $SEL"

J=$LOG/${TAG}_jobs_test.txt; : > "$J"
for ds in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do echo "$CK|${XP}_${ds}.yaml|${ds}|$S23/${TAG}_sel_${ds}.csv|" >> "$J"; done
for ds in VALmix VALmixV2 FFpp_val; do
  echo "$CK|${XP}_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|${XP}_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_rs06.csv|resize:0.6" >> "$J"
done
for ds in CDFv2 DFDCP; do
  echo "$CK|${XP}_${ds}.yaml|${ds}|$S2/p_v3_${TAG}_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|${XP}_${ds}.yaml|${ds}|$S2/p_v3_${TAG}_${ds}_rs06.csv|resize:0.6" >> "$J"
done
say "test and processed exports ($(wc -l < "$J") jobs)"; lanes "$J"
echo "$SEL" > "$LOG/${TAG}_DONE"; say "done"

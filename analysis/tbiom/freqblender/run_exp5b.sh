#!/usr/bin/env bash
# Experiment 5, corrected arms (2026-09-18): wait for sbi_armD training, select its checkpoint by the unchanged v3.1
# rule, export it, and run the comparison against the criteria frozen before either arm existed. The control's
# exports (tag fbB, sbi_armB at its development-selected epoch 8) are produced by a separate queue; this script waits
# for them too. Anything already on disk is skipped, so re-running is safe.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
FBLOG=$ROOT/logs/tbiom/freqblender; S23=$ROOT/logs/tbiom/stage23; S2=$ROOT/logs/tbiom/stage2
RUN=$EXT/runs/discern_ext_train/sbi_armD_seed42
LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

say "waiting for sbi_armD training"
while ! grep -aq "Trainer.fit\` stopped\|Trainer.fit stopped" "$FBLOG/armD_train.log" 2>/dev/null; do
  grep -aq "Traceback" "$FBLOG/armD_train.log" && { say "arm D training FAILED"; exit 1; }
  sleep 180
done
say "arm D trained: $(ls "$RUN/checkpoints" | tr '\n' ' ')"

# development exports for every candidate, then the unchanged v3.1 selection
J=$FBLOG/jobs_fbD_dev.txt; : > "$J"
for ck in "$RUN/checkpoints/"*.ckpt; do
  b=$(basename "$ck" .ckpt)
  if [[ $b =~ epoch=([0-9]+) ]]; then tag="sbiarmD_e${BASH_REMATCH[1]}"; else tag="sbiarmD_last"; fi
  for ds in FFpp_val VALmix VALmixV2; do
    echo "$ck|sbiarmB_${ds}.yaml|${ds}|$S23/${tag}_${ds}.csv|" >> "$J"
  done
done
say "arm D development exports ($(wc -l < "$J") jobs, 3 lanes)"
for L in 1 2 3; do BATCH=48 bash "$LANE" 1 "$J" >> "$FBLOG/fbD_dev_lane$L.log" 2>&1 & done
wait
$PYA analysis/tbiom/select_checkpoint_v3.py --arm sbiarmD > "$FBLOG/selection_armD.log" 2>&1 || {
  say "selection FAILED"; exit 1; }
read -r SEL CK < <($PYA -c "import json;s=json.load(open('$S23/sbiarmD_selection_v3.json'))['selected'];print(s['tag'],s['checkpoint'])")
say "arm D selected $SEL"

# test and processed exports for the selected checkpoint
J=$FBLOG/jobs_fbD.txt; : > "$J"
for ds in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
  echo "$CK|sbiarmB_${ds}.yaml|${ds}|$S23/fbD_sel_${ds}.csv|" >> "$J"
done
for ds in VALmix VALmixV2 FFpp_val; do
  echo "$CK|sbiarmB_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|sbiarmB_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_rs06.csv|resize:0.6" >> "$J"
done
for ds in CDFv2 DFDCP; do
  echo "$CK|sbiarmB_${ds}.yaml|${ds}|$S2/p_v3_fbD_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|sbiarmB_${ds}.yaml|${ds}|$S2/p_v3_fbD_${ds}_rs06.csv|resize:0.6" >> "$J"
done
say "arm D test and processed exports ($(wc -l < "$J") jobs, 3 lanes)"
for L in 1 2 3; do BATCH=48 bash "$LANE" 1 "$J" >> "$FBLOG/fbD_lane$L.log" 2>&1 & done
wait

say "waiting for the control's queue"
while read -r _ck _cfg _ds out _proc; do
  while [ ! -f "$out" ]; do sleep 120; done
done < "$FBLOG/jobs_fbB.txt"

say "analysis"
$PYA analysis/tbiom/freqblender/exp5_analysis.py \
    --control sbiarmB_e08 --control-sel fbB_sel --control-proc fbB \
    --freq "$SEL" --freq-sel fbD_sel --freq-proc fbD > "$FBLOG/analysis_corrected.log" 2>&1 \
  && say "EXP_FREQBLENDER.md written" || { say "analysis FAILED"; exit 1; }
touch "$FBLOG/EXP5_DONE"

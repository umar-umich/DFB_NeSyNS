#!/usr/bin/env bash
# Experiment 5 chain, unattended: FPNet -> pseudo-fakes -> Spatial-B-Freq training -> development
# selection -> exports -> analysis. GPU 1 only, one GPU job at a time. Anything already on disk is
# skipped, so the script is safe to re-run after an interruption.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; EXT=/data/umar/Repos/DISCERN_Ext
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
FBLOG=$ROOT/logs/tbiom/freqblender; S23=$ROOT/logs/tbiom/stage23; S2=$ROOT/logs/tbiom/stage2
RUN=$EXT/runs/discern_ext_train/warm_dicome_gate_BFreq_seed42
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# 1. wait for the generator
while [ ! -f "$FBLOG/fpnet/DONE" ]; do sleep 120; done
say "FPNet done (loss $(cat "$FBLOG/fpnet/DONE"))"

# 2. pseudo-fakes
if [ ! -f "$FBLOG/pseudofake_index.txt" ]; then
  say "generating FreqBlender pseudo-fakes"
  $PY analysis/tbiom/freqblender/fpnet.py generate --batch 32 --workers 8 > "$FBLOG/generate.log" 2>&1 \
    || { say "generation FAILED"; exit 1; }
fi
say "pseudo-fakes: $(wc -l < "$FBLOG/pseudofake_index.txt") images"

# 3. the FreqBlender spatial arm (matched to warm_dicome_gate_B_seed42 except arm D at 10%)
if [ ! -f "$RUN/checkpoints/last.ckpt" ]; then
  say "training warm_dicome_gate_BFreq_seed42"
  (cd "$EXT" && $PY tools/train/train_discern_ext.py fit \
      --config_path=eval_adaptation/configs/warm_dicome_gate_BFreq_seed42.yaml \
      --num_workers=8) > "$FBLOG/bfreq_train.log" 2>&1
fi
[ -f "$RUN/checkpoints/last.ckpt" ] || { say "training produced no last.ckpt"; exit 1; }
say "training done: $(ls "$RUN/checkpoints" | tr '\n' ' ')"

# 4. development exports for every saved checkpoint, then the v3.1 selection
J=$FBLOG/jobs_dev.txt; : > "$J"
for ck in "$RUN/checkpoints/"*.ckpt; do
  b=$(basename "$ck" .ckpt)
  if [[ $b =~ epoch=([0-9]+) ]]; then tag="warmgateBFreq_e${BASH_REMATCH[1]}"; else tag="warmgateBFreq_last"; fi
  for ds in FFpp_val VALmix VALmixV2; do
    echo "$ck|warmgate_${ds}.yaml|${ds}|$S23/${tag}_${ds}.csv|" >> "$J"
  done
done
say "development exports ($(wc -l < "$J") jobs)"
bash "$ROOT/logs/tbiom/fusion_audit/lane.sh" 1 "$J" >> "$FBLOG/exports.log" 2>&1
$PYA analysis/tbiom/select_checkpoint_v3.py --arm warmgateBFreq > "$FBLOG/selection.log" 2>&1 \
  || { say "selection FAILED (see $FBLOG/selection.log)"; exit 1; }
read -r SEL CK < <($PYA -c "import json;s=json.load(open('$S23/warmgateBFreq_selection_v3.json'))['selected'];print(s['tag'],s['checkpoint'])")
say "selected $SEL -> $CK"

# 5. test and processed exports for the selected checkpoint
J=$FBLOG/jobs_test.txt; : > "$J"
for ds in CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
  echo "$CK|warmgate_${ds}.yaml|${ds}|$S23/warmgateBFreq_sel_${ds}.csv|" >> "$J"
done
for ds in VALmix VALmixV2 FFpp_val; do
  echo "$CK|warmgate_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|warmgate_${ds}.yaml|${ds}|$S2/p_dev_${SEL}_${ds}_rs06.csv|resize:0.6" >> "$J"
done
for ds in CDFv2 DFDCP; do
  echo "$CK|warmgate_${ds}.yaml|${ds}|$S2/p_v3_warmgateBFreq_${ds}_jpeg40.csv|jpeg:40" >> "$J"
  echo "$CK|warmgate_${ds}.yaml|${ds}|$S2/p_v3_warmgateBFreq_${ds}_rs06.csv|resize:0.6" >> "$J"
done
say "test and processed exports ($(wc -l < "$J") jobs)"
bash "$ROOT/logs/tbiom/fusion_audit/lane.sh" 1 "$J" >> "$FBLOG/exports.log" 2>&1

# 6. analysis
say "analysis"
$PYA analysis/tbiom/freqblender/exp5_analysis.py --freq "$SEL" > "$FBLOG/analysis.log" 2>&1 \
  && say "EXP_FREQBLENDER.md written" || say "analysis FAILED (see $FBLOG/analysis.log)"
touch "$FBLOG/CHAIN_DONE"

#!/usr/bin/env bash
# Stage 2 follow-up (pre-registration amendment of 2026-09-21): after the 2x2 analysis, train a Mamba temporal head
# only for a sampling where the GRU head beat pooling by >= 0.005 on the selection part, then redo the analysis so the
# Mamba head joins the candidate set. If the trigger does not fire, nothing is trained.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; OUT=$ROOT/logs/tbiom/temporal_stage2
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python; PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while [ ! -f "$OUT/STAGE2_DONE" ]; do sleep 120; done
FIRE=$($PYA -c "import json; t=json.load(open('tbiom/reports/EXP_TEMPORAL_SAMPLING.json'))['_mamba_trigger']; print(' '.join(s for s, v in t.items() if v['fires']))")
if [ -z "$FIRE" ]; then say "Mamba trigger did not fire for any sampling; nothing trained"; touch "$OUT/MAMBA_DONE"; exit 0; fi
for smp in $FIRE; do
  say "trigger fired for $smp: training $smp x mamba"
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)" -lt 60000 ]; do sleep 60; done
  $PYV analysis/tbiom/temporal/stage2_train.py --sampling "$smp" --head mamba --diagnostics --workers 20 > "$OUT/${smp}_mamba.log" 2>&1 &
  sleep 240
done
wait
$PYA analysis/tbiom/temporal/stage2_analysis.py > "$OUT/analysis_mamba.log" 2>&1 && say "analysis redone with the Mamba head" || say "analysis FAILED"
touch "$OUT/MAMBA_DONE"

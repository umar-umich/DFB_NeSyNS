#!/usr/bin/env bash
# Finish the DF40 cross-manipulation exports and write the tables. GPU 1 only.
#
# The lanes claim jobs with mkdir and only rename the output at the end, so a job killed mid-write leaves a claim and
# no file. This waits for the lanes to drain, releases the claims of outputs that are missing AND have no live writer,
# re-runs them on the same queues (same claims directory, so no two lanes can take the same job), then exports the
# frozen temporal branch on every generator set and regenerates both result documents.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/df40; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

missing() { $PYA analysis/tbiom/df40_sweep.py "$1"; }

for pass in 1 2 3; do
  while pgrep -x python -f "export_views.py" > /dev/null 2>&1 || \
        ps -eo args | grep -q "[e]xport_views.py"; do sleep 60; done
  n=$(missing release)
  say "pass $pass: $n exports missing after the lanes drained"
  [ "$n" = "0" ] && break
  for L in 1 2 3 4 5 6 7 8; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BATCH=96 bash "$LANE" 1 "$LOG/jobs_all.txt" >> "$LOG/finish_all_$L.log" 2>&1 &
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BATCH=96 bash "$LANE" 1 "$LOG/jobs_fr.txt" >> "$LOG/finish_fr_$L.log" 2>&1 &
  done
  wait
done
say "spatial exports settled ($(missing count) still missing)"

say "temporal branch on every generator set"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PYV analysis/tbiom/temporal/export_branch.py --branch vjepa2_lora_B_seed42 \
    --datasets $(tr '\n' ' ' < "$LOG/datasets.txt") --workers 12 >> "$LOG/temporal.log" 2>&1 \
  && say "temporal done" || say "temporal FAILED"

touch "$LOG/EXPORTS_DONE" "$LOG/FR_DONE"
say "analysis"
OMP_NUM_THREADS=4 $PYA analysis/tbiom/df40_analysis.py >> "$LOG/analysis_chain.log" 2>&1 && say "df40 tables written" || say "df40 analysis FAILED"
OMP_NUM_THREADS=4 $PYA analysis/tbiom/detailed_report.py >> "$LOG/analysis_chain.log" 2>&1 && say "detailed report written" || say "detailed report FAILED"
say "finished"

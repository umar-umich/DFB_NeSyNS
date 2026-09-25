#!/usr/bin/env bash
# DiCoME's RELEASED weights on all 41 DF40 generator sets, so the reenactment comparison covers the same videos as
# ours and as our own DiCoME training. Waits for the t-SNE extraction to free GPU 1. Cached-checkpoint inference only.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/df40; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while [ ! -f "$ROOT/logs/tsne/PASS2_DONE" ]; do sleep 60; done
say "t-SNE extraction finished; starting released-weight DF40 exports"
for i in 1 2 3 4 5 6; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BATCH=96 bash "$LANE" 1 "$LOG/jobs_released.txt" >> "$LOG/rel_lane_$i.log" 2>&1 &
done
wait
n=$(ls "$ROOT"/logs/tbiom/stage23/dicomerel_DF40_*.csv 2>/dev/null | wc -l)
say "released-weight DF40 exports finished: $n / 41"
touch "$LOG/RELEASED_DONE"

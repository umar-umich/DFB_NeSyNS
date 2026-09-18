#!/usr/bin/env bash
# After Experiment 5's verdict: unseal and score FaceShifter (decision of 2026-09-18), then rebuild the results
# tables so they carry it. Nothing is tuned between the verdict and the read.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS
PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
FBLOG=$ROOT/logs/tbiom/freqblender
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

while [ ! -f "$FBLOG/EXP5_DONE" ]; do
  grep -q "FAILED" "$FBLOG/chain_b.log" 2>/dev/null && { say "Experiment 5 failed; not unsealing anything"; exit 1; }
  sleep 180
done
say "Experiment 5 verdict is in; the recipe is fixed"

say "FaceShifter: unsealing and scoring once"
bash analysis/tbiom/faceshifter_confirm.sh >> "$ROOT/logs/tbiom/faceshifter/driver.log" 2>&1 \
  && say "FaceShifter scored" || { say "FaceShifter scoring FAILED"; exit 1; }

say "rebuilding the results tables"
OMP_NUM_THREADS=4 $PYA analysis/tbiom/final_table.py >> "$FBLOG/final_table.log" 2>&1 \
  && say "FINAL_TABLE.md rebuilt" || say "table rebuild FAILED"
touch "$FBLOG/WRAPUP_DONE"

#!/usr/bin/env bash
# The reenactment half of the cross-manipulation study: same machinery, same lanes.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/df40; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
say "reenactment spatial exports ($(wc -l < "$LOG/jobs_fr.txt") jobs)"
for L in 1 2; do BATCH=96 bash "$LANE" 1 "$LOG/jobs_fr.txt" >> "$LOG/fr_lane_gpu1_$L.log" 2>&1 & done
for L in 1 2; do BATCH=64 bash "$LANE" 3 "$LOG/jobs_fr.txt" >> "$LOG/fr_lane_gpu3_$L.log" 2>&1 & done
wait
say "reenactment temporal exports"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PYV analysis/tbiom/temporal/export_branch.py --branch vjepa2_lora_B_seed42 \
    --datasets $(tr '\n' ' ' < "$LOG/datasets_fr.txt") --workers 12 > "$LOG/temporal_fr.log" 2>&1 \
  && say "temporal done" || say "temporal FAILED"
touch "$LOG/FR_DONE"; say "reenactment exports finished"

#!/usr/bin/env bash
# Cross-manipulation exports: our reference and our trained DiCoME on every DF40 face-swap generator, then the
# frozen temporal branch on the same sets. Lanes claim jobs with mkdir, so several may share one queue.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/df40; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
say "spatial exports ($(wc -l < "$LOG/jobs_all.txt") jobs)"
for L in 1 2 3 4; do BATCH=96 bash "$LANE" 1 "$LOG/jobs_all.txt" >> "$LOG/lane_gpu1_$L.log" 2>&1 & done
# GPU 3 lanes removed 2026-09-22 at UMAR's direction: GPU 1 only unless GPU 3 is genuinely needed.
wait
say "temporal branch exports"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PYV analysis/tbiom/temporal/export_branch.py --branch vjepa2_lora_B_seed42 \
    --datasets $(tr '\n' ' ' < "$LOG/datasets.txt") --workers 12 > "$LOG/temporal.log" 2>&1 \
  && say "temporal done" || say "temporal FAILED"
touch "$LOG/EXPORTS_DONE"; say "all exports finished"

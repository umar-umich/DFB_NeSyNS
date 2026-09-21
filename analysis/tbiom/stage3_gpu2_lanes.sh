#!/usr/bin/env bash
# Extra export lanes on GPU 2 for the Stage 3 arms. Pure inference: the lane script claims each job with mkdir, so
# these lanes drain the same queues the GPU 1 lanes are working on, and no opinion depends on which card wrote it.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/stage3; LANE=$ROOT/logs/tbiom/fusion_audit/lane.sh
cd "$ROOT" || exit 2
for TAG in wgUB wgUE sbiarmEs1337 sbiarmEs7; do
  (
    while [ ! -f "$LOG/${TAG}_DONE" ]; do
      for J in "$LOG/${TAG}_jobs_dev.txt" "$LOG/${TAG}_jobs_test.txt"; do
        [ -f "$J" ] || continue
        for L in 1 2; do BATCH=96 bash "$LANE" 2 "$J" >> "$LOG/${TAG}_gpu2_lane$L.log" 2>&1 & done
        wait
      done
      sleep 60
    done
  ) &
done
wait

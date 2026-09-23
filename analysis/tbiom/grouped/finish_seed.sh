#!/usr/bin/env bash
# When the second-detector-seed exports finish, run the confirmation for both arm-B seeds and refresh the outcome.
# GPU 1 only; the analysis itself is CPU.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; LOG=$ROOT/logs/tbiom/grouped
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd "$ROOT" || exit 2
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
need() { $PY - <<'PY'
from pathlib import Path
L = Path("logs/tbiom/grouped/jobs_seed.txt")
print(sum(0 if Path(l.split("|")[3]).exists() else 1 for l in L.read_text().split("\n") if l.strip()))
PY
}
while true; do
  n=$(need)
  [ "$n" = "0" ] && break
  if ! ps -eo args | grep -q "[e]xport_views.py"; then
    say "$n exports missing and no lane is running; re-queueing"
    for L in 1 2 3 4 5 6 7 8 9 10 11 12; do
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BATCH=96 bash "$ROOT/logs/tbiom/fusion_audit/lane.sh" 1 \
        "$LOG/jobs_seed.txt" >> "$LOG/requeue_$L.log" 2>&1 &
    done
    wait
  fi
  sleep 120
done
say "all 82 second-seed exports present"
OMP_NUM_THREADS=8 $PY analysis/tbiom/grouped/confirm.py --seeds armB42,armB1337 >> "$LOG/confirm_seed.log" 2>&1 \
  && say "confirmation refreshed" || say "confirmation FAILED"
touch "$LOG/SEED_DONE"; say "done"

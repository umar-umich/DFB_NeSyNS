#!/usr/bin/env bash
# Hand the grid over from 16 branches to 17 WITHOUT idling the box in between.
#
# P16ctlB is still exporting its opinions, so the 17-branch grid cannot start yet. Stopping the
# 16-branch grid now would leave 384 cores idle for the forty minutes that export takes, so the
# 16-branch run keeps going until the new branch is actually registered, and is replaced only then.
#
# The running shards are killed BY THE PIDS RECORDED BEFORE THIS LANE STARTED, never by pattern: a
# lane that greps for its own job name matches its own argv, and that has already cost this project
# two deadlocks and a stray exit code.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
M=logs/branchfusion/orchestrator
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== grid switch: waiting for P16ctlB to register ==="
n=0
until $PY -c "import sys; sys.path.insert(0,'upgrade/branchfusion'); import branches as B; sys.exit(0 if 'P16ctlB' in B.SOURCES else 1)" 2>/dev/null; do
  sleep 60; n=$((n + 1))
  [ $((n % 15)) -eq 0 ] && say "grid switch: still waiting for P16ctlB (${n} min)"
  [ "$n" -ge 180 ] && { say "grid switch: TIMEOUT -- P16ctlB never registered, 16-branch grid left running"; exit 1; }
done
say "grid switch: P16ctlB registered after ${n} min; retiring the 16-branch grid"

while read -r pid; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done < "$M/grid_all16.pids"
sleep 10
rm -f analysis/branchfusion/per_dataset_all16_shard*.json

say "grid switch: starting the 17-branch grid"
exec bash upgrade/branchfusion/lanes/grid_all17.sh

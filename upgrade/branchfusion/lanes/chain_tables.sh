#!/usr/bin/env bash
# Keep the box working while the branch names are decided. Naming is a find-and-replace in the
# generators and does not invalidate a single number, so none of this waits on it.
#
#   1. App's CE head -- the last missing piece of the CE arm (minutes, cached features)
#   2. wait for the new-set scoring to drain
#   3. regenerate Tables A and B, now over twelve datasets
#   4. regenerate Table C against the baselines
#   5. rebuild the single consolidated file
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== chain: App CE head ==="
if [ ! -f "$M/app_ce_head.done" ]; then
  CUDA_VISIBLE_DEVICES=3 $PY upgrade/train_head_ce.py --backbone pe_core_lLN --seed 0 \
      >> "$L/app_ce_head.log" 2>&1 \
    && { touch "$M/app_ce_head.done"; say "App CE head DONE"; } || say "App CE head FAILED"
fi

say "chain: waiting for the new-set scoring"
n=0
while pgrep -f "score_new_sets.sh" > /dev/null 2>&1 && [ $n -lt 240 ]; do sleep 60; n=$((n+1)); done
say "chain: scoring drained after ${n} min"

say "chain: regenerating tables"
$PY upgrade/branchfusion/proposal_tables.py >> "$L/chain_tables.log" 2>&1 || say "tables A/B FAILED"
$PY upgrade/branchfusion/sota_table1.py     >> "$L/chain_tables.log" 2>&1 || say "table C FAILED"

# The report is assembled in ONE place. These lanes each carried their own copy of the
# heredoc, which silently went stale -- the version here was still emitting three tables
# with no name key and no Table A2 long after the generators had five.
bash upgrade/branchfusion/lanes/build_report.sh
say "=== chain done -> tbiom/reports/branch_fusion/PROPOSAL_TABLES.md ==="

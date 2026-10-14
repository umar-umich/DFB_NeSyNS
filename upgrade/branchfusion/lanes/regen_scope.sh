#!/usr/bin/env bash
# Regenerate every table and the figure at ONE evaluation scope.
#
#   bash upgrade/branchfusion/lanes/regen_scope.sh v31        # the published census, as a check
#   bash upgrade/branchfusion/lanes/regen_scope.sh nolipadv
#
# `regen_all.sh` is this without the scope argument and with a commit message welded on; this is
# the version that can be run three times without each run overwriting the last one's numbers.
set -u
cd /data/umar/Repos/DFB_NeSyNS
SCOPE=${1:?usage: regen_scope.sh <v31|nolip|nolipadv|all>}
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/scope_expand
M=logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 BF_SCOPE=$SCOPE
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== regen at scope $SCOPE ==="
: > "$L/regen_$SCOPE.log"
for s in proposal_tables per_dataset sota_table1 table2_calibration uncertainty_figs \
         forensic_metrics sota_operating_points; do
  if [ "$s" = per_dataset ]; then
    $PY "upgrade/branchfusion/$s.py" --branches A,V,O,P16ctl,S16ctl --all --tag core5 \
      >> "$L/regen_$SCOPE.log" 2>&1 \
      && say "  $SCOPE $s ok" || say "  $SCOPE $s FAILED"
  else
    $PY "upgrade/branchfusion/$s.py" >> "$L/regen_$SCOPE.log" 2>&1 \
      && say "  $SCOPE $s ok" || say "  $SCOPE $s FAILED"
  fi
done
bash upgrade/branchfusion/lanes/build_report.sh >> "$L/regen_$SCOPE.log" 2>&1 \
  && say "=== regen $SCOPE done ===" || say "=== regen $SCOPE report FAILED ==="

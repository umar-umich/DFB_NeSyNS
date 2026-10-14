#!/usr/bin/env bash
# The widest grid that EXISTS, in the two forms that are actually computable.
#
# UMAR asked for every subset of every candidate on every test set. That is not reachable, and the
# reason is `fit_weights`: it is EXHAUSTIVE over the locked reliability grid, so one k-branch
# subset costs 5^k - 4^k fusions of the whole development pool. At 5.9 ms a fusion that is 2.2 s at
# size 4, 68 s at size 6, 32 minutes at size 8 and 139 YEARS at size 17. All 2^17-1 subsets is
# 3,024 core-years, or 24 years across 128 shards. No amount of hardware fixes an exponent.
#
# So, two passes, and together they leave nothing unseen:
#
#   LOCKED, sizes 1-6 (21,777 subsets).  The real B0 weight fit, directly comparable with every
#   locked number in this campaign. Size 6 is where the exponent stops being affordable: 1-6 is two
#   hours across 128 shards, 1-7 is seventeen. Deployed systems are three views and the brief's own
#   screen stopped at seven, so this covers every subset that could be a system.
#
#   EQUAL WEIGHTS, every size (131,071 subsets).  One weight vector instead of 746 billion, so the
#   cost is linear and the WHOLE space is visible. It is a different quantity from the locked rows
#   and is labelled as such everywhere; its job is to make sure no large subset is interesting
#   without us knowing, and any cell that looks interesting can be re-fitted on its own.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
BR=A,S,V,O,M,Z,P,R,Rmean,Y,Y16,Y20,P16om,P16ctl,P16ctlB,S16om,S16ctl
N=128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== final grid: waiting for P16ctlB ==="
n=0
until $PY -c "import sys; sys.path.insert(0,'upgrade/branchfusion'); import branches as B; sys.exit(0 if 'P16ctlB' in B.SOURCES else 1)" 2>/dev/null; do
  sleep 60; n=$((n + 1))
  [ $((n % 15)) -eq 0 ] && say "final grid: still waiting for P16ctlB (${n} min)"
  [ "$n" -ge 240 ] && { say "final grid: TIMEOUT waiting for P16ctlB"; exit 1; }
done
say "final grid: P16ctlB registered; 17 branches"

pass () {   # pass <tag> <sizes> <weight-grid>
  local tag=$1 sizes=$2 wg=$3 i
  [ -f "$M/grid_$tag.done" ] && { say "grid $tag already done"; return 0; }
  say "grid $tag: sizes $sizes, weight grid $wg, $N shards"
  for i in $(seq 0 $((N - 1))); do
    $PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$sizes" \
        --weight-grid "$wg" --tag "$tag" --shard "$i/$N" > "$L/grid_${tag}_shard$i.log" 2>&1 &
  done
  wait
  $PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$sizes" \
      --weight-grid "$wg" --tag "$tag" --merge $N > "$L/grid_${tag}_merge.log" 2>&1 \
    && { touch "$M/grid_$tag.done"; say "grid $tag DONE -> tbiom/reports/branch_fusion/PER_DATASET_${tag^^}.md"; } \
    || say "grid $tag MERGE FAILED -- see $L/grid_${tag}_merge.log"
}

# Cheap and complete first, so the whole space is on disk within the hour; then the locked fit.
pass all17eq 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 equal
pass all17   1,2,3,4,5,6                               locked
say "=== final grid done ==="

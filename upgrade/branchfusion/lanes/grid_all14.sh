#!/usr/bin/env bash
# EVERY subset of EVERY CLIP / PE-Core / V-JEPA candidate, scored on EVERY test set separately.
#
# UMAR, 2026-10-11: "I don't trust the mean numbers of all 10 test datasets ... I need all possible
# combinations for all possible candidates on all test datasets!" The ten-set macro is one number
# over sets that differ enormously in prevalence and difficulty, and a subset can win on it while
# losing badly on two sets. This produces the grid instead of the summary.
#
# CANDIDATES (14). Everything on the three trunks he named, controls included, with NO pre-filtering
# -- pre-filtering is exactly what he is pushing back against:
#     CLIP      O
#     V-JEPA    V
#     PE-Core   A S P R Rmean Y Y16 Y20 P16om P16ctl S16om S16ctl
# EXCLUDED, with the evidence rather than by assumption:
#     Z   DINOv3 off-manifold -- solo macro 0.6826 against A's 0.8736, and the fit zeroes it in 5 of
#         5 folds in both A+V+O+Z and A+V+Z. UMAR's read is correct.
#     M   VideoMAE -- not one of the three trunks. It is in the older ALL7 grid, so this one is not
#         a superset of that; adding it back doubles the subset count.
#
# 2**14 - 1 = 16383 subsets x 10 test sets. About eight seconds each, so a day and a half in one
# process and roughly an hour across 32. The box has 384 cores and 1.7 TB free, and each shard is
# capped at 4 threads.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
BR=A,S,V,O,P,R,Rmean,Y,Y16,Y20,P16om,P16ctl,S16om,S16ctl
SZ=1,2,3,4,5,6,7,8,9,10,11,12,13,14
N=32
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== all-14 per-dataset grid: 16383 subsets across $N shards ==="
for i in $(seq 0 $((N - 1))); do
  $PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$SZ" \
      --tag all14 --shard "$i/$N" > "$L/grid_all14_shard$i.log" 2>&1 &
done
wait
say "all-14 grid: shards done, merging"
$PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$SZ" --tag all14 --merge $N \
    > "$L/grid_all14_merge.log" 2>&1 \
  && say "=== all-14 grid done -> reports/branch_fusion/PER_DATASET_ALL14.md ===" \
  || say "all-14 grid MERGE FAILED -- see $L/grid_all14_merge.log"

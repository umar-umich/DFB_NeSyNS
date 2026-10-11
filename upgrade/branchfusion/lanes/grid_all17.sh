#!/usr/bin/env bash
# EVERY subset of EVERY CLIP / PE-Core / V-JEPA candidate, scored on EVERY test set separately.
#
# UMAR, 2026-10-11: "I don't trust the mean numbers of all 10 test datasets ... I need all possible
# combinations for all possible candidates on all test datasets!" The ten-set macro is one number
# over sets that differ enormously in prevalence and difficulty, and a subset can win on it while
# losing badly on two sets. This produces the grid instead of the summary.
#
# CANDIDATES: EVERY branch in the ledger, nothing excluded. "Include M as well, I want to see
# everything" (UMAR, 2026-10-11), and nothing is pre-filtered because pre-filtering is what he has
# pushed back on twice.
#     CLIP      O
#     V-JEPA    V
#     VideoMAE  M            -- added on his instruction; it is not one of the three trunks he named
#     DINOv3    Z            -- added although he and I both read it as a dead candidate
#     PE-Core   A S P R Rmean Y Y16 Y20 P16om P16ctl P16ctlB S16om S16ctl
#
# P16ctlB is P16ctl's SEED REPLICATE, not a new construction. It is in the grid on UMAR's
# instruction, and it earns its place for a reason worth stating: when both seeds of the same
# branch are in the same subset space, every pair of subsets differing only by seed is a direct
# read of how much of any gap is the branch and how much is the draw -- per dataset, which is the
# whole point of this grid. Seed 42 selected 0.8678 and seed 1 selected 0.8671 on development.
#
# Z IS IN ON PURPOSE. The case for dropping it -- solo macro 0.6826 against A's 0.8736, zeroed in
# 5 of 5 folds in A+V+O+Z and A+V+Z -- rests on exactly the ten-set MACRO he has said he does not
# trust. Putting it in the grid lets that claim be checked per dataset instead of taken on my word.
#
# 2**16 - 1 = 131071 subsets x 10 test sets. Several days in one process; roughly four hours across
# 64 shards. The box has 384 cores and 1.7 TB free, and each shard is capped at 4 threads.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
BR=A,S,V,O,M,Z,P,R,Rmean,Y,Y16,Y20,P16om,P16ctl,P16ctlB,S16om,S16ctl
SZ=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
N=128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== all-17 per-dataset grid: 131071 subsets across $N shards ==="
for i in $(seq 0 $((N - 1))); do
  $PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$SZ" \
      --tag all17 --shard "$i/$N" > "$L/grid_all17_shard$i.log" 2>&1 &
done
wait
say "all-17 grid: shards done, merging"
$PY upgrade/branchfusion/per_dataset.py --branches "$BR" --sizes "$SZ" --tag all17 --merge $N \
    > "$L/grid_all17_merge.log" 2>&1 \
  && say "=== all-17 grid done -> reports/branch_fusion/PER_DATASET_ALL17.md ===" \
  || say "all-17 grid MERGE FAILED -- see $L/grid_all17_merge.log"

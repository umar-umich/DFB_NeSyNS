#!/usr/bin/env bash
# One extraction lane: work through a job list on one GPU, sequentially.
#   ./run_lane.sh <lane index> <cuda device index>
# Finished shards are skipped on restart, so a killed lane resumes where it stopped.
set -u
LANE=$1; DEV=$2
ROOT=/data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$ROOT" || exit 1
while read -r DS SHARD NSH; do
  [ -z "${DS:-}" ] && continue
  echo "=== $(date '+%F %T') lane $LANE gpu $DEV -> $DS shard $SHARD/$NSH"
  CUDA_VISIBLE_DEVICES=$DEV $PY branches/ijepa/extract_residuals.py \
      --dataset "$DS" --shard "$SHARD" --num-shards "$NSH" --chunk-frames 8192 --batch-size 16
done < "analysis/ijepa/logs/lane${LANE}.jobs"
echo "=== $(date '+%F %T') lane $LANE done"

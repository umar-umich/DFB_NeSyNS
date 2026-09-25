#!/usr/bin/env bash
# Part 1 of the independent-views brief: everything remaining in the joint IBDC study.
# One lane, jobs run sequentially so the GPU is not oversubscribed by 20-epoch head runs.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while read -r RUN SEED LAM; do
  [ -z "${RUN:-}" ] && continue
  OUT="analysis/joint_ibdc/run_${RUN}_s${SEED}"
  [ "$RUN" = "J2a" ] || [ "$RUN" = "J2b" ] && OUT="${OUT}_lam${LAM}"
  if [ -f "${OUT}.json" ]; then say "skip $RUN seed $SEED lam $LAM (exists)"; continue; fi
  say "$RUN seed $SEED lambda $LAM"
  CUDA_VISIBLE_DEVICES=${GPU:-1} $PY training/joint_ibdc/runs.py --run "$RUN" --seed "$SEED" --lam "$LAM" \
      >> "logs/joint_ibdc/${RUN}_s${SEED}_lam${LAM}.log" 2>&1 \
    && say "$RUN s$SEED lam$LAM done" || say "$RUN s$SEED lam$LAM FAILED"
done < "$1"
say "part 1 lane finished"

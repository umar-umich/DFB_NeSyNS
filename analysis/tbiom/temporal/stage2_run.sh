#!/usr/bin/env bash
# Stage 2, unattended: dense frames -> partition -> configurations (as many at once as GPU 1 holds) -> analysis.
# Configurations and budget are frozen in tbiom/docs/stage2-temporal-preregistration.md. Safe to re-run.
set -uo pipefail
ROOT=/data/umar/Repos/DFB_NeSyNS; DENSE=/data/umar/Datasets/tbiom_dense; OUT=$ROOT/logs/tbiom/temporal_stage2
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python; PYA=/data/umar/miniconda3/envs/discern_ext/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
cd "$ROOT" || exit 2; mkdir -p "$OUT"
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
run() {  # <sampling> <head>
  local d="$OUT/$1_$2_seed42"
  [ -f "$d/DONE" ] && { say "skip $1 x $2 (done)"; return 0; }
  say "train $1 x $2"
  $PYV analysis/tbiom/temporal/stage2_train.py --sampling "$1" --head "$2" --diagnostics --workers 20 > "$OUT/$1_$2.log" 2>&1 \
    && say "done $1 x $2 ($(cat "$d/DONE"))" || say "FAILED $1 x $2 (see $OUT/$1_$2.log)"
}
# Launch only when GPU 1 has room: a V-JEPA 2 LoRA training at the frozen batch of 32 peaks near 50 GB, so a job starts
# once 60 GB are free, and the next check waits for the previous job to reach its working set.
when_free() {
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)" -lt 60000 ]; do sleep 60; done
}
while [ ! -f "$DENSE/dev2_dense_manifest.json" ]; do sleep 120; done
say "development frames ready; fixing the partition"
$PYV -c "import sys; sys.path[:0]=['analysis/tbiom/temporal','analysis/tbiom']; import stage2_train as S; p=S.partition(); \
import collections; print(collections.Counter(p.values()))" >> "$OUT/partition.log" 2>&1
cat "$OUT/partition.log"

say "reference exports + sparse x GRU"
( [ -f "$OUT/reference_sparse_pool/DONE" ] || $PYV analysis/tbiom/temporal/stage2_train.py --sampling sparse --head pool --reference \
    --workers 20 > "$OUT/reference.log" 2>&1 ) &
when_free; run sparse gru & sleep 240

while [ ! -f "$DENSE/ffpp_train_dense_manifest.json" ]; do sleep 60; done
say "training frames ready"
for cfg in "dense pool" "dense gru" "dsub pool"; do
  when_free; run $cfg & sleep 240
done
wait

say "analysis"
$PYA analysis/tbiom/temporal/stage2_analysis.py > "$OUT/analysis.log" 2>&1 && say "EXP_TEMPORAL_SAMPLING.md written" \
  || say "analysis FAILED"
touch "$OUT/STAGE2_DONE"

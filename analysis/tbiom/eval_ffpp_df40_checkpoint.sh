#!/usr/bin/env bash
# Evaluate one FF++(+)DF40 checkpoint on every standard cross-dataset benchmark.
#
#   bash analysis/tbiom/eval_ffpp_df40_checkpoint.sh <epoch> [out_root]
#
# WHY MID-TRAINING. FF++ VAL_select is saturated for this model — video AUROC sits at 0.98-0.99
# from epoch 0 — so the in-domain curve cannot say whether training is going well. The question
# is whether adding 32 DF40 methods to the training set buys cross-dataset generalisation, and
# only the OOD suite answers it. Reading it early is worth a few GPU-hours because the run is
# ~2.4h per epoch and there are ten of them.
#
# THESE NUMBERS ARE DIAGNOSTIC, NOT A RESULT. They are test splits, read mid-run, on a checkpoint
# chosen by nothing. Using them to pick an epoch would be selecting on test; selection stays on
# FF++ VAL_select (and VALmix), exactly as the FF++-only arm does. The comparison table they feed
# is `tbiom/CROSSDATASET.md`, whose header carries the same warning.
#
# Concurrency is capped at 2 because eval_v1 peaked at 63 GiB on one DFDC job and OOM-ed 13 jobs
# of the earlier sweep, and a training run is resident on the same card.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

EPOCH=${1:?epoch number, e.g. 2}
OUT=${2:-logs/tbiom/ffpp_df40_eval/epoch_$(printf '%03d' "$EPOCH")}
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
RUN=logs/tbiom/v1_ffpp_df40_seed42
CKPT=$RUN/epoch_$(printf '%03d' "$EPOCH").pth
DEVICE=${DEVICE:-cuda:3}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-16}
PAR=${PAR:-2}

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
mkdir -p "$OUT"
echo "evaluating $CKPT on $DEVICE (batch $BATCH, $PAR at a time)"

DATASETS=(FaceForensics++ Celeb-DF-v2 Celeb-DF-v1 DFDCP DFDC DeepFakeDetection \
          Deepfake-Eval-2024 UADFV)

running=0
for ds in "${DATASETS[@]}"; do
  safe=${ds//[^A-Za-z0-9]/_}
  if ls "$OUT/$safe"/*.parquet >/dev/null 2>&1; then echo "  skip $safe (done)"; continue; fi
  echo "  launch $safe"
  $PY -u training/eval_v1.py --checkpoint "$CKPT" --seed 42 --overwrite \
      --datasets "$ds" --output "$OUT/$safe" \
      --batch-size "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
      > "$OUT/$safe.log" 2>&1 &
  running=$((running + 1))
  while [ "$running" -ge "$PAR" ]; do wait -n 2>/dev/null || true; running=$((running - 1)); done
done
wait
echo "done"
for d in "$OUT"/*/; do
  ls "$d"/*.parquet >/dev/null 2>&1 && echo "  OK $(basename "$d")" || echo "  -- $(basename "$d") EMPTY"
done

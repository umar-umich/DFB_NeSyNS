#!/usr/bin/env bash
# Score the anchor and both FS-VFM students on every standard cross-dataset benchmark.
#
#   bash analysis/tbiom/sweep_crossdataset.sh [out_root]
#
# WHY. Stage 1 decided branch membership on DF40-Dev alone. DF40 is a generator zoo; it is not
# the distribution the cross-dataset literature reports on, and a branch that carries no
# recoverable signal against DF40 forgeries may still carry it against DFDC's compression and
# in-the-wild capture, or against Deepfake-Eval-2024's genuinely uncurated media. Deciding
# membership on one OOD axis and calling it "no complementarity" would be an overclaim.
#
# WHAT THESE NUMBERS MAY AND MAY NOT DECIDE. For DFDCP, Celeb-DF-v1/v2 and UADFV the dataset JSON
# gives val and test the SAME video list (verified: 652/652, 518/518, 100/100, 98/98), and DFDC
# ships 2 val videos against 4,704 test. So there is no honest held-out partition inside these
# files. Everything this script produces is therefore a TEST measurement and belongs in the final
# table. It is here to show whether the Stage-1 verdict travels, and to size the effect. It must
# NOT be the set a membership decision is fit on — that is what VALmix exists for, being carved
# from videos absent from all three test lists.
#
# Sharded by (model, dataset) because the bottleneck is CPU-side PNG decode, not GPU compute:
# batch 32 -> 256 buys +17% while 6 processes brought GPU 3 to full utilisation. See
# analysis/tbiom/shard_score.sh for the measurement.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

OUT=${1:-logs/tbiom/crossdataset}
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
DEVICE=${DEVICE:-cuda:3}
WORKERS=${WORKERS:-24}

# The two scorers are NOT interchangeable in memory. `score_fpad` runs one ViT and holds ~8 GiB
# at batch 128; `eval_v1` runs the full multi-branch V1 model and was measured at up to 63 GiB on
# one DFDC job. Running both at one parallelism OOM-ed GPU 3 and killed 13 of 24 jobs, so each
# side gets its own batch size and its own concurrency cap, and the anchor runs in a second phase
# rather than competing with the students for the card.
STU_BATCH=${STU_BATCH:-128}
STU_PAR=${STU_PAR:-10}
CLIP_BATCH=${CLIP_BATCH:-32}
CLIP_PAR=${CLIP_PAR:-2}

ANCHOR_CKPT=logs/v1/stage_b_seed42/epoch_007.pth
PRESERVE=logs/fpad/studentA_preserve_seed42
ORDINARY=logs/fpad/studentA_ordinary_seed42

# Ordered biggest-first so the long poles start immediately and the tail is short jobs.
DATASETS=(DFDC DeepFakeDetection Deepfake-Eval-2024 FaceForensics++ DFDCP Celeb-DF-v2 \
          Celeb-DF-v1 UADFV)

mkdir -p "$OUT"
running=0; cap=1
launch() {   # launch <logfile> <cmd...>
  local log=$1; shift
  "$@" > "$log" 2>&1 &
  running=$((running + 1))
  while [ "$running" -ge "$cap" ]; do
    wait -n 2>/dev/null || true
    running=$((running - 1))
  done
}

# Resumable: a directory that already holds a parquet is finished work. Re-running the sweep
# after an OOM therefore picks up only what died, instead of redoing the hours that survived.
done_already() { ls "$1"/*.parquet >/dev/null 2>&1; }

# --- phase 1: the two students, high parallelism -------------------------------------------
cap=$STU_PAR
for ds in "${DATASETS[@]}"; do
  safe=${ds//[^A-Za-z0-9]/_}
  for arm in preserve ordinary; do
    case $arm in preserve) run=$PRESERVE;; ordinary) run=$ORDINARY;; esac
    if done_already "$OUT/${arm}_$safe"; then echo "  skip ${arm}_$safe (done)"; continue; fi
    echo "  launch ${arm}_$safe"
    launch "$OUT/${arm}_$safe.log" \
      $PY -u training/score_fpad.py --student "$run" --epoch 9 --seed 42 \
          --datasets "$ds" --split test --output "$OUT/${arm}_$safe" \
          --batch-size "$STU_BATCH" --workers "$WORKERS" --device "$DEVICE"
  done
done
wait; running=0
echo "phase 1 (students) finished"

# --- phase 2: the anchor, low parallelism ---------------------------------------------------
# eval_v1 runs mode="test" unconditionally, so the split is fixed by construction.
cap=$CLIP_PAR
for ds in "${DATASETS[@]}"; do
  safe=${ds//[^A-Za-z0-9]/_}
  if done_already "$OUT/clip_$safe"; then echo "  skip clip_$safe (done)"; continue; fi
  echo "  launch clip_$safe"
  launch "$OUT/clip_$safe.log" \
    $PY -u training/eval_v1.py --checkpoint "$ANCHOR_CKPT" --seed 42 --overwrite \
        --datasets "$ds" --output "$OUT/clip_$safe" \
        --batch-size "$CLIP_BATCH" --workers "$WORKERS" --device "$DEVICE"
done
wait
echo "sweep finished"
for d in "$OUT"/*/; do
  done_already "$d" && echo "  OK $(basename "$d")" || echo "  -- $(basename "$d") EMPTY"
done

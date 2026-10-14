#!/usr/bin/env bash
# The DiCoME baseline on the seven sets its row left empty, plus CDFv3 on OUR census.
#
# WHICH CHECKPOINT, and how it was established. `weights/dicome-best.ckpt` -- the released one --
# gives 0.9729 on CDFv2; the table's row says 0.9644. `ffpp_reproduce_seed42` epoch 5 gives
# 0.9679. Epoch 1 gives 0.9644 EXACTLY, and it is the `best` by val AUROC (0.9960), which is what
# `checkpoint_for_testing: best` selects. So the row is our reproduction of DiCoME in their repo
# at epoch 1, and that is what fills the rest of it -- a row must not mix two models.
#
# CDFv3 IS RE-RUN because the existing export covers 1,918 videos against our 5,418: the cell was
# never on our census, which Table 1 disclosed but could not fix. It lands under a DIFFERENT run
# name so the 1,918-video export stays on disk and the v3.1 numbers stay reproducible.
#
# GPU 3 with a small footprint: CLIP-L inference, a few GB, on a card kutub shares.
set -u
cd /data/umar/Repos/DiCoME
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
CK="runs/dicome_train/ffpp_reproduce_seed42/checkpoints/dicome-best-epoch=01-val_auroc_video=0.9960.ckpt"
M=/data/umar/Repos/DFB_NeSyNS/logs/branchfusion/orchestrator
L=/data/umar/Repos/DFB_NeSyNS/logs/scope_expand/dicome
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== DiCoME baseline: filling seven sets, and CDFv3 on our census ==="
for DS in UADFV FFIW DFDM DeepSpeak-1.1 KoDF DeepSpeak-2.0 IDForge; do
  out="runs/dicome_trained/dicome_trained_${DS}/csv_logs/version_0/test_video_predictions.csv"
  [ -s "$out" ] && { say "  DiCoME $DS already there"; continue; }
  CUDA_VISIBLE_DEVICES=3 $PY tools/train/train_dicome.py test --checkpoint_path "$CK" \
      --config_path "eval_adaptation/configs/ours_new/${DS}.yaml" \
      --run_name "dicome_trained_${DS}" --devices "[0]" >> "$L/${DS}.log" 2>&1 \
    && say "  DiCoME $DS done" || say "  DiCoME $DS FAILED"
done
CUDA_VISIBLE_DEVICES=3 $PY tools/train/train_dicome.py test --checkpoint_path "$CK" \
    --config_path "eval_adaptation/configs/ours_new/CDFv3.yaml" \
    --run_name "dicome_trained_CDFv3_full" --devices "[0]" >> "$L/CDFv3.log" 2>&1 \
  && say "  DiCoME CDFv3 (our census) done" || say "  DiCoME CDFv3 FAILED"
say "=== DiCoME baseline filled ==="

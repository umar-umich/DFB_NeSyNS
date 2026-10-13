#!/usr/bin/env bash
# Score the five branches on the three newly packed GenD sets, so their columns stop reading
# "in progress". Each branch is replayed from its own saved weights -- nothing is retrained.
#
#   A        head over a feature cache, so the trunk is extracted first (GPU), then replayed
#   O        DISCERN_Ext artifact view, exported from the evidential checkpoint
#   V        checkpoint replay through Dr Kutub's trainer
#   P16ctl   `train_om_pecore --export-only`, the path built for exactly this
#   S16ctl   the same with the real-timestamp temporal stage
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
SETS="UADFV DeepSpeak-1.1 DeepSpeak-2.0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== scoring the five branches on $SETS ==="

# 1. A's trunk: one feature cache per set, then the head replays off it with no GPU.
for DS in $SETS; do
  CUDA_VISIBLE_DEVICES=0 $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN \
      --dataset "$DS" >> "$L/score_new_A.log" 2>&1 \
    && say "A trunk cached: $DS" || say "A trunk FAILED: $DS"
done

# 2. The three temporal branches, replayed from their checkpoints.
export CUDA_VISIBLE_DEVICES=1
$PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 --export-only \
    --model pecore_l14 --lora --augment B --seed 42 --workers 5 >> "$L/score_new_P16ctl.log" 2>&1 \
  && say "P16ctl exported" || say "P16ctl export FAILED"
$PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 --temporal seq \
    --export-only --model pecore_l14 --lora --augment B --seed 42 --workers 5 \
    >> "$L/score_new_S16ctl.log" 2>&1 \
  && say "S16ctl exported" || say "S16ctl export FAILED"

say "=== new-set scoring lane done; re-run proposal_tables next ==="

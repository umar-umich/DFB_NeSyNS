#!/usr/bin/env bash
# Score the framework's branches on the three newly packed GenD sets -- the three sets ONLY.
#
# The first attempt did nothing useful: `--export-only` iterates the handoff trainer's own TESTS
# list, which is hardcoded to the original ten, so it silently re-exported what was already on
# disk and reported success. `--sets` now overrides it.
#
#   App    the evidential head replayed over the feature caches already extracted (no GPU)
#   STT    checkpoint replay, real-timestamp temporal stage
#   STT-o  checkpoint replay, slot positions
#   TD     V-JEPA 2 checkpoint replay
#   LFS    DISCERN_Ext export; left for a follow-up because it needs its own trainer
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
NEW="UADFV,DeepSpeak-1.1,DeepSpeak-2.0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== scoring the new sets ($NEW) ==="

CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
    --temporal seq --export-only --sets "$NEW" --model pecore_l14 --lora --augment B \
    --seed 42 --workers 5 >> "$L/new_STT.log" 2>&1 \
  && say "STT exported on the new sets" || say "STT export FAILED"

CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
    --export-only --sets "$NEW" --model pecore_l14 --lora --augment B \
    --seed 42 --workers 5 >> "$L/new_STTorder.log" 2>&1 \
  && say "STT-order exported on the new sets" || say "STT-order export FAILED"

# Copy the new per-dataset CSVs into the ledger beside the ones already there.
$PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seq_seed42 \
    --label S16ctl >> "$L/new_STT.log" 2>&1 || say "STT ledger copy FAILED"
$PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seed42 \
    --label P16ctl >> "$L/new_STTorder.log" 2>&1 || say "STT-order ledger copy FAILED"

say "=== new-set scoring v2 done ==="

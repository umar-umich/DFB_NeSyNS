#!/usr/bin/env bash
# Re-export STT, STT-order and TD opinions for CDFv3 and KoDF at the WIDER scope.
#
# WHY EACH BRANCH DIFFERS. App needs nothing: its `pe_core_lLN_*` cache already holds every
# video and its reader applies no filter. LFS needs nothing either -- its exclusion is a
# read-time predicate, now `branches.in_scope()`, over an export that was always complete.
# STT, STT-order and TD filtered when their OPINIONS were written, so those have to be
# rewritten; TD's features also had to be re-encoded first, which is already done.
#
# BF_SCOPE=nolipadv is the widest scope UMAR asked about, so one export serves both KoDF
# variants -- with and without adv-real/adv-fake -- as a subset at scoring time.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/scope_expand
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BF_SCOPE=nolipadv
SETS="CDFv3,KoDF"
say () { echo "[$(date -Is)] $*" >> "$L/chain.log"; }

say "=== re-export at scope nolipadv: $SETS ==="
( CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
      --temporal seq --export-only --sets "$SETS" --model pecore_l14 --lora --augment B \
      --seed 42 --workers 5 >> "$L/STT.log" 2>&1 \
    && { $PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seq_seed42 \
           --label S16ctl >> "$L/STT.log" 2>&1; say "STT re-exported"; } || say "STT FAILED"
  CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
      --export-only --sets "$SETS" --model pecore_l14 --lora --augment B \
      --seed 42 --workers 5 >> "$L/STTorder.log" 2>&1 \
    && { $PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seed42 \
           --label P16ctl >> "$L/STTorder.log" 2>&1; say "STT-order re-exported"; } || say "STT-order FAILED"
) &
( CUDA_VISIBLE_DEVICES=3 $PYV analysis/tbiom/temporal/train_branch.py --model vjepa2 --lora \
      --augment B --seed 42 --export-shuffled-only >> "$L/TD.log" 2>&1 \
    && say "TD re-exported" || say "TD FAILED -- see TD.log" ) &
wait
say "=== re-export finished ==="

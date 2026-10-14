#!/usr/bin/env bash
# STT and STT-order opinions on DF40's generator sets.
#
#   bash upgrade/branchfusion/lanes/df40_stt.sh "DF40_uniface_cdf DF40_uniface_ff ..."
#
# Both branches run the PE-Core trunk over CLIPS -- `train_branch.opinions` builds the clips and
# the installed stage taps block 16 inside the forward -- so this is a GPU pass over frames, not a
# cache replay, and it is the same code path training used.
#
# `--sets` overrides the trainer's hardcoded TESTS list, and with it `export_only` skips the
# development exports, the CDFv2/DFDCP corruption controls and the shuffled-order control: those
# are properties of the RUN, not of the named sets, and 16 generator sets would otherwise have
# recomputed and overwritten 16 identical copies of each.
set -u
cd /data/umar/Repos/DFB_NeSyNS
SETS_SP=${1:?usage: df40_stt.sh "<space-separated dataset names>"}
SETS=$(echo $SETS_SP | tr ' ' ',')
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/scope_expand/df40
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$L/feats.log"; }

# `om_pecore_export.py` copies DEV+TEST when given no --sets, so the DF40 CSVs would stay in the
# run directory and never reach the ledger the branch readers open -- the export would report
# success and the branch would still be "unavailable" on every generator. The standard fifteen
# are passed WITH them so the provenance JSON keeps listing the whole coverage rather than being
# overwritten with a DF40-only record.
LEDGER="VALmix,VALmixV2,FFpp_val,CDFv2,CDFv3,DFD,DFDC,DFDCP,FFIW,KoDF,IDForge,DFDM,UADFV,DeepSpeak-1.1,DeepSpeak-2.0,$SETS"

say "=== DF40 STT exports: $(echo $SETS_SP | wc -w) sets ==="
CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
    --temporal seq --export-only --sets "$SETS" --model pecore_l14 --lora --augment B \
    --seed 42 --workers 5 >> "$L/stt.log" 2>&1 \
  && { $PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seq_seed42 \
         --label S16ctl --sets "$LEDGER" >> "$L/stt.log" 2>&1; say "STT exported"; } || say "STT FAILED"

CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
    --export-only --sets "$SETS" --model pecore_l14 --lora --augment B \
    --seed 42 --workers 5 >> "$L/sttorder.log" 2>&1 \
  && { $PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seed42 \
         --label P16ctl --sets "$LEDGER" >> "$L/sttorder.log" 2>&1; say "STT-order exported"; } || say "STT-order FAILED"
say "=== DF40 STT exports finished ==="

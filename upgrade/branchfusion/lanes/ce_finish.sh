#!/usr/bin/env bash
# Finish the CE arm: the three corpora packed on 2026-10-13, and the widened CDFv3/KoDF census.
#
# WHY THE CE ROW SAID "in progress" ON THREE SETS. `train_ce.py` could only TRAIN -- it had no
# replay path -- so when UADFV, DeepSpeak-1.1 and DeepSpeak-2.0 were packed, the CE counterparts
# could not follow without retraining all three runs. The row quietly stayed on nine sets.
#
# AND THE WORSE ONE, WHICH NOBODY ASKED ABOUT. VCE, S16ctlCE and P16ctlCE were exported at the
# v3.1 census (CDFv3 5118, KoDF 1395) and LFSCE applied a FIFTH hardcoded copy of the scope
# filter. At the widened scopes that put the CE row and the evidential rows of one table over
# DIFFERENT videos, with nothing saying so.
#
# AppCE and LFSCE need no GPU: one reads cached features through a CE head, the other reduces an
# export that already covers all sixteen datasets.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/scope_expand/ce
M=logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BF_SCOPE=nolipadv
SETS="CDFv3,KoDF,UADFV,DeepSpeak-1.1,DeepSpeak-2.0"
LEDGER="VALmix,VALmixV2,FFpp_val,CDFv2,CDFv3,DFD,DFDC,DFDCP,FFIW,KoDF,IDForge,DFDM,UADFV,DeepSpeak-1.1,DeepSpeak-2.0"
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== CE arm: finishing $SETS at scope nolipadv ==="

# The run DIRECTORY is named explicitly for the ledger copy; `train_ce.py` derives the same name
# from its own argv, so the two must agree and are written side by side here to make that visible.
replay () {   # replay <label> <run-dir> <train_ce args...>
  local label=$1 run=$2; shift 2
  CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_ce.py "$@" --export-only --sets "$SETS" \
      --lora --augment B --seed 42 --workers 5 >> "$L/$label.log" 2>&1 \
    || { say "CE $label replay FAILED"; return 1; }
  $PY upgrade/branchfusion/om_pecore_export.py --run "$run" --label "$label" --sets "$LEDGER" \
      >> "$L/$label.log" 2>&1 && say "CE $label done" || say "CE $label ledger copy FAILED"
}

replay VCE      vjepa2_lora_B_ce_seed42             --model vjepa2
replay S16ctlCE pecore_l14_lora_B_ce_b16_seq_seed42 --model pecore_l14 --block 16 --temporal seq
replay P16ctlCE pecore_l14_lora_B_ce_b16_seed42     --model pecore_l14 --block 16

$PY upgrade/branchfusion/register_ce_branches.py >> "$L/register.log" 2>&1 \
  && say "CE AppCE+LFSCE registered" || say "CE registration FAILED"
say "=== CE arm finished ==="

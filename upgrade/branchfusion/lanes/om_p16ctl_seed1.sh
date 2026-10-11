#!/usr/bin/env bash
# SEED REPLICATE for the one live positive left in this campaign.
#
# P16ctl -- the block-16 tap with a rebuilt temporal mixer and NO residual -- is the only fourth
# view the fusion fit keeps in all five folds (A+V+O+P16ctl macro 0.9069 against the deployed
# 0.9047, pooled 0.9382 against 0.9361), and it beats Dr Kutub's final-pool P branch on test
# AUROC 0.9314 to 0.9057. That is +0.0022 macro, which is small enough to be a draw rather than a
# result: single-branch seed movement on this trunk measured 0.0126 in the depth work.
#
# So one more seed, and nothing else. The off-manifold runs are not replicated -- the fit zeroes
# P16om in four or five folds of every subset it appears in, and replicating a branch the fusion
# throws away would buy nothing.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

name=om_P16ctl_seed1
if [ ! -f "$M/$name.done" ]; then
  say "seed replicate start: $name on GPU 1"
  $PYV upgrade/branchfusion/train_om_pecore.py --operator none --block 16 \
      --model pecore_l14 --lora --augment B --seed 1 --workers 6 >> "$L/$name.log" 2>&1 \
    && { touch "$M/$name.done"; say "$name DONE"; } \
    || { say "$name FAILED -- see $L/$name.log"; exit 1; }
fi

# Into the ledger, then the only comparison that matters: does the deployed trio still gain?
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
$PY upgrade/branchfusion/om_pecore_export.py \
    --run pecore_l14_lora_B_omnone_b16_seed1 --label P16ctlB --register >> "$L/$name.log" 2>&1 \
  || { say "$name export FAILED"; exit 1; }
$PY upgrade/branchfusion/subsets.py --screen --branches A,V,O,P,P16ctl,P16ctlB \
    --sizes 3,4 >> "$L/${name}_screen.log" 2>&1 || say "$name screen FAILED"
say "=== $name lane done ==="

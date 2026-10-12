#!/usr/bin/env bash
# The CE counterparts of our own branches: same trunk, same budget, logits + cross-entropy.
#
# Three runs, because A's CE head is cheap over cached features and O's already exists as the
# artifact view of `run1d_simplece_seed42`. What has to be trained is the three branches whose
# head lives inside the model: P16ctl, S16ctl and V.
#
# GPU 0 and 1 take one each; V waits for GPU 3 so it does not contend with the O-CE export
# already running there. GPU 0 is ours until Awais starts something on it.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

run () {  # run <gpu> <name> <args...>
  local gpu=$1 name=$2; shift 2
  [ -f "$M/$name.done" ] && { say "$name already done"; return 0; }
  say "CE arm start: $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PYV upgrade/branchfusion/train_ce.py "$@" \
      --lora --augment B --seed 42 --workers 5 >> "$L/$name.log" 2>&1 \
    && { touch "$M/$name.done"; say "$name DONE"; } \
    || say "$name FAILED -- see $L/$name.log"
}

run 0 ce_P16ctl --block 16                  --model pecore_l14 &
run 1 ce_S16ctl --block 16 --temporal seq   --model pecore_l14 &
( # V shares GPU 3 with the O-CE export; start it once that lane has drained.
  n=0
  while pgrep -f "export_views.py" > /dev/null 2>&1 && [ $n -lt 180 ]; do sleep 60; n=$((n+1)); done
  run 3 ce_V --model vjepa2 ) &
wait
say "=== CE arm lane done ==="

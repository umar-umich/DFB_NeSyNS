#!/usr/bin/env bash
# THE PE-CORE OFF-MANIFOLD CAMPAIGN, RE-RUN. Everything before 2026-10-10 is void.
#
# `train_om_pecore.install` rebuilt the temporal stage and added the beta-VAE by rebinding the
# Branch attributes, while `train_branch.main` builds its optimiser and its checkpoint from
# `trainable_blocks` -- which still held the mixer that `__init__` made and the forward never
# called. So every earlier run trained LoRA and the head on a RANDOM temporal stage, and the same
# forward had dropped `softplus`, exporting logits as evidence (p reached -0.186) and killing
# three runs at 0/0. Both are fixed; none of the old numbers mean anything.
#
# Four runs, because two constructions each need their own no-OM control at the same depth:
# gate 1 is "beats its own control", and the concat number alone cannot say whether the residual
# or the block-16 tap is doing the work.
#
# SPREAD ACROSS GPUS, not serialised: ~4 h of training each, so serial would be most of a day and
# four at once is about five hours. GPU 1 is ours and takes the headline run; 3, then 0, are the
# next in the agreed order; 2 takes the lightest of the four beside kutub's jobs. Each needs about
# 35 GB and the smallest free block across the four is ~62 GB. Threads stay capped at 4 per job.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
mkdir -p "$L" "$M"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

run () {  # run <gpu> <logname> <args...>
  local gpu=$1 name=$2; shift 2
  [ -f "$M/$name.done" ] && { say "$name already done"; return 0; }
  say "v3 start: $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PYV upgrade/branchfusion/train_om_pecore.py "$@" \
      --model pecore_l14 --lora --augment B --seed 42 --workers 5 >> "$L/$name.log" 2>&1 \
    && { touch "$M/$name.done"; say "$name DONE"; } \
    || say "$name FAILED -- see $L/$name.log"
}

run 1 om_P_b16_cat_v3     --operator bvae --block 16 --om-mode concat                 &
run 3 om_P_b16_control_v3 --operator none --block 16                                 &
run 0 om_S_b16_cat_v3     --operator bvae --block 16 --om-mode concat --temporal seq &
run 2 om_S_b16_control_v3 --operator none --block 16 --temporal seq                  &
wait
say "=== v3 LANE DONE ==="

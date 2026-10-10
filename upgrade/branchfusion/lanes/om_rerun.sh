#!/usr/bin/env bash
# RE-RUN the three jobs that died at "development videos without branch opinions".
#
# P+OM(concat) 0.8296 and P+OM(residual) 0.4942 both finished; what is missing is every CONTROL,
# so gate 1 -- "beats its own no-OM control at the same depth" -- cannot be evaluated for either
# construction, and branch S has no run at all. Without the controls the concat number is
# uninterpretable: it could be the residual helping or the block-16 tap helping.
#
# `install_opinion_retry` now retries a short scoring pass and names the dataset if it stays short,
# so a repeat of last night's failure is diagnosable instead of fatal.
#
# GPU 1 first: kutub has taken 2 and 3 heavily and has only ~14G on 1. Runs are serial because
# three concurrent PE-Core runs is what the box was doing when these passes started coming back
# short.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

run () {  # run <logname> <args...>
  local name=$1; shift
  [ -f "$M/$name.done" ] && { echo "[$(date -Is)] $name already done" >> "$M/orchestrator.log"; return 0; }
  echo "[$(date -Is)] rerun: $name" >> "$M/orchestrator.log"
  $PYV upgrade/branchfusion/train_om_pecore.py "$@" \
      --model pecore_l14 --lora --augment B --seed 42 --workers 6 >> "$L/$name.log" 2>&1 \
    && { touch "$M/$name.done"; echo "[$(date -Is)] $name DONE" >> "$M/orchestrator.log"; } \
    || echo "[$(date -Is)] $name FAILED again -- see $L/$name.log" >> "$M/orchestrator.log"
}

# Gate 1's bar for the P construction, then branch S and its own bar.
run om_P_b16_control_v2 --operator none  --block 16
run om_S_inloop_v2      --operator bvae  --block 16 --temporal seq --om-mode concat
run om_S_control_v2     --operator none  --block 16 --temporal seq

echo "[$(date -Is)] RERUN LANE DONE" >> "$M/orchestrator.log"

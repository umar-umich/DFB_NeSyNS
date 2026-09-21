#!/usr/bin/env bash
# Stage 3 (both routes UMAR chose): four spatial arms on GPU 1, each started once 45 GB are free.
set -uo pipefail
cd /data/umar/Repos/DFB_NeSyNS || exit 2; mkdir -p logs/tbiom/stage3
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
for spec in "wgUB warm_dicome_gate_unfrozenB_seed42 warm_dicome_gate_unfrozenB_seed42.yaml warmgate" \
            "wgUE warm_dicome_gate_unfrozenE_seed42 warm_dicome_gate_unfrozenE_seed42.yaml warmgate" \
            "sbiarmEs1337 sbi_armE_seed1337 sbi_armE_seed1337.yaml sbiarmB" \
            "sbiarmEs7 sbi_armE_seed7 sbi_armE_seed7.yaml sbiarmB"; do
  set -- $spec
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)" -lt 45000 ]; do sleep 120; done
  say "launching $1"
  bash analysis/tbiom/arm_pipeline.sh "$@" > "logs/tbiom/stage3/$1_chain.log" 2>&1 &
  sleep 420
done
wait; say "all four arms finished"; touch logs/tbiom/stage3/ARMS_DONE

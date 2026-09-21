#!/usr/bin/env bash
# Starts the fourth arm (E seed 7) as soon as GPU 1 can hold it, then marks all four arms done for the analysis.
cd /data/umar/Repos/DFB_NeSyNS || exit 2
sleep 300                                              # let the three running trainings reach their working sets
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)" -lt 40000 ]; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] launching sbiarmEs7"
bash analysis/tbiom/arm_pipeline.sh sbiarmEs7 sbi_armE_seed7 sbi_armE_seed7.yaml sbiarmB > logs/tbiom/stage3/sbiarmEs7_chain.log 2>&1
while [ "$(ls logs/tbiom/stage3/{wgUB,wgUE,sbiarmEs1337,sbiarmEs7}_DONE 2>/dev/null | wc -l)" -lt 4 ]; do sleep 120; done
touch logs/tbiom/stage3/ARMS_DONE; echo "[$(date '+%m-%d %H:%M')] all four arms done"

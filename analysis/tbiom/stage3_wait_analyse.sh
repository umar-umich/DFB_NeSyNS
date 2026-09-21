#!/usr/bin/env bash
cd /data/umar/Repos/DFB_NeSyNS || exit 2
while [ ! -f logs/tbiom/stage3/ARMS_DONE ]; do sleep 180; done
for t in wgUB wgUE sbiarmEs1337 sbiarmEs7; do [ -f logs/tbiom/stage3/${t}_DONE ] || { echo "$t did not finish"; exit 1; }; done
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /data/umar/miniconda3/envs/discern_ext/bin/python analysis/tbiom/stage3_analysis.py > logs/tbiom/stage3/analysis.log 2>&1 \
  && echo "EXP_COMBINE_CONFIRM.md written" || echo "analysis FAILED"
touch logs/tbiom/stage3/STAGE3_DONE

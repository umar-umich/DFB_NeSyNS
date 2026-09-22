#!/usr/bin/env bash
# Runs the cross-manipulation analysis once BOTH halves (face swap and reenactment) are exported.
cd /data/umar/Repos/DFB_NeSyNS || exit 2
while [ ! -f logs/tbiom/df40/EXPORTS_DONE ] || [ ! -f logs/tbiom/df40/FR_DONE ]; do sleep 120; done
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /data/umar/miniconda3/envs/discern_ext/bin/python analysis/tbiom/df40_analysis.py \
  > logs/tbiom/df40/analysis.log 2>&1 && echo "EXP_DF40_CROSS_MANIPULATION.md written" || echo "analysis FAILED"
touch logs/tbiom/df40/DF40_DONE

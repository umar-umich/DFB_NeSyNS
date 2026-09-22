#!/usr/bin/env bash
cd /data/umar/Repos/DFB_NeSyNS || exit 2
while [ ! -f logs/tbiom/df40/EXPORTS_DONE ]; do sleep 120; done
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /data/umar/miniconda3/envs/discern_ext/bin/python analysis/tbiom/df40_analysis.py \
  > logs/tbiom/df40/analysis.log 2>&1 && echo "EXP_DF40_CROSS_MANIPULATION.md written" || echo "analysis FAILED"
touch logs/tbiom/df40/DF40_DONE

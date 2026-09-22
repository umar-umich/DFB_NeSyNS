#!/usr/bin/env bash
# Regenerate the decision page once Stage 3 lands, so §3's open decisions carry real numbers.
cd /data/umar/Repos/DFB_NeSyNS || exit 2
while [ ! -f logs/tbiom/stage3/STAGE3_DONE ]; do sleep 300; done
OMP_NUM_THREADS=4 /data/umar/miniconda3/envs/discern_ext/bin/python analysis/tbiom/decision_summary.py

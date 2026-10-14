#!/usr/bin/env bash
# The seven GenD-repo baselines on the three corpora packed on 2026-10-13.
#
# Tables 1 and 2 carried `in progress` for every baseline on UADFV, DeepSpeak-1.1 and
# DeepSpeak-2.0 -- not a rendering artefact and not our rows: those models were simply never run
# on corpora that did not exist when the baseline matrix was built. The runner and the
# checkpoints both exist, so this fills the cells rather than explaining them.
#
# PATHS ARE ABSOLUTE ON PURPOSE. `run_gend_baselines.py` does `os.chdir(GEND)` at import, because
# the GenD checkpoints are relative to that repo -- so a relative --lists_root resolves against
# the GenD tree and the run dies on "no lists for UADFV" with the lists sitting right there.
#
# Frames first: the baselines read PNGs from our h5 crops, so every model sees exactly the videos
# our rows are scored on. ~152k frames, ~7 GB, and /data has about 105 GB free.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYD=/data/umar/miniconda3/envs/discern_ext_timm/bin/python   # h5 -> png, no GenD imports
PY=/data/umar/miniconda3/envs/GenD/bin/python                 # the GenD repo's own env

L=logs/scope_expand/baselines
M=logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
SETS="UADFV DeepSpeak-1.1 DeepSpeak-2.0"
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== baselines on $SETS: dumping frames ==="
for d in $SETS; do
  $PYD analysis/tbiom/baselines/dump_h5_frames.py "$d" 16 >> "$L/dump.log" 2>&1 \
    && say "  frames dumped: $d" || { say "  frame dump FAILED: $d"; exit 1; }
done

say "=== baselines: inference, 7 models x 3 corpora ==="
CUDA_VISIBLE_DEVICES=1 $PY analysis/tbiom/baselines/run_gend_baselines.py \
    --datasets $SETS \
    --lists_root /data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_lists_ours \
    --run_dir /data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_runs_ours \
    >> "$L/run.log" 2>&1 \
  && say "=== baselines done ===" || say "=== baselines FAILED -- see $L/run.log ==="

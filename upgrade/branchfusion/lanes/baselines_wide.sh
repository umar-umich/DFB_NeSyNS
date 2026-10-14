#!/usr/bin/env bash
# The seven GenD-repo baselines on the WIDENED CDFv3 and KoDF census.
#
# Separate from `baselines_new.sh`, which fills the three corpora that were simply never run.
# This fixes a different defect: every baseline's CDFv3 cell covered 5,118 videos against our
# 5,418, and every KoDF cell 1,509 against our 2,239 -- so Table 1 and Table 2 compared rows
# measured on different populations. Two causes, both now removed:
#   * `dump_h5_frames.py` skipped CDFv3's IP_LAP outright, so those frames were never on disk.
#   * KoDF's baseline lists never held `adv-real`/`adv-fake`.
#
# KoDF MOVES TO OUR CROPS. Its baseline cells came from `gend_runs` -- GenD's own crops -- and
# the adversarial subsets exist only in our h5. Mixing two crop sources inside one dataset would
# be worse than re-running it, so KoDF is dumped from our h5 like every other set and
# `sota_table1.GEND_CROPS` drops it. FFIW, IDForge and DFDM already match our census and stay.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYD=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PY=/data/umar/miniconda3/envs/GenD/bin/python
L=logs/scope_expand/baselines
M=logs/branchfusion/orchestrator
mkdir -p "$L"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
R=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== baselines, widened census: dumping CDFv3 (with IP_LAP) and KoDF (with adv) ==="
for d in CDFv3 KoDF; do
  $PYD analysis/tbiom/baselines/dump_h5_frames.py "$d" 16 >> "$L/dump_wide.log" 2>&1 \
    && say "  frames dumped: $d" || { say "  frame dump FAILED: $d"; exit 1; }
done

# The existing CDFv3 runs are at the narrow census and must not be read as the new ones.
for m in GenD_CLIP GenD_PE GenD_DINO FSFM FSFM_Adapter Effort ForAda; do
  for d in CDFv3 KoDF; do
    old="$R/gend_runs_ours/${m}--${d}"
    [ -d "$old" ] && mv "$old" "$R/gend_runs_ours/${m}--${d}.v31" 2>/dev/null
  done
done
say "  previous narrow-census runs kept as *.v31"

say "=== baselines, widened census: inference ==="
CUDA_VISIBLE_DEVICES=1 $PY analysis/tbiom/baselines/run_gend_baselines.py \
    --datasets CDFv3 KoDF \
    --lists_root "$R/gend_lists_ours" --run_dir "$R/gend_runs_ours" \
    >> "$L/run_wide.log" 2>&1 \
  && say "=== baselines widened census done ===" || say "=== baselines widened FAILED ==="

#!/usr/bin/env bash
# ADAPTIVE depth search for the encoder-in-the-loop OM on the PE spatial trunk. GPU 1.
#
# UMAR's rule, 2026-10-09: go to block 12; if it improves, go to block 8; if it drops, probe the
# other side instead. For THIS trunk the other side is already measured, so the rule collapses:
#
#     b16  0.8673   (done)      b20  0.8613   (done, worst)      last  0.8664   (done)
#
# so block 20 needs no re-run and the only live question is whether EARLIER is better. Run b12; if
# it beats b16, run b8; if it does not, the trunk has been bracketed on both sides and the search
# stops rather than spending three more GPU-hours confirming a known loser.
#
# The mechanism being tested: deeper blocks are more semantic, and a reconstruction residual has
# most to find where the representation still carries texture and frequency structure. If that is
# right, b12 > b16. If b12 < b16 as well, the residual is not depth-limited and the whole depth
# axis is closed for this construction.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
OUT=analysis/upgrade/om
LOG=logs/branchfusion/lanes
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

score () {  # score <readout> -> full-dev residual v3.1, or "nan"
  $PY - "$1" <<'PYEOF'
import json, sys, pathlib
t = sys.argv[1]
f = pathlib.Path("analysis/upgrade/om") / (
    "om_pe_core_l_LN_bvae_s0.json" if t == "last" else f"om_pe_core_l_LN_bvae_s0_{t}.json")
print(f"{json.load(open(f))['full_dev']['r']['dev_v31']:.4f}" if f.exists() else "nan")
PYEOF
}

run () {
  echo "=== $1 start $(date -Is)"
  $PY upgrade/adapt_om.py --backbone pe_core_l --recipe LN --operator bvae \
      --seed 0 --readout "$1" --select r >> "$LOG/om_eil_$1.log" 2>&1 \
    && echo "=== $1 done $(date -Is)" || { echo "=== $1 FAILED"; return 1; }
}

B16=$(score b16)
echo "reference: b16 $B16"
run b12 || exit 1
B12=$(score b12)
echo "b12 $B12 against b16 $B16"
if $PY -c "import sys; sys.exit(0 if float('$B12') > float('$B16') else 1)"; then
  echo "b12 IMPROVES on b16 -- going earlier, to block 8"
  run b8
  echo "b8 $(score b8)"
else
  echo "b12 does NOT improve on b16, and b20 ($(score b20)) was already the worst of the three."
  echo "The trunk is bracketed on both sides; the depth axis is closed for this construction."
fi
echo "OM_DEPTH_SEARCH_DONE $(date -Is)"

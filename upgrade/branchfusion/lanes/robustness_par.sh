#!/usr/bin/env bash
# The robustness sweep, PARALLEL across GPUs instead of 52 passes in single file.
#
# The serial version put one extraction on one GPU and left the other three idle; at ~1.6 min a
# pass that is well over an hour of wall time spent waiting rather than computing. Each pass needs
# about 4 GB and the GPUs have 80-140 GB free, so the only reason to run them one at a time was
# that nothing told them not to.
#
# GPU 1 and GPU 3 carry six workers each and GPU 0 takes four, per UMAR: GPU 1 and 3 at maximum,
# GPU 0 as the overflow. Threads stay capped at 4 per job.
#
# IDEMPOTENT. `extract_adapted` returns early on an existing cache and writes through a .partial
# file replaced atomically, so a cache on disk is complete and a re-run costs one line of output.
# That makes this lane safe to restart and safe to run over the work the serial version finished.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
CK=logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt
SETS="VALmix VALmixV2 FFpp_val CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM"
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

jobs=$L/robustness_jobs.txt
: > "$jobs"
i=0
for PROC in "jpeg:40" "resize:0.6"; do
  for DS in $SETS; do
    for TRUNK in A Y16; do
      # Six slots on 1, six on 3, four on 0 -- the cycle below lands jobs in that ratio.
      case $((i % 16)) in
        0|1|2|3|4|5)     G=1 ;;
        6|7|8|9|10|11)   G=3 ;;
        *)               G=0 ;;
      esac
      echo "$G|$TRUNK|$PROC|$DS" >> "$jobs"
      i=$((i + 1))
    done
  done
done
say "=== robustness (parallel): $(wc -l < "$jobs") passes across GPUs 1, 3 and 0 ==="

run_one () {
  local spec=$1 g trunk proc ds
  IFS='|' read -r g trunk proc ds <<< "$spec"
  local args=(--backbone pe_core_l --recipe LN --dataset "$ds" --process "$proc")
  [ "$trunk" = "Y16" ] && args+=(--ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt
                                 --readout b16 --cache-tag omEIL_b16)
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    /data/umar/miniconda3/envs/discern_ext_timm/bin/python upgrade/extract_adapted.py "${args[@]}" \
    >> "logs/branchfusion/lanes/robustness_par.log" 2>&1 \
    || echo "[$(date -Is)] FAILED $spec" >> "logs/branchfusion/orchestrator/orchestrator.log"
}
export -f run_one

xargs -a "$jobs" -I{} -P 16 bash -c 'run_one "$@"' _ {}
say "robustness (parallel): extraction done; exporting Y16 opinions"

# The cache suffix is exactly what `extract_adapted` builds: process.replace(":", ""), so the
# DOT survives -- jpeg:40 -> jpeg40 but resize:0.6 -> resize0.6, not rs06.
for TAG in jpeg40 resize0.6; do
  $PY upgrade/branchfusion/om_eil_export.py --ckpt "$CK" --cache "omEIL_b16_${TAG}" \
      --label "Y16_${TAG}" >> "$L/robustness_par.log" 2>&1 \
    || say "robustness (parallel): Y16 export failed for $TAG"
done
$PY upgrade/branchfusion/robustness.py >> "$L/robustness_par.log" 2>&1 || say "robustness report failed"
say "=== robustness (parallel) done ==="

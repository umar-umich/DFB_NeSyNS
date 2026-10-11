#!/usr/bin/env bash
# EXTENSION: the jpeg:40 development slice separated Y16 from A (phi +0.706 -> +0.528), so the
# cheap slice did its job and the full comparison is now worth its cost. Adds resize:0.6 on the
# development sets and BOTH conditions on the ten test sets, which is what a table row needs.
#
# At ~1.6 min a pass this is hours, not the day the original scoping assumed.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }
log=$L/robustness_Y16_full.log
CK=logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt

disk_ok () {
  local free; free=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge 40 ] || { say "robustness-full: only ${free}G free -- stopping"; return 1; }
}

say "=== robustness lane (full): both conditions, development and test ==="
for PROC in "jpeg:40" "resize:0.6"; do
  for DS in VALmix VALmixV2 FFpp_val CDFv2 CDFv3 DFD DFDC DFDCP DFEval24 FFIW KoDF IDForge DFDM; do
    disk_ok || exit 1
    $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --dataset "$DS" \
        --process "$PROC" >> "$log" 2>&1 || say "robustness-full: A $PROC failed on $DS"
    $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --dataset "$DS" \
        --ckpt "$CK" --readout b16 --cache-tag omEIL_b16 --process "$PROC" >> "$log" 2>&1 \
      || say "robustness-full: Y16 $PROC failed on $DS"
  done
  TAG=$(echo "$PROC" | tr -d ':.' | sed 's/resize0/rs0/')
  $PY upgrade/branchfusion/om_eil_export.py --ckpt "$CK" \
      --cache "omEIL_b16_${TAG}" --label "Y16${TAG}" >> "$log" 2>&1 \
    || say "robustness-full: export failed for $PROC"
done
say "=== robustness lane (full) done; caches and Y16 exports ready for the report ==="

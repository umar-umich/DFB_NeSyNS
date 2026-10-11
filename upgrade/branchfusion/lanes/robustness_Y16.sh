#!/usr/bin/env bash
# DOES Y16 EARN ITS ROW UNDER DEGRADATION? The one open question about keeping it.
#
# Y16 is a near-duplicate of A on CLEAN data: phi +0.678 development, +0.669 test, against a 0.60
# gate. UMAR's position is that a branch which merely EQUALS A can still be worth keeping when it
# is built differently, because it may hold where A does not -- and that is a hypothesis with a
# measurement attached, not a matter of taste. phi on clean data cannot settle it: two branches can
# fail on the same clean videos and come apart entirely under compression.
#
# So: re-extract both trunks under jpeg:40, replay both heads, and compare.
#
#   * if phi(Y16, A) FALLS under degradation, the two views separate exactly where robustness
#     matters, and keeping Y16 is supported by evidence rather than by hope;
#   * if phi stays at +0.67, Y16 duplicates A under attack too, and the case for the row is
#     novelty alone -- which is UMAR's call to make, but should be made knowing that.
#
# DEVELOPMENT SETS ONLY, and jpeg:40 only. That is the cheapest slice that can answer it: six
# extraction passes instead of forty. resize:0.6 and the test sets follow only if this separates.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
#: `extract_adapted` builds the backbone through `adapt_backbone`, which imports timm -- absent
#: from discern_ext. The extraction runs in the timm env; the opinion replay and the report stay
#: in discern_ext, which is where every other analysis in this campaign runs.
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }
log=$L/robustness_Y16.log

disk_ok () {                       # the box sat at 99% through this campaign
  local free; free=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge 40 ] || { say "robustness: only ${free}G free -- stopping"; return 1; }
}

say "=== robustness lane: Y16 vs A under jpeg:40 ==="
for DS in VALmix VALmixV2 FFpp_val; do
  disk_ok || exit 1
  # A's trunk, the appearance view's own features.
  $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --dataset "$DS" \
      --process jpeg:40 >> "$log" 2>&1 || { say "robustness: A extract failed on $DS"; exit 1; }
  # Y16's trunk: the adapters trained WITH the beta-VAE, tapped at block 16, same as training.
  $PYT upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --dataset "$DS" \
      --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt --readout b16 \
      --cache-tag omEIL_b16 --process jpeg:40 >> "$log" 2>&1 \
    || { say "robustness: Y16 extract failed on $DS"; exit 1; }
done

$PY upgrade/branchfusion/om_eil_export.py --ckpt logs/upgrade/om/om_pe_core_l_LN_bvae_s0_b16.pt \
    --cache omEIL_b16_jpeg40 --label Y16j40 >> "$log" 2>&1 \
  || { say "robustness: Y16 opinion export failed"; exit 1; }
$PY upgrade/branchfusion/robustness.py >> "$log" 2>&1 || say "robustness: report failed"
say "=== robustness lane done -- see reports/branch_fusion/ROBUSTNESS_Y16.md ==="

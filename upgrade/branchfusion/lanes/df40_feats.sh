#!/usr/bin/env bash
# App's feature cache for DF40's individual generators.
#
#   bash upgrade/branchfusion/lanes/df40_feats.sh "uniface blendface mobileswap e4s ..."
#
# WHY ONLY APP. Of the framework's four branches, App is the only one that reads a FEATURE CACHE:
# `branches._head_opinions` runs `pe_core_lLN_<set>.npz` through the trained evidential head. The
# other three do not.
#   TD    already has DF40 opinions for all 41 generator sets from the 2026-09-22 campaign.
#   LFS   reads the stage-23 export, which that campaign wrote as `ours_DF40_<gen>_<half>.csv` --
#         the same checkpoint and the same columns as `warmgate_sel_*`; see `branches.sel_csv`.
#   STT   runs the PE-Core trunk over CLIPS inside `train_branch.opinions`, tapping block 16 in the
#         forward. It never opens a cache, so the `pe_core_lLN-b16_*` extraction this lane used to
#         run alongside was pure waste -- 20 GPU-minutes producing a file no reader would open.
#         Its export is `lanes/df40_stt.sh`, which goes through the trunk as training did.
set -u
cd /data/umar/Repos/DFB_NeSyNS
GENS=${1:-"uniface blendface mobileswap e4s facedancer fsgan inswap simswap"}
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
L=logs/scope_expand/df40
mkdir -p "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$L/feats.log"; }

# A generator with no `<gen>-<half>.txt` split is named without a half -- DeepFaceLab ships one
# source and its own real pool. `adapt_backbone.df40_manifests` resolves both forms; this picks
# the names that exist so the lane does not log a failure for a set that was never packed.
SETS=""
for g in $GENS; do
  for h in cdf ff; do
    [ -f "/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits/DF40/$g-$h.txt" ] \
      && SETS="$SETS DF40_${g}_${h}"
  done
  [ -f "/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits/DF40/$g.txt" ] \
    && SETS="$SETS DF40_${g}"
done
say "=== DF40 App features: $(echo $SETS | wc -w) sets ==="

for d in $SETS; do
  CUDA_VISIBLE_DEVICES=1 $PY upgrade/extract_adapted.py --backbone pe_core_l --recipe LN \
    --dataset "$d" >> "$L/app.log" 2>&1 && say "App $d ok" || say "App $d FAILED"
done
say "=== DF40 App features finished ==="

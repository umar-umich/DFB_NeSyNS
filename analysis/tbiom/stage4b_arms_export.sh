#!/usr/bin/env bash
# Held-out processed exports for each trained arm, so Stage 4's processing test can be rerun
# per arm rather than only on the frozen arm B.
#
# Only what the test needs: the dev baseline (for tau) and the two held-out severities on the
# two small OOD sets. The full fit grid is NOT re-run per arm -- those curves exist to define
# reliability, and this test only needs unseen conditions.
set -uo pipefail
EXT=/data/umar/Repos/DISCERN_Ext
OUT=/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage2
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
export CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "$OUT"; cd "$EXT" || exit 2

run() {  # <prefix> <run_dir> <dataset> <spec> <tag>
  local pre=$1 rd=$2 ds=$3 spec=$4 tag=$5
  local out="$OUT/p_${pre}_${ds}_${tag}.csv"
  [ -f "$out" ] && { echo "skip $pre/$ds/$tag"; return 0; }
  local ck; ck=$(ls -t "runs/discern_ext_train/${rd}/checkpoints/"*best*.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "  no checkpoint for $rd"; return 1; }
  local arg=""; [ "$spec" != "none" ] && arg="--process $spec"
  echo "[$(date +%H:%M)] $pre / $ds / $tag"
  $PY eval_adaptation/export_views.py --checkpoint "$ck" \
      --config "eval_adaptation/configs/sbiarm${pre#arm}_${ds}.yaml" --dataset "$ds" \
      --device cuda:0 --batch-size 64 $arg --out "$out" \
      >> "$OUT/${pre}_${ds}_${tag}.log" 2>&1 || echo "  !! FAILED $pre/$ds/$tag"
}

for A in A B C; do
  run "arm$A" "sbi_arm${A}_seed42" FFpp_val none base
  for ds in CDFv2 DFDCP; do
    run "arm$A" "sbi_arm${A}_seed42" "$ds" none         base
    run "arm$A" "sbi_arm${A}_seed42" "$ds" "jpeg:40"    jpeg40
    run "arm$A" "sbi_arm${A}_seed42" "$ds" "resize:0.6" rs06
  done
done
echo "[$(date +%H:%M)] per-arm held-out exports done"

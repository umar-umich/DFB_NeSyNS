#!/bin/bash
# Lane 5: everything that waits on an extraction. Restartable -- each step skips if its output exists.
# GPU priority 1, then 3, then 0. Never GPU 2 (kutub).
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU="${1:-1}"
L=logs/branchfusion/lanes

wait_for () {  # wait_for <file> <pattern> <label> <max_minutes>
  local f="$1" pat="$2" lab="$3" max="${4:-240}" n=0
  while ! grep -q "$pat" "$f" 2>/dev/null; do
    n=$((n+1)); [ $((n % 20)) -eq 0 ] && echo "  waiting on $lab ($((n/2)) min)"
    [ $n -gt $((max*2)) ] && { echo "TIMEOUT waiting on $lab"; return 1; }
    sleep 30
  done
  echo "  $lab ready"; return 0
}

echo "=== [1] wait for R extraction $(date -u +%FT%TZ) ==="
wait_for $L/lane2_patch.log "LANE2 DONE" "R patch extraction" 300 || true

echo "=== [2] register R caches and train its head and control $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/extract_patch.py --register || echo FAILED_register
for bb in Rcov Rmean; do
  if [ ! -f "analysis/upgrade/heads/view_${bb}_s0.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/train_head.py --backbone $bb --seed 0 || echo "FAILED head $bb"
  else
    echo "  view_${bb}_s0.pt exists"
  fi
done

echo "=== [3] section 2.3 keep rule for R $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/extract_patch.py --gate
R_IN=$?; echo "  R gate exit $R_IN (0 = R enters the subset search)"

echo "=== [4] wait for dense features $(date -u +%FT%TZ) ==="
wait_for $L/lane3_dense2.log "LANE3 EXTRACT DONE" "dense features" 180 || true

echo "=== [5] saturation report, then every S variant $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --saturation || echo FAILED_sat
# The specified S, both input variants, on the deployed sparse frames.
for v in adapt frozen; do
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --variant $v --seed 0 \
      || echo "FAILED S-$v"
done
# The dense pilot: the same readouts on stride-1 consecutive frames. This is the variant that tests
# whether P's order-invariance comes from frame SPACING rather than from the architecture.
for v in dense dense-mid; do
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --variant $v --seed 0 \
      || echo "FAILED S-$v"
done
# The order-discrimination intervention, dense and sparse, so the two can be compared.
for v in dense adapt; do
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --variant $v --seed 0 \
      --order-loss 0.3 || echo "FAILED S-$v-ord"
done
# Section 7 controls 5 and 6: positional encoding removed, transformer replaced.
for extra in "--no-pos" "--pool mean" "--pool mlp"; do
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --variant adapt --seed 0 \
      $extra || echo "FAILED S-adapt $extra"
done

echo "=== [6] export the best S $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/branchfusion/seq_branch.py --export auto || echo FAILED_export

echo "=== [7] re-screen with S and R included $(date -u +%FT%TZ) ==="
BR=A,S,V,O,M,P
[ $R_IN -eq 0 ] && BR=$BR,R
echo "  branches: $BR"
export CUDA_VISIBLE_DEVICES=""
$PY upgrade/branchfusion/subsets.py --screen --branches $BR --sizes 1,2,3 || echo FAILED_screen2
$PY upgrade/branchfusion/subsets.py --nested --branches $BR || echo FAILED_nested2

echo "=== [8] sequence controls on every trained S $(date -u +%FT%TZ) ==="
TAGS=$(ls logs/branchfusion/seq/S-*_s0.json 2>/dev/null | xargs -n1 basename 2>/dev/null \
       | sed 's/\.json$//' | paste -sd, -)
[ -n "$TAGS" ] && $PY upgrade/branchfusion/temporal_controls.py --branches V,M,P --seq "$TAGS" \
       --negative-control || echo "no S tags yet"

echo "=== [9] fusion matrix and uncertainty on the shortlist $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/fusion_matrix.py --branches $BR \
    --subsets A+V+O,A+S+V,A+V+P,A+V+O+M+P --ops all || echo FAILED_fusion2
$PY upgrade/branchfusion/uncertainty.py --subset A+V+P --schedules all --selective \
    || echo FAILED_uncertainty

echo "=== [10] final tables $(date -u +%FT%TZ) ==="
$PY upgrade/branchfusion/tables.py --all || echo FAILED_tables
echo "LANE5 DONE $(date -u +%FT%TZ)"

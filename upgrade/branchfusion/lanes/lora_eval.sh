#!/bin/bash
# Section 8's second half: evaluate each swept LoRA cell STANDALONE and by its FUSED contribution in
# one fixed reference subset (B0's triple with A swapped in). The standalone numbers are already in
# each adapt run's JSON; this produces the fused ones, which need the cell's own feature caches.
#
# Waits for P-dense to clear GPU 1 first, so two ~50 GB jobs never overlap. Per the brief, the subset
# and operator search is NOT re-run per cell -- `screen_eval.py` scores the one fixed subset.
# setsid-detached.
set -u
PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
cd /data/umar/Repos/DFB_NeSyNS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU=1

echo "=== waiting for the GPU 1 heavy queue to clear $(date -u +%FT%TZ) ==="
n=0
while pgrep -u mufarooq -f "train_p_dense.py --train" >/dev/null 2>&1; do
  n=$((n+1)); [ $((n % 20)) -eq 0 ] && echo "  still waiting ($((n/2)) min)"
  [ $n -gt 600 ] && { echo "TIMEOUT"; break; }
  sleep 30
done
echo "  GPU 1 clear of P-dense"

# cell: r alpha dropout -> checkpoint suffix (empty for the centre, which is B0 and already evaluated)
cells="16 32 0.1
4 8 0.1
2 4 0.1
8 16 0
8 16 0.2
8 32 0.1
8 8 0.1
8 16 0.05
8 4 0.1"

echo "$cells" | while read r al dr; do
  sfx="_r${r}a${al}d${dr}"
  tag="pe_core_lLN${sfx}"
  if [ ! -f "analysis/upgrade/adapt/pe_core_l_LN_s0${sfx}.json" ]; then
    echo "=== skip $sfx: adaptation not finished"; continue
  fi
  echo "=== $sfx $(date -u +%FT%TZ) ==="
  for ds in FFpp_train FFpp_val VALmix VALmixV2; do
    BF_LORA_R=$r BF_LORA_ALPHA=$al BF_LORA_DROPOUT=$dr CUDA_VISIBLE_DEVICES=$GPU \
      $PY upgrade/extract_adapted.py --backbone pe_core_l --recipe LN --seed 0 \
      --ckpt-suffix "$sfx" --dataset $ds --batch-size 64 || echo "FAILED extract $sfx $ds"
  done
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/train_head.py --backbone "$tag" --seed 0 \
    || echo "FAILED head $sfx"
  CUDA_VISIBLE_DEVICES=$GPU $PY upgrade/screen_eval.py --backbone "$tag" --seeds 1 \
    || echo "FAILED screen $sfx"
done

echo "=== rebuild the LoRA table and the final report $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" $PY upgrade/branchfusion/tables.py --all || echo FAILED_tables
echo "LORA_EVAL_DONE $(date -u +%FT%TZ)"

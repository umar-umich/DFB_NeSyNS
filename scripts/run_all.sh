#!/usr/bin/env bash
# =============================================================================
# Single-command driver for the DeFakeNet ablation pipeline.
# Update the ABLATIONS list below if your checkpoint paths differ from the
# defaults under `checkpoints/<name>.pth`.
# =============================================================================
set -u

ABLATIONS=(
  "full_defakenet:checkpoints/defakenet_full.pth"
  "no_ibdc:checkpoints/ablation_no_ibdc.pth"
  "no_cmef:checkpoints/ablation_no_cmef.pth"
  "no_pbas:checkpoints/ablation_no_pbas.pth"
  "no_causal:checkpoints/ablation_no_causal.pth"
  "no_symbolic:checkpoints/ablation_no_symbolic.pth"
  "visual_edl_only:checkpoints/ablation_visual_edl_only.pth"
)

mkdir -p results/faithfulness results/selective

for entry in "${ABLATIONS[@]}"; do
  name="${entry%%:*}"
  ckpt="${entry##*:}"
  mkdir -p results/ablations/$name
  python scripts/run_ablation_eval.py \
    --ablation-name $name --checkpoint-path $ckpt \
    > results/ablations/$name/stdout.log \
    2> results/ablations/$name/stderr.log \
    || echo "FAILED: $name (continuing)"
done

python scripts/run_faithfulness.py \
  > results/faithfulness/stdout.log 2>&1 \
  || echo "FAILED: faithfulness"

python scripts/run_selective.py \
  > results/selective/stdout.log 2>&1 \
  || echo "FAILED: selective"

python scripts/aggregate.py
echo "DONE. See results/ABLATION_SUMMARY.md"

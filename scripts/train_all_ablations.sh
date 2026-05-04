#!/usr/bin/env bash
# =============================================================================
# DeFakeNet ablation training launcher.
#
# Each entry trains a single ablation variant from the corresponding YAML
# in configs/ablations/. Order is by paper-priority — the most load-bearing
# rows for the headline claim run first, so an interrupted training run
# leaves the most-important variants complete.
#
# CONFIG PATHS:
#   configs/ablations/<name>.yaml   — one per ablation
# CHECKPOINT LOCATION:
#   training/train.py auto-derives the output dir as
#   logs/train/<config-basename>_<timestamp>_exp/best_avg.pth
#   After each successful run, copy/symlink the resulting best_avg.pth to
#   checkpoints/<name>.pth before running scripts/run_all.sh evaluation.
#
# PRE-REQUISITE PATCHES (apply once, before this launcher):
#   patches/ablation_no_cmef.diff       — adds cmef_disable_modulation flag
#   patches/ablation_no_symbolic.diff   — adds concept_branch.disabled flag
# Apply with:   cd <repo_root> && patch -p1 < patches/<name>.diff
# =============================================================================
set -u

CONFIGS=(
  "full_defakenet_18rules"   # 1. NEW headline checkpoint (12-rule old ckpt incompatible)
  "no_ibdc"                  # 2. headline-claim ablation
  "no_symbolic"              # 3. component contribution
  "no_causal"                # 4. component contribution
  "no_cmef"                  # 5. supporting mechanism
  "no_pbas"                  # 6. supporting mechanism
  "visual_edl_only"          # 7. anchor baseline
)

mkdir -p logs/ablations checkpoints

for config in "${CONFIGS[@]}"; do
  echo "=================================================="
  echo "Training: $config"
  echo "Started:  $(date)"
  echo "=================================================="

  python training/train.py \
    --detector_path configs/ablations/${config}.yaml \
    > logs/ablations/${config}.log 2>&1

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED: $config (rc=$rc). Continuing."
  else
    echo "OK: $config trained. See logs/ablations/${config}.log."
    echo "    Copy/symlink the resulting best_avg.pth to checkpoints/${config}.pth"
    echo "    before running scripts/run_all.sh."
  fi
done

echo "=================================================="
echo "All training runs dispatched. Finished: $(date)"
echo "=================================================="

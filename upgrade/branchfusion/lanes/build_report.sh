#!/usr/bin/env bash
set -u
cd /data/umar/Repos/DFB_NeSyNS
{
  echo "# Proposal results — all tables, every dataset"; echo
  echo "Regenerated $(date -Is). Framework: **App + TD + LFS + STT**, fused with PCR6+-R."; echo
  echo "| id | thesis name | what it is |"; echo "|---|---|---|"
  echo "| A | **App** | Appearance — PE-Core-L/336, LoRA + LayerNorm, frame mean |"
  echo "| V | **TD** | Temporal Dynamics — V-JEPA 2 LoRA, 16 frames |"
  echo "| O | **LFS** | Latent Facial Structure — CLIP/β-VAE orthogonal residual |"
  echo "| S16ctl | **STT** | Spatio-Temporal Texture — PE-Core block 16 + real relative timestamps |"
  echo "| P16ctl | **STT-order** | the same with learned slot positions |"
  echo
  echo "Figure: \`reports/branch_fusion/figures/uncertainty_riskcoverage.png\`"; echo
  for t in TABLE_A_ABLATION TABLE_B_FUSION TABLE1_DETECTION TABLE2_CALIBRATION; do
    echo "---"; echo; cat "reports/branch_fusion/$t.md"; echo
  done
} > reports/branch_fusion/PROPOSAL_TABLES.md

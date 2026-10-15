#!/usr/bin/env bash
# One consolidated file from the per-table generators. Assembly only -- it recomputes nothing,
# so running it after a rename or a reword cannot move a number.
#
# THE SCOPE PICKS THE OUTPUT DIRECTORY, through the same `C.REPORT` the generators write to, so a
# wider census cannot overwrite the v3.1 report with numbers over a different set of videos.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
R=$($PY -c "import sys; sys.path.insert(0,'upgrade/branchfusion'); import common as C; print(C.REPORT.relative_to(C.ROOT))")
NOTE=$($PY -c "import sys; sys.path.insert(0,'upgrade/branchfusion'); import common as C; print(C.scope_note())")
for t in TABLE_A_ABLATION PER_DATASET_CORE5 TABLE_B_FUSION TABLE1_DETECTION TABLE2_CALIBRATION; do
  [ -f "$R/$t.md" ] || { echo "build_report: missing $R/$t.md" >&2; exit 1; }
done
{
  echo "# Proposal results — all tables, every dataset"; echo
  echo "Regenerated $(date -Is). Framework: **App + TD + LFS + STT**, fused with ES-PCR6+."; echo
  echo "**Evaluation scope \`${BF_SCOPE:-v31}\`** — $NOTE."
  echo "The scope changes the video census of CDFv3 and KoDF only; the other ten sets are unaffected."
  echo "A number here is comparable to one in another scope's report only where the census matches."
  echo
  echo "| id | thesis name | what it is |"; echo "|---|---|---|"
  echo "| A | **App** | Appearance — PE-Core-L/336, LoRA + LayerNorm, frame mean |"
  echo "| V | **TD** | Temporal Dynamics — V-JEPA 2 LoRA, 16 frames |"
  echo "| O | **LFS** | Latent Facial Structure — CLIP/β-VAE orthogonal residual |"
  echo "| S16ctl | **STT** | Spatio-Temporal Texture — PE-Core block 16 + real relative timestamps |"
  echo "| P16ctl | **STT-order** | the same with learned slot positions |"
  echo
  echo "Twelve test sets. Every number is video level. Weights are fitted once on the"
  echo "development pool (VALmix, VALmixV2, FF++ val) and applied unchanged to every test set."
  echo
  echo "## How to read the mean columns"; echo
  echo "Tables 1 and 2 carry three. A row's own \`mean\` is over that row's coverage and is NOT"
  echo "comparable across rows — ours spans twelve sets, DiCoME five. \`mean (all)\` and"
  echo "\`mean (wide)\` are matched supports and are the only columns a ranking may be read from."
  echo "Each section states which datasets each one covers."
  echo
  echo "Figure: \`$R/figures/uncertainty_riskcoverage.png\`"; echo
  echo "| table | what it answers |"; echo "|---|---|"
  echo "| A | what each branch and each hand-picked group contributes |"
  echo "| A2 | the same question settled by exhaustion — all 31 non-empty combinations of the five |"
  echo "| B | which decision rule to fuse with, the four-branch set held fixed |"
  echo "| 1 | where the framework stands against the field, on detection |"
  echo "| 2 | and on probability quality and uncertainty |"
  echo
  for t in TABLE_A_ABLATION PER_DATASET_CORE5 TABLE_B_FUSION TABLE1_DETECTION TABLE2_CALIBRATION; do
    echo "---"; echo
    if [ "$t" = PER_DATASET_CORE5 ]; then
      echo "# Table A2 — all 31 combinations of the five branches"; echo
      echo "The exhaustive version of Table A: every non-empty subset of {App, TD, LFS, STT-order,"
      echo "STT}, each one fitted on development and scored on all twelve test sets. It is here so"
      echo "the ten hand-picked groups in Table A cannot be read as a favourable selection — the"
      echo "best cell in this grid IS the framework."; echo
      # the grid's own title and the exploratory banner would restate the heading above
      sed -e '1,2d' "$R/$t.md"
    else
      cat "$R/$t.md"
    fi
    echo
  done
} > "$R/PROPOSAL_TABLES.md"
echo "-> $R/PROPOSAL_TABLES.md"

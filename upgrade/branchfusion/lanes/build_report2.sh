#!/usr/bin/env bash
# PROPOSAL_TABLES_2.md -- the proposal's second results pass, on ONE census.
#
#   bash upgrade/branchfusion/lanes/build_report2.sh            # nolipadv, the default
#   bash upgrade/branchfusion/lanes/build_report2.sh nolip
#
# UMAR took the adversarial subsets on 2026-10-14, so the report carries that census alone. The
# `scope_nolip/` tree is still generated and still on disk; it is simply not reprinted here,
# because two censuses side by side invite a cell from one to be quoted against a cell from the
# other. `SCOPE_COMPARISON.md` remains the one place both are shown, which is its whole job.
#
# Assembly only. Every number is produced by the generator that owns it and copied verbatim.
# PROPOSAL_TABLES_1.md is the v3.1 report and is NOT touched.
set -u
cd /data/umar/Repos/DFB_NeSyNS
SCOPE=${1:-nolipadv}
R=tbiom/reports/branch_fusion
S=$R/scope_$SCOPE
TABLES="TABLE_A_ABLATION PER_DATASET_CORE5 TABLE_B_FUSION TABLE1_DETECTION TABLE2_CALIBRATION"
for f in "$R/SCOPE_COMPARISON.md" "$R/DF40_CROSSMANIP.md" "$R/CHOOSE_THREE.md"; do
  [ -f "$f" ] || { echo "build_report2: missing $f" >&2; exit 1; }
done
for t in $TABLES; do
  [ -f "$S/$t.md" ] || { echo "build_report2: missing $S/$t.md" >&2; exit 1; }
done

{
  echo "# Proposal results, pass 2"; echo
  echo "Regenerated $(date -Is). Census: **\`$SCOPE\`** — the lip-sync / audio-driven exclusion"
  echo "removed, and the adversarial subsets admitted. CDFv3 **5,418** videos, KoDF **2,239**."; echo
  echo "\`PROPOSAL_TABLES_1.md\` holds the protocol-v3.1 census and is unchanged; this file is the"
  echo "widened one. Only CDFv3 and KoDF differ between them — the other ten test sets contain"
  echo "none of the excluded families, so their cells are identical in both."; echo
  echo "| part | what it answers |"; echo "|---|---|"
  echo "| I | what admitting each family cost, and where the false alarms come from |"
  echo "| II | which three branches to take forward, if three is the number |"
  echo "| III | every table on this census |"
  echo "| IV | DF40 cross-manipulation — the structure of DiCoME Table 1's second half |"
  echo
  echo "## One caveat, and it decides which table to quote where"; echo
  echo "**The baselines were never run on the admitted videos.** The GenD runs' prediction files"
  echo "stop at the v3.1 census for CDFv3 and at the no-adversarial census for KoDF, and DiCoME's"
  echo "CDFv3 export covers 1,918 videos against our 5,418. Widening the census widens OUR row and"
  echo "leaves theirs where it was; Table 1 below names every cell this affects."; echo
  echo "So: quote **Table 1 from \`PROPOSAL_TABLES_1.md\`** for the comparison against the field,"
  echo "where every cell is on one census, and the tables here for everything about our own"
  echo "system — the ablation, the fusion operator, calibration, and DF40."; echo
  echo "---"; echo
  echo "# Part I — what the census costs"; echo
  sed -e '1,2d' "$R/SCOPE_COMPARISON.md"
  echo; echo "---"; echo
  echo "# Part II — choosing three branches"; echo
  sed -e '1,2d' -e 's/^## /### /' "$R/CHOOSE_THREE.md"
  echo; echo "---"; echo
  echo "# Part III — every table on this census"; echo
  echo "Risk-coverage figure: \`$S/figures/uncertainty_riskcoverage.png\`"; echo
  [ -f "$S/FIGURE_NOTES.md" ] && { sed -e 's/^# /## /' "$S/FIGURE_NOTES.md"; echo; }
  for t in $TABLES; do
    echo "---"; echo
    if [ "$t" = PER_DATASET_CORE5 ]; then
      echo "## Table A2 — all 31 combinations of the five branches"; echo
      sed -e '1,2d' "$S/$t.md"
    else
      sed -e 's/^# /## /' "$S/$t.md"
    fi
    echo
  done
  echo "---"; echo
  echo "# Part IV — DF40 cross-manipulation"; echo
  sed -e '1,2d' "$R/DF40_CROSSMANIP.md"
} > "$R/PROPOSAL_TABLES_2.md"
echo "-> $R/PROPOSAL_TABLES_2.md  (census $SCOPE)"

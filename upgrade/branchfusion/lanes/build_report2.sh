#!/usr/bin/env bash
# PROPOSAL_TABLES_2.md -- the proposal's second results pass: the widened evaluation scope, both
# KoDF variants, and DF40 cross-manipulation.
#
# Assembly only. Every number is produced by the generator that owns it and is copied verbatim:
#   SCOPE_COMPARISON.md              scope_compare.py      what admitting each family costs
#   scope_nolip/*.md                 regen_scope.sh nolip      WITHOUT adversarial
#   scope_nolipadv/*.md              regen_scope.sh nolipadv   WITH adversarial
#   DF40_CROSSMANIP.md               df40_crossmanip.py    DiCoME Table 1's second half
#
# PROPOSAL_TABLES_1.md is the v3.1 report and is NOT touched: the two are different censuses and
# a reader has to be able to see both.
set -u
cd /data/umar/Repos/DFB_NeSyNS
R=tbiom/reports/branch_fusion
TABLES="TABLE_A_ABLATION PER_DATASET_CORE5 TABLE_B_FUSION TABLE1_DETECTION TABLE2_CALIBRATION"
for f in "$R/SCOPE_COMPARISON.md" "$R/DF40_CROSSMANIP.md"; do
  [ -f "$f" ] || { echo "build_report2: missing $f" >&2; exit 1; }
done
for s in nolip nolipadv; do
  for t in $TABLES; do
    [ -f "$R/scope_$s/$t.md" ] || { echo "build_report2: missing $R/scope_$s/$t.md" >&2; exit 1; }
  done
done

emit_scope () {                       # $1 = scope dir, $2 = heading
  echo "# $2"; echo
  echo "Risk-coverage figure: \`$R/$1/figures/uncertainty_riskcoverage.png\`"; echo
  if [ -f "$R/$1/FIGURE_NOTES.md" ]; then
    # The figure's verdict is COMPUTED from the figure's own AURC data, so it belongs with the
    # tables rather than only beside the image a reader of this file may never open.
    sed -e 's/^# /## /' "$R/$1/FIGURE_NOTES.md"; echo
  fi
  for t in $TABLES; do
    echo "---"; echo
    if [ "$t" = PER_DATASET_CORE5 ]; then
      echo "## Table A2 — all 31 combinations of the five branches"; echo
      sed -e '1,2d' "$R/$1/$t.md"
    else
      sed -e 's/^# /## /' "$R/$1/$t.md"
    fi
    echo
  done
}

{
  echo "# Proposal results, pass 2 — widened scope, both KoDF variants, and DF40"; echo
  echo "Regenerated $(date -Is)."; echo
  echo "This supersedes nothing. \`PROPOSAL_TABLES_1.md\` holds the protocol-v3.1 census and stays"
  echo "as it is; this file holds the census with the lip-sync / audio-driven exclusion REMOVED,"
  echo "reported twice — once without the adversarial subsets and once with them."; echo
  echo "| part | what it answers |"; echo "|---|---|"
  echo "| I | what admitting each excluded family costs, on the only two sets that have any |"
  echo "| II | every table WITHOUT the adversarial subsets (KoDF = 1,509 videos) |"
  echo "| III | every table WITH them (KoDF = 2,239 videos) |"
  echo "| IV | DF40 cross-manipulation — the structure of DiCoME Table 1's second half |"
  echo
  echo "Only CDFv3 and KoDF change between parts II and III. The other ten test sets contain none"
  echo "of the excluded families, so their cells are identical in both and a difference between"
  echo "the two parts is a difference in which videos were scored, never in the system."; echo
  echo "## One caveat, and it decides which table to quote where"; echo
  echo "**The baselines were never run on the admitted videos.** The GenD runs' prediction files"
  echo "stop at the v3.1 census for CDFv3 and at the no-adversarial census for KoDF, and DiCoME's"
  echo "CDFv3 export covers 1,918 videos against our 5,418. Widening the scope widens OUR row and"
  echo "leaves theirs where it was; each Table 1 below names every cell this affects."; echo
  echo "So: quote **Table 1 from \`PROPOSAL_TABLES_1.md\`** for the comparison against the field,"
  echo "where every cell is on one census, and quote the tables here for everything about our own"
  echo "system — the ablation, the fusion operator, calibration, and the two KoDF variants. The"
  echo "ten unaffected datasets are identical in both files either way."; echo
  echo "---"; echo
  echo "# Part I — what the scope costs"; echo
  sed -e '1,2d' "$R/SCOPE_COMPARISON.md"
  echo; echo "---"; echo
  emit_scope scope_nolip "Part II — all tables WITHOUT the adversarial subsets"
  echo "---"; echo
  emit_scope scope_nolipadv "Part III — all tables WITH the adversarial subsets"
  echo "---"; echo
  echo "# Part IV — DF40 cross-manipulation"; echo
  sed -e '1,2d' "$R/DF40_CROSSMANIP.md"
} > "$R/PROPOSAL_TABLES_2.md"
echo "-> $R/PROPOSAL_TABLES_2.md"

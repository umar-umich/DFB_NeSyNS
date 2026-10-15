#!/usr/bin/env bash
# When DiCoME's pooled DF40 run lands, fold it into both report files and verify.
#
# It is the last outstanding row: PROPOSAL_TABLES_3 omits it rather than showing a mean over six
# generators beside means over twenty-one, and DF40_RESULTS marks it partial.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
M=logs/branchfusion/orchestrator
L=logs/scope_expand
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

# ONLY MARKERS WRITTEN FROM NOW ON. The log keeps every attempt: the first pooled run died on a
# relative-path error and left a FAILED line behind, so matching anywhere in the file killed this
# chain instantly while the real run was eight rows in -- and matching the LAST marker failed the
# same way, because the running job has not written one yet. Anchoring on the line count at start
# is the only version that asks the right question.
FROM=$(wc -l < "$M/orchestrator.log")
n=0
while [ $n -lt 240 ]; do
  new_lines=$(tail -n +$((FROM + 1)) "$M/orchestrator.log")
  case "$new_lines" in
    *"DiCoME DF40 pooled done"*)   break ;;
    *"DiCoME DF40 pooled FAILED"*) say "after_dicome: the run FAILED"; exit 1 ;;
  esac
  sleep 30; n=$((n+1))
done

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$PY upgrade/branchfusion/df40_table.py --ours "V+O+S16ctl,V+O+P16ctl,A+V+O+S16ctl" \
  >> "$L/after_dicome.log" 2>&1 && say "after_dicome: DF40_RESULTS rebuilt" \
  || say "after_dicome: DF40_RESULTS FAILED"
$PY upgrade/branchfusion/proposal_tables_3.py >> "$L/after_dicome.log" 2>&1 \
  && say "after_dicome: PROPOSAL_TABLES_3 rebuilt" || say "after_dicome: _3 FAILED"
bash upgrade/branchfusion/lanes/build_report2.sh nolipadv >> "$L/after_dicome.log" 2>&1 \
  && say "after_dicome: PROPOSAL_TABLES_2 rebuilt" || say "after_dicome: _2 FAILED"

for f in PROPOSAL_TABLES_3 PROPOSAL_TABLES_2; do
  $PY upgrade/branchfusion/audit_report.py "tbiom/reports/branch_fusion/$f.md" \
    > "$L/audit_$f.txt" 2>&1
  say "after_dicome: audit $f -- $(tail -1 "$L/audit_$f.txt")"
done

git add -A upgrade tbiom/reports analysis logs/tbiom/df40 2>/dev/null
git diff --cached --quiet 2>/dev/null || git commit -q -m "df40: DiCoME's own checkpoint, 21 pooled generator rows

The last outstanding row. Both report files carried it as absent or partial rather than showing
a mean over six generators beside means over twenty-one.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push -q origin phase3/discern-v2-tbiom 2>/dev/null && say "after_dicome: pushed" || say "after_dicome: push FAILED"
say "=== after_dicome: DONE ==="

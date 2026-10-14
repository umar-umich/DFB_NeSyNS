#!/usr/bin/env bash
# Wait for every fill job, regenerate everything on the chosen census, rebuild the report, and
# VERIFY it -- then commit. Written to run unattended: UMAR is offline for 3-4 hours and expects
# complete tables, so every stage logs a verdict and the last stage is an audit, not a claim.
set -u
cd /data/umar/Repos/DFB_NeSyNS
SCOPE=nolipadv
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
M=logs/branchfusion/orchestrator
L=logs/scope_expand
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

# Wait on LOG MARKERS, not PIDs: a prerequisite may already have exited by the time this starts,
# and `pgrep -f` matches the Bash tool's own wrapper, which spun a chain forever earlier today.
wait_for () {   # wait_for <done-marker> <fail-marker> <minutes>
  local ok=$1 bad=$2 limit=$(( $3 * 2 )) n=0
  while [ $n -lt $limit ]; do
    grep -q "$ok" "$M/orchestrator.log" && return 0
    grep -q "$bad" "$M/orchestrator.log" && { say "finish_all: '$bad' seen -- continuing anyway"; return 1; }
    sleep 30; n=$((n+1))
  done
  say "finish_all: TIMED OUT waiting for '$ok' after $3 min -- continuing"
  return 1
}

say "=== finish_all: waiting for the fill jobs ==="
wait_for "DiCoME baseline filled"          "DiCoME CDFv3 FAILED"        90
wait_for "baselines widened census done"   "baselines widened FAILED"  240

say "=== finish_all: regenerating at $SCOPE ==="
bash upgrade/branchfusion/lanes/regen_scope.sh "$SCOPE"
# `regen_scope` runs proposal_tables FIRST and the rest after; a single earlier failure there
# left Table B stale for an hour. Re-run it explicitly so the report cannot assemble from a
# table that failed at the head of the list.
BF_SCOPE=$SCOPE OMP_NUM_THREADS=8 $PY upgrade/branchfusion/proposal_tables.py \
  >> "$L/regen_$SCOPE.log" 2>&1 && say "  proposal_tables ok (second pass)" \
  || say "  proposal_tables FAILED (second pass)"

$PY upgrade/branchfusion/scope_compare.py >> "$L/regen_$SCOPE.log" 2>&1 \
  && say "  scope_compare ok" || say "  scope_compare FAILED"
$PY upgrade/branchfusion/df40_crossmanip.py >> "$L/regen_$SCOPE.log" 2>&1 \
  && say "  df40_crossmanip ok" || say "  df40_crossmanip FAILED"
$PY upgrade/branchfusion/choose_three.py >> "$L/regen_$SCOPE.log" 2>&1 \
  && say "  choose_three ok" || say "  choose_three FAILED"

bash upgrade/branchfusion/lanes/build_report2.sh "$SCOPE" >> "$L/regen_$SCOPE.log" 2>&1 \
  && say "  report assembled" || say "  report assembly FAILED"

say "=== finish_all: auditing the report ==="
$PY upgrade/branchfusion/audit_report.py > "$L/audit.txt" 2>&1
tail -3 "$L/audit.txt" | while read -r l; do say "  audit: $l"; done

git add -A upgrade tbiom/reports analysis logs/branchfusion/branches/registry.json 2>/dev/null
git diff --cached --quiet 2>/dev/null || git commit -q -m "reports: every baseline cell filled, on one census

The baselines were never run on UADFV, DeepSpeak-1.1 and DeepSpeak-2.0, and their CDFv3 and KoDF
cells sat at the narrow census. Both are now closed: seven GenD-repo baselines and DiCoME over
the three corpora, and the same seven re-run on the widened CDFv3 and KoDF.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push -q origin phase3/discern-v2-tbiom 2>/dev/null && say "  pushed" || say "  push FAILED"
say "=== finish_all: DONE ==="

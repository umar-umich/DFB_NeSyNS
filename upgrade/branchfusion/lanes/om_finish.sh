#!/usr/bin/env bash
# FINISHER: waits for the rerun lane, then applies the gates and writes the campaign report.
#
# The first orchestrator pass completed with every control missing, so gate 1 was HELD for both P
# constructions and branch S had no run at all. That is not a verdict, it is an absence. This lane
# turns the reruns into one, and is resumable the same way.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

commit () {
  mkdir -p tbiom/reports/branch_fusion
  { echo "# Orchestrator decision log"; echo; echo '```'; cat "$M/orchestrator.log"; echo '```'; } \
    > tbiom/reports/branch_fusion/ORCHESTRATOR_LOG.md
  for v in depth gate; do
    [ -f "$M/${v}_verdict.txt" ] && cp "$M/${v}_verdict.txt" "tbiom/reports/branch_fusion/${v}_verdict.txt"
  done
  git add -A analysis tbiom/reports upgrade 2>/dev/null
  git diff --cached --quiet 2>/dev/null && return 0
  git commit -q -m "$1

Committed unattended; see tbiom/reports/branch_fusion/ORCHESTRATOR_LOG.md.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && say "committed: $1"
}

# Only python counts: a shell that wrote a lane script carries the pattern in its own argv.
py_running () {
  local pid
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) return 0 ;; esac
  done
  return 1
}

say "=== finisher waiting on the rerun lane ==="
n=0
while py_running "train_om_pecore.py" || pgrep -f "lanes/om_rerun.sh" > /dev/null 2>&1; do
  sleep 120; n=$((n + 2))
  [ $((n % 60)) -eq 0 ] && say "finisher still waiting (${n} min)"
  [ "$n" -ge 900 ] && { say "finisher TIMEOUT after ${n} min"; break; }
done
say "rerun lane finished"

$PY upgrade/branchfusion/om_report.py --stage depth > "$M/depth_verdict.txt" 2>&1 || true
$PY upgrade/branchfusion/om_report.py --stage gates > "$M/gate_verdict.txt" 2>&1 || true
say "gate verdict:"; sed 's/^/    /' "$M/gate_verdict.txt" >> "$M/orchestrator.log"

if grep -q "^REGISTER:" "$M/gate_verdict.txt" 2>/dev/null; then
  for B in $(grep "^REGISTER:" "$M/gate_verdict.txt" | cut -d: -f2); do
    say "$B passed both gates -- running the full subset screen"
    $PY upgrade/branchfusion/subsets.py --screen --branches "A,S,V,O,M,P,$B" \
        --sizes 1,2,3,4,5,6,7 >> "$L/screen_with_new.log" 2>&1 || say "screen FAILED for $B"
  done
else
  say "no branch passed both gates; the deployed selection stands"
fi

$PY upgrade/branchfusion/om_report.py --stage final > tbiom/reports/branch_fusion/OM_CAMPAIGN.md 2>&1 || true
commit "fusion: off-manifold campaign, controls included"
say "=== finisher done ==="

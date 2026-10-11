#!/usr/bin/env bash
# FINISHER for the v3 lane: wait, apply the gates, write the report, commit.
#
# Same shape as the first finisher, pointed at `om_v3.sh`. The gate table it reads is now empty of
# the voided runs -- they were moved under `logs/branchfusion/om_pecore/voided_2026-10-10_*/`,
# which the glob in `om_report.branch_rows` does not descend into -- so a verdict here is about the
# four v3 runs or about nothing.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

commit () {
  mkdir -p reports/branch_fusion
  { echo "# Orchestrator decision log"; echo; echo '```'; cat "$M/orchestrator.log"; echo '```'; } \
    > reports/branch_fusion/ORCHESTRATOR_LOG.md
  for v in depth gate; do
    [ -f "$M/${v}_verdict.txt" ] && cp "$M/${v}_verdict.txt" "reports/branch_fusion/${v}_verdict.txt"
  done
  git add -A analysis reports upgrade 2>/dev/null
  git diff --cached --quiet 2>/dev/null && return 0
  git commit -q -m "$1

Committed unattended; see reports/branch_fusion/ORCHESTRATOR_LOG.md.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && say "committed: $1"
}

# Only python counts: a shell that carries the pattern in its own argv is not a running job.
py_running () {
  local pid
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) return 0 ;; esac
  done
  return 1
}

say "=== v3 finisher waiting ==="
n=0
while py_running "train_om_pecore.py" || pgrep -f "lanes/om_v3.sh" > /dev/null 2>&1; do
  sleep 120; n=$((n + 2))
  [ $((n % 60)) -eq 0 ] && say "v3 finisher still waiting (${n} min)"
  [ "$n" -ge 900 ] && { say "v3 finisher TIMEOUT after ${n} min"; break; }
done
say "v3 lane finished"

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

$PY upgrade/branchfusion/om_report.py --stage final > reports/branch_fusion/OM_CAMPAIGN.md 2>&1 || true
commit "fusion: the PE-Core off-manifold campaign re-run with the residual in the optimiser"
say "=== v3 finisher done ==="

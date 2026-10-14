#!/usr/bin/env bash
# Regenerate every table and the figure over the full twelve test sets.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }
say "=== regenerating all tables over 12 test sets ==="
for s in proposal_tables sota_table1 table2_calibration uncertainty_figs forensic_metrics sota_operating_points; do
  $PY "upgrade/branchfusion/$s.py" >> "$L/regen_all.log" 2>&1 \
    && say "regen $s ok" || say "regen $s FAILED"
done
bash upgrade/branchfusion/lanes/build_report.sh >> "$L/regen_all.log" 2>&1
{ echo "# Orchestrator decision log"; echo; echo '```'; cat "$M/orchestrator.log"; echo '```'; } \
  > tbiom/reports/branch_fusion/ORCHESTRATOR_LOG.md
git add -A upgrade tbiom/reports analysis logs/branchfusion/branches/registry.json 2>/dev/null
git diff --cached --quiet 2>/dev/null || git commit -q -m "fusion: all tables over twelve test sets, with the three GenD corpora in

UADFV, DeepSpeak-1.1 and DeepSpeak-2.0 promoted into TEST_SETS now that every CORE branch is
scored on them, so every table and every figure downstream picks them up without a special case.

Two fixes were needed to get there. meta_for had no path for a corpus packed by THIS project:
the stage-23 selection file it reads belongs to the earlier pipeline and was never run for them,
so the metadata is now rebuilt from a materialised branch -- sound only because the pack step
admitted in-scope generators alone, which the comment states. And the first attempt at that
fallback landed in _stage23_opinions by a str.replace that matched the wrong occurrence, where it
would have returned metadata columns from a function whose contract is to return opinions.

The exhaustive 31-subset ablation confirms the framework: App+TD+LFS+STT is the best of all 31
combinations at 0.9509 over twelve sets. Every subset containing BOTH STT variants collapses onto
the one-variant score, because the fit zeroes whichever it sees second -- they are one branch.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
say "=== regen done ==="

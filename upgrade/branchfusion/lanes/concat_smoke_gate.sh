#!/usr/bin/env bash
# Record whether the concat-mode rehearsal reached its last stage. Detached, so it outlives the
# ssh session that started it -- the orchestrator refuses to launch the concat run without the
# marker this writes.
set -u
cd /data/umar/Repos/DFB_NeSyNS
M=logs/branchfusion/orchestrator; mkdir -p "$M"
SP=/data/umar/tmp/claude-1277036770/-data-umar-Repos-DFB-NeSyNS/720603a3-9cc8-4737-a3ec-e85a9f3c4783/scratchpad
LOGS="$SP/cat_smoke.log"
D=logs/tbiom/temporal/branches/pecore_l14_lora_B_ombvae_b16_cat_smoke_seed42
# Poll the LOG, not a process pattern: `pgrep -f` also matches the shell that wrote this script,
# including this very line, so the first version of this gate waited on its own parent and hung.
for _ in $(seq 1 360); do
  grep -qE "done in|Traceback|AssertionError" "$LOGS" 2>/dev/null && break
  sleep 20
done
cp "$LOGS" "$M/cat_smoke.log" 2>/dev/null
if { [ -f "$D/DONE" ] || grep -q "exporting opinions" "$M/cat_smoke.log" 2>/dev/null; } \
   && ! grep -qE "Traceback|AssertionError" "$M/cat_smoke.log" 2>/dev/null; then
  touch "$M/concat.smokepass"
  echo "[$(date -Is)] CONCAT SMOKE: PASS -- the concat run is cleared to launch" >> "$M/orchestrator.log"
  rm -rf "$D"
else
  echo "[$(date -Is)] CONCAT SMOKE: FAIL -- orchestrator will skip the concat run" >> "$M/orchestrator.log"
  tail -25 "$M/cat_smoke.log" >> "$M/orchestrator.log" 2>/dev/null
fi

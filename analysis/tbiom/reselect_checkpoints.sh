#!/usr/bin/env bash
# Re-select every run's checkpoint under ONE declared dev-only rule (stage23_select.py:
# VALmix macro AUROC, FF++ val logged as a guardrail). Replaces the undeclared newest-file rule
# the SBI drivers used. Exports every saved checkpoint on FF++ val and VALmix only; no OOD set is
# touched. Sequential, one export process at a time.
set -uo pipefail
cd /data/umar/Repos/DFB_NeSyNS
for arm in run1auxedl sbiarmA sbiarmB sbiarmC; do
  echo "[$(date +%H:%M)] select-phase exports: $arm"
  bash analysis/tbiom/stage23_export.sh "$arm" select >> "logs/tbiom/sbi/reselect_${arm}.log" 2>&1
done
echo "[$(date +%H:%M)] dev exports for re-selection complete"

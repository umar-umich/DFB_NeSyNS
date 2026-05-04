#!/usr/bin/env bash
# =============================================================================
# Single-command driver for the DeFakeNet ablation pipeline.
#
# Streams child output to the console AND a log file (via `tee`), prints a
# banner with start time + elapsed time per stage, and ends with a summary
# table. Failures in any stage are tolerated so a single bad checkpoint
# doesn't kill the whole sweep.
#
# Usage:
#   bash scripts/run_all.sh           # run everything
#   FORCE=1 bash scripts/run_all.sh   # re-run ablations even if metrics.json exists
# =============================================================================
set -u
set -o pipefail
export PYTHONUNBUFFERED=1   # so tqdm bars / prints flush in real time

ABLATIONS=(
  "full_defakenet_18rules:checkpoints/full_defakenet_18rules.pth"
  "no_ibdc:checkpoints/no_ibdc.pth"
  # "no_cmef:checkpoints/no_cmef.pth"
  # "no_pbas:checkpoints/no_pbas.pth"
  "no_causal:checkpoints/no_causal.pth"
  "no_symbolic:checkpoints/no_symbolic.pth"
  "visual_edl_only:checkpoints/visual_edl_only.pth"
)

mkdir -p results/ablations results/faithfulness results/selective

# ── helpers ────────────────────────────────────────────────────────────────
_BOLD=$'\033[1m'; _DIM=$'\033[2m'; _GRN=$'\033[32m'; _RED=$'\033[31m'
_YEL=$'\033[33m'; _CYA=$'\033[36m'; _RST=$'\033[0m'
[ -t 1 ] || { _BOLD=""; _DIM=""; _GRN=""; _RED=""; _YEL=""; _CYA=""; _RST=""; }

_fmt_dur() {
  # seconds -> "HH:MM:SS"
  local s=$1
  printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

declare -A STATUS_MAP
declare -A ELAPSED_MAP
TOTAL_START=$(date +%s)

_run_stage() {
  # _run_stage <stage_name> <log_path> -- <command...>
  local name="$1"; shift
  local log="$1"; shift
  shift  # discard the literal "--"
  local start=$(date +%s)
  printf '%s\n' "${_CYA}${_BOLD}==================================================${_RST}"
  printf '%s %s%s%s\n'        "${_CYA}▶${_RST}"     "${_BOLD}" "$name" "${_RST}"
  printf '   started : %s\n'  "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '   log     : %s\n'  "$log"
  printf '%s\n' "${_CYA}${_BOLD}--------------------------------------------------${_RST}"

  # tee streams to console + log file. pipefail propagates the python exit code.
  ( "$@" 2>&1 | tee "$log" )
  local rc=${PIPESTATUS[0]}
  local elapsed=$(( $(date +%s) - start ))

  if [ "$rc" -eq 0 ]; then
    STATUS_MAP[$name]="OK"
    printf '%s\n' "${_GRN}${_BOLD}✓ ${name} OK   (elapsed $(_fmt_dur "$elapsed"))${_RST}"
  else
    STATUS_MAP[$name]="FAIL(rc=$rc)"
    printf '%s\n' "${_RED}${_BOLD}✗ ${name} FAILED rc=$rc   (elapsed $(_fmt_dur "$elapsed"))${_RST}"
  fi
  ELAPSED_MAP[$name]=$elapsed
  return 0
}

# ── per-ablation evaluation ────────────────────────────────────────────────
for entry in "${ABLATIONS[@]}"; do
  name="${entry%%:*}"
  ckpt="${entry##*:}"
  mkdir -p "results/ablations/$name"

  metrics_json="results/ablations/$name/metrics.json"
  if [ -z "${FORCE:-}" ] && [ -f "$metrics_json" ]; then
    printf '%s\n' "${_YEL}↷ skipping $name — $metrics_json already exists (set FORCE=1 to re-run)${_RST}"
    STATUS_MAP[$name]="SKIP"
    ELAPSED_MAP[$name]=0
    continue
  fi

  if [ ! -f "$ckpt" ]; then
    printf '%s\n' "${_RED}✗ $name — checkpoint missing: $ckpt${_RST}"
    STATUS_MAP[$name]="MISSING_CKPT"
    ELAPSED_MAP[$name]=0
    continue
  fi

  _run_stage "ablation/$name" "results/ablations/$name/run.log" -- \
    python scripts/run_ablation_eval.py \
      --ablation-name "$name" --checkpoint-path "$ckpt"
done

# ── faithfulness ───────────────────────────────────────────────────────────
_run_stage "faithfulness" "results/faithfulness/run.log" -- \
  python scripts/run_faithfulness.py

# ── selective ──────────────────────────────────────────────────────────────
_run_stage "selective" "results/selective/run.log" -- \
  python scripts/run_selective.py

# ── aggregate ──────────────────────────────────────────────────────────────
_run_stage "aggregate" "results/aggregate.log" -- \
  python scripts/aggregate.py

# ── summary ────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
printf '\n%s\n' "${_BOLD}==================================================${_RST}"
printf '%s\n'   "${_BOLD}                Run summary${_RST}"
printf '%s\n'   "${_BOLD}==================================================${_RST}"
printf '%-32s  %-14s  %s\n' "stage" "status" "elapsed"
printf '%-32s  %-14s  %s\n' "-----" "------" "-------"
for name in "${!STATUS_MAP[@]}"; do
  st="${STATUS_MAP[$name]}"
  el=$(_fmt_dur "${ELAPSED_MAP[$name]}")
  case "$st" in
    OK)            color="$_GRN" ;;
    SKIP)          color="$_YEL" ;;
    MISSING_CKPT)  color="$_RED" ;;
    FAIL*)         color="$_RED" ;;
    *)             color="" ;;
  esac
  printf '%-32s  %s%-14s%s  %s\n' "$name" "$color" "$st" "$_RST" "$el"
done | sort
printf '%-32s  %-14s  %s\n' "TOTAL" "" "$(_fmt_dur "$TOTAL_ELAPSED")"
printf '\n'
printf '%s\n' "${_BOLD}DONE. See results/ABLATION_SUMMARY.md${_RST}"

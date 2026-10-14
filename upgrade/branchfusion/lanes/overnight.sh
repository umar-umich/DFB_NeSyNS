#!/usr/bin/env bash
# Everything that can finish unattended, in order, with markers so a restart resumes.
#
#   1. wait for STT and STT-order on the three new sets (already running)
#   2. TD on the new sets -- checkpoint replay from Dr Kutub's temporal branches directory
#   3. LFS on the new sets -- DISCERN_Ext export, then video-level reduction
#      (App needs nothing: its head reads the feature caches, which are already extracted)
#   4. regenerate Tables A, B, C and 2, and the consolidated file
#   5. regenerate the uncertainty figure for the FOUR-branch framework with the thesis names
#   6. commit
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PYT=/data/umar/miniconda3/envs/discern_ext_timm/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
NEW="UADFV DeepSpeak-1.1 DeepSpeak-2.0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }
disk_ok () { local f; f=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9'); [ "${f:-0}" -ge 40 ] || { say "overnight: ${f}G free -- stopping"; return 1; }; }

say "=== overnight chain start ==="
n=0
while pgrep -f "score_new_v2.sh" > /dev/null 2>&1 && [ $n -lt 180 ]; do sleep 60; n=$((n+1)); done
say "overnight: STT lane drained after ${n} min"

# ---- TD on the new sets
if [ ! -f "$M/new_TD.done" ]; then
  disk_ok || exit 1
  CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --export-only \
      --sets "UADFV,DeepSpeak-1.1,DeepSpeak-2.0" \
      --run-dir /data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/branches/vjepa2_lora_B_seed42 \
      --model vjepa2 --lora --augment B --seed 42 --workers 5 >> "$L/new_TD.log" 2>&1 \
    && { touch "$M/new_TD.done"; say "TD exported on the new sets"; } || say "TD export FAILED"
fi

# ---- LFS on the new sets
CK=runs/discern_ext_train/run1_auxedl_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt
for DS in $NEW; do
  f="logs/branchfusion/o_evid/${DS}.csv"
  [ -s "$f" ] && continue
  disk_ok || exit 1
  mkdir -p logs/branchfusion/o_evid
  ( cd /data/umar/Repos/DISCERN_Ext && OMP_NUM_THREADS=4 $PYT eval_adaptation/export_views.py \
      --checkpoint "$CK" --config "eval_adaptation/configs/run1auxedl_${DS}.yaml" --dataset "$DS" \
      --out "/data/umar/Repos/DFB_NeSyNS/$f" --device cuda:3 --num-workers 4 ) \
      >> "$L/new_LFS.log" 2>&1 && say "LFS exported: $DS" || say "LFS export FAILED: $DS"
done
$PY upgrade/branchfusion/register_ce_branches.py >> "$L/new_LFS.log" 2>&1 || true

# ---- tables
say "overnight: regenerating tables"
$PY upgrade/branchfusion/proposal_tables.py    >> "$L/overnight.log" 2>&1 || say "tables A/B FAILED"
$PY upgrade/branchfusion/sota_table1.py        >> "$L/overnight.log" 2>&1 || say "table C FAILED"
$PY upgrade/branchfusion/table2_calibration.py >> "$L/overnight.log" 2>&1 || say "table 2 FAILED"
$PY upgrade/branchfusion/uncertainty_figs.py   >> "$L/overnight.log" 2>&1 || say "figure FAILED"

# The report is assembled in ONE place. These lanes each carried their own copy of the
# heredoc, which silently went stale -- the version here was still emitting three tables
# with no name key and no Table A2 long after the generators had five.
bash upgrade/branchfusion/lanes/build_report.sh

mkdir -p tbiom/reports/branch_fusion
{ echo "# Orchestrator decision log"; echo; echo '```'; cat "$M/orchestrator.log"; echo '```'; } \
  > tbiom/reports/branch_fusion/ORCHESTRATOR_LOG.md
git add -A upgrade tbiom/reports analysis logs/branchfusion/branches/registry.json 2>/dev/null
git diff --cached --quiet 2>/dev/null || git commit -q -m "fusion: new GenD sets scored, all tables and the figure regenerated

Committed unattended overnight; see tbiom/reports/branch_fusion/ORCHESTRATOR_LOG.md.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
say "=== overnight chain done -> tbiom/reports/branch_fusion/PROPOSAL_TABLES.md ==="

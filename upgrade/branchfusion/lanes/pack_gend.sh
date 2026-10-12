#!/usr/bin/env bash
# Pack the two DeepSpeak sets. CPU and disk only -- no GPU, so it runs beside the CE training.
set -u
cd /data/umar/Repos/DFB_NeSyNS
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
M=logs/branchfusion/orchestrator
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }
disk_ok () { local f; f=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9'); [ "${f:-0}" -ge 45 ] || { say "pack: only ${f}G free -- stopping"; return 1; }; }
for DS in DeepSpeak-2.0 DeepSpeak-1.1; do
  disk_ok || exit 1
  say "packing $DS"
  $PY upgrade/branchfusion/pack_gend.py --dataset "$DS" >> logs/branchfusion/lanes/pack_$DS.log 2>&1 \
    && say "$DS packed" || say "$DS pack FAILED"
done
say "=== GenD packing done ==="

#!/usr/bin/env bash
# Finish the three new GenD sets. Everything here is a replay or a copy -- nothing retrains.
#
# Four gaps closed, each of which reported success while doing nothing:
#   TD      `train_om_pecore` installed the PE-Core front end unconditionally and died on V-JEPA
#   LFS     exported to logs/branchfusion/o_evid, where the `O` reader never looks
#   STT/-o  exported into the run dirs, but the ledger copy iterated the old TEST_SETS
#   tables  computed over C.TEST_SETS, so filled cells still printed "in progress"
set -u
cd /data/umar/Repos/DFB_NeSyNS
PYV=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
PY=/data/umar/miniconda3/envs/discern_ext/bin/python
L=logs/branchfusion/lanes
M=logs/branchfusion/orchestrator
NEW="UADFV,DeepSpeak-1.1,DeepSpeak-2.0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say () { echo "[$(date -Is)] $*" >> "$M/orchestrator.log"; }

say "=== finishing the new sets ==="

# 1. LFS: reduce the evidential per-frame export to one opinion per video, into the ledger.
$PY - <<'PYEOF' >> "$L/finish_new.log" 2>&1
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "upgrade/branchfusion"); sys.path.insert(0, "analysis/tbiom")
import common as C
from video_id import to_video_level
out = C.BRANCH_OUT / "O"; out.mkdir(parents=True, exist_ok=True)
for ds in ("UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0"):
    f = C.ROOT / "logs/branchfusion/o_evid" / f"{ds}.csv"
    if not f.exists():
        print(f"{ds}: no evidential export"); continue
    d = pd.read_csv(f)
    g = to_video_level(d.rename(columns={"p_artifact": "_p"}), "_p",
                       key_col="key", label_col="label").set_index("video")
    u = to_video_level(d.rename(columns={"u_artifact": "_u"}), "_u",
                       key_col="key", label_col="label").set_index("video")
    pd.DataFrame({"v": [str(v) for v in g.index], "y": g.y.to_numpy(),
                  "p": g.p.to_numpy(), "u": u.p.reindex(g.index).to_numpy()}
                 ).to_csv(out / f"{ds}.csv", index=False)
    print(f"LFS {ds}: {len(g)} videos")
PYEOF
say "LFS reduced into the ledger"

# 2. TD: replay V-JEPA now that the install is skipped for non-PE trunks.
if [ ! -f "$M/new_TD.done" ]; then
  CUDA_VISIBLE_DEVICES=1 $PYV upgrade/branchfusion/train_om_pecore.py --operator none \
      --export-only --sets "$NEW" \
      --run-dir /data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/branches/vjepa2_lora_B_seed42 \
      --model vjepa2 --lora --augment B --seed 42 --workers 5 >> "$L/new_TD.log" 2>&1 \
    && { touch "$M/new_TD.done"; say "TD exported"; } || say "TD export FAILED AGAIN"
fi

# 3. Copy every temporal branch's new-set CSVs into the ledger.
$PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seq_seed42 \
    --label S16ctl --sets "$NEW" >> "$L/finish_new.log" 2>&1 || say "STT copy FAILED"
$PY upgrade/branchfusion/om_pecore_export.py --run pecore_l14_lora_B_omnone_b16_seed42 \
    --label P16ctl --sets "$NEW" >> "$L/finish_new.log" 2>&1 || say "STT-order copy FAILED"
$PY upgrade/branchfusion/om_pecore_export.py --run vjepa2_lora_B_seed42 \
    --label V --sets "$NEW" >> "$L/finish_new.log" 2>&1 || say "TD copy FAILED"

# 4. Regenerate everything.
for s in proposal_tables sota_table1 table2_calibration uncertainty_figs; do
  $PY "upgrade/branchfusion/$s.py" >> "$L/finish_new.log" 2>&1 || say "$s FAILED"
done
bash upgrade/branchfusion/lanes/build_report.sh >> "$L/finish_new.log" 2>&1 || say "report FAILED"
say "=== new sets finished ==="

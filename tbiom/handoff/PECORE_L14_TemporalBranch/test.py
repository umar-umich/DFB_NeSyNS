#!/usr/bin/env python3
"""Evaluate a trained branch: export one opinion (p, u) per video for named datasets.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
      /data/umar/miniconda3/envs/temporal_vjepa2/bin/python test.py \
        --branch pecore_l14_lora_B_review_seed42 --model pecore_l14 --datasets CDFv3 CDFv2

Thin wrapper over code/tbiom/temporal/export_branch.py, which loads a run's development-selected
`best.pt` and writes `opinions/<dataset>.csv` with columns (v, y, p, u). It SKIPS datasets whose
CSV already exists, so it is safe to re-run.

NOTE ON WHICH WEIGHTS IT LOADS. export_branch.py reads from the original repo's run directory
(code/tbiom/temporal/export_branch.py line 18). To score the checkpoint shipped HERE without
touching that path, use `verify.py --rerun <dataset>` instead -- it loads weights/best.pt directly
and diffs the result against the reference opinions.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path[:0] = [str(CODE / "tbiom"), str(CODE / "tbiom/xclip"), str(CODE / "tbiom/temporal"),
                "/data/umar/Repos/DISCERN_Ext"]

import export_branch  # noqa: E402

if __name__ == "__main__":
    print(f"entry point: {export_branch.__file__}", flush=True)
    export_branch.main()

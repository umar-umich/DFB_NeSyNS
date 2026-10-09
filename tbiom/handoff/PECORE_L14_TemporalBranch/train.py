#!/usr/bin/env python3
"""Train the temporal branch. Thin wrapper -- the real code is code/tbiom/temporal/train_branch.py.

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
      /data/umar/miniconda3/envs/temporal_vjepa2/bin/python train.py \
        --model pecore_l14 --lora --augment B --seed 42 --run-suffix _review

This file exists so the folder has an obvious entry point; it adds NOTHING to the recipe. Every
argument is passed straight through, and the training loop, the model and the data loader are all
in `train_branch.py`, which is a byte-identical copy of the live repo. See "Where everything is"
in README.md for the file:line map.

Run ./setup_local.sh first in your own copy, or training will try to write into the original repo.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path[:0] = [str(CODE / "tbiom"), str(CODE / "tbiom/xclip"), str(CODE / "tbiom/temporal"),
                "/data/umar/Repos/DISCERN_Ext"]

import train_branch  # noqa: E402

if __name__ == "__main__":
    print(f"entry point: {train_branch.__file__}", flush=True)
    print(f"selection CSVs: {train_branch.S23}", flush=True)
    train_branch.main()

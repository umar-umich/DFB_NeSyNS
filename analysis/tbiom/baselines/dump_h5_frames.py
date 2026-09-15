#!/usr/bin/env python3
"""Write our DISCERN test frames out of the h5 files as PNGs so the GenD-repo baselines see exactly our crops.
Layout /data/umar/Datasets/tbiom_frames/<DS>/<subset>/<video>/<frame> — GenD reads the label from <subset>
('real' substring = genuine), which matches the h5 label rule. Writes one GenD file list per subset.
Skips files that already exist, so it can be resumed. CDFv3 IP_LAP (lip-sync only) is out of scope and not dumped."""
import sys, os
from multiprocessing import Pool
from pathlib import Path
import h5py, numpy as np
from PIL import Image

EXT = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data")
ROOT = Path("/data/umar/Datasets/tbiom_frames")
LISTS = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_lists_ours")


def work(args):
    ds, keys = args
    n = 0
    with h5py.File(EXT / f"h5/{ds}.h5", "r") as h:
        for k in keys:
            out = ROOT / k
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.png")
            Image.fromarray(np.asarray(h[k])).save(tmp)
            os.replace(tmp, out); n += 1
    return n


def main():
    ds = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    jobs = []
    (LISTS / ds).mkdir(parents=True, exist_ok=True)
    for f in sorted((EXT / "splits" / ds).glob("*.txt")):
        if "IP_LAP" in f.name:
            continue
        keys = [k for k in f.read_text().split("\n") if k]
        (LISTS / ds / f.name).write_text("\n".join(str(ROOT / k) for k in keys) + "\n")
        for i in range(0, len(keys), 500):
            jobs.append((ds, keys[i:i + 500]))
    with Pool(procs) as p:
        total = sum(p.imap_unordered(work, jobs))
    print(f"{ds}: wrote {total} frames", flush=True)


if __name__ == "__main__":
    main()

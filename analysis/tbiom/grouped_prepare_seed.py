#!/usr/bin/env python3
"""Second-detector-seed exports for the grouped-fusion confirmation (Stage 3 of the grouped-fusion brief).

    python analysis/tbiom/grouped_prepare_seed.py

The deployed reference exists at one detector seed only, and no second warm+gate checkpoint was ever trained, so the
seed check uses the one three-branch family that WAS trained at several seeds: SBI arm B, seeds 42 and 1337, each at
its own v3.1-selected epoch, paired with the V-JEPA 2 temporal branch at the matching seed. Within that family the
seed is the only thing that differs, which is what a seed check needs; the deployed warm+gate recipe is a different
recipe and is not mixed into the pair.

Writes per-generator configs for the `sbiarmB` config family and the export job list for both seeds.
"""
from __future__ import annotations
import json, re
from pathlib import Path

EXT = Path("/data/umar/Repos/DISCERN_Ext")
CFG, SPL = EXT / "eval_adaptation/configs", "eval_adaptation/data/splits/DF40"
LOG = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/grouped")
S23 = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/stage23")
D40 = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40")
SEEDS = {"armB42": "sbiarmB", "armB1337": "sbiarmBs1337"}          # export tag -> selection record
TEMPLATE = "sbiarmB_CDFv2.yaml"


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    ds = json.load(open(D40 / "meta.json"))
    fake_of = {}
    for name in ds:
        m = re.match(r"DF40_(.+)_(cdf|ff)$", name)
        fake_of[name] = (f"{m.group(1)}-{m.group(2)}", f"DF40-{m.group(2)}-real") if m else ("deepfacelab", "deepfacelab-real")
    jobs, n = [], 0
    for tag, arm in SEEDS.items():
        sel = json.load(open(S23 / f"{arm}_selection_v3.json"))["selected"]
        ck = sel["checkpoint"]
        print(f"{tag}: {arm} epoch {sel['epoch']} -> {Path(ck).name}")
        for name in ds:
            fake, real = fake_of[name]
            out = CFG / f"{arm}_{name}.yaml"
            if not out.exists():
                t = (CFG / TEMPLATE).read_text()
                t = re.sub(r"^run_name: .*$", f"run_name: {arm}_{name}", t, flags=re.M)
                t = re.sub(r"^tst_h5_path: .*$", "tst_h5_path: eval_adaptation/data/h5/DF40.h5", t, flags=re.M)
                files = "\n".join(f"  - {SPL}/{s}.txt" for s in (real, fake))
                t = re.sub(r"^tst_files:\n(?:  \S+:\n)(?:  - .*\n)+", f"tst_files:\n  {name}:\n{files}\n", t, flags=re.M)
                out.write_text(t); n += 1
            jobs.append(f"{ck}|{arm}_{name}.yaml|{name}|{S23}/{tag}_{name}.csv|")
    (LOG / "jobs_seed.txt").write_text("\n".join(jobs) + "\n")
    json.dump({t: json.load(open(S23 / f"{a}_selection_v3.json"))["selected"] for t, a in SEEDS.items()},
              open(LOG / "seed_checkpoints.json", "w"), indent=1)
    print(f"{n} configs written, {len(jobs)} export jobs -> {LOG / 'jobs_seed.txt'}")


if __name__ == "__main__":
    main()

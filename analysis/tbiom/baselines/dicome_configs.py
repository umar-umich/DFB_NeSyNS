#!/usr/bin/env python3
"""Eval configs so the DiCoME baseline can be scored on the sets it was never run on.

    python analysis/tbiom/baselines/dicome_configs.py

Table 1 carried `in progress` for `DiCoME (trained, their repo)` on seven of twelve sets. The
DiCoME repo has the h5 AND the splits for every one of them -- only the config was missing, and
a config is a template with two fields changed. Nothing in the repo's code is touched: the files
land in `eval_adaptation/configs/ours_new/` and the runs under the existing `runs/dicome_trained`.

The template is the CDFv2 config the published five were produced from, so the model, the
backbone, the batch size and the precision are whatever those rows already used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

DICOME = Path("/data/umar/Repos/DiCoME")
CFG = DICOME / "eval_adaptation/configs"
OUT = CFG / "ours_new"
MISSING = ["FFIW", "KoDF", "IDForge", "DFDM", "UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0"]


def main() -> int:
    base = yaml.safe_load((CFG / "CDFv2.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for ds in [a for a in sys.argv[1:]] or MISSING:
        splits = sorted((DICOME / "eval_adaptation/data/splits" / ds).glob("*.txt"))
        h5 = DICOME / "eval_adaptation/data/h5" / f"{ds}.h5"
        if not splits or not h5.exists():
            print(f"  {ds}: SKIPPED ({len(splits)} splits, h5 {h5.exists()})")
            continue
        c = dict(base)
        c["run_name"] = f"dicome_trained_{ds}"
        c["run_dir"] = "runs/dicome_trained"
        c["tst_h5_path"] = str(h5.relative_to(DICOME))
        c["tst_files"] = {ds: [str(p.relative_to(DICOME)) for p in splits]}
        f = OUT / f"{ds}.yaml"
        f.write_text(yaml.safe_dump(c, sort_keys=False))
        made.append(ds)
        print(f"  {ds}: {len(splits)} split files -> {f.relative_to(DICOME)}")
    print(f"{len(made)} configs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

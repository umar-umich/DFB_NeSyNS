#!/usr/bin/env python3
"""Combined DF40 configs for DiCoME: one row per generator, both source halves together.

    python analysis/tbiom/baselines/dicome_df40_combined.py

`eval_df40.py` reports AUROC per row and keeps no per-video scores, so a generator's pooled
number cannot be recovered from its two per-half rows -- and averaging those two AUROCs would
make this row a different quantity from every other row in the table, which are all pooled from
videos. A config is only a file list, so the pooled number is obtained directly: list BOTH
halves' fake splits and BOTH real pools in one row.

It is also cheaper. 21 combined rows cover the same frames as 41 per-half rows without the
per-row model setup, and nothing in DiCoME's code is touched -- the configs land in their own
subdirectory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

DICOME = Path("/data/umar/Repos/DiCoME")
CFG = DICOME / "eval_adaptation/configs/DF40"
OUT = CFG / "ours_combined"
SPL = "eval_adaptation/data/splits/DF40"
META = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40/meta.json")


def main() -> int:
    meta = json.loads(META.read_text())
    gens = {}
    for name, m in meta.items():
        g = m["method"]
        n = name[len("DF40_"):]
        half = n.rsplit("_", 1)[1] if n.rsplit("_", 1)[-1] in ("cdf", "ff") else None
        gens.setdefault(g, []).append(half)

    base = yaml.safe_load((CFG / "blendface-cdf.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for g, halves in sorted(gens.items()):
        files, missing = [], []
        for h in sorted(x for x in halves if x):
            for f in (f"{g}-{h}.txt", f"DF40-{h}-real.txt"):
                (files if (DICOME / SPL / f).exists() else missing).append(f"{SPL}/{f}")
        if None in halves:                       # DeepFaceLab: one source, its own real pool
            for f in (f"{g}.txt", f"{g}-real.txt"):
                (files if (DICOME / SPL / f).exists() else missing).append(f"{SPL}/{f}")
        files = sorted(set(files))               # one real pool listed once
        if missing or not files:
            print(f"  {g}: SKIPPED ({len(missing)} splits missing)")
            continue
        c = dict(base)
        c["run_name"] = f"df40c_{g}"
        c["tst_files"] = {f"DF40-{g}": files}
        (OUT / f"{g}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        made.append(g)
        print(f"  {g}: {len(files)} split files")
    print(f"{len(made)} combined configs -> {OUT}")
    print(",".join(f"ours_combined/{g}" for g in made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

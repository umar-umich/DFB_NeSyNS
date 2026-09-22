#!/usr/bin/env python3
"""Which DF40 exports are still missing, and release the claims of the ones nobody is writing.

    python analysis/tbiom/df40_sweep.py count     # print the number missing, touch nothing
    python analysis/tbiom/df40_sweep.py release   # also release claims with no live writer, then print the number

A lane renames its output only when the export finishes, so "missing" and "in flight" look the same on disk. The
live-writer set is read from the process table, which is the only thing that tells the two apart; releasing a claim
whose job is still running is what creates two writers on one temporary file.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

L = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/df40")


def live():
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.split("\n")
    out = set()
    for ln in ps:
        if "export_views.py" in ln and "--out " in ln and "ps -eo" not in ln:
            out.add(Path(ln.split("--out ")[1].split()[0]).name.replace(".tmp.csv", ""))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "count"
    running, n = live(), 0
    for q in ("jobs_all", "jobs_fr"):
        for ln in (L / f"{q}.txt").read_text().split("\n"):
            if not ln.strip():
                continue
            out = Path(ln.split("|")[3])
            if out.exists():
                continue
            n += 1
            if mode != "release" or out.stem in running:
                continue
            d = L / f"{q}.claims" / out.stem
            if d.exists():
                d.rmdir()
            t = out.with_suffix(".tmp.csv")
            if t.exists():
                t.unlink()
    print(n)


if __name__ == "__main__":
    main()

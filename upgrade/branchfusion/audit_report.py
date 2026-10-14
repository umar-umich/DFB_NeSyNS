#!/usr/bin/env python3
"""Verify the assembled report, rather than claiming it is complete.

    <discern_ext python> upgrade/branchfusion/audit_report.py [report.md]

I told UMAR twice that the tables were done while `in progress` cells were still sitting in them,
because I checked the tables I had just touched instead of the file I handed over. This parses
every markdown table in the report and reports, per table: empty cells by row and column, and any
duplicate table. It exits non-zero when anything is missing, so a chain cannot report success
over an incomplete file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
DEFAULT = ROOT / "tbiom/reports/branch_fusion/PROPOSAL_TABLES_2.md"
#: A cell is empty when it is a placeholder, NOT when prose inside it happens to contain an em
#: dash -- "DF40 cross-manipulation — the structure of ..." is a filled cell.
def is_empty(c: str) -> bool:
    c = c.strip().strip("*` ")
    # A truly blank cell is a deliberate one -- a "mean over 8" row has no video count. The
    # generators write `*in progress*` or an em dash for data that is MISSING, never "".
    return ("in progress" in c) or c in ("-", "—", "n/a", "N/A", "TODO")


def tables(lines):
    """Yield (section, header_line_no, header_cells, rows) for each markdown table."""
    sec = None
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#"):
            sec = l.lstrip("# ").strip()
        is_sep = (l.startswith("|") and set(l.replace("|", "").strip()) <= set("-: "))
        if is_sep and i > 0 and lines[i - 1].startswith("|"):
            hdr = [c.strip() for c in lines[i - 1].split("|")[1:-1]]
            rows, j = [], i + 1
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].split("|")[1:-1]])
                j += 1
            yield sec, i, hdr, rows
            i = j
            continue
        i += 1


def main() -> int:
    f = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    lines = f.read_text().splitlines()
    gaps, seen, dups = [], {}, []
    n_tables = n_cells = 0
    for sec, ln, hdr, rows in tables(lines):
        n_tables += 1
        sig = (tuple(hdr), tuple(tuple(r) for r in rows))
        if sig in seen and len(rows) > 2:
            dups.append((sec, ln, seen[sig]))
        else:
            seen[sig] = (sec, ln)
        for r in rows:
            for k, c in enumerate(r):
                n_cells += 1
                if is_empty(c) and k > 0:
                    gaps.append((sec, ln, r[0], hdr[k] if k < len(hdr) else f"col{k}"))

    print(f"{f}")
    print(f"{n_tables} tables, {n_cells} cells")
    if dups:
        print(f"\nDUPLICATE TABLES ({len(dups)}):")
        for sec, ln, (osec, oln) in dups:
            print(f"  line {ln}: '{sec}' repeats '{osec}' (line {oln})")
    if gaps:
        by = {}
        for sec, ln, row, col in gaps:
            by.setdefault((sec, ln), []).append((row, col))
        print(f"\nEMPTY CELLS ({len(gaps)}):")
        for (sec, ln), items in by.items():
            rows = sorted({r for r, _ in items})
            cols = sorted({c for _, c in items})
            print(f"  '{sec}' (line {ln}): {len(items)} cells; rows {rows}; columns {cols}")
    else:
        print("\nNo empty cells.")
    bad = len(gaps) + len(dups)
    print(f"\nVERDICT: {'INCOMPLETE' if bad else 'COMPLETE'} "
          f"({len(gaps)} empty cells, {len(dups)} duplicate tables)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

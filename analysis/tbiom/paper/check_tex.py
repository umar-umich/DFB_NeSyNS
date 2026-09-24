#!/usr/bin/env python3
"""Structural checks on the manuscript, since no LaTeX toolchain is available on this machine.

    python analysis/tbiom/paper/check_tex.py

Catches what a compile would catch cheaply: unbalanced environments, table column-count mismatches, \\input files
that do not exist, \\cite keys with no bib entry, \\ref labels with no \\label, and unescaped percent signs.
It is not a substitute for compiling; it is a substitute for shipping something obviously broken.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

P = Path("tbiom/paper")


def split_spec(s):
    """Split a tabular's column spec from its body, honouring nested braces in the spec."""
    assert s.lstrip().startswith("{")
    i = s.index("{"); depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], s[j + 1:]
    raise ValueError("unbalanced column spec")


def main():
    main_tex = (P / "main.tex").read_text()
    bodies = {"main.tex": main_tex}
    for m in re.finditer(r"\\input\{([^}]+)\}", main_tex):
        f = P / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        if not f.exists():
            print(f"FAIL missing \\input target: {f}")
            return 1
        bodies[m.group(1)] = f.read_text()
    bad = 0

    # environments balanced, per file
    for name, b in bodies.items():
        for env in set(re.findall(r"\\begin\{(\w+\*?)\}", b)):
            o = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", b))
            c = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", b))
            if o != c:
                print(f"FAIL {name}: {env} begin={o} end={c}"); bad += 1

    # tabular column counts
    for name, b in bodies.items():
        for m in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", b, re.S):
            spec, body = split_spec(m.group(1))
            ncol = len(re.findall(r"[lcrp]", re.sub(r"@\{[^{}]*\}|\|", "", spec)))
            for ln in body.split("\n"):
                s = ln.strip()
                if not s.endswith(r"\\") or s.startswith("%") or "multicolumn" in s or "cmidrule" in s:
                    continue
                cells = len(re.split(r"(?<!\\)&", s[:-2])) if s != r"\\" else 0
                if cells and cells != ncol:
                    print(f"FAIL {name}: expected {ncol} columns, found {cells}: {s[:70]}"); bad += 1

    # cites and refs
    keys = set(re.findall(r"@\w+\{([^,]+),", (P / "refs.bib").read_text()))
    cited = {k.strip() for b in bodies.values() for m in re.finditer(r"\\cite\{([^}]*)\}", b)
             for k in m.group(1).split(",")}
    for k in sorted(cited - keys):
        print(f"FAIL \\cite{{{k}}} has no bib entry"); bad += 1
    unused = sorted(keys - cited)
    labels = {l for b in bodies.values() for l in re.findall(r"\\label\{([^}]*)\}", b)}
    refs = {r for b in bodies.values() for m in re.finditer(r"\\(?:ref|autoref)\{([^}]*)\}", b) for r in [m.group(1)]}
    for r in sorted(refs - labels):
        print(f"FAIL \\ref{{{r}}} has no \\label"); bad += 1

    # unescaped % outside comments
    for name, b in bodies.items():
        for i, ln in enumerate(b.split("\n"), 1):
            code = ln.split("%")[0]
            if re.search(r"(?<!\\)%", ln) and code.strip() and not ln.lstrip().startswith("%"):
                if re.search(r"\d\s*%", code):
                    print(f"WARN {name}:{i} possible unescaped %: {ln.strip()[:60]}")

    fills = [(n, i) for n, b in bodies.items() for i, ln in enumerate(b.split("\n"), 1) if "FILL" in ln]
    print(f"\n{len(bodies)} files, {len(cited)} distinct citations, {len(labels)} labels, {len(refs)} refs")
    print(f"unused bib entries: {unused if unused else 'none'}")
    print(f"FILL markers in tex: {len(fills)}" + (f" -> {fills}" if fills else ""))
    print("RESULT:", "FAIL" if bad else "structural checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

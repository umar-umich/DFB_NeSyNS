# CURRENT_PAPER.pdf — compile status

**Compiled successfully from this snapshot.** 12 pages, 28 references, no unresolved citations.

| item | value |
|---|---|
| engine | `tectonic` 0.17.0 (installed into an isolated conda env `texbuild` for this purpose) |
| command | `tectonic -k --outdir <dir> main.tex` |
| source | `tbiom/paper/main.tex` at the commit recorded in `HANDOFF_MANIFEST.md` |
| pages | 12 (US letter, two-column IEEEtran journal) |
| bibliography | 28 `\bibitem`s in `main.bbl`; zero `[?]` in the output |
| figures | 2 external PDFs + 1 inline TikZ, all rendered |

`pdflatex` was not available on the machine; `tectonic` produces the same output for this document and runs BibTeX
internally, so no separate `bibtex` invocation is needed.

## Layout issues the compile revealed — NOT fixed, by instruction

This handoff is packaging only, so the manuscript was not modified to address these. They are listed here in
descending severity so a revision pass can act on them.

**Overfull `\hbox` — tables wider than their column.** Ten of the sixteen tables overflow. The generated tables must
be fixed in `analysis/tbiom/paper/build_tables.py`, not by hand.

| table | overflow | note |
|---|---:|---|
| `tab_configs` | 193.9 pt | configuration names are long; abbreviate or move to a `table*` |
| `tab_conflict` | 118.3 pt | case labels are long sentences |
| `tab_efficiency` | 70.1 pt | component descriptions are long |
| `tab_reliability` | 61.3 pt | six numeric columns |
| `tab_slices` | 60.5 pt | six columns |
| `tab_perview` | 42.7 pt | six columns |
| `tab_viewrole` | 30.0 pt | |
| `tab_views` | 14.3 pt | |
| `tab_crossdataset` | 12.9 pt | already a `table*`; 12 columns at `\small` |
| `tab_operators` | 11.0 pt | |
| `tab_uncertainty` | 1.5 pt | marginal |

**Overfull `\hbox` in prose:** `main.tex` line 140 (67.1 pt, a long paragraph in the method), and lines 200 and 207
(25.9 pt and 39.5 pt, both inside the `algorithmic` block — the pseudocode lines are too long for one column).

**Underfull boxes:** 157 in total, dominated by `tab_extension`, whose `p{}` columns contain full sentences and so
stretch badly. Cosmetic, but that table is the worst offender.

**Benign warnings, no action needed:** `LaTeX Font Warning: Font shape TU/ptm/... undefined` (hyperref's Unicode
encoding against IEEEtran's Times; adding `\usepackage[T1]{fontenc}` silences it), and `Object @table.N already
defined` from hyperref across multi-pass runs.
